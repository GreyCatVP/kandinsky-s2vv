"""Kandinsky S2V 1.0 pipeline: reference frame + speech → avatar video."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..core.algo.audio_io import (
    audio_duration,
    chunk_duration_sec,
    num_chunks_for,
    pad_with_silence,
    read_audio,
)
from ..core.algo.chunks import split_audio_embeds
from ..core.algo.denoise_loop import ChunkContext, TextContext, denoise_chunk
from ..core.algo.mux import mux_video_audio
from ..core.algo.piflow_sampler import piflow_denoise_chunk
from ..core.algo.postprocess import decode_latent
from ..core.algo.prepare_latents import (
    crop_reference_latent,
    encode_reference_image,
    init_memory,
    init_noise,
    roll_memory,
    select_best_resolution,
)
from ..core.algo.prepare_ropes import (
    audio_rope_pos,
    memory_rope_pos,
    ref_rope_pos,
    text_rope_pos,
    visual_rope_pos,
)
from ..core.parallel import SequenceParallel
from .config import S2VPipelineConfig

logger = logging.getLogger(__name__)

TEMPORAL_COMPRESSION_RATIO = 4


@dataclass
class S2VPipelineOutput:
    frames: Tensor  # (3, T, H, W) uint8
    path: Path | None = None


class KandinskyS2VPipeline:
    def __init__(  # noqa: PLR0913
        self,
        cfg: S2VPipelineConfig,
        dit,
        vae,
        embedder,
        device,
        sequence_parallel: SequenceParallel | None = None,
    ):
        self.cfg = cfg
        self.dit = dit
        self.vae = vae
        self.embedder = embedder
        self.device = torch.device(device)
        self.offload = cfg.offload.strategy == "module"
        # Sequence parallel: every rank runs this whole loop on identical inputs
        # (rank 0's are broadcast) and only the DiT forward is split; rank 0 saves.
        self.sequence_parallel = sequence_parallel

    @property
    def is_main(self) -> bool:
        return self.sequence_parallel is None or self.sequence_parallel.rank == 0

    def _sync(self, *tensors: Tensor) -> None:
        if self.sequence_parallel is not None:
            self.sequence_parallel.broadcast(*tensors)

    @property
    def _memory_coefs(self):
        dit_cfg = self.cfg.dit
        return (dit_cfg.patch_proj1x_coef, dit_cfg.patch_proj2x_coef, dit_cfg.patch_proj4x_coef)

    @contextlib.contextmanager
    def _staged(self, *modules):
        """Move modules onto the GPU for one stage, then park them back on CPU.

        Note the cost: the DiT and the VAE cross the bus once per chunk each, so
        a multi-chunk run with ``offload: module`` pays ~38 GiB of host-to-device
        traffic per chunk on Pro. The stage order inside the chunk loop
        (dit → denoise → vae → decode) is what keeps peak VRAM within one card.
        """
        if not self.offload:
            yield
            return
        for module in modules:
            module.to(self.device)
        try:
            yield
        finally:
            for module in modules:
                module.to("cpu")
            torch.cuda.empty_cache()

    def _encode_text_audio(self, text, negative_text, audio_path, end_audio):
        with self._staged(self.embedder):
            text_embeds, pooled, audio_embeds = self.embedder.encode(text, audio_path, 0.0, end_audio)
            null_text_embeds, null_pooled, _ = self.embedder.encode(negative_text)
        if audio_embeds is None:
            raise ValueError("S2V requires an audio track")
        return (
            text_embeds.to(self.device),
            pooled.to(self.device),
            audio_embeds.to(self.device),
            null_text_embeds.to(self.device),
            null_pooled.to(self.device),
        )

    def __call__(  # noqa: PLR0913, PLR0915
        self,
        audio: str | Path,
        image: str | Path,
        text: str | None = None,
        negative_text: str | None = None,
        num_steps: int | None = None,
        guidance_weight: float | None = None,
        scheduler_scale: float | None = None,
        seed: int | None = None,
        chunks: int | None = None,
        save_path: str | Path | None = None,
        show_progress: bool = False,
    ) -> S2VPipelineOutput:
        gen = self.cfg.generation
        dit_cfg = self.cfg.dit
        text = gen.prompt if text is None else text
        negative_text = gen.negative_prompt if negative_text is None else negative_text
        num_steps = gen.num_steps if num_steps is None else num_steps
        guidance_weight = gen.guidance_weight if guidance_weight is None else guidance_weight
        if self.cfg.piflow.enabled and scheduler_scale is not None and self.cfg.piflow.shift is not None:
            raise ValueError("scheduler_scale has no effect: the config sets piflow.shift, which pi-Flow uses instead")
        scheduler_scale = gen.scheduler_scale if scheduler_scale is None else scheduler_scale
        seed = gen.seed if seed is None else seed
        if self.cfg.piflow.enabled and guidance_weight != 1.0:
            raise ValueError("pi-Flow expects guidance_weight=1.0 (CFG is distilled into the student)")
        # Euler takes num_steps - 1 updates, so one step would decode the initial noise.
        min_steps = 1 if self.cfg.piflow.enabled else 2
        if num_steps < min_steps:
            raise ValueError(f"num_steps must be >= {min_steps}, got {num_steps}")
        if chunks is not None and chunks < 1:
            raise ValueError(f"chunks must be >= 1, got {chunks}")

        duration = gen.latent_frames
        chunk_seconds = chunk_duration_sec(duration, gen.fps, TEMPORAL_COMPRESSION_RATIO)
        track_seconds = audio_duration(audio)
        needed = num_chunks_for(track_seconds, chunk_seconds)
        if chunks is not None and chunks > needed:
            # The video is trimmed to the track, so these are computed and then dropped.
            logger.warning(
                "%d chunk(s) past the end of the speech track are generated and then trimmed; pass chunks=%d to "
                "save the time",
                chunks - needed,
                needed,
            )
        chunks = needed if chunks is None else chunks
        target_seconds = chunks * chunk_seconds
        extractor = self.embedder.processor.feature_extractor
        max_seconds = extractor.n_samples / extractor.sampling_rate
        if target_seconds > max_seconds + 0.01:
            raise ValueError(
                f"{chunks} chunks need {target_seconds:g} s of speech, but the Qwen2.5-Omni feature extractor "
                f"truncates at {max_seconds:g} s; trim the track or pass fewer chunks"
            )

        resolution = select_best_resolution(image, gen.allowed_resolutions)
        with self._staged(self.vae):
            # Its own generator, so the noise stream below stays what ``seed`` gave before.
            ref_generator = torch.Generator(device=self.device).manual_seed(seed)
            ref = encode_reference_image(image, resolution, self.vae, self.device, ref_generator)
        ref = crop_reference_latent(ref, dit_cfg.patch_size).to(self.device)
        self._sync(ref)
        latent_height, latent_width = ref.shape[1], ref.shape[2]

        padded_audio = None
        try:
            padded_audio = pad_with_silence(audio, target_seconds)
            audio_for_encode = padded_audio or Path(audio)
            (
                text_embeds,
                pooled,
                audio_embeds,
                null_text_embeds,
                null_pooled,
            ) = self._encode_text_audio(text, negative_text, audio_for_encode, target_seconds)
        finally:
            if padded_audio is not None:
                padded_audio.unlink(missing_ok=True)
        self._sync(text_embeds, pooled, audio_embeds, null_text_embeds, null_pooled)

        audio_per_chunk = split_audio_embeds(audio_embeds, chunks)

        text_ctx = TextContext(
            text_embeds=text_embeds.to(torch.bfloat16),
            pooled_embed=pooled.to(torch.bfloat16),
            rope_pos=text_rope_pos(text_embeds.shape[0], self.device),
        )
        null_ctx = TextContext(
            text_embeds=null_text_embeds.to(torch.bfloat16),
            pooled_embed=null_pooled.to(torch.bfloat16),
            rope_pos=text_rope_pos(null_text_embeds.shape[0], self.device),
        )

        memory = init_memory(
            dit_cfg.frame_buckets,
            self._memory_coefs,
            dit_cfg.patch_size,
            latent_height,
            latent_width,
            dit_cfg.in_visual_dim,
            self.device,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)

        vis_pos = visual_rope_pos(duration, latent_height, latent_width, dit_cfg.patch_size, self.device)
        ref_pos = ref_rope_pos(
            dit_cfg.index_rope_ref_image,
            latent_height,
            latent_width,
            dit_cfg.patch_size,
            self.device,
        )

        # Euler runs num_steps-1 updates; pi-Flow runs one per segment.
        steps_per_chunk = num_steps if self.cfg.piflow.enabled else num_steps - 1
        progress = self._progress_bar(chunks, steps_per_chunk) if show_progress and self.is_main else None
        decoded_chunks: list[Tensor] = []

        for index in range(chunks):
            latent = init_noise(
                duration,
                latent_height,
                latent_width,
                dit_cfg.in_visual_dim,
                dit_cfg.patch_size,
                generator,
                self.device,
            )
            self._sync(latent)
            ctx = ChunkContext(
                ref=ref,
                memory=memory,
                audio_embeds=audio_per_chunk[index].to(torch.bfloat16),
                visual_rope_pos=vis_pos,
                ref_rope_pos=ref_pos,
                memory_rope_pos=memory_rope_pos(
                    memory,
                    dit_cfg.frame_buckets,
                    self._memory_coefs,
                    dit_cfg.patch_size,
                    self.device,
                ),
                audio_rope_pos=audio_rope_pos(audio_per_chunk[index].shape[0], duration, self.device),
                duration=duration,
                scale_factor=gen.scale_factor,
            )

            with self._staged(self.dit), torch.autocast("cuda", dtype=torch.bfloat16):
                if self.cfg.piflow.enabled:
                    latent = piflow_denoise_chunk(
                        latent,
                        self.dit,
                        ctx,
                        text_ctx,
                        null_ctx,
                        num_steps=num_steps,
                        scheduler_scale=scheduler_scale,
                        guidance_weight=guidance_weight,
                        num_policy_substeps=self.cfg.piflow.num_policy_substeps,
                        final_step_size_scale=self.cfg.piflow.final_step_size_scale,
                        shift=self.cfg.piflow.shift,
                        eps=self.cfg.piflow.eps,
                        progress_callback=progress,
                    )
                else:
                    latent = denoise_chunk(
                        latent,
                        self.dit,
                        ctx,
                        text_ctx,
                        null_ctx,
                        num_steps=num_steps,
                        guidance_weight=guidance_weight,
                        scheduler_scale=scheduler_scale,
                        progress_callback=progress,
                    )

            memory = roll_memory(latent, dit_cfg.frame_buckets, self._memory_coefs, dit_cfg.patch_size)
            with self._staged(self.vae), torch.autocast("cuda", dtype=torch.bfloat16):
                decoded_chunks.append(decode_latent(latent, self.vae).cpu())

        frames = torch.cat(decoded_chunks, dim=1)
        frames = self._trim(frames, track_seconds, gen.fps)

        path = None
        if save_path is not None and self.is_main:
            samples, sample_rate = read_audio(audio)
            samples = samples[: int(frames.shape[1] / gen.fps * sample_rate)]
            path = mux_video_audio(frames, samples, save_path, fps=gen.fps, audio_sample_rate=sample_rate)
        return S2VPipelineOutput(frames=frames, path=path)

    @staticmethod
    def _trim(frames: Tensor, track_seconds: float, fps: int) -> Tensor:
        """Drop the frames generated over the silence padding."""
        keep = min(frames.shape[1], int(track_seconds * fps))
        return frames[:, :keep]

    @staticmethod
    def _progress_bar(chunks: int, steps_per_chunk: int):
        from tqdm import tqdm  # noqa: PLC0415

        bar = tqdm(total=chunks * steps_per_chunk, desc="s2v")
        return bar.update
