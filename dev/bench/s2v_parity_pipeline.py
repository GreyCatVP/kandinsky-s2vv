"""GPU parity check: this pipeline against the kandinsky-5 reference pipeline.

[dev/bench/s2v_parity_cpu.py](s2v_parity_cpu.py) compares a single DiT forward.
This one compares the whole generation loop — sampler schedule, chunking,
FramePack memory carry-over, RoPE positions, audio slicing — by running the
reference `generate_sample` and this pipeline side by side and diffing the
latents *before* the VAE decode. Decoding would hide small differences and add
its own noise.

Everything the two sides could legitimately differ on is pinned so that only the
generation loop is under test:

* both DiTs are built from the **same** safetensors file — parameter names match
  the training checkpoints on both sides, so no conversion is involved;
* text, pooled and audio embeddings are computed once and handed to both;
* the reference latent is computed once (the VAE posterior is sampled, so
  recomputing it would differ by construction);
* both draw their noise from `torch.Generator` seeded identically.

Requires a GPU and the converted weights. The reference imports `flash_attn`
(FA2) at module level but prefers FA3 at runtime, so FA2 is stubbed for the
import and the real work goes through FA3.

Usage:
    KANDINSKY5_REPO=/path/to/kandinsky-5 \
        uv run python dev/bench/s2v_parity_pipeline.py [--config ...] [--chunks N]
"""

from __future__ import annotations

import argparse
import importlib.machinery
import os
import sys
import types
from pathlib import Path

import torch

_DEFAULT_REFERENCE_REPO = Path(__file__).resolve().parents[2] / "dev" / "ref" / "kandinsky-5"
REFERENCE_REPO = Path(os.environ.get("KANDINSKY5_REPO", _DEFAULT_REFERENCE_REPO))
# Distilled configs are compared against the distillation fork: it owns the
# pi-Flow sampler the students were trained with. Its `generate_sample` takes a
# `piflow` dict and dispatches to `generate_piflow`.
PIFLOW_REPO = Path(os.environ.get("KANDINSKY5_PIFLOW_REPO", REFERENCE_REPO))

# Exact equality is not achievable here: the two sides run different attention
# kernels (dense FA3 against varlen FA3, plus padded SDPA for the audio adapter),
# and bf16 accumulates differently depending on reduction order. A single DiT
# forward already differs by ~3e-3 relative for that reason alone, while the CPU
# check — same kernels on both sides — matches exactly.
#
# The criterion is the relative L2 norm, not the worst element: on a 3M-element
# latent a max-based bound is dominated by one outlier and grows as the latent
# shrinks during denoising, which says nothing about agreement.
TOLERANCE = 2e-2

DEFAULT_CONFIG = "src/kandinsky_s2v/configs/s2v_lite.yaml"
DEFAULT_AUDIO = "/home/jovyan/semenov/projects/kandinsky-5/tests/sbs_RU_2402026_PROBLEMED/100__TEST_10s.wav"
DEFAULT_IMAGE = "/home/jovyan/semenov/projects/kandinsky-5/tests/sbs_RU_2402026_PROBLEMED/100__TEST_frame.png"


def _install_stubs() -> None:
    """Satisfy the reference's module-level imports without installing FA2."""
    if "flash_attn" not in sys.modules:
        flash = types.ModuleType("flash_attn")
        # The reference takes FA3 for self-attention when it is installed, but its
        # cross-attention calls FA2's varlen unconditionally. FA2 ships no wheels,
        # so that entry point is served by FA3, which takes the same maths with
        # different argument names.
        flash.flash_attn_varlen_func = _fa2_varlen_via_fa3
        for name in ("flash_attn_varlen_qkvpacked_func", "flash_attn_func"):
            setattr(flash, name, None)  # unreachable while FA3 is present
        # diffusers probes installed packages with importlib.util.find_spec, which
        # raises on a module whose __spec__ is None.
        flash.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
        sys.modules["flash_attn"] = flash

    logger = types.ModuleType("kandinsky.model.logger")

    class ModelLogger:
        @staticmethod
        def log_attention(target):
            return target

        @staticmethod
        def log_transformer_block(target):
            return target

        @staticmethod
        def log_transformer(target):
            return target

    logger.ModelLogger = ModelLogger
    sys.modules["kandinsky.model.logger"] = logger


class Fa3ReturnsPair:
    """Make FA3 varlen return ``(out, lse)`` while the reference runs.

    The reference unpacks a pair; this FA3 build returns the tensor alone. The
    patch is scoped to the reference run because transformers calls the same
    symbol with its own conventions — patching it globally breaks the encoder.
    Only the call wrapper changes; the kernel and its arguments do not, so both
    sides still run the same attention.
    """

    def __enter__(self):
        import flash_attn_interface  # noqa: PLC0415

        self.module = flash_attn_interface
        self.original = flash_attn_interface.flash_attn_varlen_func

        def varlen(*args, **kwargs):
            out = self.original(*args, **kwargs)
            return out if isinstance(out, tuple) else (out, None)

        flash_attn_interface.flash_attn_varlen_func = varlen
        return self

    def __exit__(self, *exc):
        self.module.flash_attn_varlen_func = self.original
        return False


def _fa2_varlen_via_fa3(  # noqa: PLR0913
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs
):
    """FA2's varlen signature, served by FA3; returns FA2's 3-tuple."""
    del args, kwargs
    import flash_attn_interface  # noqa: PLC0415

    out = flash_attn_interface.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
    )
    out = out[0] if isinstance(out, tuple) else out
    return out, None, None


def reference_dit_params(cfg, *, flat_attention_params: bool = False) -> dict:
    """Our DiTConfig in the argument names the reference constructor expects.

    ``flat_attention_params`` selects the distillation fork's older shape: it
    reads ``model.attention_params.type`` directly, while the main repository
    indexes by resolution bucket first.
    """
    d = cfg.dit
    return dict(
        in_visual_dim=d.in_visual_dim,
        out_visual_dim=d.out_visual_dim,
        in_text_dim=d.in_text_dim,
        in_text_dim2=d.in_text_dim2,
        in_audio_dim=d.in_audio_dim,
        time_dim=d.time_dim,
        patch_size=d.patch_size,
        frame_buckets=d.frame_buckets,
        enable_framepack=d.enable_framepack,
        index_rope_ref_image=d.index_rope_ref_image,
        patch_proj1x_coef=d.patch_proj1x_coef,
        patch_proj2x_coef=d.patch_proj2x_coef,
        patch_proj4x_coef=d.patch_proj4x_coef,
        model_dim=d.model_dim,
        ff_dim=d.ff_dim,
        num_text_blocks=d.num_text_blocks,
        num_audio_blocks=d.num_audio_blocks,
        num_visual_blocks=d.num_visual_blocks,
        axes_dims=d.axes_dims,
        audio_inject_layers=list(d.audio_inject_layers),
        audio_adaln=False,
        # This port never uses NABLA (it applies to the 1024 bucket, which S2V
        # does not use), so the reference is pinned to dense attention in both
        # buckets — otherwise the two sides would run different kernels.
        attention_params=(
            types.SimpleNamespace(type="flash")
            if flat_attention_params
            else {
                512: types.SimpleNamespace(type="flash"),
                1024: types.SimpleNamespace(type="flash"),
            }
        ),
    )


class FrozenEmbedder:
    """Hands both sides the same embeddings, in the reference's return shape."""

    def __init__(self, text, pooled, audio, null_text, null_pooled, device):  # noqa: PLR0913
        self.text, self.pooled, self.audio = text, pooled, audio
        self.null_text, self.null_pooled = null_text, null_pooled
        self.device = device

    def encode(self, texts=None, audios=None, start_audio=None, end_audio=None):
        del start_audio, end_audio
        is_null = audios is None
        text = self.null_text if is_null else self.text
        pooled = self.null_pooled if is_null else self.pooled
        cu_text = torch.tensor([0, text.shape[0]], dtype=torch.int32, device=self.device)
        audio = None if is_null else self.audio
        cu_audio = None if audio is None else torch.tensor([0, audio.shape[0]], dtype=torch.int32, device=self.device)
        return {"text_embeds": text, "pooled_embed": pooled}, cu_text, {"audio_embeds": audio}, cu_audio


class LatentCapture:
    """Replaces ``vae.decode`` to record what each side would have decoded."""

    def __init__(self, vae):
        self.vae = vae
        self.latents: list[torch.Tensor] = []
        self._original = vae.decode

    def __enter__(self):
        def decode(z, *args, **kwargs):
            self.latents.append(z.detach().float().cpu())
            # The callers only read `.sample`; shape does not matter for parity.
            return types.SimpleNamespace(sample=torch.zeros(1, 3, 1, 8, 8, device=z.device))

        self.vae.decode = decode
        return self

    def __exit__(self, *exc):
        self.vae.decode = self._original
        return False


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--audio", default=DEFAULT_AUDIO)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="fewer steps make the check quicker")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg_probe = None
    reference_repo = REFERENCE_REPO
    try:
        import yaml  # noqa: PLC0415

        cfg_probe = yaml.safe_load(Path(args.config).read_text())
    except OSError:
        pass
    distilled = bool((cfg_probe or {}).get("piflow", {}).get("enabled"))
    if distilled:
        reference_repo = PIFLOW_REPO

    if not (reference_repo / "kandinsky" / "metrics" / "utils_s2v.py").is_file():
        which = "KANDINSKY5_PIFLOW_REPO" if distilled else "KANDINSKY5_REPO"
        print(
            f"reference checkout not found at {reference_repo}.\n"
            f"Set {which}=/path/to/checkout to run this parity check.",
            file=sys.stderr,
        )
        return 77
    if not torch.cuda.is_available():
        print("This parity check needs a GPU: the reference attention is FA3-only.", file=sys.stderr)
        return 77

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(reference_repo))
    _install_stubs()

    from kandinsky.metrics.utils_s2v import generate_sample  # noqa: PLC0415
    from kandinsky.model.dit_s2v_tp import get_dit  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    from kandinsky_s2v.core.algo.prepare_latents import (  # noqa: PLC0415
        crop_reference_latent,
        encode_reference_image,
        select_best_resolution,
    )
    from kandinsky_s2v.core.components.omni_embedder import S2VTextAudioEmbedder  # noqa: PLC0415
    from kandinsky_s2v.pipeline.config import load_config  # noqa: PLC0415
    from kandinsky_s2v.pipeline.factory import build_vae, create_bare_dit  # noqa: PLC0415
    from kandinsky_s2v.pipeline.pipeline import KandinskyS2VPipeline  # noqa: PLC0415
    from kandinsky_s2v.pipeline.weights import resolve  # noqa: PLC0415

    device = torch.device(args.device)
    cfg = load_config(args.config)
    gen = cfg.generation
    num_steps = args.steps or gen.num_steps
    duration = gen.latent_frames

    print(f"config {Path(args.config).name} | chunks {args.chunks} | steps {num_steps}", flush=True)

    vae = build_vae(resolve(cfg.paths.vae), device)
    # The reference reads this off the VAE object rather than its config.
    if not hasattr(vae, "temporal_compression_ratio"):
        vae.temporal_compression_ratio = vae.config.temporal_compression_ratio

    resolution = select_best_resolution(args.image, gen.allowed_resolutions)
    # Seeded as the pipeline seeds it. An unseeded draw differs between processes, and the comparison is so
    # sensitive to its input that this alone spread one command's result from 6e-3 to 2e-2.
    ref_generator = torch.Generator(device=device).manual_seed(gen.seed)
    reference_latent = crop_reference_latent(
        encode_reference_image(args.image, resolution, vae, device, ref_generator), cfg.dit.patch_size
    )
    latent_h, latent_w = reference_latent.shape[1], reference_latent.shape[2]
    print(f"resolution {resolution} -> ref latent {tuple(reference_latent.shape)}", flush=True)

    embedder = S2VTextAudioEmbedder(resolve(cfg.paths.qwen_omni), resolve(cfg.paths.clip), device=device)
    chunk_seconds = ((duration - 1) * vae.temporal_compression_ratio + 1) / gen.fps
    text, pooled, audio = embedder.encode(gen.prompt, args.audio, 0.0, args.chunks * chunk_seconds)
    null_text, null_pooled, _ = embedder.encode(gen.negative_prompt)
    del embedder
    torch.cuda.empty_cache()
    print(f"embeddings: text {tuple(text.shape)} audio {tuple(audio.shape)}", flush=True)

    frozen = FrozenEmbedder(text, pooled, audio, null_text, null_pooled, device)

    # --- reference side ------------------------------------------------------
    if distilled:
        from kandinsky.model.dit_s2v_dx import DXDiTWrapper  # noqa: PLC0415

        reference_dit = DXDiTWrapper(reference_dit_params(cfg, flat_attention_params=True), cfg.piflow.n_grid)
    else:
        reference_dit = get_dit(reference_dit_params(cfg))
    # assign=True swaps in the loaded tensors, so the move to GPU has to come after.
    reference_dit.load_state_dict(load_file(resolve(cfg.paths.dit), device="cpu"), strict=True, assign=True)
    reference_dit = reference_dit.to(device).eval()

    shape = (args.chunks, duration, latent_h, latent_w, cfg.dit.in_visual_dim)
    piflow_kwargs = (
        {
            "piflow": {
                "n_grid": cfg.piflow.n_grid,
                "num_policy_substeps": cfg.piflow.num_policy_substeps,
                "final_step_size_scale": cfg.piflow.final_step_size_scale,
                "eps": cfg.piflow.eps,
            }
        }
        if distilled
        else {}
    )
    with Fa3ReturnsPair(), LatentCapture(vae) as captured_reference, torch.no_grad():
        generate_sample(
            shape,
            gen.fps,
            reference_latent,
            [gen.prompt],
            [gen.negative_prompt],
            [args.audio],
            reference_dit,
            vae,
            frozen,
            gen.scale_factor,
            num_steps,
            gen.guidance_weight,
            scheduler_scale=gen.scheduler_scale,
            seed=gen.seed,
            device=str(device),
            tp_mesh=None,
            **piflow_kwargs,
        )
    del reference_dit
    torch.cuda.empty_cache()
    print(f"reference produced {len(captured_reference.latents)} chunk latents", flush=True)

    # --- our side ------------------------------------------------------------
    pipeline = KandinskyS2VPipeline(
        cfg=cfg,
        dit=create_bare_dit(cfg, device, "flash_attention_3"),
        vae=vae,
        embedder=PipelineEmbedder(text, pooled, audio, null_text, null_pooled),
        device=device,
    )
    with FixedReferenceLatent(vae, reference_latent), LatentCapture(vae) as captured_ours:
        pipeline(
            audio=args.audio,
            image=args.image,
            chunks=args.chunks,
            num_steps=num_steps,
            save_path=None,
            show_progress=False,
        )
    print(f"ours produced {len(captured_ours.latents)} chunk latents", flush=True)

    # --- compare -------------------------------------------------------------
    if len(captured_reference.latents) != len(captured_ours.latents):
        print("MISMATCH: different chunk counts", file=sys.stderr)
        return 1

    worst = 0.0
    for index, (want, got) in enumerate(zip(captured_reference.latents, captured_ours.latents, strict=True)):
        if want.shape != got.shape:
            print(f"MISMATCH chunk {index}: shapes {tuple(want.shape)} vs {tuple(got.shape)}")
            return 1
        rel_l2 = ((want - got).norm() / want.norm()).item()
        rel_max = ((want - got).abs().max() / want.abs().max()).item()
        cosine = torch.nn.functional.cosine_similarity(want.flatten().unsqueeze(0), got.flatten().unsqueeze(0)).item()
        worst = max(worst, rel_l2)
        print(f"  chunk {index}: rel L2 {rel_l2:.3e}  rel max {rel_max:.3e}  cosine {cosine:.6f}")

    print(f"worst relative L2: {worst:.3e}  (tolerance {TOLERANCE:.0e})")
    passed = worst < TOLERANCE
    print("PIPELINE PARITY OK" if passed else "PIPELINE PARITY FAILED")
    return 0 if passed else 1


class PipelineEmbedder:
    """Our pipeline's embedder interface, returning the same frozen tensors."""

    def __init__(self, text, pooled, audio, null_text, null_pooled):
        self.text, self.pooled, self.audio = text, pooled, audio
        self.null_text, self.null_pooled = null_text, null_pooled
        # The pipeline reads the speech encoder's input limit from here; Qwen2.5-Omni's is 300 s at 16 kHz.
        self.processor = types.SimpleNamespace(
            feature_extractor=types.SimpleNamespace(n_samples=4_800_000, sampling_rate=16_000)
        )

    def to(self, device):
        del device
        return self

    def encode(self, text, audio_path=None, audio_start=0.0, audio_end=None):
        del text, audio_start, audio_end
        if audio_path is None:
            return self.null_text, self.null_pooled, None
        return self.text, self.pooled, self.audio


class FixedReferenceLatent:
    """Make the pipeline's own VAE encode return the latent we already computed.

    ``encode_reference_image`` does
    ``encode(x).latent_dist.sample(generator).permute(0, 2, 3, 4, 1) * scaling`` then
    ``squeeze(1)``, so to hand it a known ``(1, H, W, C)`` latent the posterior
    sample has to be that latent unscaled and back in ``(B, C, T, H, W)``.
    """

    def __init__(self, vae, latent):
        self.vae, self.latent = vae, latent
        self._original = vae.encode

    def __enter__(self):
        sample = (self.latent / self.vae.config.scaling_factor).unsqueeze(1)  # (1, 1, H, W, C)
        sample = sample.permute(0, 4, 1, 2, 3).contiguous()  # (1, C, 1, H, W)

        def encode(x, *args, **kwargs):
            del x, args, kwargs
            return types.SimpleNamespace(latent_dist=types.SimpleNamespace(sample=lambda generator=None: sample))

        self.vae.encode = encode
        return self

    def __exit__(self, *exc):
        self.vae.encode = self._original
        return False


if __name__ == "__main__":
    raise SystemExit(main())
