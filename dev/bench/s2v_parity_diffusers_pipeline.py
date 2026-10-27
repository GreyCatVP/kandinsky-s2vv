"""Parity of the Diffusers (modular) pipeline against the native S2V pipeline, on GPU.

Both sides are driven from the same YAML config, the same reference frame,
speech track and seed, and their decoded frames are compared.

The encoders (Qwen2.5-Omni, CLIP) and the VAE are *shared* between the two
pipelines — loading them twice would double 16 GiB for no benefit, and they are
the same weights either way. What is under test is everything the port rewrote:
prompt and speech slicing, latent preparation, the RoPE grids, the FramePack
memory, the chunk loop, the sampler schedule and classifier-free guidance.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
        dev/bench/s2v_parity_diffusers_pipeline.py \\
        --config src/kandinsky_s2v/configs/s2v_lite.yaml \\
        --audio speech.wav --image reference.png --steps 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from diffusers import ClassifierFreeGuidance, ComponentsManager, FlowMatchEulerDiscreteScheduler
from s2v_parity_pipeline import FixedReferenceLatent

from kandinsky_s2v.core.algo.prepare_latents import encode_reference_image, select_best_resolution
from kandinsky_s2v.pipeline.config import load_config
from kandinsky_s2v.pipeline.factory import build_pipeline
from kandinsky_s2v.ports.diffusers.modular import (
    KandinskyS2VAutoBlocks,
    KandinskyS2VDistilledAutoBlocks,
    KandinskyS2VDistilledModularPipeline,
    KandinskyS2VModularPipeline,
)
from kandinsky_s2v.ports.diffusers.scheduling_piflow import KandinskyS2VPiflowScheduler
from kandinsky_s2v.ports.diffusers.transformer_kandinsky_s2v import KandinskyS2VTransformer3DModel

TOLERANCE = 2e-2


def build_scheduler(cfg):
    """The distilled configs need the pi-Flow policy sampler; the base ones do not."""
    if not cfg.piflow.enabled:
        return FlowMatchEulerDiscreteScheduler(shift=cfg.generation.scheduler_scale)
    return KandinskyS2VPiflowScheduler(
        shift=cfg.piflow.shift if cfg.piflow.shift is not None else cfg.generation.scheduler_scale,
        n_grid=cfg.piflow.n_grid,
        eps=cfg.piflow.eps,
        final_step_size_scale=cfg.piflow.final_step_size_scale,
        num_policy_substeps=cfg.piflow.num_policy_substeps,
    )


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--transformer", required=True, help="the bundle's transformer/ folder")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="override classifier-free guidance on both sides, to isolate its amplification",
    )
    parser.add_argument(
        "--offload",
        choices=["none", "module"],
        default=None,
        help="override the config's offload strategy on both sides, to isolate that path",
    )
    parser.add_argument(
        "--attention",
        default=None,
        help="override the native attention engine, e.g. sdpa, to match the port's backend",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)
    if args.offload is not None:
        # The config is frozen, so the override has to go through a copy.
        cfg = cfg.model_copy(update={"offload": cfg.offload.model_copy(update={"strategy": args.offload})})
    if args.guidance is not None:
        cfg = cfg.model_copy(
            update={"generation": cfg.generation.model_copy(update={"guidance_weight": args.guidance})}
        )
    gen_cfg = cfg.generation

    started = time.time()
    native = build_pipeline(cfg, device, args.attention)
    print(f"[parity] native pipeline loaded in {time.time() - started:.0f}s", flush=True)

    # The VAE posterior is sampled, and the two pipelines draw it from different
    # generators, so each would otherwise start from a different reference latent.
    # Pin it so the comparison measures the port and not that draw.
    resolution = select_best_resolution(args.image, gen_cfg.allowed_resolutions)
    # With module offload the VAE waits on the CPU until its stage runs, so it has
    # to be brought over for this one encode and then put back.
    vae_device = next(native.vae.parameters()).device
    native.vae.to(device)
    with torch.no_grad():
        # Seeded, not only shared: an unseeded draw differs from one process to the next, and that alone moved
        # the result by tens of percent between runs of the same code.
        ref_generator = torch.Generator(device=device).manual_seed(gen_cfg.seed)
        reference_latent = encode_reference_image(args.image, resolution, native.vae, device, ref_generator)
    native.vae.to(vae_device)
    pinned = FixedReferenceLatent(native.vae, reference_latent)

    started = time.time()
    with pinned:
        native_out = native(
            audio=args.audio,
            image=args.image,
            num_steps=args.steps,
            chunks=args.chunks,
            show_progress=False,
        )
    native_frames = native_out.frames.float()  # (3, T, H, W) in 0..255
    print(f"[parity] native {tuple(native_frames.shape)} in {time.time() - started:.0f}s", flush=True)

    # Reuse the loaded encoders and VAE; only the transformer is built anew.
    transformer = (
        KandinskyS2VTransformer3DModel.from_pretrained(args.transformer, torch_dtype=torch.bfloat16).to(device).eval()
    )
    distilled = cfg.piflow.enabled
    blocks, pipeline_class = (
        (KandinskyS2VDistilledAutoBlocks(), KandinskyS2VDistilledModularPipeline)
        if distilled
        else (KandinskyS2VAutoBlocks(), KandinskyS2VModularPipeline)
    )
    manager = ComponentsManager() if cfg.offload.strategy == "module" else None
    ported = pipeline_class(blocks=blocks, components_manager=manager)
    ported.update_components(
        transformer=transformer,
        vae=native.vae,
        text_encoder=native.embedder.thinker,
        tokenizer=native.embedder.processor,
        text_encoder_2=native.embedder.clip,
        tokenizer_2=native.embedder.clip_tokenizer,
        scheduler=build_scheduler(cfg),
        # Guidance is a component here, not an argument; the config's weight on both sides.
        guider=ClassifierFreeGuidance(guidance_scale=gen_cfg.guidance_weight),
    )
    if manager is not None:
        # The config asks for module offload, and Pro does not fit otherwise: the
        # transformer alone is 40 GiB and the shared Qwen another 17. One model on the
        # device at a time; enabled after the components are in, since adding one
        # resets the strategy to the default.
        manager.enable_auto_cpu_offload(device=device, offload_strategy=lambda hooks, **_: hooks)
    else:
        ported.to(device)

    started = time.time()
    generator = torch.Generator(device=device).manual_seed(gen_cfg.seed)
    with pinned:
        ported_frames = ported(
            image=args.image,
            audio=args.audio,
            prompt=gen_cfg.prompt,
            negative_prompt=gen_cfg.negative_prompt,
            num_latent_frames=gen_cfg.latent_frames,
            fps=gen_cfg.fps,
            num_chunks=args.chunks,
            num_inference_steps=args.steps,
            scale_factor=tuple(gen_cfg.scale_factor),
            generator=generator,
            output_type="np",
            output="videos",
        )
    # (1, T, H, W, 3) in 0..1 -> (3, T, H, W) in 0..255
    ported_frames = torch.from_numpy(np.asarray(ported_frames[0])).permute(3, 0, 1, 2).float() * 255.0
    print(f"[parity] ported {tuple(ported_frames.shape)} in {time.time() - started:.0f}s", flush=True)

    if native_frames.shape != ported_frames.shape:
        raise SystemExit(f"shape mismatch: native {tuple(native_frames.shape)} vs ported {tuple(ported_frames.shape)}")

    diff = ported_frames - native_frames
    rel = diff.norm() / native_frames.norm()

    # The verdict is per chunk, not per clip. With classifier-free guidance the
    # deviation grows from one chunk to the next, so an average over the whole
    # video flatters a long run and would fail a short one for the same code.
    total_frames = native_frames.shape[1]
    worst = rel
    if args.chunks > 1 and total_frames % args.chunks == 0:
        per_chunk = total_frames // args.chunks
        worst = 0.0
        for index in range(args.chunks):
            window = slice(index * per_chunk, (index + 1) * per_chunk)
            chunk_rel = (diff[:, window].norm() / native_frames[:, window].norm()).item()
            worst = max(worst, chunk_rel)
            print(f"[parity]   chunk {index}: relative L2 {chunk_rel:.3e}")

    print(f"[parity] relative L2: {rel:.3e} overall, {worst:.3e} worst chunk")
    print(f"[parity] max |diff|: {diff.abs().max():.1f}/255")
    print("[parity] PASSED" if worst < TOLERANCE else f"[parity] FAILED (tolerance {TOLERANCE:.0e} per chunk)")


if __name__ == "__main__":
    main()
