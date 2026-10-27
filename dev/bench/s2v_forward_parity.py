"""Deviation of a *single* transformer forward, native against ported, on GPU in bf16.

The pipeline-level parity check reports what a whole generation accumulates. This
reports what one forward costs, which is what separates "the port differs" from
"the difference compounds over a hundred forwards".

The two models are built one at a time — Pro is 40 GiB, and both at once do not
fit — so the native output is kept on the host while the ported one runs.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python dev/bench/s2v_forward_parity.py \\
        --config src/kandinsky_s2v/configs/s2v_pro.yaml \\
        --transformer /path/to/bundle/transformer
"""

from __future__ import annotations

import argparse
import gc

import torch

from kandinsky_s2v.pipeline.config import load_config
from kandinsky_s2v.pipeline.factory import create_bare_dit
from kandinsky_s2v.ports.diffusers.transformer_kandinsky_s2v import KandinskyS2VTransformer3DModel

HEIGHT, WIDTH = 512, 768
NUM_TEXT_TOKENS = 64
AUDIO_TOKENS_PER_FRAME = 4


def build_inputs(cfg, device: torch.device) -> dict:
    """Shapes as the pipeline produces them for one chunk at 512x768."""
    dit = cfg.dit
    duration = cfg.generation.latent_frames
    latent_h, latent_w = HEIGHT // 8, WIDTH // 8
    patch_h, patch_w = dit.patch_size[1], dit.patch_size[2]
    generator = torch.Generator(device="cpu").manual_seed(0)

    def randn(*shape: int, dtype=torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

    buckets, coefs = dit.frame_buckets, (dit.patch_proj1x_coef, dit.patch_proj2x_coef, dit.patch_proj4x_coef)
    memory, memory_pos = [], []
    bound = 0
    for bucket, coef in zip(buckets, coefs, strict=True):
        h = (latent_h // (patch_h * coef[1])) * patch_h * coef[1]
        w = (latent_w // (patch_w * coef[2])) * patch_w * coef[2]
        memory.append(randn(bucket, h, w, dit.in_visual_dim, dtype=torch.float32))
        memory_pos.append(
            [
                torch.arange(-(bound + bucket), -bound, coef[0], device=device),
                torch.arange(0, h // patch_h, coef[1], device=device),
                torch.arange(0, w // patch_w, coef[2], device=device),
            ]
        )
        bound += bucket

    num_audio = 1 + (duration - 1) * AUDIO_TOKENS_PER_FRAME
    return {
        "latent": randn(duration, latent_h, latent_w, dit.in_visual_dim, dtype=torch.float32),
        "ref": randn(1, latent_h, latent_w, dit.in_visual_dim),
        "memory": memory,
        "memory_pos": memory_pos,
        "text": randn(NUM_TEXT_TOKENS, dit.in_text_dim),
        "audio": randn(num_audio, dit.in_audio_dim),
        "pooled": randn(1, dit.in_text_dim2),
        "time": torch.tensor([700.0], device=device),
        "visual_pos": [
            torch.arange(duration, device=device),
            torch.arange(latent_h // patch_h, device=device),
            torch.arange(latent_w // patch_w, device=device),
        ],
        "ref_pos": [
            torch.tensor([dit.index_rope_ref_image], device=device),
            torch.arange(latent_h // patch_h, device=device),
            torch.arange(latent_w // patch_w, device=device),
        ],
        "text_pos": torch.arange(NUM_TEXT_TOKENS, device=device),
        "audio_pos": torch.arange(duration, device=device).repeat_interleave(
            torch.cat(
                [
                    torch.ones(1, dtype=torch.long, device=device),
                    torch.diff(torch.arange(duration, device=device) * (num_audio - 1) // (duration - 1)),
                ]
            )
        ),
        "duration": duration,
        "scale_factor": tuple(cfg.generation.scale_factor),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--transformer", required=True, help="the bundle's transformer/ folder")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)
    inputs = build_inputs(cfg, device)

    native = create_bare_dit(cfg, device, "sdpa")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = native(
            inputs["latent"],
            inputs["ref"],
            inputs["memory"][2],
            inputs["memory"][1],
            inputs["memory"][0],
            inputs["text"],
            inputs["audio"],
            inputs["pooled"],
            inputs["time"],
            inputs["visual_pos"],
            inputs["ref_pos"],
            inputs["memory_pos"][2],
            inputs["memory_pos"][1],
            inputs["memory_pos"][0],
            inputs["text_pos"],
            inputs["audio_pos"],
            inputs["scale_factor"],
            inputs["duration"],
        )
    expected = expected.float().cpu()
    print(f"[forward] native {tuple(expected.shape)}", flush=True)

    del native
    gc.collect()
    torch.cuda.empty_cache()

    ported = (
        KandinskyS2VTransformer3DModel.from_pretrained(args.transformer, torch_dtype=torch.bfloat16).to(device).eval()
    )
    with torch.no_grad():
        actual = ported(
            hidden_states=inputs["latent"].unsqueeze(0).to(torch.bfloat16),
            encoder_hidden_states=inputs["text"].unsqueeze(0),
            audio_hidden_states=inputs["audio"].unsqueeze(0),
            reference_hidden_states=inputs["ref"].unsqueeze(0),
            memory_hidden_states=tuple(m.unsqueeze(0).to(torch.bfloat16) for m in reversed(inputs["memory"])),
            timestep=inputs["time"],
            pooled_projections=inputs["pooled"],
            visual_rope_pos=inputs["visual_pos"],
            reference_rope_pos=inputs["ref_pos"],
            memory_rope_pos=tuple(reversed(inputs["memory_pos"])),
            text_rope_pos=inputs["text_pos"],
            audio_rope_pos=inputs["audio_pos"],
            scale_factor=inputs["scale_factor"],
            duration=inputs["duration"],
            return_dict=False,
        )[0].squeeze(0)
    actual = actual.float().cpu()
    print(f"[forward] ported {tuple(actual.shape)}", flush=True)

    rel = (actual - expected).norm() / expected.norm()
    print(f"[forward] relative L2 of one forward: {rel:.3e}")
    print(f"[forward] max |diff|: {(actual - expected).abs().max():.4f}")


if __name__ == "__main__":
    main()
