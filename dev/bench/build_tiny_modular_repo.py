"""Build the tiny repositories the Diffusers modular tests load their components from.

Modular pipeline tests never build components in code: the tester mixins load
them with `blocks.init_pipeline(repo).load_components()` and
`ModularPipeline.from_pretrained(repo)`, from a repository laid out like the
real checkpoints — a `modular_model_index.json` naming the pipeline and blocks,
and a folder per component. One repository per variant, as for the real ones:
the base setup, and with `--distilled` the pi-Flow one (the head widened to its
grid, `KandinskyS2VPiflowScheduler`, the distilled blocks).

    PYTHONPATH=src .venv/bin/python dev/bench/build_tiny_modular_repo.py \\
        --out /tmp/tiny-kandinsky-s2v-modular-pipe --repo-id kandinskylab/tiny-kandinsky-s2v-modular-pipe
    PYTHONPATH=src .venv/bin/python dev/bench/build_tiny_modular_repo.py --distilled \\
        --out /tmp/tiny-kandinsky-s2v-distilled-modular-pipe \\
        --repo-id kandinskylab/tiny-kandinsky-s2v-distilled-modular-pipe

`--repo-id` is what the modular index points each component at: the Hub id the
folder is going to be uploaded under, or the folder itself for a local run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from kandinsky_s2v.ports.diffusers import KandinskyS2VPiflowScheduler, KandinskyS2VTransformer3DModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_diffusers_bundle import write_modular_index  # noqa: E402

TINY_OMNI = "tiny-random/qwen2.5-omni"
TINY_CLIP_TOKENIZER = "hf-internal-testing/tiny-random-clip"
TINY_CHANNELS = 4
TINY_GRID = 2


def tiny_components(distilled: bool) -> dict:
    """The tiny components the Diffusers tests run on; deterministic, so a rebuild matches the published repo."""
    from diffusers import AutoencoderKLHunyuanVideo, FlowMatchEulerDiscreteScheduler  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        CLIPTextConfig,
        CLIPTextModel,
        CLIPTokenizer,
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    torch.manual_seed(0)
    vae = AutoencoderKLHunyuanVideo(
        act_fn="silu",
        block_out_channels=[8, 8, 8],
        down_block_types=["HunyuanVideoDownBlock3D"] * 3,
        in_channels=3,
        latent_channels=TINY_CHANNELS,
        layers_per_block=1,
        mid_block_add_attention=False,
        norm_num_groups=4,
        out_channels=3,
        scaling_factor=0.476986,
        spatial_compression_ratio=8,
        temporal_compression_ratio=4,
        up_block_types=["HunyuanVideoUpBlock3D"] * 3,
    )
    torch.manual_seed(0)
    text_encoder_2 = CLIPTextModel(
        CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=16,
            intermediate_size=16,
            layer_norm_eps=1e-05,
            num_attention_heads=2,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
            projection_dim=16,
        )
    )
    torch.manual_seed(0)
    transformer = KandinskyS2VTransformer3DModel(
        in_visual_dim=TINY_CHANNELS,
        out_visual_dim=TINY_CHANNELS * TINY_GRID if distilled else TINY_CHANNELS,
        in_text_dim=16,
        in_text_dim2=16,
        in_audio_dim=16,
        time_dim=16,
        patch_size=(1, 2, 2),
        frame_buckets=(1, 2, 4),
        enable_framepack=True,
        index_rope_ref_image=40,
        model_dim=32,
        ff_dim=64,
        num_text_blocks=1,
        num_audio_blocks=1,
        num_visual_blocks=2,
        axes_dims=(4, 6, 6),
        audio_inject_layers=(0,),
        text_rope_max_pos=128,
    )
    scheduler = (
        KandinskyS2VPiflowScheduler(shift=7.0, n_grid=TINY_GRID)
        if distilled
        else FlowMatchEulerDiscreteScheduler(shift=7.0)
    )
    return {
        "vae": vae,
        "text_encoder": Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(TINY_OMNI).to(torch.float32),
        "tokenizer": Qwen2_5OmniProcessor.from_pretrained(TINY_OMNI),
        "text_encoder_2": text_encoder_2,
        "tokenizer_2": CLIPTokenizer.from_pretrained(TINY_CLIP_TOKENIZER),
        "transformer": transformer,
        "scheduler": scheduler,
    }


README = """---
library_name: diffusers
tags:
- test
---

# Tiny Kandinsky S2V 1.0{variant} (random weights, for tests)

Randomly initialised, tiny versions of every Kandinsky S2V 1.0{variant} component. It generates noise, not
video: it exists only so the Diffusers modular pipeline tests can load a complete pipeline in seconds.
For the real models see the Kandinsky S2V 1.0 collection.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", help="where the modular index points; defaults to --out")
    parser.add_argument("--distilled", action="store_true", help="the pi-Flow variant")
    args = parser.parse_args()

    out = args.out.resolve()
    repo_id = args.repo_id or str(out)
    components = tiny_components(distilled=args.distilled)
    for name, component in components.items():
        component.save_pretrained(out / name)
    write_modular_index(out, repo_id, distilled=args.distilled)
    (out / "README.md").write_text(README.format(variant=" distilled" if args.distilled else ""))

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    kind = "distilled" if args.distilled else "base"
    print(f"[tiny] {out}: {kind}, {len(components)} components, {size / 2**20:.1f} MiB, index -> {repo_id}")


if __name__ == "__main__":
    main()
