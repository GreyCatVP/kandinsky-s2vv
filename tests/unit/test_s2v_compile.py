"""Selective ``torch.compile`` of the visual blocks."""

from __future__ import annotations

import torch

from kandinsky_s2v.core.components.dit import (
    DiffusionTransformer3DS2V,
    TransformerAudioEncoderBlock,
    TransformerDecoderBlock,
    TransformerTextEncoderBlock,
)
from kandinsky_s2v.core.utils.compile import compile_visual_blocks

TINY = dict(
    in_visual_dim=16,
    out_visual_dim=16,
    in_text_dim=32,
    in_text_dim2=16,
    in_audio_dim=32,
    time_dim=16,
    patch_size=(1, 2, 2),
    frame_buckets=(2, 4, 24),
    enable_framepack=True,
    index_rope_ref_image=40,
    patch_proj1x_coef=(1, 1, 1),
    patch_proj2x_coef=(2, 2, 2),
    patch_proj4x_coef=(4, 4, 4),
    model_dim=32,
    ff_dim=64,
    num_text_blocks=1,
    num_audio_blocks=1,
    num_visual_blocks=3,
    axes_dims=(4, 6, 6),
    audio_inject_layers=(0, 2),
    attention_engine="sdpa",
)


def _forwards(model: torch.nn.Module, block_type: type) -> list:
    return [m.forward for m in model.modules() if isinstance(m, block_type)]


def test_only_visual_blocks_are_compiled():
    """Text and audio encoders are ~0.1% of a step; compiling them only adds warm-up."""
    dit = DiffusionTransformer3DS2V(**TINY)
    before_text = _forwards(dit, TransformerTextEncoderBlock)
    before_audio = _forwards(dit, TransformerAudioEncoderBlock)
    before_visual = _forwards(dit, TransformerDecoderBlock)

    compile_visual_blocks(dit)

    assert _forwards(dit, TransformerTextEncoderBlock) == before_text
    assert _forwards(dit, TransformerAudioEncoderBlock) == before_audio
    after_visual = _forwards(dit, TransformerDecoderBlock)
    assert len(after_visual) == len(before_visual) == TINY["num_visual_blocks"]
    assert all(new is not old for new, old in zip(after_visual, before_visual, strict=True))


def test_compile_returns_the_same_module():
    dit = DiffusionTransformer3DS2V(**TINY)
    assert compile_visual_blocks(dit) is dit
