"""RoPE position grids for the packed ``[memory | ref | visual]`` sequence.

Temporal layout (in patch units, before the DiT scales them):

    memory 4x   memory 2x   memory 1x   visual        ref
    [-t3, -t2)  [-t2, -t1)  [-t1, 0)    [0, duration) {index_rope_ref_image}

Memory sits at negative positions, i.e. "before" the chunk being generated, and
the reference frame is pinned to a fixed index far from both.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..components.dit import audio_group_lengths


def visual_rope_pos(duration: int, height: int, width: int, patch_size: tuple[int, int, int], device) -> list[Tensor]:
    """``height``/``width`` are latent (pre-patch) sizes."""
    return [
        torch.arange(duration, device=device),
        torch.arange(height // patch_size[1], device=device),
        torch.arange(width // patch_size[2], device=device),
    ]


def ref_rope_pos(
    index_rope_ref_image: int,
    height: int,
    width: int,
    patch_size: tuple[int, int, int],
    device,
) -> list[Tensor]:
    return [
        torch.tensor([index_rope_ref_image], device=device),
        torch.arange(height // patch_size[1], device=device),
        torch.arange(width // patch_size[2], device=device),
    ]


def memory_rope_pos(
    memory: list[Tensor],
    frame_buckets: tuple[int, int, int],
    coefs: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    patch_size: tuple[int, int, int],
    device,
) -> list[list[Tensor]]:
    """Positions for ``[proj1x, proj2x, proj4x]``, in that order."""
    bounds = [0]
    for bucket in frame_buckets:
        bounds.append(bounds[-1] + bucket)

    out = []
    for i, (buffer, coef) in enumerate(zip(memory, coefs, strict=True)):
        start, stop = -bounds[i + 1], -bounds[i]
        height = buffer.shape[1] // patch_size[1]
        width = buffer.shape[2] // patch_size[2]
        out.append(
            [
                torch.arange(start, stop, coef[0], device=device),
                torch.arange(0, height, coef[1], device=device),
                torch.arange(0, width, coef[2], device=device),
            ]
        )
    return out


def text_rope_pos(num_text_tokens: int, device) -> Tensor:
    return torch.arange(num_text_tokens, device=device)


def audio_rope_pos(num_audio_tokens: int, duration: int, device) -> Tensor:
    """Audio token → index of the latent frame it belongs to.

    Uses the same uneven split as the frame-aligned cross-attention, so each
    audio token carries the temporal position of its frame.
    """
    lengths = audio_group_lengths(num_audio_tokens, duration, device)
    return torch.arange(duration, device=device).repeat_interleave(lengths).to(torch.int32)
