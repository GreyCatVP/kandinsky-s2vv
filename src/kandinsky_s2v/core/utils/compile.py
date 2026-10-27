"""Optional ``torch.compile`` of the visual blocks.

Only the visual blocks are compiled. A step profile at 51k visual tokens put
2047 ms of a 2050 ms forward inside them — the text encoder, the audio encoder
and the RoPE/patchify work are ~0.1% each, so compiling those would only add
warm-up time.

Mode: plain ``dynamic=True``, matching the reference's
``@torch.compile(dynamic=True)`` decorator. Measured on Lite / 512x768:
1.31x (2047 → 1568 ms) for ~26 s of warm-up, against 1.32x for ~98 s with
``max-autotune-no-cudagraphs`` — the autotuning does not pay for itself here.
"""

from __future__ import annotations

import torch
from torch import nn

from ..components.dit import TransformerDecoderBlock


def compile_visual_blocks(dit: nn.Module) -> nn.Module:
    """Compile every visual block's forward in place; returns the same module."""
    for module in dit.modules():
        if isinstance(module, TransformerDecoderBlock):
            module.forward = torch.compile(module.forward, dynamic=True)
    return dit
