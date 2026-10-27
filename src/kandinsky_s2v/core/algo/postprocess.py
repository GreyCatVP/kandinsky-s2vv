"""Latent → uint8 frames."""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def decode_latent(latent: Tensor, vae) -> Tensor:
    """``(T, H, W, C)`` latent → ``(3, T*4-3, H*8, W*8)`` uint8 frames."""
    images = latent.reshape(1, *latent.shape)
    images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
    decoded = vae.decode(images).sample
    frames = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    return frames.squeeze(0)
