"""Euler flow-matching sampler for one S2V chunk."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

_CFG_EPS = 1e-6


@dataclass
class TextContext:
    """One text branch (positive or negative)."""

    text_embeds: Tensor  # (S_text, in_text_dim)
    pooled_embed: Tensor  # (1, in_text_dim2)
    rope_pos: Tensor  # (S_text,)


@dataclass
class ChunkContext:
    """Everything that stays fixed while one chunk is denoised."""

    ref: Tensor  # (1, H, W, C)
    memory: list[Tensor]  # [proj1x, proj2x, proj4x]
    audio_embeds: Tensor  # (S_audio, in_audio_dim)
    visual_rope_pos: list[Tensor]
    ref_rope_pos: list[Tensor]
    memory_rope_pos: list[list[Tensor]]  # [proj1x, proj2x, proj4x]
    audio_rope_pos: Tensor
    duration: int
    scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0)


def shift_timesteps(t: Tensor, shift: float) -> Tensor:
    """Flow-matching timestep warp; ``shift`` is the config ``scheduler_scale``."""
    return shift * t / (1 + (shift - 1) * t)


def dit_forward(  # noqa: PLR0913
    dit: nn.Module,
    latent: Tensor,
    time: Tensor,
    ctx: ChunkContext,
    text: TextContext,
    audio_scale: float = 1.0,
) -> Tensor:
    memory_1x, memory_2x, memory_4x = ctx.memory
    pos_1x, pos_2x, pos_4x = ctx.memory_rope_pos
    return dit(
        latent,
        ctx.ref,
        memory_4x,
        memory_2x,
        memory_1x,
        text.text_embeds,
        audio_scale * ctx.audio_embeds,
        text.pooled_embed,
        time * 1000,
        ctx.visual_rope_pos,
        ctx.ref_rope_pos,
        pos_4x,
        pos_2x,
        pos_1x,
        text.rope_pos,
        ctx.audio_rope_pos,
        ctx.scale_factor,
        ctx.duration,
    )


@torch.no_grad()
def denoise_chunk(  # noqa: PLR0913
    latent: Tensor,
    dit: nn.Module,
    ctx: ChunkContext,
    text: TextContext,
    null_text: TextContext,
    num_steps: int,
    guidance_weight: float,
    scheduler_scale: float,
    progress_callback=None,
) -> Tensor:
    """Denoise one chunk with ``num_steps - 1`` Euler updates.

    Guidance is joint over text and audio: the unconditional pass swaps in the
    negative prompt *and* zeroes the audio features, as in K5 ``get_velocity``.
    """
    device = latent.device
    timesteps = torch.linspace(1, 0, num_steps, device=device)
    timesteps = shift_timesteps(timesteps, scheduler_scale)

    use_cfg = abs(guidance_weight - 1.0) > _CFG_EPS
    for timestep, step in zip(timesteps[:-1], torch.diff(timesteps), strict=True):
        time = timestep.unsqueeze(0)
        velocity = dit_forward(dit, latent, time, ctx, text)
        if use_cfg:
            uncond = dit_forward(dit, latent, time, ctx, null_text, audio_scale=0.0)
            velocity = uncond + guidance_weight * (velocity - uncond)
        latent = latent + step * velocity
        if progress_callback is not None:
            progress_callback()
    return latent
