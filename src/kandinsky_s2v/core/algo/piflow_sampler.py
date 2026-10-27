"""π-Flow sampler for distilled S2V checkpoints.

Each of the ``num_steps`` segments costs exactly one student evaluation: the
network predicts a grid of x0 values, which is then integrated over
``num_policy_substeps`` substeps for free.  CFG is baked into the student during
distillation, so ``guidance_weight=1.0`` (one forward per segment) is the normal
mode; a different weight applies CFG to the predicted grid at the cost of a
second forward, which is only useful for sanity-checking.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .denoise_loop import _CFG_EPS, ChunkContext, TextContext, dit_forward, shift_timesteps
from .piflow_math import DXPolicy, policy_rollout_fm


@torch.no_grad()
def piflow_denoise_chunk(  # noqa: PLR0913
    latent: Tensor,
    dit: nn.Module,
    ctx: ChunkContext,
    text: TextContext,
    null_text: TextContext,
    num_steps: int,
    scheduler_scale: float,
    guidance_weight: float = 1.0,
    *,
    num_policy_substeps: int = 128,
    final_step_size_scale: float = 0.5,
    shift: float | None = None,
    eps: float = 1e-6,
    progress_callback=None,
) -> Tensor:
    """Denoise one chunk in ``num_steps`` DX segments (NFE == ``num_steps``)."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    device = latent.device
    shift = scheduler_scale if shift is None else shift
    num_tokens = latent.shape[0]
    extra_dims = (latent.dim() - 1) * [1]

    raw_t_src = torch.ones(1, device=device, dtype=torch.float32)
    base_segment_size = 1.0 / (float(num_steps) - (1.0 - final_step_size_scale))
    use_cfg = abs(guidance_weight - 1.0) > _CFG_EPS

    for step_id in range(num_steps):
        is_final_step = step_id == num_steps - 1
        segment_size = base_segment_size * (final_step_size_scale if is_final_step else 1.0)
        # Clamped to eps here while DXPolicy clamps the same quantity to 0 —
        # that asymmetry exists in the distillation fork too, so the student's
        # final segment behaves as it did during training. Keep it.
        # eps here, 0 inside DXPolicy — see the note in piflow_math.py.
        raw_t_dst = (raw_t_src - segment_size).clamp(min=eps)

        sigma_t_src = shift_timesteps(raw_t_src, shift)
        sigma_t_src_tok = sigma_t_src.repeat_interleave(num_tokens).reshape(num_tokens, *extra_dims)
        segment_size_tok = torch.full_like(sigma_t_src_tok, segment_size)

        grid = dit_forward(dit, latent, sigma_t_src, ctx, text)
        if use_cfg:
            uncond = dit_forward(dit, latent, sigma_t_src, ctx, null_text, audio_scale=0.0)
            grid = uncond + guidance_weight * (grid - uncond)

        policy = DXPolicy(
            denoising_output=grid,
            x_t_src=latent,
            sigma_t_src=sigma_t_src_tok,
            segment_size=segment_size_tok,
            shift=shift,
            eps=eps,
        ).detach()
        latent = policy_rollout_fm(
            x_t_start=latent,
            sigma_t_start=sigma_t_src_tok,
            raw_t_start=raw_t_src.repeat_interleave(num_tokens),
            raw_t_end=raw_t_dst.repeat_interleave(num_tokens),
            total_substeps=num_policy_substeps,
            policy=policy,
        )

        raw_t_src = raw_t_dst
        if progress_callback is not None:
            progress_callback()
    return latent
