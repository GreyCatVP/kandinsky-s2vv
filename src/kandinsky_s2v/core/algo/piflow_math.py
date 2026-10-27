"""π-Flow (arXiv 2510.14974) DX policy used by distilled S2V checkpoints.

The student predicts a *grid* of x0 values over the segment ``[t_dst, t_src]``
in a single forward; the policy interpolates that grid, so integrating the ODE
across the segment costs no further network evaluations.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .denoise_loop import shift_timesteps

_MIN_GRID_POINTS = 2


class DXPolicy:
    """Dynamic-x0 policy: a grid of x0 predictions, linearly interpolated in raw time."""

    def __init__(  # noqa: PLR0913
        self,
        denoising_output: Tensor,
        x_t_src: Tensor,
        sigma_t_src: Tensor,
        segment_size: Tensor,
        shift: float = 1.0,
        eps: float = 1e-4,
    ):
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.shift = shift
        self.eps = eps

        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.raw_t_src = self._unwarp_t(self.sigma_t_src)
        seg = segment_size
        if isinstance(seg, Tensor) and seg.dim() < self.raw_t_src.dim():
            seg = seg.reshape(*seg.size(), *((self.ndim - seg.dim()) * [1]))
        # clamp(min=0) here while the sampler clamps to eps — that asymmetry is
        # the fork's (piflow_s2v.py vs utils_s2v_piflow.py). Parity, not an
        # oversight: do not align the two.
        self.raw_t_dst = (self.raw_t_src - seg).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)

        self.denoising_output_x_0 = self._u_to_x_0(denoising_output, self.x_t_src, self.sigma_t_src)

    @staticmethod
    def _interpolate(x: Tensor, t: Tensor) -> Tensor:
        n = x.size(1)
        if n < _MIN_GRID_POINTS:
            return x.squeeze(1)
        t = t.clamp(min=0, max=1) * (n - 1)
        t0 = t.floor().to(torch.long).clamp(min=0, max=n - 2)
        t1 = t0 + 1
        t0t1 = torch.stack([t0, t1], dim=1)
        x0x1 = torch.gather(x, dim=1, index=t0t1.expand(-1, -1, *x.shape[2:]))
        return (t1 - t) * x0x1[:, 0] + (t - t0) * x0x1[:, 1]

    def _unwarp_t(self, sigma_t: Tensor) -> Tensor:
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    @staticmethod
    def _u_to_x_0(denoising_output: Tensor, x_t: Tensor, sigma_t: Tensor) -> Tensor:
        return x_t.unsqueeze(1) - sigma_t.unsqueeze(1) * denoising_output

    def pi(self, x_t: Tensor, sigma_t: Tensor) -> Tensor:
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        raw_t = self._unwarp_t(sigma_t)
        x_0 = self._interpolate(self.denoising_output_x_0, (raw_t - self.raw_t_dst) / self.segment_size)
        return (x_t - x_0) / sigma_t.clamp(min=self.eps)

    def detach(self) -> DXPolicy:
        clone = DXPolicy.__new__(DXPolicy)
        clone.__dict__.update(self.__dict__)
        clone.denoising_output_x_0 = self.denoising_output_x_0.detach()
        return clone


def policy_rollout_fm(  # noqa: PLR0913
    x_t_start: Tensor,
    sigma_t_start: Tensor,
    raw_t_start: Tensor,
    raw_t_end: Tensor,
    total_substeps: int,
    policy: DXPolicy,
) -> Tensor:
    """Integrate the policy ODE across a segment with a fixed substep size."""
    num_tokens = x_t_start.size(0)
    ndim = x_t_start.dim()
    raw_t_start = raw_t_start.reshape(num_tokens, *((ndim - 1) * [1]))
    raw_t_end = raw_t_end.reshape(num_tokens, *((ndim - 1) * [1]))

    delta_raw_t = raw_t_start - raw_t_end
    num_substeps = (delta_raw_t * total_substeps).round().to(torch.long).clamp(min=1)
    substep_size = delta_raw_t / num_substeps

    raw_t, sigma_t, x_t = raw_t_start, sigma_t_start, x_t_start
    for substep_id in range(int(num_substeps.max().item())):
        u = policy.pi(x_t, sigma_t)

        raw_t_minus = (raw_t - substep_size).clamp(min=0)
        sigma_t_minus = shift_timesteps(raw_t_minus, policy.shift)
        x_t_minus = x_t + u * (sigma_t_minus - sigma_t)

        active = num_substeps > substep_id
        x_t = torch.where(active, x_t_minus, x_t)
        sigma_t = torch.where(active, sigma_t_minus, sigma_t)
        raw_t = torch.where(active, raw_t_minus, raw_t)
    return x_t
