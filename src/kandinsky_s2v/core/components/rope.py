"""Rotary position embeddings for S2V (ported from K5 ``nn_s2v.py``)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


@torch.autocast(device_type="cuda", enabled=False)
def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


def _to_rope(args: Tensor) -> Tensor:
    rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
    return rope.view(*rope.shape[:-1], 2, 2).unsqueeze(-4)


class RoPE1D(nn.Module):
    def __init__(self, dim: int, max_pos: int = 1024, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos
        self.max_period = max_period
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, pos: Tensor) -> Tensor:
        return _to_rope(self.args[pos])

    def materialize_buffers(self) -> None:
        freq = get_freqs(self.dim // 2, self.max_period).to(self.args.device)
        pos = torch.arange(self.max_pos, dtype=freq.dtype, device=freq.device)
        self.args = torch.outer(pos, freq)


class RoPE3D(nn.Module):
    """3-D RoPE; the temporal axis spans negative positions for memory frames."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period
        self.offsets_t = max_pos[0]

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos, strict=True)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = (
                torch.arange(-ax_max_pos, ax_max_pos, dtype=freq.dtype)
                if i == 0
                else torch.arange(ax_max_pos, dtype=freq.dtype)
            )
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        shape: tuple[int, int, int],
        pos: list[Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        duration, height, width = shape
        args_t = self.args_0[pos[0] + self.offsets_t] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(duration, 1, 1, -1).repeat(1, height, width, 1),
                args_h.view(1, height, 1, -1).repeat(duration, 1, width, 1),
                args_w.view(1, 1, width, -1).repeat(duration, height, 1, 1),
            ],
            dim=-1,
        )
        return _to_rope(args)

    def materialize_buffers(self) -> None:
        for i, (axes_dim, ax_max_pos) in enumerate(zip(self.axes_dims, self.max_pos, strict=True)):
            freq = get_freqs(axes_dim // 2, self.max_period).to(self.args_0.device)
            pos = (
                torch.arange(-ax_max_pos, ax_max_pos, dtype=freq.dtype, device=freq.device)
                if i == 0
                else torch.arange(ax_max_pos, dtype=freq.dtype, device=freq.device)
            )
            setattr(self, f"args_{i}", torch.outer(pos, freq))
