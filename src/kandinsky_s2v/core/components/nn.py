"""Leaf modules of the S2V DiT (ported from K5 ``nn_s2v.py``).

Parameter names are kept identical to the reference checkpoints so that weight
conversion is a pure shard gather with no key remapping.

Batch size is always 1 in S2V inference, so tensors are packed ``(S, C)`` and
the varlen/``cu_seqlens`` plumbing of the reference collapses into dense
attention.  The one exception is the frame-aligned audio cross-attention, which
takes explicit group lengths.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..parallel import SequenceShard
from .attention import AttentionEngine, grouped_attention
from .rope import get_freqs


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_scale_shift_norm(norm, x: Tensor, scale: Tensor, shift: Tensor, idx: Tensor) -> Tensor:
    return norm(x) * (scale.index_select(0, idx) + 1.0) + shift.index_select(0, idx)


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_gate_sum(x: Tensor, out: Tensor, gate: Tensor, idx: Tensor) -> Tensor:
    return x + gate.index_select(0, idx) * out


@torch.autocast(device_type="cuda", enabled=False)
def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = rope[..., 0] * x_[..., 0] + rope[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape)


class TimeEmbeddings(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0):
        super().__init__()
        if model_dim % 2 != 0:
            raise ValueError(f"model_dim must be even, got {model_dim}")
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer("freqs", get_freqs(model_dim // 2, max_period), persistent=False)

        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, time: Tensor) -> tuple[Tensor, Tensor]:
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = self.out_layer(self.activation(self.in_layer(time_embed)))
        time_embed_idx = torch.arange(time_embed.shape[0], device=time_embed.device, dtype=torch.int32)
        return time_embed, time_embed_idx

    def materialize_buffers(self) -> None:
        """Rebuild the non-persistent ``freqs`` buffer after a meta-device build."""
        self.freqs = get_freqs(self.model_dim // 2, self.max_period).to(self.freqs.device)


class GeneralEmbeddings(nn.Module):
    """Linear projection + LayerNorm; used for text, pooled text and audio."""

    def __init__(self, dim: int, model_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, embed: Tensor) -> Tensor:
        embed = self.in_layer(embed)
        return self.norm(embed).type_as(embed)


class VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        """``(T, H, W, C)`` → ``(T/p0, H/p1, W/p2, model_dim)``."""
        duration, height, width, dim = x.shape
        x = (
            x.view(
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 2, 4, 1, 3, 5, 6)
            .flatten(3, 6)
        )
        return self.in_layer(x)


class Modulation(nn.Module):
    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.activation(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.activation(self.in_layer(x)))


class _QKVAttention(nn.Module):
    """Shared projections + QK-norm for the self/cross attention modules."""

    def __init__(self, num_channels: int, head_dim: int):
        super().__init__()
        if num_channels % head_dim != 0:
            raise ValueError(f"num_channels={num_channels} not divisible by head_dim={head_dim}")
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    def _norm_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k


class MultiheadSelfAttention(_QKVAttention):
    def __init__(self, num_channels: int, head_dim: int, engine: str = "auto"):
        super().__init__(num_channels, head_dim)
        self.attention = AttentionEngine(engine)

    def forward(self, x: Tensor, rope: Tensor, shard: SequenceShard | None = None) -> Tensor:
        """``shard`` (sequence parallel): ``x`` and ``rope`` are this rank's tokens."""
        shape = x.shape[:-1]
        query = self.to_query(x).reshape(*shape, self.num_heads, -1)
        key = self.to_key(x).reshape(*shape, self.num_heads, -1)
        value = self.to_value(x).reshape(*shape, self.num_heads, -1)
        query, key = self._norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        if shard is not None:
            # All heads of my tokens → my heads of all tokens, so the kernel
            # attends over the whole sequence; then back.
            query, key, value = shard.tokens_to_heads(torch.stack((query, key, value), dim=1)).unbind(1)
            out = self.attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)).squeeze(0)
            return self.out_layer(shard.heads_to_tokens(out).flatten(-2, -1))

        out = self.attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0))
        return self.out_layer(out.squeeze(0).flatten(-2, -1))


class MultiheadCrossAttention(_QKVAttention):
    def __init__(self, num_channels: int, head_dim: int, engine: str = "auto"):
        super().__init__(num_channels, head_dim)
        self.attention = AttentionEngine(engine)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        q_lengths: Tensor | None = None,
        kv_lengths: Tensor | None = None,
    ) -> Tensor:
        """Dense cross-attention, or block-diagonal when group lengths are given."""
        shape, cond_shape = x.shape[:-1], cond.shape[:-1]
        query = self.to_query(x).reshape(*shape, self.num_heads, -1)
        key = self.to_key(cond).reshape(*cond_shape, self.num_heads, -1)
        value = self.to_value(cond).reshape(*cond_shape, self.num_heads, -1)
        query, key = self._norm_qk(query, key)

        if q_lengths is None:
            out = self.attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)).squeeze(0)
        else:
            if kv_lengths is None:
                raise ValueError("kv_lengths is required when q_lengths is given")
            out = grouped_attention(query, key, value, q_lengths, kv_lengths)
        return self.out_layer(out.flatten(-2, -1))


class OutLayer(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        if patch_size[0] != 1:
            # K5 OutLayer re-expands the temporal patch with a repeat_interleave that
            # depends on cu_seqlens; every S2V config uses patch_size[0] == 1.
            raise NotImplementedError(f"temporal patch size must be 1, got {patch_size[0]}")
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True)

    def forward(self, visual_embed: Tensor, time_embed: Tensor, time_embed_idx: Tensor) -> Tensor:
        """``(T, H, W, model_dim)`` → ``(T*p0, H*p1, W*p2, visual_dim)``."""
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        visual_embed = apply_scale_shift_norm(
            self.norm, visual_embed, scale[:, None, None], shift[:, None, None], time_embed_idx
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)

        duration, height, width, _ = x.shape
        x = (
            x.view(duration, height, width, -1, *self.patch_size)
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )
        return x
