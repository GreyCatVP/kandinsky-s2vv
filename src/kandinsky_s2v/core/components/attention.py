"""Attention kernel dispatch for S2V.

Two access patterns are needed:

``dense``   — one packed sequence (batch size is always 1 in S2V inference), so
              self-attention and text cross-attention are plain dense attention.
``grouped`` — the audio cross-attention is frame-aligned: latent frame ``i``
              attends only to the audio tokens belonging to that frame.  The
              reference implementation expresses this with varlen flash-attention
              and ``cu_seqlens``; here the groups are padded into a batch
              dimension and masked, which is equivalent and needs no varlen
              kernel.  Group sizes are uneven (K5 ``get_spatial_audio_cu_seqlens``
              gives the first frame a single audio token and splits the rest),
              so padding is not avoidable by reshaping.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

try:
    from flash_attn import flash_attn_func as _fa2  # type: ignore[import]
except Exception:  # noqa: BLE001
    _fa2 = None

try:
    from flash_attn_interface import flash_attn_func as _fa3  # type: ignore[import]
except Exception:  # noqa: BLE001
    _fa3 = None


def _sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """(B,S,H,D) in/out."""
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous()


def resolve_attention_engine(requested: str) -> str:
    """Resolve a config/CLI engine name against what is installed."""
    if requested == "flash_attention_3" and _fa3 is None:
        print("[attention] flash_attention_3 requested but not installed; falling back to auto")
        return "auto"
    if requested == "flash_attention_2" and _fa2 is None:
        print("[attention] flash_attention_2 requested but not installed; falling back to auto")
        return "auto"
    return requested


class AttentionEngine:
    """Picks the dense attention kernel once, at construction time."""

    _ENGINES = ("flash_attention_3", "flash_attention_2", "sdpa", "auto")

    def __init__(self, engine: str = "auto"):
        if engine not in self._ENGINES:
            raise ValueError(f"Unknown attention engine: {engine!r}")
        engine = resolve_attention_engine(engine)

        if engine == "flash_attention_3":
            self._fn = _fa3
        elif engine == "flash_attention_2":
            self._fn = _fa2
        elif engine == "sdpa":
            self._fn = _sdpa
        else:  # auto
            self._fn = _fa3 or _fa2 or _sdpa

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        out = self._fn(q, k, v)
        # FA3 returns (out, softmax_lse) on some builds.
        return out[0] if isinstance(out, tuple) else out


def pad_groups(x: Tensor, lengths: Tensor, max_len: int) -> tuple[Tensor, Tensor]:
    """Scatter a packed ``(S, ...)`` tensor into ``(G, max_len, ...)`` + validity mask.

    Rows of ``x`` must be ordered group by group, which is how K5 packs them.
    """
    positions = torch.arange(max_len, device=x.device)
    valid = positions[None, :] < lengths[:, None]  # (G, max_len)
    out = x.new_zeros(lengths.shape[0], max_len, *x.shape[1:])
    out[valid] = x
    return out, valid


def grouped_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_lengths: Tensor,
    kv_lengths: Tensor,
) -> Tensor:
    """Block-diagonal attention over uneven groups.

    ``q``/``k``/``v`` are packed ``(S, heads, head_dim)``; group ``i`` of the
    queries attends exactly to group ``i`` of the keys.  Returns packed
    ``(S_q, heads, head_dim)``.
    """
    q_max = int(q_lengths.max())
    kv_max = int(kv_lengths.max())
    q_pad, q_valid = pad_groups(q, q_lengths, q_max)
    k_pad, kv_valid = pad_groups(k, kv_lengths, kv_max)
    v_pad, _ = pad_groups(v, kv_lengths, kv_max)

    # (G, heads, S, D) for SDPA; mask out padded keys.
    attn_mask = kv_valid[:, None, None, :]
    out = F.scaled_dot_product_attention(
        q_pad.transpose(1, 2),
        k_pad.transpose(1, 2),
        v_pad.transpose(1, 2),
        attn_mask=attn_mask,
    ).transpose(1, 2)
    return out[q_valid]
