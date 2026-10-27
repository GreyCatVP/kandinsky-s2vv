"""Kandinsky S2V 1.0 diffusion transformer (ported from K5 ``dit_s2v_tp.py``).

Differences from the reference, all of which are consequences of running
single-device with batch size 1:

* no tensor-parallel ``replicate``/``shard`` identity hooks;
* ``cu_seqlens`` plumbing collapses to plain segment lengths;
* the frame-aligned audio cross-attention uses padded group attention instead of
  varlen flash-attention (see ``attention.grouped_attention``);
* NABLA sparse attention is dropped — it only applies to the 1024 resolution
  bucket, while S2V runs at 512.

The ``audio_adaln`` branch of the reference is not ported: both shipped S2V
configs set ``audio_adaln: false``, so those weights do not exist.  Module and
parameter names match the reference checkpoints exactly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..parallel import SequenceParallel, SequenceShard
from .nn import (
    FeedForward,
    GeneralEmbeddings,
    Modulation,
    MultiheadCrossAttention,
    MultiheadSelfAttention,
    OutLayer,
    TimeEmbeddings,
    VisualEmbeddings,
    apply_gate_sum,
    apply_scale_shift_norm,
)
from .rope import RoPE1D, RoPE3D

# The first latent frame takes a single audio token, so the rest need at least one frame.
_MIN_DURATION = 2


def split_lengths(total: int, groups: int, device: torch.device) -> Tensor:
    """Split ``total`` items into ``groups`` as evenly as K5 does.

    Mirrors ``get_spatial_visual_cu_seqlens``: boundaries are ``total*i // groups``.
    """
    bounds = torch.arange(groups + 1, device=device) * total // groups
    return torch.diff(bounds)


def audio_group_lengths(num_audio_tokens: int, duration: int, device: torch.device) -> Tensor:
    """K5 ``get_spatial_audio_cu_seqlens``: first frame gets one token, rest split evenly."""
    if duration < _MIN_DURATION:
        raise ValueError(f"duration must be >= {_MIN_DURATION} to split audio tokens, got {duration}")
    rest = split_lengths(num_audio_tokens - 1, duration - 1, device)
    first = torch.ones(1, dtype=rest.dtype, device=device)
    return torch.cat([first, rest])


class TransformerTextEncoderBlock(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int, engine: str):
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim, engine)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, x: Tensor, time_embed: Tensor, rope: Tensor, time_embed_idx: Tensor) -> Tensor:
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.self_attention(out, rope)
        x = apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.feed_forward(out)
        return apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)


class TransformerAudioEncoderBlock(nn.Module):
    """Plain pre-norm transformer block — no time modulation."""

    def __init__(self, model_dim: int, ff_dim: int, head_dim: int, engine: str):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim, engine)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        x = x + self.self_attention(self.self_attention_norm(x), rope)
        return x + self.feed_forward(self.feed_forward_norm(x))


class TransformerDecoderBlock(nn.Module):
    """Visual block; ``enable_audio`` adds the frame-aligned audio adapter."""

    def __init__(  # noqa: PLR0913
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        engine: str,
        enable_audio: bool = False,
    ):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim, engine)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim, engine)

        self.enable_audio = enable_audio
        if enable_audio:
            self.cross_attention_norm_v2a = nn.LayerNorm(model_dim, elementwise_affine=True)
            self.cross_attention_v2a = MultiheadCrossAttention(model_dim, head_dim, engine)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(  # noqa: PLR0913
        self,
        visual_embed_full: Tensor,
        text_embed: Tensor,
        audio_embed: Tensor,
        time_embed_full: Tensor,
        visual_rope_full: Tensor,
        time_embed_idx_full: Tensor,
        num_visual_tokens: int,
        visual_group_lengths: Tensor,
        audio_group_lengths_: Tensor,
        shard: SequenceShard | None = None,
    ) -> Tensor:
        """With ``shard``, the per-token inputs are this rank's slice of the sequence."""
        self_attn_params, cross_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed_full), 3, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(
            self.self_attention_norm, visual_embed_full, scale, shift, time_embed_idx_full
        ).type_as(visual_embed_full)
        out = self.self_attention(out, visual_rope_full, shard)
        visual_embed_full = apply_gate_sum(visual_embed_full, out, gate, time_embed_idx_full).type_as(visual_embed_full)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(
            self.cross_attention_norm, visual_embed_full, scale, shift, time_embed_idx_full
        ).type_as(visual_embed_full)
        out = self.cross_attention(out, text_embed)
        visual_embed_full = apply_gate_sum(visual_embed_full, out, gate, time_embed_idx_full).type_as(visual_embed_full)

        if self.enable_audio and shard is not None:
            # Same adapter on the frames this shard covers: its visual tokens are
            # the suffix of the local slice, possibly cut mid-frame at both ends.
            if shard.frame_lengths:
                head, visual_embed = visual_embed_full[: shard.visual_offset], visual_embed_full[shard.visual_offset :]
                out = self.cross_attention_v2a(
                    self.cross_attention_norm_v2a(visual_embed),
                    audio_embed[shard.audio_start : shard.audio_end],
                    torch.tensor(shard.frame_lengths, device=audio_embed.device),
                    torch.tensor(shard.audio_lengths, device=audio_embed.device),
                )
                visual_embed_full = torch.cat([head, visual_embed + out], dim=0)
        elif self.enable_audio:
            # Audio conditions only the noisy part of the sequence, frame by frame.
            # The ungated residual matches the reference (audio_adaln=False).
            head, visual_embed = (
                visual_embed_full[:-num_visual_tokens],
                visual_embed_full[-num_visual_tokens:],
            )
            out = self.cross_attention_v2a(
                self.cross_attention_norm_v2a(visual_embed),
                audio_embed,
                visual_group_lengths,
                audio_group_lengths_,
            )
            visual_embed_full = torch.cat([head, visual_embed + out], dim=0)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(
            self.feed_forward_norm, visual_embed_full, scale, shift, time_embed_idx_full
        ).type_as(visual_embed_full)
        out = self.feed_forward(out)
        return apply_gate_sum(visual_embed_full, out, gate, time_embed_idx_full).type_as(visual_embed_full)


class DiffusionTransformer3DS2V(nn.Module):
    """Speech-to-video DiT.

    The visual stream is one packed sequence ``[memory | ref | visual]``:

    * ``memory`` — FramePack context from the previous chunk at 4x / 2x / 1x
      temporal downsampling, placed at negative temporal RoPE positions;
    * ``ref``    — the reference frame, pinned to temporal RoPE index
      ``index_rope_ref_image``;
    * ``visual`` — the noisy latent being denoised.

    ``memory`` and ``ref`` are clean, so they are modulated with ``t = 0`` while
    ``visual`` uses the real timestep; ``time_embed_full`` holds both rows and
    ``time_embed_idx_full`` selects per token.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_visual_dim: int = 16,
        out_visual_dim: int = 16,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        in_audio_dim: int = 3584,
        time_dim: int = 512,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        frame_buckets: tuple[int, int, int] = (2, 4, 24),
        enable_framepack: bool = True,
        index_rope_ref_image: int = 40,
        patch_proj1x_coef: tuple[int, int, int] = (1, 1, 1),
        patch_proj2x_coef: tuple[int, int, int] = (2, 2, 2),
        patch_proj4x_coef: tuple[int, int, int] = (4, 4, 4),
        model_dim: int = 1792,
        ff_dim: int = 7168,
        num_text_blocks: int = 2,
        num_audio_blocks: int = 2,
        num_visual_blocks: int = 32,
        axes_dims: tuple[int, int, int] = (16, 24, 24),
        audio_inject_layers: tuple[int, ...] = (),
        text_rope_max_pos: int = 4096,
        attention_engine: str = "auto",
    ):
        super().__init__()
        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.frame_buckets = frame_buckets
        self.enable_framepack = enable_framepack
        self.index_rope_ref_image = index_rope_ref_image
        self.patch_proj1x_coef = patch_proj1x_coef
        self.patch_proj2x_coef = patch_proj2x_coef
        self.patch_proj4x_coef = patch_proj4x_coef
        self.audio_inject_layers = tuple(audio_inject_layers)

        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = GeneralEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = GeneralEmbeddings(in_text_dim2, time_dim)
        self.audio_embeddings = GeneralEmbeddings(in_audio_dim, model_dim)

        self.visual_embeddings = VisualEmbeddings(in_visual_dim, model_dim, patch_size)
        for name, coef in (
            ("proj1x", patch_proj1x_coef),
            ("proj2x", patch_proj2x_coef),
            ("proj4x", patch_proj4x_coef),
        ):
            scaled = (
                patch_size[0] * coef[0],
                patch_size[1] * coef[1],
                patch_size[2] * coef[2],
            )
            setattr(
                self,
                f"visual_embeddings_{name}",
                VisualEmbeddings(in_visual_dim, model_dim, scaled),
            )
        # 0 = visual, 1 = reference frame, 2 = memory
        self.mask_embeddings = nn.Embedding(3, model_dim)

        self.text_rope_embeddings = RoPE1D(head_dim, max_pos=text_rope_max_pos)
        self.text_transformer_blocks = nn.ModuleList(
            [
                TransformerTextEncoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine)
                for _ in range(num_text_blocks)
            ]
        )

        self.audio_rope_embeddings = RoPE1D(head_dim)
        self.audio_transformer_blocks = nn.ModuleList(
            [
                TransformerAudioEncoderBlock(model_dim, ff_dim, head_dim, attention_engine)
                for _ in range(num_audio_blocks)
            ]
        )

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    model_dim,
                    time_dim,
                    ff_dim,
                    head_dim,
                    attention_engine,
                    enable_audio=(i in self.audio_inject_layers),
                )
                for i in range(num_visual_blocks)
            ]
        )

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)
        # Set by the factory under torchrun; None keeps the single-process path.
        self.sequence_parallel: SequenceParallel | None = None

    def _embed_memory(  # noqa: PLR0913
        self,
        memory_proj4x: Tensor,
        memory_proj2x: Tensor,
        memory_proj1x: Tensor,
        rope_pos_4x: list[Tensor],
        rope_pos_2x: list[Tensor],
        rope_pos_1x: list[Tensor],
        scale_factor: tuple[float, float, float],
    ) -> tuple[Tensor, Tensor]:
        """Returns flattened memory tokens and their RoPE."""
        parts = [(self.visual_embeddings_proj1x(memory_proj1x), rope_pos_1x)]
        if self.enable_framepack:
            parts = [
                (self.visual_embeddings_proj4x(memory_proj4x), rope_pos_4x),
                (self.visual_embeddings_proj2x(memory_proj2x), rope_pos_2x),
                *parts,
            ]

        embeds, ropes = [], []
        for embed, rope_pos in parts:
            rope = self.visual_rope_embeddings(embed.shape[:-1], rope_pos, scale_factor)
            embeds.append(embed.flatten(0, 2))
            ropes.append(rope.flatten(0, 2))
        return torch.cat(embeds, dim=0), torch.cat(ropes, dim=0)

    def forward(  # noqa: PLR0913, PLR0915
        self,
        x: Tensor,
        ref: Tensor,
        memory_proj4x: Tensor,
        memory_proj2x: Tensor,
        memory_proj1x: Tensor,
        text_embed: Tensor,
        audio_embed: Tensor,
        pooled_text_embed: Tensor,
        time: Tensor,
        visual_rope_pos: list[Tensor],
        ref_rope_pos: list[Tensor],
        memory_rope_pos_proj4x: list[Tensor],
        memory_rope_pos_proj2x: list[Tensor],
        memory_rope_pos_proj1x: list[Tensor],
        text_rope_pos: Tensor,
        audio_rope_pos: Tensor,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        duration: int | None = None,
    ) -> Tensor:
        device = x.device
        duration = x.shape[0] if duration is None else duration

        text_embed = self.text_embeddings(text_embed)
        audio_embed = self.audio_embeddings(audio_embed)

        time_embed, _ = self.time_embeddings(time)
        time_embed_clean, _ = self.time_embeddings(torch.zeros_like(time))
        pooled = self.pooled_text_embeddings(pooled_text_embed)
        time_embed = time_embed + pooled
        time_embed_clean = time_embed_clean + pooled
        # row 0 → memory / ref (t=0), row 1 → visual (t)
        time_embed_full = torch.stack((time_embed_clean, time_embed), dim=1).flatten(0, 1)

        text_rope = self.text_rope_embeddings(text_rope_pos)
        text_time_embed_idx = torch.zeros(text_embed.shape[0], dtype=torch.int32, device=device)
        for block in self.text_transformer_blocks:
            text_embed = block(text_embed, time_embed, text_rope, text_time_embed_idx)

        audio_rope = self.audio_rope_embeddings(audio_rope_pos)
        for block in self.audio_transformer_blocks:
            audio_embed = block(audio_embed, audio_rope)

        visual_embed = self.visual_embeddings(x)
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        visual_embed = visual_embed.flatten(0, 2)
        visual_rope = visual_rope.flatten(0, 2)

        ref_embed = self.visual_embeddings(ref)
        ref_rope = self.visual_rope_embeddings(ref_embed.shape[:-1], ref_rope_pos, scale_factor)
        ref_embed = ref_embed.flatten(0, 2)
        ref_rope = ref_rope.flatten(0, 2)

        memory_embed, memory_rope = self._embed_memory(
            memory_proj4x,
            memory_proj2x,
            memory_proj1x,
            memory_rope_pos_proj4x,
            memory_rope_pos_proj2x,
            memory_rope_pos_proj1x,
            scale_factor,
        )

        visual_embed = visual_embed + self.mask_embeddings(torch.tensor(0, device=device))
        ref_embed = ref_embed + self.mask_embeddings(torch.tensor(1, device=device))
        memory_embed = memory_embed + self.mask_embeddings(torch.tensor(2, device=device))

        num_visual_tokens = visual_embed.shape[0]
        num_clean_tokens = memory_embed.shape[0] + ref_embed.shape[0]
        visual_embed_full = torch.cat([memory_embed, ref_embed, visual_embed], dim=0)
        visual_rope_full = torch.cat([memory_rope, ref_rope, visual_rope], dim=0)
        time_embed_idx_full = torch.cat(
            [
                torch.zeros(num_clean_tokens, dtype=torch.int32, device=device),
                torch.ones(num_visual_tokens, dtype=torch.int32, device=device),
            ]
        )

        visual_groups = split_lengths(num_visual_tokens, duration, device)
        audio_groups = audio_group_lengths(audio_embed.shape[0], duration, device)

        shard = None
        if self.sequence_parallel is not None:
            shard = SequenceShard.build(
                self.sequence_parallel, visual_embed_full.shape[0], num_clean_tokens, visual_groups, audio_groups
            )
            visual_embed_full = shard.local(visual_embed_full)
            visual_rope_full = shard.local(visual_rope_full)
            time_embed_idx_full = shard.local(time_embed_idx_full)

        for block in self.visual_transformer_blocks:
            visual_embed_full = block(
                visual_embed_full,
                text_embed,
                audio_embed,
                time_embed_full,
                visual_rope_full,
                time_embed_idx_full,
                num_visual_tokens,
                visual_groups,
                audio_groups,
                shard,
            )

        if shard is not None:
            visual_embed_full = shard.gather(visual_embed_full)

        visual_embed = visual_embed_full[-num_visual_tokens:].reshape(*visual_shape, self.model_dim)
        visual_time_embed_idx = torch.zeros(visual_shape[0], dtype=torch.int32, device=device)
        return self.out_layer(visual_embed, time_embed, visual_time_embed_idx)
