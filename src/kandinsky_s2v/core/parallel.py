"""Sequence parallelism for the visual stream of the S2V DiT (Ulysses layout).

The packed visual sequence ``[memory | ref | visual]`` is cut into one
contiguous shard per rank. Every rank holds the full weights and runs every
block on its own shard; the only exchange is inside self-attention, where an
``all_to_all`` turns "all heads of my tokens" into "my heads of all tokens" and
back, so the attention kernel itself sees the whole sequence. Text cross-
attention needs no exchange (its keys are replicated), and the frame-aligned
audio adapter attends per frame, so each rank serves the frames its shard
covers. The shards are gathered once before the output head.

This is the same layout as the reference ``parallelize_seq`` in kandinsky-5,
written with plain collectives instead of DTensor. Heads must divide by the
world size: 28 on Lite (2, 4), 32 on Pro (2, 4, 8).

Activated by the launcher: under ``torchrun`` (``WORLD_SIZE`` > 1) the factory
attaches a :class:`SequenceParallel` to the DiT; a plain ``python`` run never
touches ``torch.distributed``. One process group generates one video, so two
runs on separate GPU groups never share a collective.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import Tensor

_TIMEOUT = timedelta(hours=3)


@dataclass(frozen=True)
class SequenceParallel:
    rank: int
    world_size: int

    @classmethod
    def from_env(cls) -> SequenceParallel | None:
        """The group ``torchrun`` describes, or None when not launched by it."""
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:  # noqa: PLR2004
            return None
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            if backend == "nccl":
                torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            dist.init_process_group(backend, timeout=_TIMEOUT)
        return cls.current()

    @classmethod
    def current(cls) -> SequenceParallel:
        return cls(dist.get_rank(), dist.get_world_size())

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available() and dist.get_backend() == "nccl":
            return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        return torch.device("cpu")

    def broadcast(self, *tensors: Tensor) -> None:
        """Overwrite every rank's copy with rank 0's, in place."""
        for tensor in tensors:
            buffer = tensor.contiguous()  # collectives refuse strided views
            dist.broadcast(buffer, src=0)
            if buffer is not tensor:
                tensor.copy_(buffer)

    def check_heads(self, num_heads: int) -> None:
        if num_heads % self.world_size:
            raise ValueError(
                f"{num_heads} attention heads do not divide across {self.world_size} processes; "
                "use a world size that divides the head count (28 on Lite, 32 on Pro)"
            )


def shard_bounds(total: int, parts: int) -> list[int]:
    """Boundaries of ``parts`` contiguous shards; the remainder goes to the first ranks."""
    base, extra = divmod(total, parts)
    bounds = [0]
    for i in range(parts):
        bounds.append(bounds[-1] + base + (1 if i < extra else 0))
    return bounds


@dataclass(frozen=True)
class SequenceShard:
    """One rank's slice of the packed sequence, fixed for one DiT forward."""

    sp: SequenceParallel
    sizes: list[int]  # shard length per rank
    start: int
    end: int
    # Audio adapter: the local tokens from ``visual_offset`` on are noisy visual
    # tokens of frames ``frame_lengths`` (query groups), fed by audio rows
    # ``audio_start:audio_end`` split as ``audio_lengths``.
    visual_offset: int
    frame_lengths: list[int]
    audio_start: int
    audio_end: int
    audio_lengths: list[int]

    @classmethod
    def build(  # noqa: PLR0913
        cls,
        sp: SequenceParallel,
        total: int,
        num_clean: int,
        visual_groups: Tensor,
        audio_groups: Tensor,
    ) -> SequenceShard:
        bounds = shard_bounds(total, sp.world_size)
        start, end = bounds[sp.rank], bounds[sp.rank + 1]
        # Visual tokens are the suffix of the sequence, so within a contiguous
        # shard they are a suffix too.
        first_visual = max(start, num_clean)
        g0, g1 = first_visual - num_clean, max(end, num_clean) - num_clean
        frame_lengths: list[int] = []
        audio_lengths: list[int] = []
        audio_start = audio_end = 0
        if g1 > g0:
            vb = [0, *torch.cumsum(visual_groups, 0).tolist()]
            ab = [0, *torch.cumsum(audio_groups, 0).tolist()]
            f0 = bisect.bisect_right(vb, g0) - 1
            f1 = bisect.bisect_right(vb, g1 - 1) - 1
            frame_lengths = [min(vb[f + 1], g1) - max(vb[f], g0) for f in range(f0, f1 + 1)]
            audio_lengths = audio_groups[f0 : f1 + 1].tolist()
            audio_start, audio_end = ab[f0], ab[f1 + 1]
        return cls(
            sp=sp,
            sizes=[bounds[i + 1] - bounds[i] for i in range(sp.world_size)],
            start=start,
            end=end,
            visual_offset=first_visual - start,
            frame_lengths=frame_lengths,
            audio_start=audio_start,
            audio_end=audio_end,
            audio_lengths=audio_lengths,
        )

    @property
    def local_size(self) -> int:
        return self.end - self.start

    def local(self, x: Tensor) -> Tensor:
        return x[self.start : self.end]

    def gather(self, x: Tensor) -> Tensor:
        """Concatenate every rank's shard back into the full sequence."""
        # all_gather wants equal buffers: pad to the largest shard, trim after.
        width = max(self.sizes)
        padded = x.new_zeros(width, *x.shape[1:])
        padded[: x.shape[0]] = x
        parts = [torch.empty_like(padded) for _ in self.sizes]
        dist.all_gather(parts, padded)
        return torch.cat([part[:size] for part, size in zip(parts, self.sizes, strict=True)], dim=0)

    def tokens_to_heads(self, qkv: Tensor) -> Tensor:
        """``(S_local, 3, H, D)`` → ``(S, 3, H / world, D)`` for this rank's heads."""
        n = self.sp.world_size
        s_local, three, heads, dim = qkv.shape
        send = (
            qkv.view(s_local, three, n, heads // n, dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
            .view(n * s_local, three, heads // n, dim)
        )
        recv = qkv.new_empty(sum(self.sizes), three, heads // n, dim)
        dist.all_to_all_single(recv, send, output_split_sizes=self.sizes, input_split_sizes=[s_local] * n)
        return recv

    def heads_to_tokens(self, out: Tensor) -> Tensor:
        """``(S, H / world, D)`` → ``(S_local, H, D)``: the inverse exchange."""
        n = self.sp.world_size
        s_local = self.local_size
        _, heads_local, dim = out.shape
        recv = out.new_empty(n * s_local, heads_local, dim)
        dist.all_to_all_single(recv, out.contiguous(), output_split_sizes=[s_local] * n, input_split_sizes=self.sizes)
        return recv.view(n, s_local, heads_local, dim).permute(1, 0, 2, 3).reshape(s_local, n * heads_local, dim)
