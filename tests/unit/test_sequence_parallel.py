"""Sequence-parallel DiT forward must equal the single-process forward.

Two processes run the tiny DiT of ``test_s2v_dit`` with the sequence sharded
between them; rank 0 also runs the same weights unsharded and compares. This
covers the all_to_all head exchange, the per-shard audio adapter (its frame
groups are cut mid-frame at the shard boundary) and the final gather, on an
uneven split. Always over gloo on CPU; over NCCL on two GPUs when the machine
has them, which is the exchange the real runs use.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kandinsky_s2v.core.parallel import SequenceParallel, SequenceShard, shard_bounds

from .test_s2v_dit import TINY, _inputs, _randomize_modulations

WORLD = 2  # TINY has model_dim 32 / head_dim 16 = 2 heads
# Eight heads (64 / 8), so eight shards on CPU: the GPU runs at 8 are not bit-identical
# to 1 GPU, and this is what shows the sharding itself is exact there.
TINY8 = {**TINY, "model_dim": 64, "ff_dim": 128, "axes_dims": (4, 2, 2)}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init(rank: int, port: int, backend: str, world: int = WORLD) -> torch.device:
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    return torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")


def _exchange_roundtrip(rank: int, port: int, backend: str) -> None:
    device = _init(rank, port, backend)
    sp = SequenceParallel.current()
    total, heads, dim = 7, 4, 3  # uneven: shards of 4 and 3
    bounds = shard_bounds(total, WORLD)
    torch.manual_seed(0)
    full = torch.randn(total, 3, heads, dim, device=device)
    shard = SequenceShard(sp, [bounds[1], total - bounds[1]], bounds[rank], bounds[rank + 1], 0, [], 0, 0, [])
    by_heads = shard.tokens_to_heads(shard.local(full))
    expected = full[:, :, rank * heads // WORLD : (rank + 1) * heads // WORLD]
    assert torch.equal(by_heads, expected)
    back = shard.heads_to_tokens(by_heads[:, 0])
    assert torch.equal(back, shard.local(full[:, 0]))
    assert torch.equal(shard.gather(shard.local(full)), full)
    dist.destroy_process_group()


def _to(obj, device: torch.device):
    """Move the tensors inside a context dataclass; leave ints, floats and tuples of them alone."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [_to(item, device) for item in obj]
    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            setattr(obj, field.name, _to(getattr(obj, field.name), device))
    return obj


def _forward_matches(rank: int, port: int, backend: str, world: int = WORLD, config: dict = TINY) -> None:
    from kandinsky_s2v.core.algo.denoise_loop import dit_forward  # noqa: PLC0415
    from kandinsky_s2v.core.components.dit import DiffusionTransformer3DS2V  # noqa: PLC0415

    device = _init(rank, port, backend, world)
    sp = SequenceParallel.current()
    torch.manual_seed(0)
    dit = DiffusionTransformer3DS2V(**config).eval()
    _randomize_modulations(dit)
    dit = dit.to(device)
    latent, ctx, text = _inputs()
    latent, ctx, text = latent.to(device), _to(ctx, device), _to(text, device)
    time = torch.tensor([0.7], device=device)

    sharded = copy.deepcopy(dit)
    sharded.sequence_parallel = sp
    with torch.no_grad():
        parallel = dit_forward(sharded, latent, time, ctx, text)
        if rank == 0:
            single = dit_forward(dit, latent, time, ctx, text)
            assert parallel.shape == single.shape
            assert torch.allclose(parallel, single, atol=1e-5, rtol=1e-4), (parallel - single).abs().max()
    dist.destroy_process_group()


BACKENDS = ["gloo"] + (["nccl"] if torch.cuda.is_available() and torch.cuda.device_count() >= WORLD else [])


def _spawn(target, backend: str, *extra) -> None:
    os.environ.pop("WORLD_SIZE", None)
    world = extra[0] if extra else WORLD
    mp.spawn(target, args=(_free_port(), backend, *extra), nprocs=world, join=True)


def test_shard_bounds_give_remainder_to_first_ranks():
    assert shard_bounds(10, 3) == [0, 4, 7, 10]
    assert shard_bounds(8, 4) == [0, 2, 4, 6, 8]


def test_shard_cuts_audio_frames_at_the_boundary():
    """A shard's audio groups are the frames its visual tokens touch, clipped."""
    sp = SequenceParallel(rank=1, world_size=2)
    visual_groups = torch.tensor([3, 3, 3, 3])  # 4 frames × 3 tokens
    audio_groups = torch.tensor([1, 2, 2, 2])
    # 2 clean + 12 visual = 14 tokens → rank 1 holds tokens 7..14 = visual 5..12
    shard = SequenceShard.build(sp, total=14, num_clean=2, visual_groups=visual_groups, audio_groups=audio_groups)
    assert (shard.start, shard.end, shard.visual_offset) == (7, 14, 0)
    assert shard.frame_lengths == [1, 3, 3]  # frame 1 cut to its last token
    assert shard.audio_lengths == [2, 2, 2]
    assert (shard.audio_start, shard.audio_end) == (1, 7)

    rank0 = SequenceShard.build(
        SequenceParallel(0, 2), total=14, num_clean=2, visual_groups=visual_groups, audio_groups=audio_groups
    )
    assert (rank0.start, rank0.end, rank0.visual_offset) == (0, 7, 2)
    assert rank0.frame_lengths == [3, 2]
    assert (rank0.audio_start, rank0.audio_end) == (0, 3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_head_exchange_roundtrip(backend):
    _spawn(_exchange_roundtrip, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sharded_forward_matches_single_process(backend):
    _spawn(_forward_matches, backend)


def test_eight_shards_match_single_process():
    _spawn(_forward_matches, "gloo", 8, TINY8)


@pytest.mark.parametrize("heads", [28, 32])
def test_world_size_must_divide_heads(heads):
    SequenceParallel(0, 4).check_heads(heads)
    with pytest.raises(ValueError, match="do not divide"):
        SequenceParallel(0, 8).check_heads(28)
