"""Audio chunking and the FramePack memory carry-over between chunks."""

from __future__ import annotations

import pytest
import torch

from kandinsky_s2v.core.algo.audio_io import chunk_duration_sec, num_chunks_for
from kandinsky_s2v.core.algo.chunks import split_audio_embeds
from kandinsky_s2v.core.algo.prepare_latents import init_memory, roll_memory

CPU = torch.device("cpu")
FRAME_BUCKETS = (2, 4, 24)
COEFS = ((1, 1, 1), (2, 2, 2), (4, 4, 4))
PATCH = (1, 2, 2)


def test_chunk_duration_matches_reference():
    # 32 latent frames → (32-1)*4+1 = 125 pixel frames → 5.0 s at 25 fps
    assert chunk_duration_sec(32, 25) == pytest.approx(5.0)


def test_num_chunks_rounds_up_and_is_at_least_one():
    assert num_chunks_for(12.0, 5.0) == 3  # noqa: PLR2004
    assert num_chunks_for(5.0, 5.0) == 1
    assert num_chunks_for(0.2, 5.0) == 1


def test_split_audio_embeds_drops_ragged_tail():
    embeds = torch.arange(26 * 3, dtype=torch.float32).reshape(26, 3)
    parts = split_audio_embeds(embeds, 5)
    assert len(parts) == 5  # noqa: PLR2004
    assert all(p.shape == (5, 3) for p in parts)
    assert torch.equal(parts[0], embeds[:5])


def test_split_audio_embeds_rejects_too_many_chunks():
    with pytest.raises(ValueError):
        split_audio_embeds(torch.zeros(3, 2), 5)


def test_init_memory_shapes_follow_buckets_and_coefs():
    memory = init_memory(FRAME_BUCKETS, COEFS, PATCH, 32, 48, 16, CPU)
    assert [m.shape[0] for m in memory] == list(FRAME_BUCKETS)
    assert all(m.abs().sum() == 0 for m in memory)


def test_roll_memory_takes_the_tail_of_the_chunk():
    duration = 32
    latent = torch.arange(duration, dtype=torch.float32).reshape(duration, 1, 1, 1)
    latent = latent.expand(duration, 32, 48, 16).contiguous()
    memory_1x, memory_2x, memory_4x = roll_memory(latent, FRAME_BUCKETS, COEFS, PATCH)

    # 1x holds the last 2 frames, 2x the 4 before them, 4x the 24 before those.
    assert memory_1x[:, 0, 0, 0].tolist() == [30, 31]
    assert memory_2x[:, 0, 0, 0].tolist() == [26, 27, 28, 29]
    assert memory_4x[:, 0, 0, 0].tolist() == list(range(2, 26))


def test_roll_memory_crops_spatially_for_downsampled_buckets():
    latent = torch.randn(32, 30, 46, 16)
    memory_1x, memory_2x, memory_4x = roll_memory(latent, FRAME_BUCKETS, COEFS, PATCH)
    assert memory_1x.shape[1:3] == (30, 46)  # multiple of patch 2
    assert memory_2x.shape[1:3] == (28, 44)  # multiple of 4
    assert memory_4x.shape[1:3] == (24, 40)  # multiple of 8


def test_roll_memory_rejects_a_chunk_shorter_than_the_window():
    """Pure algo guards its own precondition — a port calls it without the config."""
    short = torch.randn(sum(FRAME_BUCKETS) - 1, 32, 48, 16)
    with pytest.raises(ValueError, match="FramePack memory window"):
        roll_memory(short, FRAME_BUCKETS, COEFS, PATCH)


def test_roll_memory_accepts_exactly_the_window():
    exact = torch.randn(sum(FRAME_BUCKETS), 32, 48, 16)
    assert [m.shape[0] for m in roll_memory(exact, FRAME_BUCKETS, COEFS, PATCH)] == list(FRAME_BUCKETS)
