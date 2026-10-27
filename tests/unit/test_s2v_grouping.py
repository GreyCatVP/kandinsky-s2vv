"""Frame-alignment maths shared by the audio cross-attention and audio RoPE."""

from __future__ import annotations

import pytest
import torch

from kandinsky_s2v.core.algo.prepare_ropes import audio_rope_pos
from kandinsky_s2v.core.components.dit import audio_group_lengths, split_lengths

CPU = torch.device("cpu")


def test_split_lengths_is_even_when_divisible():
    assert split_lengths(32 * 768, 32, CPU).tolist() == [768] * 32


def test_split_lengths_distributes_remainder():
    lengths = split_lengths(10, 4, CPU)
    assert lengths.sum().item() == 10  # noqa: PLR2004
    assert lengths.tolist() == [2, 3, 2, 3]


def test_audio_groups_give_first_frame_one_token():
    # 125 audio tokens over 32 latent frames: 1 + 31*4 (K5 get_spatial_audio_cu_seqlens)
    lengths = audio_group_lengths(125, 32, CPU)
    assert lengths[0].item() == 1
    assert lengths[1:].tolist() == [4] * 31
    assert lengths.sum().item() == 125  # noqa: PLR2004


def test_audio_rope_repeats_frame_index_per_token():
    duration, tokens = 8, 25
    positions = audio_rope_pos(tokens, duration, CPU)
    lengths = audio_group_lengths(tokens, duration, CPU)
    assert positions.shape[0] == tokens
    assert positions.tolist() == torch.arange(duration).repeat_interleave(lengths).tolist()


def test_audio_groups_and_rope_agree_for_uneven_splits():
    for duration, tokens in [(32, 125), (16, 77), (8, 25)]:
        lengths = audio_group_lengths(tokens, duration, CPU)
        assert lengths.sum().item() == tokens
        assert audio_rope_pos(tokens, duration, CPU).shape[0] == tokens


def test_audio_groups_reject_single_frame():
    """The first frame consumes one token, so a second frame is required."""
    with pytest.raises(ValueError, match="duration must be >= 2"):
        audio_group_lengths(125, 1, CPU)
