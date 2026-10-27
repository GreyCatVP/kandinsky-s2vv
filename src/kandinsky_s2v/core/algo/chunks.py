"""Splitting the speech features across autoregressive chunks."""

from __future__ import annotations

from torch import Tensor


def split_audio_embeds(audio_embeds: Tensor, chunks: int) -> list[Tensor]:
    """Split ``(S_audio, D)`` evenly over ``chunks``, dropping the ragged tail.

    Matches K5 ``generate_sample``: the whole track is encoded once and the
    feature sequence is cut into equal pieces, one per chunk.
    """
    if chunks < 1:
        raise ValueError(f"chunks must be >= 1, got {chunks}")
    usable = (audio_embeds.shape[0] // chunks) * chunks
    if usable == 0:
        raise ValueError(f"{audio_embeds.shape[0]} audio tokens cannot be split into {chunks} chunks")
    return list(audio_embeds[:usable].view(chunks, -1, audio_embeds.shape[1]).unbind(0))
