"""Speech-track helpers: duration, chunk count, silence padding."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np


def audio_duration(path: str | Path) -> float:
    import soundfile as sf  # noqa: PLC0415

    info = sf.info(str(path))
    return info.frames / info.samplerate


def chunk_duration_sec(latent_frames: int, fps: int, temporal_compression_ratio: int = 4) -> float:
    """Seconds of video produced by one chunk."""
    return ((latent_frames - 1) * temporal_compression_ratio + 1) / fps


def num_chunks_for(audio_seconds: float, chunk_seconds: float) -> int:
    return max(1, math.ceil(audio_seconds / chunk_seconds))


def pad_with_silence(path: str | Path, target_duration: float) -> Path | None:
    """Append silence so the track covers ``target_duration``.

    Returns the path of a temporary wav, or ``None`` when no padding is needed
    (the caller then keeps using the original file).
    """
    import soundfile as sf  # noqa: PLC0415

    original = audio_duration(path)
    if target_duration <= original + 0.01:
        return None

    data, sample_rate = sf.read(str(path))
    pad_samples = int((target_duration - original) * sample_rate)
    shape = (pad_samples,) if data.ndim == 1 else (pad_samples, data.shape[1])
    padded = np.concatenate([data, np.zeros(shape, dtype=data.dtype)])

    handle, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    try:
        sf.write(temp_path, padded, sample_rate)
    except BaseException:
        os.unlink(temp_path)
        raise
    return Path(temp_path)


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float32 waveform + sample rate, as written into the output file."""
    import soundfile as sf  # noqa: PLC0415

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sample_rate
