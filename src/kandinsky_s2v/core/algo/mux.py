"""Write generated frames plus the driving speech track into an mp4 (PyAV)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
from torch import Tensor


def _require_av():
    try:
        import av  # noqa: PLC0415
    except (ImportError, OSError) as exc:
        raise RuntimeError("S2V video output requires PyAV and FFmpeg. Install with `pip install av`.") from exc
    return av


def mux_video_audio(  # noqa: PLR0913
    frames: Tensor,
    audio: np.ndarray | None,
    output_path: str | Path,
    fps: int = 25,
    audio_sample_rate: int = 44100,
    video_crf: int = 18,
) -> Path:
    """``frames``: (3, T, H, W) uint8. ``audio``: mono float32 in [-1, 1] or None."""
    av = _require_av()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_np = frames.detach().permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8, copy=False)

    with av.open(str(output_path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=fps)
        video_stream.width = video_np.shape[2]
        video_stream.height = video_np.shape[1]
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": str(video_crf)}

        audio_stream = None
        if audio is not None:
            audio_stream = container.add_stream("aac", rate=audio_sample_rate)
            audio_stream.layout = "mono"
            audio_stream.bit_rate = 192_000

        for i, frame_np in enumerate(video_np):
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in video_stream.encode(frame):
                container.mux(packet)

        if audio_stream is not None:
            samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
            audio_frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="fltp", layout="mono")
            audio_frame.sample_rate = audio_sample_rate
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, audio_sample_rate)
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

        for packet in video_stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            for packet in audio_stream.encode():
                container.mux(packet)

    return output_path
