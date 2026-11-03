"""Hugging Face Space entry point: Kandinsky S2V 1.0 speech-to-video demo.

The ``kandinsky_s2v`` package is uploaded next to this file (see README.md).
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import librosa
import soundfile as sf
import spaces

import kandinsky_s2v
from kandinsky_s2v.pipeline.factory import get_s2v_pipeline

CONFIG = os.environ.get("S2V_CONFIG", "s2v_lite_distill.yaml")
# ZeroGPU allots GPU time per call, so the track is cut to one or two chunks (5 s each).
# GPU seconds requested per clip length, Lite distilled on a ZeroGPU xlarge slot (RTX Pro 6000,
# SDPA): one chunk took 70-77 s, two took 139 s; the rest is headroom. xlarge bills the request
# twice, and a free account's daily quota is 300 s, so only the 5 s clip fits it.
GPU_DURATION = {5: 100, 10: 180}
DEFAULT_SECONDS = 5
GPU_SIZE = os.environ.get("S2V_ZEROGPU_SIZE", "xlarge")
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
# Gradio copies a returned video into its own cache, so ours only has to outlive the call.
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="kandinsky_s2v_"))
OUTPUT_TTL_SECONDS = 3600

# Built at module level: on ZeroGPU the weights are placed on the emulated CUDA
# device here, and the GPU worker forked per call inherits them. FA3 is not
# available on Spaces, so attention runs through SDPA.
pipe = get_s2v_pipeline(
    Path(kandinsky_s2v.__file__).parent / "configs" / CONFIG,
    device="cuda",
    attention_engine="sdpa",
)


def trimmed_wav(audio: str, seconds: float, path: Path) -> str:
    """The first ``seconds`` of ``audio`` as a wav at ``path``.

    Trimming the track, not passing fewer ``chunks``: the pipeline splits the
    embeddings of the whole track across the chunks it is given.
    """
    samples, sample_rate = librosa.load(audio, sr=None, mono=True, duration=seconds)
    sf.write(path, samples, sample_rate)
    return str(path)


def prune_outputs() -> None:
    """Drop results old enough that Gradio has long since copied them."""
    cutoff = time.time() - OUTPUT_TTL_SECONDS
    for old in OUTPUT_DIR.glob("*.mp4"):
        if old.stat().st_mtime < cutoff:
            old.unlink(missing_ok=True)


def full_prompt(extra: str | None) -> str:
    """The config's default prompt, with the user's text appended to it rather than replacing it."""
    default = pipe.cfg.generation.prompt
    extra = (extra or "").strip()
    return f"{default} {extra}" if extra else default


def gpu_duration(image, audio, prompt, seed, seconds, *_, **__) -> int:
    """GPU seconds to request: ZeroGPU calls this with ``generate``'s arguments."""
    return GPU_DURATION[int(seconds)]


@spaces.GPU(duration=gpu_duration, size=GPU_SIZE)
def generate(  # noqa: PLR0913 - one argument per Gradio input
    image: str | None,
    audio: str | None,
    prompt: str | None,
    seed: float | None,
    seconds: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),  # noqa: B008 - Gradio injects the tracker
) -> str:
    if not image or not audio:
        raise gr.Error("Upload a reference frame and a speech track.")
    progress(0, desc="Encoding the reference frame and the speech")
    prune_outputs()
    out = OUTPUT_DIR / f"avatar_{uuid.uuid4().hex}.mp4"
    with tempfile.TemporaryDirectory(prefix="kandinsky_s2v_") as work:
        # The pipeline's tqdm bar (one tick per denoising step, all chunks) is mirrored into the UI.
        result = pipe(
            audio=trimmed_wav(audio, seconds, Path(work) / "speech.wav"),
            image=image,
            text=full_prompt(prompt),
            seed=None if seed is None else int(seed),  # a cleared field falls back to the config's seed
            save_path=out,
            show_progress=True,
        )
    return str(result.path)


def example_rows() -> list[list[str | int]]:
    """``examples/<name>.png`` paired with ``examples/<name>.wav``."""
    return [
        [str(image), str(image.with_suffix(".wav")), "", 5, DEFAULT_SECONDS]
        for image in sorted(EXAMPLES_DIR.glob("*.png"))
        if image.with_suffix(".wav").exists()
    ]


def badge(label: str, value: str, color: str, url: str) -> str:
    """A shields.io-style badge drawn in HTML: an external image did not render in the page."""
    part = "padding:2px 7px; font:11px Verdana, sans-serif; color:#fff"
    return (
        f'<a href="{url}" target="_blank" style="display:inline-flex; border-radius:3px; overflow:hidden; '
        f'text-decoration:none"><span style="{part}; background:#555">{label}</span>'
        f'<span style="{part}; background:{color}">{value}</span></a>'
    )


BADGES = "".join(
    (
        badge("Project", "Page", "#4c1", "https://kandinskyavatar.github.io"),
        badge(
            "🤗 Hugging Face",
            "Models",
            "#dfb317",
            "https://huggingface.co/collections/kandinskylab/kandinsky-s2v-10-6ab65ebe8f765b6dd9d706d7",
        ),
        badge("GitHub", "Code", "#007ec6", "https://github.com/kandinskylab/kandinsky-s2v"),
    )
)

with gr.Blocks(title="Kandinsky S2V 1.0") as demo:
    # Sized to fit one screen: a short header, and fixed heights for the frame and the result.
    gr.HTML(
        '<div style="display:flex; align-items:baseline; gap:14px; flex-wrap:wrap">'
        '<h2 style="margin:0">Kandinsky S2V 1.0</h2>'
        '<span style="font-size:16px; opacity:.75">An avatar video from a <b>reference frame</b> and a '
        "<b>speech track</b>; the first 5 or 10 s of the track are used.</span>"
        "</div>"
        f'<div style="display:flex; gap:6px; margin-top:8px">{BADGES}</div>'
    )
    with gr.Row():
        with gr.Column():
            # No fullscreen button: leaving fullscreen left the uploaded frame blank (Gradio 6.27).
            image = gr.Image(type="filepath", label="Reference frame", height=280, buttons=["download"])
            audio = gr.Audio(type="filepath", label="Speech")
            seconds = gr.Radio(
                choices=list(GPU_DURATION),
                value=DEFAULT_SECONDS,
                label="Video length, s",
                info="10 s needs more GPU time than a free account's daily ZeroGPU quota; "
                "it works with PRO or a Team/Enterprise organisation.",
            )
            with gr.Accordion("Advanced settings", open=False):
                prompt = gr.Textbox(label="Prompt", placeholder="appended to the default prompt")
                seed = gr.Number(value=5, precision=0, minimum=0, label="Seed")
            run = gr.Button("Generate", variant="primary")
            rows = example_rows()
            if rows:
                gr.Examples(rows, inputs=[image, audio, prompt, seed, seconds])
        with gr.Column():
            video = gr.Video(label="Result", interactive=False, autoplay=True, height=520)
    run.click(generate, inputs=[image, audio, prompt, seed, seconds], outputs=video)

# Dark regardless of the viewer's system setting: Gradio styles everything under body.dark.
DARK_THEME_JS = "() => document.body.classList.add('dark')"

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(js=DARK_THEME_JS, ssr_mode=False, show_error=True)
