from __future__ import annotations

from pathlib import Path

import cyclopts

app = cyclopts.App(help="Kandinsky S2V 1.0 (speech-to-video) inference CLI")

_DEFAULT_CONFIG = str(Path(__file__).parent / "configs" / "s2v_lite.yaml")


@app.command
def generate(  # noqa: PLR0913
    *,
    audio: str,
    image: str,
    config: str = _DEFAULT_CONFIG,
    out: str = "output.mp4",
    prompt: str | None = None,
    negative: str | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    scheduler_scale: float | None = None,
    seed: int | None = None,
    chunks: int | None = None,
    device: str = "cuda",
    attention_engine: str | None = None,
    progress: bool = True,
):
    """Generate an avatar video from a speech track and a reference frame.

    ``--audio`` is the driving speech (wav), ``--image`` the reference frame.
    Geometry, step count and guidance default to the YAML config; the number of
    chunks is derived from the audio length unless ``--chunks`` is given.
    ``--prompt`` / ``--negative`` override the canned prompts in the config.
    """
    from kandinsky_s2v.pipeline.factory import get_s2v_pipeline  # noqa: PLC0415

    pipe = get_s2v_pipeline(config, device=device, attention_engine=attention_engine)
    result = pipe(
        audio=audio,
        image=image,
        text=prompt,
        negative_text=negative,
        num_steps=steps,
        guidance_weight=guidance,
        scheduler_scale=scheduler_scale,
        seed=seed,
        chunks=chunks,
        save_path=out,
        show_progress=progress,
    )
    if result.path is not None:
        print(f"Saved {result.frames.shape[1]} frames → {result.path}")


@app.command
def convert(
    *,
    checkpoint: str,
    out: str,
):
    """Gather an FSDP-sharded training checkpoint into one bf16 safetensors file.

    ``--checkpoint`` is the directory holding ``0.pt``…``N-1.pt``; shards are
    concatenated along dim 0 in rank order.  Works for any shard count, so a
    64-shard checkpoint needs no resharding first.
    """
    from kandinsky_s2v.ckpt.convert import convert as convert_checkpoint  # noqa: PLC0415

    path = convert_checkpoint(checkpoint, out)
    print(f"Wrote {path}")


def main() -> None:
    app()


if __name__ == "__main__":  # torchrun -m kandinsky_s2v.cli generate ...
    main()
