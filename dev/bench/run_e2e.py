"""Run one S2V generation without the CLI.

`just generate` is the normal entry point. This script exists for environments
that carry torch and the runtime deps but not the CLI's own (cyclopts) — for
example a CUDA environment matched to an older driver. It also prints timings
and peak VRAM, which makes it the convenient way to benchmark a config:

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        /path/to/python dev/bench/run_e2e.py \
            --config src/kandinsky_s2v/configs/s2v_lite_distill.yaml \
            --audio speech.wav --image reference.png --out outputs/out.mp4
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunks", type=int, default=None, help="default: from the audio length")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Imported here so --help works without the package on sys.path.
    from kandinsky_s2v.pipeline.factory import get_s2v_pipeline  # noqa: PLC0415

    started = time.time()
    pipe = get_s2v_pipeline(args.config, device=args.device)
    print(f"[e2e] pipeline loaded in {time.time() - started:.0f}s", flush=True)

    started = time.time()
    result = pipe(
        audio=args.audio,
        image=args.image,
        chunks=args.chunks,
        num_steps=args.steps,
        save_path=args.out,
        show_progress=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(
        f"[e2e] {result.frames.shape[1]} frames {tuple(result.frames.shape)} "
        f"in {time.time() - started:.0f}s -> {result.path}"
    )
    if torch.cuda.is_available():
        print(f"[e2e] peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
