"""How long one chunk costs each configuration, warm, with and without compile.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python dev/bench/time_configs.py \\
        --audio speech.wav --image reference.png

Every pipeline generates the same chunk twice and only the second is reported:
the first pays for lazy CUDA initialisation, allocator growth, attention
autotuning and — with `compile.strategy: torch` — the compilation itself. That
is the number a reader sizing a job needs, since everything but the first chunk
of the first run is paid once.

Sequential on one device on purpose: run these side by side and every figure
grows by whatever the neighbours are doing.

The result is seconds per second of video; a chunk is five seconds at 25 fps.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

CONFIGS = (
    "s2v_lite.yaml",
    "s2v_lite_distill.yaml",
    "s2v_pro.yaml",
    "s2v_pro_distill.yaml",
)
FPS = 25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--configs", nargs="*", default=CONFIGS)
    parser.add_argument("--out", default="outputs/timing")
    args = parser.parse_args()

    from kandinsky_s2v.pipeline.config import load_config  # noqa: PLC0415
    from kandinsky_s2v.pipeline.factory import build_pipeline  # noqa: PLC0415

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.configs:
        for strategy in ("none", "torch"):
            cfg = load_config(f"src/kandinsky_s2v/configs/{name}")
            cfg = cfg.model_copy(update={"compile": cfg.compile.model_copy(update={"strategy": strategy})})
            pipe = build_pipeline(cfg, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            for run in ("warm-up", "measured"):
                started = time.time()
                result = pipe(
                    audio=args.audio,
                    image=args.image,
                    chunks=1,
                    save_path=out / f"{Path(name).stem}.mp4",
                    show_progress=False,
                )
                elapsed = time.time() - started
                print(f"[timing] {name:<28} compile={strategy:<5} {run:<8} {elapsed:7.1f} s", flush=True)
            rate = elapsed / (result.frames.shape[1] / FPS)
            peak = torch.cuda.max_memory_allocated() / 2**30
            results[name, strategy] = (rate, peak)
            print(f"[timing] {name:<28} compile={strategy:<5} → {rate:5.1f} s/s, peak {peak:.0f} GiB", flush=True)
            del pipe
            torch.cuda.empty_cache()

    print("\n| Config | s / s of video | compiled | Peak VRAM |")
    print("|---|---|---|---|")
    for name in args.configs:
        plain, peak = results[name, "none"]
        compiled, _ = results[name, "torch"]
        print(f"| `{name}` | {plain:.0f} s | {compiled:.0f} s | {peak:.0f} GiB |")


if __name__ == "__main__":
    main()
