"""Speed-up and parity of sequence-parallel generation against one GPU.

    .venv/bin/python dev/bench/time_parallel.py --audio speech.wav --image reference.png \\
        --configs src/kandinsky_s2v/configs/s2v_lite_distill.yaml --gpus 1 2 4 --out outputs/parallel

Each (config, N) runs the CLI once, under ``torchrun`` for N > 1, on GPUs
``0..N-1`` of ``CUDA_VISIBLE_DEVICES``. Two numbers per run: the wall time of
the whole process — model loading, warm-up and the VAE decode included, which
is what a user waits for — and the sampling time of the DiT loop from the
progress bar. The N > 1 videos are then diffed against the N = 1 one: with the
head exchange the kernel sees the same sequence, so frames should agree to
kernel noise (a few grey levels), never structurally.

Sequential on purpose; nothing else should be on the node while it runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np

_PROGRESS = re.compile(r"s2v: 100%\|[^|]*\| (\d+)/\1 \[([0-9:]+)<")


def frames(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def diff(a: Path, b: Path) -> dict:
    x, y = frames(a), frames(b)
    n = min(len(x), len(y))
    d = x[:n].astype(np.int16) - y[:n].astype(np.int16)
    mse = (d.astype(np.float64) ** 2).mean(axis=(1, 2, 3))
    psnr = 10 * np.log10(255**2 / np.maximum(mse, 1e-9))
    return {"frames": [len(x), len(y)], "max_abs": int(np.abs(d).max()), "psnr_min": round(float(psnr.min()), 2)}


def seconds(clock: str) -> int:
    parts = [int(p) for p in clock.split(":")]
    return sum(p * 60**i for i, p in enumerate(reversed(parts)))


def run(config: str, gpus: int, args, out: Path) -> dict:
    cli = ["-m", "kandinsky_s2v.cli", "generate", "--config", config, "--audio", args.audio, "--image", args.image]
    cli += ["--chunks", str(args.chunks), "--out", str(out)]
    if gpus == 1:
        cmd = [sys.executable, *cli]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpus}", *cli]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(8))).split(",")[:gpus]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(visible))
    started = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)  # noqa: S603
    wall = time.time() - started
    out.with_suffix(".log").write_text(proc.stdout + proc.stderr)
    bars = _PROGRESS.findall((proc.stdout + proc.stderr).replace("\r", "\n"))
    return {"rc": proc.returncode, "wall_s": round(wall, 1), "sampling_s": seconds(bars[-1][1]) if bars else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--out", default="outputs/parallel")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for config in args.configs:
        name = Path(config).stem
        for gpus in args.gpus:
            out = out_dir / f"{name}_gpu{gpus}.mp4"
            res = run(config, gpus, args, out)
            if res["rc"] == 0 and gpus != 1 and (out_dir / f"{name}_gpu1.mp4").exists():
                res["vs_gpu1"] = diff(out_dir / f"{name}_gpu1.mp4", out)
            results[f"{name}/{gpus}"] = res
            print(f"[parallel] {name:<24} gpus={gpus} {json.dumps(res)}", flush=True)
            (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    print("\n| Config | GPUs | Wall (warm-up incl.) | Sampling | Speed-up (wall / sampling) | max abs diff vs 1 GPU |")
    print("|---|---|---|---|---|---|")
    for config in args.configs:
        name = Path(config).stem
        base = results.get(f"{name}/1")
        for gpus in args.gpus:
            res = results[f"{name}/{gpus}"]
            if res["rc"] != 0:
                print(f"| `{name}.yaml` | {gpus} | failed (rc {res['rc']}) | | | |")
                continue
            speed = ""
            if base and base["rc"] == 0 and res["sampling_s"] and base["sampling_s"]:
                speed = f"{base['wall_s'] / res['wall_s']:.2f}x / {base['sampling_s'] / res['sampling_s']:.2f}x"
            vs = res.get("vs_gpu1", {}).get("max_abs", "")
            print(f"| `{name}.yaml` | {gpus} | {res['wall_s']:.0f} s | {res['sampling_s']} s | {speed} | {vs} |")


if __name__ == "__main__":
    main()
