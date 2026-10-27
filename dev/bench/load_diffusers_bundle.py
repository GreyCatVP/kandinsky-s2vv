"""Load a Diffusers bundle built by `build_diffusers_bundle.py`, and optionally generate.

`ModularPipeline.from_pretrained` resolves the class names in
`modular_model_index.json` against the installed `diffusers` package, which does
not know about this port yet. This script does the same resolution but looks in
the port first — the pipeline, the blocks and the components — so the bundle can
be used, and checked, before the port is injected into a Diffusers install.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
        dev/bench/load_diffusers_bundle.py --bundle /path/to/...-Diffusers \\
        --audio speech.wav --image reference.png --out out.mp4
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import torch

from kandinsky_s2v.ports.diffusers import modular, scheduling_piflow, transformer_kandinsky_s2v

PORT_MODULES = (transformer_kandinsky_s2v, scheduling_piflow, modular)


def resolve(library: str, name: str):
    """Find a component class: the port owns some names, the named library the rest."""
    for module in PORT_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    return getattr(importlib.import_module(library), name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--audio")
    parser.add_argument("--image")
    parser.add_argument("--out")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = Path(args.bundle)
    index = json.loads((root / "modular_model_index.json").read_text())
    pipeline_class = resolve("diffusers", index["_class_name"])
    blocks = resolve("diffusers", index["_blocks_class_name"])()

    started = time.time()
    components = {}
    for name, entry in index.items():
        if name.startswith("_"):
            continue
        library, class_name = entry[0], entry[1]
        cls = resolve(library, class_name)
        # Only the weighted components take a dtype; tokenizers and the scheduler do not.
        kwargs = {} if name.startswith("tokenizer") or name == "scheduler" else {"torch_dtype": torch.bfloat16}
        components[name] = cls.from_pretrained(root, subfolder=name, **kwargs)
        print(f"[bundle] {name:<15} {class_name}", flush=True)
    # The VAE is trained in fp16 and the native pipeline runs it that way.
    components["vae"] = components["vae"].to(torch.float16)

    pipe = pipeline_class(blocks=blocks)
    pipe.update_components(**components)
    pipe.to(args.device)
    print(f"[bundle] {pipeline_class.__name__} + {type(blocks).__name__} built in {time.time() - started:.0f}s")

    if not (args.audio and args.image):
        return

    started = time.time()
    inputs = {"image": args.image, "audio": args.audio, "num_chunks": args.chunks, "output_type": "np"}
    if args.steps is not None:
        inputs["num_inference_steps"] = args.steps
    frames = pipe(**inputs, output="videos")[0]
    print(f"[bundle] {frames.shape[0]} frames {frames.shape} in {time.time() - started:.0f}s")
    if torch.cuda.is_available():
        print(f"[bundle] peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    if args.out:
        # The project's own muxer rather than `export_to_video`: it needs no opencv
        # and it puts the speech track back on the result, which is the point here.
        from kandinsky_s2v.core.algo.audio_io import read_audio  # noqa: PLC0415
        from kandinsky_s2v.core.algo.mux import mux_video_audio  # noqa: PLC0415

        video = torch.from_numpy(frames).permute(3, 0, 1, 2).mul(255).round().clamp(0, 255).to(torch.uint8)
        samples, sample_rate = read_audio(args.audio)
        samples = samples[: int(video.shape[1] / 25 * sample_rate)]
        mux_video_audio(video, samples, args.out, fps=25, audio_sample_rate=sample_rate)
        print(f"[bundle] wrote {args.out}")


if __name__ == "__main__":
    main()
