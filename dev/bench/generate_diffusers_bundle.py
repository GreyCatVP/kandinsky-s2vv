"""Generate from a Diffusers bundle the way a user of the published model would.

    PYTHONPATH=$DIFFUSERS/src:src CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
        dev/bench/generate_diffusers_bundle.py \\
        --bundle /path/to/Kandinsky-S2V-1.0-Pro-Diffusers \\
        --image reference.png --audio speech.wav --out out.mp4 --chunks 2 --offload

The point is what is *not* here: `ModularPipeline.from_pretrained` resolves the
class names in `modular_model_index.json` against the installed `diffusers` —
the pipeline and the blocks, base or distilled, both come from the bundle — so
this runs only if the port has been injected into the Diffusers on the path; put
the branch's `src` first. Nothing of this project is imported except the muxer, which
puts the speech track back on the result; the script says so before generating.

[load_diffusers_bundle.py](load_diffusers_bundle.py) is the other half: it builds
the port's standard pipeline out of the bundle, which works *before* injection but
proves nothing about it.

Steps and guidance are left to the blocks: the distilled ones take 16 evaluations
and a guider at 1.0, since the guidance is distilled into the student.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import ComponentsManager, ModularPipeline

FPS = 25


def offload_all_others(hooks, model_id, model, execution_device):
    """Offload strategy: every other resident model leaves before the next one loads."""
    return hooks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunks", type=int, default=1, help="5 s of video each")
    parser.add_argument("--steps", type=int, default=None, help="default: the scheduler's own")
    parser.add_argument(
        "--offload",
        action="store_true",
        help="required for Pro on an 80 GB card, and what the Pro configs do anyway",
    )
    args = parser.parse_args()

    started = time.time()
    manager = ComponentsManager() if args.offload else None
    pipe = ModularPipeline.from_pretrained(args.bundle, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    # The VAE is trained in fp16 and the native pipeline runs it that way.
    pipe.vae.to(torch.float16)
    if manager is not None:
        # One model on the device at a time, as `enable_model_cpu_offload` does for a
        # standard pipeline. The default strategy only moves models aside when the next
        # one's weights do not fit; the Pro transformer fits beside the speech encoder,
        # and the VAE, resident since it encoded the reference frame, never triggers a
        # move before the decode that needs tens of GiB with the transformer gone.
        # Enabled only now, after the components are in: adding one re-enables the
        # offload with the default strategy, dropping the one passed here.
        manager.enable_auto_cpu_offload(device="cuda", offload_strategy=offload_all_others)
    else:
        pipe.to("cuda")

    kind = f"{type(pipe).__name__} + {type(pipe.blocks).__name__} + {type(pipe.scheduler).__name__}"
    print(f"[gen] {kind} in {time.time() - started:.0f}s", flush=True)
    print(f"[gen] imported from {type(pipe).__module__}", flush=True)

    started = time.time()
    inputs = {"image": args.image, "audio": args.audio, "num_chunks": args.chunks, "output_type": "np"}
    if args.steps is not None:
        inputs["num_inference_steps"] = args.steps
    frames = pipe(**inputs, output="videos")[0]
    print(f"[gen] {frames.shape[0]} frames {frames.shape[1:]} in {time.time() - started:.0f}s", flush=True)
    if torch.cuda.is_available():
        print(f"[gen] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    # The project's own muxer rather than `export_to_video`: it needs no opencv and
    # it puts the speech back on the result, which is what makes the clip watchable.
    from kandinsky_s2v.core.algo.audio_io import read_audio  # noqa: PLC0415
    from kandinsky_s2v.core.algo.mux import mux_video_audio  # noqa: PLC0415

    video = torch.from_numpy(frames).permute(3, 0, 1, 2).mul(255).round().clamp(0, 255).to(torch.uint8)
    samples, sample_rate = read_audio(args.audio)
    samples = samples[: int(video.shape[1] / FPS * sample_rate)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mux_video_audio(video, samples, args.out, fps=FPS, audio_sample_rate=sample_rate)
    print(f"[gen] wrote {args.out}")


if __name__ == "__main__":
    main()
