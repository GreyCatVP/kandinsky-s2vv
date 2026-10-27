"""Do the two pipelines hand the transformer the same conditioning?

The second link in the chain, after the transformer forward itself. Both sides
use the same encoder weights, so what is under test is the port's own prompt
slicing, speech extraction and CLIP pooling — and the answer should be bitwise
identical, not merely close.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
        dev/bench/s2v_conditioning_parity.py \\
        --config src/kandinsky_s2v/configs/s2v_lite.yaml --audio speech.wav
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from kandinsky_s2v.pipeline.config import load_config
from kandinsky_s2v.pipeline.factory import build_pipeline
from kandinsky_s2v.ports.diffusers.modular.encoders import encode_clip, encode_qwen_omni

TEMPORAL_COMPRESSION_RATIO = 4


def compare(name: str, native: torch.Tensor, ported: torch.Tensor) -> bool:
    if native.shape != ported.shape:
        print(f"[cond] {name:<9} SHAPES DIFFER: {tuple(native.shape)} vs {tuple(ported.shape)}")
        return False
    if torch.equal(native, ported):
        print(f"[cond] {name:<9} {tuple(native.shape)}  identical")
        return True
    rel = (ported.float() - native.float()).norm() / native.float().norm()
    print(f"[cond] {name:<9} {tuple(native.shape)}  DIFFERS, relative L2 {rel:.3e}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)
    gen = cfg.generation

    native = build_pipeline(cfg, device, "sdpa")
    # The encoders are shared on purpose: only the port's own extraction is under test.
    components = SimpleNamespace(
        text_encoder=native.embedder.thinker,
        tokenizer=native.embedder.processor,
        text_encoder_2=native.embedder.clip,
        tokenizer_2=native.embedder.clip_tokenizer,
    )

    chunk_seconds = ((gen.latent_frames - 1) * TEMPORAL_COMPRESSION_RATIO + 1) / gen.fps
    target_seconds = args.chunks * chunk_seconds

    native_text, native_pooled, native_audio = native.embedder.encode(gen.prompt, args.audio, 0.0, target_seconds)
    ported_text, ported_audio = encode_qwen_omni(components, gen.prompt, args.audio, target_seconds, device)
    ported_pooled = encode_clip(components, gen.prompt, device)

    native_negative, _, _ = native.embedder.encode(gen.negative_prompt)
    ported_negative, _ = encode_qwen_omni(components, gen.negative_prompt, None, None, device)

    ok = all(
        [
            compare("text", native_text, ported_text),
            compare("audio", native_audio, ported_audio),
            compare("pooled", native_pooled, ported_pooled),
            compare("negative", native_negative, ported_negative),
        ]
    )
    print("[cond] PASSED" if ok else "[cond] FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
