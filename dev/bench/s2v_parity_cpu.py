"""CPU parity check: kandinsky_s2v DiT vs the kandinsky-5 reference DiT.

The reference implementation calls flash-attention varlen kernels, which need a
GPU.  To make the comparison runnable on CPU its two attention entry points are
replaced with mathematically equivalent SDPA implementations driven by the same
``cu_seqlens``.  Everything else — block order, modulation, packing, RoPE,
role embeddings, the output layer — is exercised unchanged, so a match here
means the port reproduces the reference architecture.

Usage:
    python dev/bench/s2v_parity_cpu.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

# The kandinsky-5 reference checkout is not part of this repo; point at it with
# KANDINSKY5_REPO (CLAUDE.md expects reference material under dev/ref/, which is
# where it lands once vendored).
_DEFAULT_REFERENCE_REPO = Path(__file__).resolve().parents[2] / "dev" / "ref" / "kandinsky-5"
REFERENCE_REPO = Path(os.environ.get("KANDINSKY5_REPO", _DEFAULT_REFERENCE_REPO))


def _install_flash_attn_stub() -> None:
    """Stub flash-attn: the reference imports it at module level, but this script
    replaces every kernel that would actually call it (see _patch_reference_attention),
    so the package itself is not needed — and it is GPU-only anyway."""
    if importlib.util.find_spec("flash_attn") is not None:
        return
    stub = types.ModuleType("flash_attn")
    for name in ("flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_func", "flash_attn_func"):
        setattr(stub, name, _unavailable)
    sys.modules["flash_attn"] = stub


def _unavailable(*_args, **_kwargs):
    raise RuntimeError("flash-attn is stubbed in this parity script; attention is patched to SDPA")


def _install_reference_stubs() -> None:
    """Stub the reference logger (it imports lightning, which we do not need)."""
    stub = types.ModuleType("kandinsky.model.logger")

    class ModelLogger:
        @staticmethod
        def log_attention(cls):  # noqa: PLW0211
            return cls

        @staticmethod
        def log_transformer_block(cls):  # noqa: PLW0211
            return cls

        @staticmethod
        def log_transformer(cls):  # noqa: PLW0211
            return cls

    stub.ModelLogger = ModelLogger
    sys.modules["kandinsky.model.logger"] = stub


def _patch_reference_attention() -> None:
    """Swap the reference varlen flash kernels for CPU-runnable SDPA."""
    from kandinsky.model import nn_s2v  # noqa: PLC0415

    def self_attn(self, query, key, value, cu_seqlens, return_attn_probs=False):  # noqa: PLR0913
        outs = []
        for start, stop in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
            q, k, v = query[start:stop], key[start:stop], value[start:stop]
            out = (
                F.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    k.transpose(0, 1).unsqueeze(0),
                    v.transpose(0, 1).unsqueeze(0),
                )
                .squeeze(0)
                .transpose(0, 1)
            )
            outs.append(out)
        return torch.cat(outs, dim=0).flatten(-2, -1)

    def cross_attn(self, query, key, value, cu_seqlens, cond_cu_seqlens, return_attn_probs=False):  # noqa: PLR0913
        outs = []
        q_bounds = list(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True))
        kv_bounds = list(zip(cond_cu_seqlens[:-1].tolist(), cond_cu_seqlens[1:].tolist(), strict=True))
        for (qs, qe), (ks, ke) in zip(q_bounds, kv_bounds, strict=True):
            q, k, v = query[qs:qe], key[ks:ke], value[ks:ke]
            out = (
                F.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    k.transpose(0, 1).unsqueeze(0),
                    v.transpose(0, 1).unsqueeze(0),
                )
                .squeeze(0)
                .transpose(0, 1)
            )
            outs.append(out)
        return torch.cat(outs, dim=0).flatten(-2, -1)

    nn_s2v.MultiheadSelfAttention.scaled_dot_product_attention = self_attn
    nn_s2v.MultiheadCrossAttention.scaled_dot_product_attention = cross_attn


def _randomize_modulations(model: torch.nn.Module) -> None:
    """Fill zero-initialised Modulation heads so the blocks actually do something."""
    generator = torch.Generator().manual_seed(1234)
    for name, module in model.named_modules():
        if name.endswith("modulation") or name.endswith("visual_modulation") or name.endswith("text_modulation"):
            for param in module.parameters():
                param.data = torch.randn(param.shape, generator=generator) * 0.1


# Small but structurally faithful model: framepack on, two audio-injected layers.
PATCH_SIZE = (1, 2, 2)
FRAME_BUCKETS = (2, 4, 24)
COEFS = ((1, 1, 1), (2, 2, 2), (4, 4, 4))
DURATION, LAT_H, LAT_W, LAT_C = 8, 32, 48, 16
NUM_AUDIO_TOKENS, NUM_TEXT_TOKENS = 25, 11
SCALE_FACTOR = (1.0, 2.0, 2.0)

PARAMS = dict(
    in_visual_dim=LAT_C,
    out_visual_dim=LAT_C,
    in_text_dim=32,
    in_text_dim2=16,
    in_audio_dim=32,
    time_dim=16,
    patch_size=PATCH_SIZE,
    frame_buckets=FRAME_BUCKETS,
    enable_framepack=True,
    index_rope_ref_image=40,
    patch_proj1x_coef=COEFS[0],
    patch_proj2x_coef=COEFS[1],
    patch_proj4x_coef=COEFS[2],
    model_dim=32,
    ff_dim=64,
    num_text_blocks=1,
    num_audio_blocks=1,
    num_visual_blocks=3,
    axes_dims=(4, 6, 6),
    audio_inject_layers=[0, 2],
    audio_adaln=False,
)


def main() -> int:
    if not (REFERENCE_REPO / "kandinsky" / "model" / "dit_s2v_tp.py").is_file():
        print(
            f"kandinsky-5 reference checkout not found at {REFERENCE_REPO}.\n"
            "Set KANDINSKY5_REPO=/path/to/kandinsky-5 to run this parity check.",
            file=sys.stderr,
        )
        return 77  # conventional "skipped"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(REFERENCE_REPO))
    _install_flash_attn_stub()
    _install_reference_stubs()
    _patch_reference_attention()
    torch._dynamo.config.disable = True

    from kandinsky.model.dit_s2v_tp import DiffusionTransformer3D as RefDiT  # noqa: PLC0415

    from kandinsky_s2v.core.algo.prepare_latents import init_memory  # noqa: PLC0415
    from kandinsky_s2v.core.algo.prepare_ropes import (  # noqa: PLC0415
        audio_rope_pos,
        memory_rope_pos,
        ref_rope_pos,
        text_rope_pos,
        visual_rope_pos,
    )
    from kandinsky_s2v.core.components.dit import DiffusionTransformer3DS2V  # noqa: PLC0415

    torch.manual_seed(0)
    device = torch.device("cpu")

    reference = RefDiT(**PARAMS).eval()
    # Modulation layers are zero-initialised, which would make every block an
    # identity and the comparison trivially true. Give them real values first.
    _randomize_modulations(reference)
    ported_params = {k: v for k, v in PARAMS.items() if k != "audio_adaln"}
    ported_params["audio_inject_layers"] = tuple(PARAMS["audio_inject_layers"])
    ported = DiffusionTransformer3DS2V(**ported_params, attention_engine="sdpa").eval()

    missing, unexpected = ported.load_state_dict(reference.state_dict(), strict=True)
    assert not missing and not unexpected

    latent = torch.randn(DURATION, LAT_H, LAT_W, LAT_C)
    ref_latent = torch.randn(1, LAT_H, LAT_W, LAT_C)
    memory = init_memory(FRAME_BUCKETS, COEFS, PATCH_SIZE, LAT_H, LAT_W, LAT_C, device)
    memory = [torch.randn_like(m) for m in memory]  # non-zero, so memory really matters
    memory_1x, memory_2x, memory_4x = memory

    text_embed = torch.randn(NUM_TEXT_TOKENS, 32)
    audio_embed = torch.randn(NUM_AUDIO_TOKENS, 32)
    pooled = torch.randn(1, 16)
    time = torch.tensor([700.0])

    vis_pos = visual_rope_pos(DURATION, LAT_H, LAT_W, PATCH_SIZE, device)
    rpos = ref_rope_pos(PARAMS["index_rope_ref_image"], LAT_H, LAT_W, PATCH_SIZE, device)
    mpos_1x, mpos_2x, mpos_4x = memory_rope_pos(memory, FRAME_BUCKETS, COEFS, PATCH_SIZE, device)
    tpos = text_rope_pos(NUM_TEXT_TOKENS, device)
    apos = audio_rope_pos(NUM_AUDIO_TOKENS, DURATION, device)

    def cu(n: int) -> torch.Tensor:
        return torch.tensor([0, n], dtype=torch.int32)

    with torch.no_grad():
        expected = reference(
            latent,
            ref_latent,
            memory_4x,
            memory_2x,
            memory_1x,
            text_embed,
            audio_embed,
            pooled,
            time,
            cu(DURATION),
            cu(1),
            cu(memory_4x.shape[0]),
            cu(memory_2x.shape[0]),
            cu(memory_1x.shape[0]),
            cu(NUM_TEXT_TOKENS),
            cu(NUM_AUDIO_TOKENS),
            vis_pos,
            rpos,
            mpos_4x,
            mpos_2x,
            mpos_1x,
            tpos,
            apos,
            SCALE_FACTOR,
            DURATION,
            None,
        )
        got = ported(
            latent,
            ref_latent,
            memory_4x,
            memory_2x,
            memory_1x,
            text_embed,
            audio_embed,
            pooled,
            time,
            vis_pos,
            rpos,
            mpos_4x,
            mpos_2x,
            mpos_1x,
            tpos,
            apos,
            SCALE_FACTOR,
            DURATION,
        )

    diff = (got - expected).abs().max().item()
    scale = expected.abs().max().item()
    print(f"reference {tuple(expected.shape)}  ported {tuple(got.shape)}")
    print(f"max |diff| = {diff:.3e}   max |ref| = {scale:.3e}   rel = {diff / scale:.3e}")
    passed = torch.allclose(got, expected, atol=1e-5, rtol=1e-4)
    print("PARITY OK" if passed else "PARITY FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
