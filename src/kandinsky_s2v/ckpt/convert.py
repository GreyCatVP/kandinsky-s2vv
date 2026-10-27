"""Convert an FSDP-sharded S2V training checkpoint into a single safetensors file.

Training writes one ``<rank>.pt`` per FSDP shard, each holding the rank-local
slice of every parameter along dim 0 (see K5
``gather_distributed_model_state_dict``).  Inference loads a single file, so the
shards are concatenated in rank order and cast to bf16.

Module and parameter names already match ``DiffusionTransformer3DS2V``, so no
key remapping is involved.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file


def shard_paths(checkpoint_dir: str | Path) -> list[Path]:
    """``<dir>/*.pt`` ordered by rank (numeric, so 10.pt sorts after 2.pt)."""
    directory = Path(checkpoint_dir)
    shards = [p for p in directory.iterdir() if p.suffix == ".pt" and p.stem.isdigit()]
    if not shards:
        raise FileNotFoundError(f"No <rank>.pt shards under {directory}")
    return sorted(shards, key=lambda p: int(p.stem))


# Kept in fp32 rather than cast down with the rest: the time embedding and every
# normalisation feed AdaLN modulation, which runs in fp32 anyway, and they are
# small — 206 of 1013 tensors on Lite, almost all vectors of length model_dim.
# RoPE tables need no entry: they are non-persistent buffers, absent from the
# checkpoint, and get rebuilt in fp32 at load time.
_FP32_KEY_PARTS = ("time_embeddings", "norm")


def tensor_dtype(key: str, default: torch.dtype) -> torch.dtype:
    """Target dtype for one parameter, honouring the fp32 carve-outs."""
    return torch.float32 if any(part in key for part in _FP32_KEY_PARTS) else default


def gather_state_dict(checkpoint_dir: str | Path, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Concatenate all shards along dim 0 and cast to ``dtype``.

    ``dtype`` applies to the bulk of the weights; see ``_FP32_KEY_PARTS`` for
    what stays in fp32.
    """
    paths = shard_paths(checkpoint_dir)
    parts: dict[str, list[torch.Tensor]] = {}
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for key, value in shard.items():
            parts.setdefault(key, []).append(value)

    counts = {len(v) for v in parts.values()}
    if counts != {len(paths)}:
        ragged = [k for k, v in parts.items() if len(v) != len(paths)]
        raise ValueError(f"{len(ragged)} key(s) are missing from some shards, e.g. {ragged[:3]}")
    return {key: torch.cat(values).to(tensor_dtype(key, dtype)) for key, values in parts.items()}


def convert(
    checkpoint_dir: str | Path,
    out_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Gather ``checkpoint_dir`` into ``out_path`` and return the written path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_dict = gather_state_dict(checkpoint_dir, dtype)
    save_file(state_dict, str(out), metadata={"format": "pt"})
    return out
