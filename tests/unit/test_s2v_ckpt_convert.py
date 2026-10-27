"""Gathering FSDP shards into a single safetensors file."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from kandinsky_s2v.ckpt.convert import convert, gather_state_dict, shard_paths
from kandinsky_s2v.core.components.dit import DiffusionTransformer3DS2V
from kandinsky_s2v.pipeline.config import load_config
from kandinsky_s2v.pipeline.factory import create_bare_dit


def _write_shards(directory: Path, num_shards: int, rows_per_shard: int = 2) -> torch.Tensor:
    """Write ``num_shards`` shards of one parameter; returns the full tensor."""
    full = torch.arange(num_shards * rows_per_shard * 3, dtype=torch.float32).reshape(num_shards * rows_per_shard, 3)
    for rank in range(num_shards):
        chunk = full[rank * rows_per_shard : (rank + 1) * rows_per_shard]
        torch.save({"layer.weight": chunk.clone()}, directory / f"{rank}.pt")
    return full


def test_shards_are_ordered_numerically(tmp_path: Path):
    _write_shards(tmp_path, num_shards=12)
    assert [p.stem for p in shard_paths(tmp_path)] == [str(i) for i in range(12)]


def test_gather_concatenates_in_rank_order(tmp_path: Path):
    full = _write_shards(tmp_path, num_shards=8)
    gathered = gather_state_dict(tmp_path, dtype=torch.float32)
    assert torch.equal(gathered["layer.weight"], full)


def test_gather_casts_to_bfloat16(tmp_path: Path):
    _write_shards(tmp_path, num_shards=4)
    gathered = gather_state_dict(tmp_path)
    assert gathered["layer.weight"].dtype == torch.bfloat16


def test_convert_writes_loadable_file(tmp_path: Path):
    source = tmp_path / "ema_model"
    source.mkdir()
    full = _write_shards(source, num_shards=4)
    out = convert(source, tmp_path / "out" / "model.safetensors")
    loaded = load_file(str(out))
    assert torch.equal(loaded["layer.weight"].float(), full)


def test_missing_shards_are_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        shard_paths(tmp_path)


def test_ragged_shards_are_rejected(tmp_path: Path):
    _write_shards(tmp_path, num_shards=2)
    torch.save({"other.weight": torch.zeros(1)}, tmp_path / "2.pt")
    with pytest.raises(ValueError, match="missing from some shards"):
        gather_state_dict(tmp_path)


def test_time_and_norm_stay_fp32(tmp_path: Path):
    """Casting these down loses precision that the fp32 AdaLN path cannot recover."""
    shard = {
        "visual_transformer_blocks.0.feed_forward.in_layer.weight": torch.randn(2, 3),
        "time_embeddings.in_layer.weight": torch.randn(2, 3),
        "audio_embeddings.norm.weight": torch.randn(2),
        "visual_transformer_blocks.0.self_attention.query_norm.weight": torch.randn(2),
    }
    torch.save(shard, tmp_path / "0.pt")

    gathered = gather_state_dict(tmp_path)

    assert gathered["visual_transformer_blocks.0.feed_forward.in_layer.weight"].dtype == torch.bfloat16
    assert gathered["time_embeddings.in_layer.weight"].dtype == torch.float32
    assert gathered["audio_embeddings.norm.weight"].dtype == torch.float32
    assert gathered["visual_transformer_blocks.0.self_attention.query_norm.weight"].dtype == torch.float32


def test_factory_preserves_stored_dtypes(tmp_path: Path):
    """The factory must load weights as stored — casting would undo the fp32 carve-outs."""
    dit_kwargs = dict(
        in_visual_dim=16,
        out_visual_dim=16,
        in_text_dim=32,
        in_text_dim2=16,
        in_audio_dim=32,
        time_dim=16,
        patch_size=[1, 2, 2],
        frame_buckets=[2, 4, 24],
        enable_framepack=True,
        index_rope_ref_image=40,
        patch_proj1x_coef=[1, 1, 1],
        patch_proj2x_coef=[2, 2, 2],
        patch_proj4x_coef=[4, 4, 4],
        model_dim=32,
        ff_dim=64,
        num_text_blocks=1,
        num_audio_blocks=1,
        num_visual_blocks=2,
        axes_dims=[4, 6, 6],
        audio_inject_layers=[0],
    )
    reference = DiffusionTransformer3DS2V(**{k: tuple(v) if isinstance(v, list) else v for k, v in dit_kwargs.items()})
    stored = {
        k: v.to(torch.float32 if ("time_embeddings" in k or "norm" in k) else torch.bfloat16)
        for k, v in reference.state_dict().items()
    }
    weights = tmp_path / "dit.safetensors"
    save_file(stored, str(weights), metadata={"format": "pt"})

    config = tmp_path / "cfg.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"dit": str(weights), "vae": "-", "qwen_omni": "-", "clip": "-"},
                "dit": dit_kwargs,
                "generation": {"prompt": "p", "negative_prompt": "n"},
            }
        )
    )

    loaded = create_bare_dit(load_config(config), "cpu", "sdpa")
    for name, tensor in loaded.state_dict().items():
        assert tensor.dtype == stored[name].dtype, f"{name} changed dtype on load"
