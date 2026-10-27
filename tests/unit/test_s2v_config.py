from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kandinsky_s2v.pipeline.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "src" / "kandinsky_s2v" / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_configs_exist():
    assert {p.name for p in CONFIGS} == {
        "s2v_lite.yaml",
        "s2v_pro.yaml",
        "s2v_lite_distill.yaml",
        "s2v_pro_distill.yaml",
        "s2v_pro_hd.yaml",
        "s2v_pro_distill_hd.yaml",
    }


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads(path: Path):
    cfg = load_config(path)
    assert cfg.dit.in_audio_dim == 3584  # Qwen2.5-Omni audio tower width  # noqa: PLR2004
    assert cfg.generation.fps == 25  # noqa: PLR2004
    assert cfg.generation.latent_frames == 32  # noqa: PLR2004
    assert cfg.dit.frame_buckets == (2, 4, 24)
    assert cfg.generation.prompt and cfg.generation.negative_prompt


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_distilled_configs_disable_cfg(path: Path):
    cfg = load_config(path)
    if cfg.piflow.enabled:
        # CFG is distilled into the student: a second forward would double-apply it.
        assert cfg.generation.guidance_weight == 1.0
        assert cfg.generation.num_steps == 16  # noqa: PLR2004
        assert cfg.piflow.n_grid == 5  # noqa: PLR2004
    else:
        assert cfg.generation.guidance_weight == 3.0  # noqa: PLR2004
        assert cfg.generation.num_steps == 50  # noqa: PLR2004


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_audio_inject_layers_within_range(path: Path):
    cfg = load_config(path)
    assert cfg.dit.audio_inject_layers
    assert max(cfg.dit.audio_inject_layers) < cfg.dit.num_visual_blocks


def test_unknown_key_is_rejected(tmp_path: Path):
    raw = yaml.safe_load(CONFIGS[0].read_text())
    raw["generation"]["not_a_field"] = 1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(bad)


def test_short_chunk_is_rejected(tmp_path: Path):
    """A chunk shorter than the FramePack window would build undersized memory."""
    raw = yaml.safe_load(CONFIGS[0].read_text())
    raw["generation"]["latent_frames"] = 25  # sum(frame_buckets) is 30
    bad = tmp_path / "short.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="FramePack memory window"):
        load_config(bad)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_chunk_covers_memory_window(path: Path):
    cfg = load_config(path)
    assert cfg.generation.latent_frames >= sum(cfg.dit.frame_buckets)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_compile_is_off_by_default(path: Path):
    """Compiling costs tens of seconds of warm-up, so configs opt in explicitly."""
    assert load_config(path).compile.strategy == "none"
