"""A configured weight is a local path, or a repository on the Hub.

The distinction is made without touching the network: a path that exists is
used, a reference that looks like `owner/name` is fetched, and anything else
raises rather than turning a typo into a download.
"""

from __future__ import annotations

import huggingface_hub
import pytest

from kandinsky_s2v.pipeline.weights import resolve


@pytest.fixture
def hub(monkeypatch):
    """Record what would have been downloaded instead of downloading it."""
    calls: list[tuple] = []

    def snapshot_download(repo_id, allow_patterns=None, **_):
        calls.append(("snapshot", repo_id, allow_patterns))
        return f"/cache/{repo_id}"

    def hf_hub_download(repo_id, filename, **_):
        calls.append(("file", repo_id, filename))
        return f"/cache/{repo_id}/{filename}"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    return calls


def test_a_local_path_is_left_alone(tmp_path, hub):
    weights = tmp_path / "model.safetensors"
    weights.touch()
    assert resolve(str(weights)) == str(weights)
    assert hub == []


def test_a_missing_path_is_not_a_download(hub):
    with pytest.raises(FileNotFoundError, match="not a Hub reference"):
        resolve("/shares/somebody-elses/weights_k6/vae")
    assert hub == []


def test_a_repository_comes_whole(hub):
    assert resolve("Qwen/Qwen2.5-Omni-7B") == "/cache/Qwen/Qwen2.5-Omni-7B"
    assert hub == [("snapshot", "Qwen/Qwen2.5-Omni-7B", None)]


def test_a_file_comes_alone(hub):
    reference = "kandinskylab/Kandinsky-S2V-1.0-Pro/model/kandinsky_s2v_pro.safetensors"
    assert resolve(reference) == "/cache/kandinskylab/Kandinsky-S2V-1.0-Pro/model/kandinsky_s2v_pro.safetensors"
    assert hub == [("file", "kandinskylab/Kandinsky-S2V-1.0-Pro", "model/kandinsky_s2v_pro.safetensors")]


def test_a_folder_brings_only_itself(hub):
    """The sibling folders in a bundle are tens of gigabytes."""
    assert resolve("kandinskylab/Kandinsky-S2V-1.0-Pro-Diffusers/vae") == (
        "/cache/kandinskylab/Kandinsky-S2V-1.0-Pro-Diffusers/vae"
    )
    assert hub == [("snapshot", "kandinskylab/Kandinsky-S2V-1.0-Pro-Diffusers", "vae/*")]
