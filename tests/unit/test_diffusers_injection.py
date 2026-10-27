"""Injecting the port into a Diffusers checkout must make it importable.

The port exists in two forms — ours, which imports Diffusers from outside, and
the one that lives inside the library with relative imports — and
`dev/bench/integrate_diffusers.py` converts the first into the second. This
exercises that conversion against a throwaway copy of a Diffusers checkout.

The check that matters is the import: Diffusers' lazy module only reads a class
name when it is accessed, so looking the classes up is what proves the files
parse, their relative imports resolve, and the registrations landed in the right
tables.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "dev/bench/integrate_diffusers.py"
CHECKOUT_ENV = "KANDINSKY_S2V_DIFFUSERS_REPO"
DEFAULT_CHECKOUT = Path(__file__).resolve().parents[3] / "diffusers-new-model-addition-kandinskyX"
TOUCHED_INITS = (
    "__init__.py",
    "models/__init__.py",
    "models/transformers/__init__.py",
    "modular_pipelines/__init__.py",
    "modular_pipelines/kandinsky_s2v/__init__.py",
    "modular_pipelines/modular_pipeline.py",
    "schedulers/__init__.py",
    "utils/dummy_pt_objects.py",
    "utils/dummy_torch_and_transformers_objects.py",
)


def diffusers_checkout() -> Path:
    """A Diffusers source tree; the port targets the main branch, so any recent one does."""
    package = Path(os.environ.get(CHECKOUT_ENV, DEFAULT_CHECKOUT)) / "src/diffusers"
    if not (package / "__init__.py").is_file():
        pytest.skip(f"no Diffusers checkout; point {CHECKOUT_ENV} at one")
    return package


def run_script(target: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--diffusers", str(target), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def digest(package: Path) -> str:
    sha = hashlib.sha256()
    for relative in TOUCHED_INITS:
        sha.update((package / relative).read_bytes())
    return sha.hexdigest()


@pytest.fixture(scope="module")
def injected(tmp_path_factory) -> Path:
    """A private copy of the branch with the port injected into it."""
    root = tmp_path_factory.mktemp("diffusers_tree")
    shutil.copytree(
        diffusers_checkout(),
        root / "diffusers",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    result = run_script(root)
    assert result.returncode == 0, result.stdout + result.stderr
    # The script imports the patched tree in a fresh interpreter and looks the
    # classes up; anything less would not prove the relative imports resolve.
    assert "exposes every class" in result.stdout, result.stdout
    return root


def test_injection_places_the_port(injected: Path):
    package = injected / "diffusers"
    for created in (
        "models/transformers/transformer_kandinsky_s2v.py",
        "modular_pipelines/kandinsky_s2v/encoders.py",
        "modular_pipelines/kandinsky_s2v/denoise.py",
    ):
        assert (package / created).is_file(), created
        # Import statements must be relative.
        lines = (package / created).read_text().splitlines()
        assert not [line for line in lines if line.startswith("from diffusers.")], created

    # Modular only: no standard pipeline package, and the blocks import nothing from `pipelines`.
    assert not (package / "pipelines/kandinsky_s2v").exists()
    for module in (package / "modular_pipelines/kandinsky_s2v").glob("*.py"):
        assert "from ...pipelines" not in module.read_text(), module.name
    assert '"KandinskyS2VModularPipeline"' in (package / "modular_pipelines/kandinsky_s2v/__init__.py").read_text()

    # The distilled bundles' scheduler ships with the port, so they need no other branch.
    scheduler = (package / "schedulers/scheduling_kandinsky_s2v_piflow.py").read_text()
    assert "class KandinskyS2VPiflowScheduler(FlowMatchEulerDiscreteScheduler):" in scheduler
    assert "from .scheduling_flow_match_euler_discrete import" in scheduler


def test_injection_is_idempotent(injected: Path):
    """Re-running must not duplicate a registration or a placed class."""
    package = injected / "diffusers"
    before = digest(package)
    result = run_script(injected, "--no-verify")
    assert result.returncode == 0, result.stdout + result.stderr
    assert digest(package) == before


def test_check_mode_detects_a_stale_copy(injected: Path):
    package = injected / "diffusers"
    assert run_script(injected, "--check").returncode == 0

    target = package / "modular_pipelines/kandinsky_s2v/encoders.py"
    original = target.read_text()
    try:
        target.write_text(original + "# edited out of band\n")
        stale = run_script(injected, "--check")
        assert stale.returncode != 0
        assert "stale" in stale.stdout
    finally:
        target.write_text(original)
