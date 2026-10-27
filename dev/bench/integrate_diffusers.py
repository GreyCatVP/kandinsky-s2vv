"""Inject the S2V port into a Diffusers checkout as a package of its own.

The port in `src/kandinsky_s2v/ports/diffusers/` is written as a *consumer* of
Diffusers: absolute `diffusers.*` imports, standing outside the package. Inside
the library the same code needs relative imports and entries in the lazy-import
tables, so that `diffusers.KandinskyS2VModularPipeline` exists — which is what
`ModularPipeline.from_pretrained` resolves the names in `modular_model_index.json`
against. This script is that transformation.

Kandinsky S2V 1.0 is versioned apart from the other Kandinsky models, so it
stands apart in the library as well rather than joining another family's files:

* the transformer is `models/transformers/transformer_kandinsky_s2v.py`;
* the pi-Flow scheduler of the distilled checkpoints is
  `schedulers/scheduling_kandinsky_s2v_piflow.py`, so the distilled bundles do
  not depend on another family's scheduler;
* the pipeline is modular only, as Diffusers prefers for new models: blocks for the
  base and the distilled checkpoints and their `ModularPipeline` classes, in the
  package `modular_pipelines/kandinsky_s2v/`;
* the documentation is three pages of its own, added to the toctree;
* the tests are Diffusers' own, kept in `dev/diffusers_tests/` under the paths
  they take in the repository's `tests/`, and copied there as they are.

The registrations are anchored on lines stock Diffusers already has (the
Kandinsky 6 entries, which ours sort right after, and the schedulers that sort
next to ours), so the target is the `main` branch of
`huggingface/diffusers-new-model-addition-kandinskyX-avatar` and needs nothing
from the other Kandinsky branches.

    PYTHONPATH=src .venv/bin/python dev/bench/integrate_diffusers.py \\
        --diffusers ~/projects/diffusers-new-model-addition-kandinskyX-avatar

It is re-runnable: every edit first removes its own previous result. `--check`
reports whether the injected copy still matches the port, changing nothing.
Nothing happens at run time; the result is a patched source tree.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PORT = Path(__file__).resolve().parents[2] / "src/kandinsky_s2v/ports/diffusers"
TESTS = Path(__file__).resolve().parents[1] / "diffusers_tests"

TRANSFORMER = "KandinskyS2VTransformer3DModel"
SCHEDULER = "KandinskyS2VPiflowScheduler"
MODULAR_PACKAGE = "modular_pipelines/kandinsky_s2v"
MODULAR_BLOCKS = "KandinskyS2VAutoBlocks"
MODULAR_DISTILLED_BLOCKS = "KandinskyS2VDistilledAutoBlocks"
MODULAR_PIPELINE = "KandinskyS2VModularPipeline"
MODULAR_DISTILLED_PIPELINE = "KandinskyS2VDistilledModularPipeline"
# In the order isort and the dummy generator put them.
MODULAR_NAMES = (MODULAR_BLOCKS, MODULAR_DISTILLED_BLOCKS, MODULAR_DISTILLED_PIPELINE, MODULAR_PIPELINE)
MODULAR_FILES = (
    "modular_pipeline.py",
    "encoders.py",
    "before_denoise.py",
    "denoise.py",
    "decoders.py",
    "modular_blocks_kandinsky_s2v.py",
    "modular_blocks_kandinsky_s2v_distilled.py",
)
# One table for every modular module: they all sit at the same depth.
MODULAR_REWRITES = (
    ("from diffusers.configuration_utils import", "from ...configuration_utils import"),
    ("from diffusers.guiders import", "from ...guiders import"),
    ("from diffusers.models import", "from ...models import"),
    ("from diffusers.modular_pipelines.modular_pipeline import", "from ..modular_pipeline import"),
    ("from diffusers.modular_pipelines.modular_pipeline_utils import", "from ..modular_pipeline_utils import"),
    ("from diffusers.schedulers import", "from ...schedulers import"),
    ("from diffusers.utils import", "from ...utils import"),
    ("from diffusers.utils.torch_utils import", "from ...utils.torch_utils import"),
    ("from diffusers.video_processor import", "from ...video_processor import"),
    ("from ..scheduling_piflow import", "from ...schedulers.scheduling_kandinsky_s2v_piflow import"),
    ("from ..transformer_kandinsky_s2v import", "from ...models.transformers.transformer_kandinsky_s2v import"),
)

# Source file, destination relative to the diffusers package root, and the import
# rewrites that destination needs. The number of leading dots follows the
# destination's depth, which is why the tables differ.
SOURCES = (
    (
        "transformer_kandinsky_s2v.py",
        "models/transformers/transformer_kandinsky_s2v.py",
        (
            ("from diffusers.configuration_utils import", "from ...configuration_utils import"),
            ("from diffusers.loaders import", "from ...loaders import"),
            ("from diffusers.models.attention_dispatch import", "from ..attention_dispatch import"),
            ("from diffusers.models.attention import", "from ..attention import"),
            ("from diffusers.models.cache_utils import", "from ..cache_utils import"),
            ("from diffusers.models.modeling_outputs import", "from ..modeling_outputs import"),
            ("from diffusers.models.modeling_utils import", "from ..modeling_utils import"),
            ("from diffusers.utils import", "from ...utils import"),
        ),
    ),
    (
        "scheduling_piflow.py",
        "schedulers/scheduling_kandinsky_s2v_piflow.py",
        (
            ("from diffusers.configuration_utils import", "from ..configuration_utils import"),
            (
                "from diffusers.schedulers.scheduling_flow_match_euler_discrete import",
                "from .scheduling_flow_match_euler_discrete import",
            ),
        ),
    ),
    *((f"modular/{name}", f"{MODULAR_PACKAGE}/{name}", MODULAR_REWRITES) for name in MODULAR_FILES),
)

# The modular family's lazy-import module, in the shape every modular family has:
# it needs transformers as well as torch, and falls back to the dummies without them.
MODULAR_PACKAGE_INIT = f'''from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {{}}
_import_structure = {{}}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["modular_blocks_kandinsky_s2v"] = ["{MODULAR_BLOCKS}"]
    _import_structure["modular_blocks_kandinsky_s2v_distilled"] = ["{MODULAR_DISTILLED_BLOCKS}"]
    _import_structure["modular_pipeline"] = ["{MODULAR_DISTILLED_PIPELINE}", "{MODULAR_PIPELINE}"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .modular_blocks_kandinsky_s2v import {MODULAR_BLOCKS}
        from .modular_blocks_kandinsky_s2v_distilled import {MODULAR_DISTILLED_BLOCKS}
        from .modular_pipeline import {MODULAR_DISTILLED_PIPELINE}, {MODULAR_PIPELINE}
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
'''

# Import-time placeholders, as `utils/check_dummies.py` generates them: (file, the
# class ours follow, our classes in order, backends). The anchors are the classes
# ours are registered right after in the root __init__.
DUMMIES = (
    ("utils/dummy_pt_objects.py", "Kandinsky6Transformer3DModel", (TRANSFORMER,), ("torch",)),
    ("utils/dummy_pt_objects.py", "IPNDMScheduler", (SCHEDULER,), ("torch",)),
    (
        "utils/dummy_torch_and_transformers_objects.py",
        "Ideogram4ModularPipeline",
        MODULAR_NAMES,
        ("torch", "transformers"),
    ),
)


def dummy_class(name: str, backends: tuple[str, ...]) -> str:
    listed = ", ".join(f'"{backend}"' for backend in backends)
    return f"""class {name}(metaclass=DummyObject):
    _backends = [{listed}]

    def __init__(self, *args, **kwargs):
        requires_backends(self, [{listed}])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, [{listed}])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, [{listed}])
"""


# Registrations, anchored on lines already in these tables: the Kandinsky 6
# entries the transformer sorts after, and for the scheduler the one it sorts after.
# (file, anchor, line, placed "after" or "before" the anchor)
REGISTRATIONS = (
    ("__init__.py", '            "Kandinsky6Transformer3DModel",', f'            "{TRANSFORMER}",', "after"),
    ("__init__.py", "            Kandinsky6Transformer3DModel,", f"            {TRANSFORMER},", "after"),
    ("__init__.py", '            "IPNDMScheduler",', f'            "{SCHEDULER}",', "after"),
    ("__init__.py", "            IPNDMScheduler,", f"            {SCHEDULER},", "after"),
    (
        "schedulers/__init__.py",
        '    _import_structure["scheduling_ipndm"] = ["IPNDMScheduler"]',
        f'    _import_structure["scheduling_kandinsky_s2v_piflow"] = ["{SCHEDULER}"]',
        "after",
    ),
    (
        "schedulers/__init__.py",
        "        from .scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler",
        f"        from .scheduling_kandinsky_s2v_piflow import {SCHEDULER}",
        "after",
    ),
    (
        "models/__init__.py",
        '    _import_structure["transformers.transformer_kandinsky6_sr"] = ["Kandinsky6SRTransformer3DModel"]',
        f'    _import_structure["transformers.transformer_kandinsky_s2v"] = ["{TRANSFORMER}"]',
        "after",
    ),
    ("models/__init__.py", "            Kandinsky6Transformer3DModel,", f"            {TRANSFORMER},", "after"),
    (
        "models/transformers/__init__.py",
        "    from .transformer_kandinsky6_sr import Kandinsky6SRTransformer3DModel",
        f"    from .transformer_kandinsky_s2v import {TRANSFORMER}",
        "after",
    ),
    # The modular classes follow Ideogram 4's in both of the root __init__'s tables.
    *(
        ("__init__.py", f'            "{anchor}",', f'            "{name}",', "after")
        for anchor, name in zip(("Ideogram4ModularPipeline", *MODULAR_NAMES), MODULAR_NAMES, strict=False)
    ),
    *(
        ("__init__.py", f"            {anchor},", f"            {name},", "after")
        for anchor, name in zip(("Ideogram4ModularPipeline", *MODULAR_NAMES), MODULAR_NAMES, strict=False)
    ),
    # `blocks.init_pipeline()` picks the pipeline class by the blocks' model name;
    # `from_pretrained` reads it from `modular_model_index.json` instead.
    (
        "modular_pipelines/modular_pipeline.py",
        '        ("helios-pyramid", _helios_pyramid_map_fn),',
        f'        ("kandinsky-s2v", _create_default_map_fn("{MODULAR_PIPELINE}")),',
        "after",
    ),
    (
        "modular_pipelines/modular_pipeline.py",
        f'        ("kandinsky-s2v", _create_default_map_fn("{MODULAR_PIPELINE}")),',
        f'        ("kandinsky-s2v-distilled", _create_default_map_fn("{MODULAR_DISTILLED_PIPELINE}")),',
        "after",
    ),
)

# Registrations that span several lines: (file, anchor line, block placed right before it).
MODULAR_LIST = "".join(f'        "{name}",\n' for name in MODULAR_NAMES)
MODULAR_IMPORTS = "".join(f"            {name},\n" for name in MODULAR_NAMES)
BLOCK_REGISTRATIONS = (
    (
        "modular_pipelines/__init__.py",
        '    _import_structure["krea2"] = [',
        f'    _import_structure["kandinsky_s2v"] = [\n{MODULAR_LIST}    ]\n',
    ),
    (
        "modular_pipelines/__init__.py",
        "        from .krea2 import (",
        f"        from .kandinsky_s2v import (\n{MODULAR_IMPORTS}        )\n",
    ),
)


# Documentation: two pages of its own, and their entries in the toctree. Paths
# here are relative to the repository, not to the package.
DOCS = "docs/source/en"
DOC_LICENSE = """<!--Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
"""
DOC_PAGES = {
    f"{DOCS}/api/pipelines/kandinsky_s2v.md": f"""{DOC_LICENSE}
# Kandinsky S2V 1.0

Kandinsky S2V 1.0 generates an avatar video from a reference frame and a speech
track. The person in the frame is animated to match the speech — the whole figure,
not only the head. The speech is a condition, not an output: the model does not
generate audio, and the soundtrack of the result is the track you pass in.

Both conditions go through [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B):
its hidden states encode the prompt and its audio tower encodes the speech, which
enters the transformer through cross-attention adapters aligned frame by frame, so
each latent frame attends only to the speech it covers. A clip is generated in
chunks of 5 s (125 frames at 25 fps), each carrying FramePack memory of the
previous ones, so longer tracks stay consistent across chunk boundaries.

Kandinsky S2V 1.0 is integrated as a [modular pipeline](../../modular_diffusers/overview)
only; there is no standard `DiffusionPipeline` class for it.

## Checkpoints

| Checkpoint | Transformer | Blocks | Scheduler | Steps | Guidance |
|---|---|---|---|---|---|
| [Lite][lite] | 2B | [`{MODULAR_BLOCKS}`] | [`FlowMatchEulerDiscreteScheduler`] | 50 | 3.0 |
| [Pro][pro] | 21B | [`{MODULAR_BLOCKS}`] | [`FlowMatchEulerDiscreteScheduler`] | 50 | 3.0 |
| [Lite distilled][lite-distill] | 2B | [`{MODULAR_DISTILLED_BLOCKS}`] | [`{SCHEDULER}`] | 16 | off |
| [Pro distilled][pro-distill] | 21B | [`{MODULAR_DISTILLED_BLOCKS}`] | [`{SCHEDULER}`] | 16 | off |

[lite]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Lite-Diffusers
[pro]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-Diffusers
[lite-distill]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Lite-distill-Diffusers
[pro-distill]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers

`ModularPipeline.from_pretrained` picks the blocks from the checkpoint. The steps
default to the values above. Classifier-free guidance is the `guider` component,
not a call argument: the base checkpoints ship it at 3.0, and the distilled ones
at 1.0, because guidance is already part of the student — they reject any other.

## Usage

```py
import torch
from diffusers import ModularPipeline
from diffusers.utils import export_to_video

pipe = ModularPipeline.from_pretrained("kandinskylab/Kandinsky-S2V-1.0-Lite-distill-Diffusers")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

video = pipe(image="reference.png", audio="speech.wav", output="videos")[0]
export_to_video(video, "avatar.mp4", fps=25)
```

`export_to_video` writes the frames only; add the speech track with any muxer, for
example `ffmpeg -i avatar.mp4 -i speech.wav -c:v copy -c:a aac -shortest out.mp4`.

The Pro checkpoints do not fit an 80 GB GPU next to the 7B speech and text encoder
when every component is on the device. Keep one model on the GPU at a time, which
brings the peak to about 46 GiB:

```py
from diffusers import ComponentsManager

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("kandinskylab/Kandinsky-S2V-1.0-Pro-Diffusers", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
manager.enable_auto_cpu_offload(device="cuda", offload_strategy=lambda hooks, **_: hooks)
```

The default strategy is not enough here: it keeps the transformer beside the speech
encoder because both fit, and the VAE decode that follows needs the room they hold.
Enable the offload after `load_components`, so the strategy applies to every component.

## Notes

- The number of chunks follows the length of the speech track unless `num_chunks`
  is passed. The video is trimmed to the track either way, so chunks past its end
  are computed and dropped, with a warning.
- With guidance, the unconditional pass swaps in the negative prompt *and* silences
  the speech, so the guidance is joint over text and speech.

## {MODULAR_PIPELINE}

[[autodoc]] {MODULAR_PIPELINE}

## {MODULAR_BLOCKS}

[[autodoc]] {MODULAR_BLOCKS}

## {MODULAR_DISTILLED_PIPELINE}

[[autodoc]] {MODULAR_DISTILLED_PIPELINE}

## {MODULAR_DISTILLED_BLOCKS}

[[autodoc]] {MODULAR_DISTILLED_BLOCKS}
""",
    f"{DOCS}/api/schedulers/kandinsky_s2v_piflow.md": f"""{DOC_LICENSE}
# {SCHEDULER}

`{SCHEDULER}` is the few-step pi-Flow scheduler of the distilled Kandinsky S2V 1.0
checkpoints. The student predicts a grid of x0 values per segment in one forward
pass, and the scheduler integrates that policy across the segment, so each
segment costs a single model evaluation.

## {SCHEDULER}
[[autodoc]] {SCHEDULER}
""",
    f"{DOCS}/api/models/kandinsky_s2v_transformer.md": f"""{DOC_LICENSE}
# {TRANSFORMER}

The Diffusion Transformer (DiT) of Kandinsky S2V 1.0, conditioned on a reference
frame and a speech track.

## {TRANSFORMER}

[[autodoc]] {TRANSFORMER}
  - all
  - forward
""",
}
# (file, anchor, block): the block goes before the anchor line.
TOCTREE = f"{DOCS}/_toctree.yml"
DOC_REGISTRATIONS = (
    (
        TOCTREE,
        "      - local: api/models/krea2_transformer2d",
        f"      - local: api/models/kandinsky_s2v_transformer\n        title: {TRANSFORMER}",
    ),
    (
        TOCTREE,
        "      - local: api/pipelines/latte",
        "      - local: api/pipelines/kandinsky_s2v\n        title: Kandinsky S2V 1.0",
    ),
    (
        TOCTREE,
        "    - local: api/schedulers/dpm_discrete_ancestral",
        f"    - local: api/schedulers/kandinsky_s2v_piflow\n      title: {SCHEDULER}",
    ),
)


# Test files relative to `TESTS` and to the repository's `tests/` alike; the
# package marker is created alongside, as every test directory there has one.
TEST_FILES = (
    "models/transformers/test_models_transformer_kandinsky_s2v.py",
    "modular_pipelines/kandinsky_s2v/test_modular_pipeline_kandinsky_s2v.py",
)
TEST_PACKAGE_INITS = ("modular_pipelines/kandinsky_s2v/__init__.py",)


def rendered_tests(root: Path) -> dict[Path, str]:
    """What each placed test file should contain, keyed by its path in the repository."""
    rendered = {root / "tests" / init: "" for init in TEST_PACKAGE_INITS}
    for relative in TEST_FILES:
        target = root / "tests" / relative
        rendered[target] = reformat((TESTS / relative).read_text(), target)
    return rendered


def place_tests(root: Path) -> None:
    for target, text in rendered_tests(root).items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"[integrate] {target.relative_to(root)}")


def resolve_package(path: Path) -> Path:
    """Accept either the diffusers package directory or a repository root."""
    path = path.expanduser().resolve()
    for candidate in (path, path / "src/diffusers", path / "diffusers"):
        if (candidate / "__init__.py").is_file() and (candidate / "pipelines").is_dir():
            return candidate
    raise SystemExit(f"no diffusers package under {path}")


def rewrite_imports(source: str, rewrites: tuple[tuple[str, str], ...]) -> str:
    """Rewrite import statements only, leaving their mentions in prose alone.

    Applied line by line and anchored at column zero, because a blind replace also
    rewrites the example inside the module docstring, where the absolute import is
    exactly what a reader needs to copy.
    """
    lines = []
    for line in source.splitlines():
        rewritten = line
        for old, new in rewrites:
            if line.startswith(old):
                rewritten = new + line[len(old) :]
                break
        lines.append(rewritten)
    return "\n".join(lines) + "\n"


def reformat(text: str, destination: Path) -> str:
    """Format as the destination repo formats, not as this one does.

    The two disagree: Diffusers sets `line-length = 119` against 120 here, and
    rewriting `from diffusers.x` to `from ...x` moves those imports out of the
    third-party isort section into the local one. Neither is something the port
    can be written to satisfy on both sides, so the destination's own ruff settles
    it — resolved from `destination`, which is why that path has to be absolute.

    Docstrings then go through `doc-builder style`, as `make style` there runs it:
    it rewraps them to 119 columns, and `make quality` fails on any it would change.
    It reads files only, so it runs on a temporary copy.
    """
    ruff = shutil.which("ruff") or str(Path(sys.executable).with_name("ruff"))
    doc_builder = shutil.which("doc-builder") or str(Path(sys.executable).with_name("doc-builder"))
    for tool, name in ((ruff, "ruff"), (doc_builder, "doc-builder")):
        if not Path(tool).is_file():
            raise SystemExit(f"{name} not found; it formats what is written for the destination repo")
    for arguments in (("check", "--fix-only", "--select", "I"), ("format",)):
        result = subprocess.run(  # noqa: S603
            [ruff, *arguments, "--stdin-filename", str(destination), "-"],
            input=text,
            capture_output=True,
            text=True,
            check=True,
        )
        text = result.stdout
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / destination.name
        copy.write_text(text)
        subprocess.run([doc_builder, "style", str(copy), "--max_len", "119"], capture_output=True, check=True)  # noqa: S603
        return copy.read_text()


def rendered_sources(package: Path) -> dict[str, str]:
    """What each placed file should contain, given the port as it is now."""
    rendered = {
        destination: reformat(rewrite_imports((PORT / source_name).read_text(), rewrites), package / destination)
        for source_name, destination, rewrites in SOURCES
    }
    modular_init = f"{MODULAR_PACKAGE}/__init__.py"
    rendered[modular_init] = reformat(MODULAR_PACKAGE_INIT, package / modular_init)
    return rendered


def place_sources(package: Path) -> None:
    for destination, source in rendered_sources(package).items():
        target = package / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        print(f"[integrate] {destination}")


def strip_class(text: str, name: str, base: str) -> str:
    """Remove a definition left by an earlier run, so re-running replaces it.

    Found by its class header rather than by markers: the placed code has to read
    like the rest of the module, which leaves nothing to delimit it with.
    """
    header = f"class {name}({base}):"
    lines = text.splitlines()
    if header not in lines:
        return text
    first = lines.index(header)
    start = first
    while start > 0 and (lines[start - 1].startswith("@") or not lines[start - 1].strip()):
        start -= 1
    end = first + 1
    while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t"))):
        end += 1
    return "\n".join(lines[:start] + lines[end:]) + "\n"


def dummy_blocks(package: Path) -> None:
    """Give our classes the same import-time placeholders the others have."""
    for relative, anchor_name, names, backends in DUMMIES:
        path = package / relative
        text = path.read_text()
        for name in names:
            text = strip_class(text, name, base="metaclass=DummyObject")
        anchor = f"class {anchor_name}(metaclass=DummyObject):"
        if anchor not in text:
            raise SystemExit(f"anchor not found in {relative}: {anchor!r}")
        block = "\n\n\n".join(dummy_class(name, backends).rstrip("\n") for name in names)
        path.write_text(place_after_class(text, anchor, block))
        print(f"[integrate] {relative}: {', '.join(names)}")


def place_after_class(text: str, anchor: str, block: str) -> str:
    """``block`` right after the class headed by ``anchor``."""
    # After the anchor class, not before it: check_dummies regenerates this file in
    # the order of the root __init__'s TYPE_CHECKING block, where ours is registered
    # right after its anchor. Inserting before makes the generator
    # disagree with what is on disk.
    lines = text.splitlines()
    index = lines.index(anchor) + 1
    while index < len(lines) and (not lines[index].strip() or lines[index].startswith((" ", "\t"))):
        index += 1
    # The walk above also consumes the blank lines that separate the classes, so
    # they are trimmed and both separators put back; the generator writes exactly two.
    head = lines[:index]
    while head and not head[-1].strip():
        head.pop()
    placed = head + ["", ""] + block.splitlines() + ["", ""] + lines[index:]
    return "\n".join(placed).rstrip("\n") + "\n"


def register(package: Path) -> None:
    for relative, anchor, line, where in REGISTRATIONS:
        path = package / relative
        lines = [existing for existing in path.read_text().splitlines() if existing != line]
        if anchor not in lines:
            raise SystemExit(f"anchor not found in {relative}: {anchor.strip()!r}")
        lines.insert(lines.index(anchor) + (1 if where == "after" else 0), line)
        path.write_text("\n".join(lines) + "\n")
        print(f"[integrate] {relative}: {line.strip()}")


def place_block(text: str, anchor: str, block: str) -> str:
    """``block`` right before the line ``anchor``, replacing an earlier run's copy."""
    text = text.replace(block, "")
    lines = text.splitlines(keepends=True)
    if anchor + "\n" not in lines:
        raise SystemExit(f"anchor not found: {anchor.strip()!r}")
    index = lines.index(anchor + "\n")
    return "".join(lines[:index]) + block + "".join(lines[index:])


def register_blocks(package: Path) -> None:
    for relative, anchor, block in BLOCK_REGISTRATIONS:
        path = package / relative
        path.write_text(place_block(path.read_text(), anchor, block))
        print(f"[integrate] {relative}: {block.splitlines()[0].strip()}")


def repository(package: Path) -> Path | None:
    """The checkout the package sits in, if it is one; an install has no `docs/`."""
    root = package.parents[1] if package.parent.name == "src" else package.parent
    return root if (root / DOCS).is_dir() else None


def register_docs(root: Path) -> None:
    for relative, page in DOC_PAGES.items():
        (root / relative).write_text(page)
        print(f"[integrate] {relative}")
    for relative, anchor, block in DOC_REGISTRATIONS:
        path = root / relative
        # Drop the block an earlier run placed, so re-running replaces it.
        lines = path.read_text().replace(block + "\n", "").splitlines()
        if anchor not in lines:
            raise SystemExit(f"anchor not found in {relative}: {anchor!r}")
        index = lines.index(anchor)
        lines[index:index] = block.splitlines()
        path.write_text("\n".join(lines) + "\n")
        print(f"[integrate] {relative}: {block.splitlines()[0].strip()}")


def check(package: Path) -> list[str]:
    """Every way the injected copy differs from what would be written now."""
    problems = []
    for destination, expected in rendered_sources(package).items():
        target = package / destination
        if not target.is_file():
            problems.append(f"missing: {destination}")
        elif target.read_text() != expected:
            problems.append(f"stale: {destination}")

    for relative, _anchor, names, _backends in DUMMIES:
        dummy = (package / relative).read_text()
        problems += [f"missing: {name} in {relative}" for name in names if f"class {name}(" not in dummy]

    for relative, _anchor, line, _where in REGISTRATIONS:
        count = (package / relative).read_text().splitlines().count(line)
        if count != 1:
            problems.append(f"{relative}: {line.strip()!r} appears {count} times, expected once")

    for relative, _anchor, block in BLOCK_REGISTRATIONS:
        count = (package / relative).read_text().count(block)
        if count != 1:
            problems.append(f"{relative}: {block.splitlines()[0].strip()!r} appears {count} times, expected once")

    root = repository(package)
    if root is not None:
        problems += check_repository(root)
    return problems


def check_repository(root: Path) -> list[str]:
    """The part of `check` that lives outside the package: documentation and tests."""
    problems = []
    for relative, page in DOC_PAGES.items():
        target = root / relative
        if not target.is_file() or target.read_text() != page:
            problems.append(f"missing or stale: {relative}")
    for relative, _anchor, block in DOC_REGISTRATIONS:
        if (root / relative).read_text().count(block + "\n") != 1:
            problems.append(f"{relative}: {block.splitlines()[0].strip()!r} not placed exactly once")
    for target, expected in rendered_tests(root).items():
        if not target.is_file() or target.read_text() != expected:
            problems.append(f"missing or stale: {target.relative_to(root)}")
    return problems


def verify(package: Path) -> None:
    """Import Diffusers from this tree in a fresh interpreter and look the classes up."""
    code = (
        "import diffusers;"
        f"names = {[TRANSFORMER, SCHEDULER, *MODULAR_NAMES]!r};"
        "missing = [n for n in names if not hasattr(diffusers, n)];"
        "print('[integrate] missing:', missing) if missing else "
        "print('[integrate] the tree exposes every class the port needs')"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(package.parent),
        check=False,
    )
    print(result.stdout.strip() or result.stderr.strip()[-2000:])
    if result.returncode != 0:
        raise SystemExit("verification failed")


# The composite blocks marked `# auto_docstring`, whose docstrings Diffusers generates.
AUTO_DOCSTRING_CLASSES = (
    ("modular_blocks_kandinsky_s2v.py", MODULAR_BLOCKS),
    ("modular_blocks_kandinsky_s2v_distilled.py", MODULAR_DISTILLED_BLOCKS),
)


def docstring_span(source: str, class_name: str) -> str:
    """The class's docstring as it appears in ``source``, quotes and indentation included."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            first = node.body[0]
            if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)):
                raise SystemExit(f"{class_name} has no docstring to replace")
            return "".join(source.splitlines(keepends=True)[first.lineno - 1 : first.end_lineno])
    raise SystemExit(f"class {class_name} not found")


def sync_docstrings(package: Path) -> None:
    """Regenerate the `# auto_docstring` docstrings in the tree, and carry them back into the port.

    They are generated from the blocks' declared inputs, components and outputs by
    Diffusers' own `utils/modular_auto_docstring.py`, which imports the classes from
    the tree, so it can only run after an injection; its CI check fails on any that
    is stale. Writing the result back into the port is what keeps the next injection
    from reverting it.
    """
    root = repository(package)
    if root is None:
        raise SystemExit("--sync-docstrings needs a repository checkout, not a bare package")
    # The tool calls `ruff` and `doc-builder` by name; ours sit next to this interpreter.
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    env["PYTHONPATH"] = f"{package.parent}{os.pathsep}{env.get('PYTHONPATH', '')}"
    # Relative to the repository: the tool turns the path into the module name it
    # imports, and on an absolute one it only warns, exits 0 and changes nothing.
    target = (package / MODULAR_PACKAGE).relative_to(root)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "utils/modular_auto_docstring.py", str(target), "--fix_and_overwrite"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    if "Could not" in result.stdout + result.stderr:
        raise SystemExit(f"modular_auto_docstring.py could not regenerate the docstrings:\n{result.stdout}")
    for name, class_name in AUTO_DOCSTRING_CLASSES:
        generated = docstring_span((package / MODULAR_PACKAGE / name).read_text(), class_name)
        source_path = PORT / "modular" / name
        source = source_path.read_text()
        source_path.write_text(source.replace(docstring_span(source, class_name), generated, 1))
        print(f"[integrate] docstring of {class_name} carried back into the port")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diffusers",
        type=Path,
        required=True,
        help="Diffusers checkout (main of diffusers-new-model-addition-kandinskyX), or its package directory",
    )
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the injected copy is up to date, changing nothing",
    )
    parser.add_argument(
        "--sync-docstrings",
        action="store_true",
        help="after injecting, regenerate the modular blocks' docstrings there and write them back into the port",
    )
    args = parser.parse_args()

    package = resolve_package(args.diffusers)
    print(f"[integrate] target: {package}")

    if args.check:
        problems = check(package)
        for problem in problems:
            print(f"[integrate] {problem}")
        if problems:
            raise SystemExit("the injected copy is out of date; re-run without --check")
        print("[integrate] the injected copy matches the port")
        return

    place_sources(package)
    dummy_blocks(package)
    register(package)
    register_blocks(package)
    root = repository(package)
    if root is not None:
        register_docs(root)
        place_tests(root)
    else:
        print(f"[integrate] no {DOCS} above the package; documentation and tests left alone")

    if args.sync_docstrings:
        sync_docstrings(package)
        # Placed again, so the tree holds exactly what the port now renders to.
        place_sources(package)

    if not args.no_verify:
        verify(package)


if __name__ == "__main__":
    main()
