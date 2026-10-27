"""Where a configured weight lives: on this machine, or on the Hub.

`paths` in a configuration used to mean an absolute path on the machine the
model was trained on, which is why none of the shipped configurations run
anywhere else. An entry may now also name a Hub repository, and the same four
keys work on a machine that has never seen the shared filesystem:

    paths:
      dit: kandinskylab/Kandinsky-S2V-1.0-Pro/model/kandinsky_s2v_pro.safetensors
      qwen_omni: Qwen/Qwen2.5-Omni-7B

A local path wins whenever it exists, so nothing about an existing checkout
changes and no configuration silently starts downloading. A path that does not
exist is a Hub reference only if it looks like one — `owner/name`, optionally
followed by a path inside the repository. Anything else is the error it always
was, rather than a download that fails half a minute later with a stranger's
message.

This is prod-only: the port under `ports/` resolves its weights the Diffusers
way, through `from_pretrained`.
"""

from __future__ import annotations

import re
from pathlib import Path

# owner/name, then an optional path inside the repository.
REFERENCE = re.compile(r"^[\w.-]+/[\w.-]+(?:/[\w./-]+)?$")


def resolve(reference: str) -> str:
    """A local path as it is, a Hub reference downloaded and pointed at."""
    if Path(reference).exists():
        return reference
    if not REFERENCE.match(reference):
        raise FileNotFoundError(f"no such path, and not a Hub reference either: {reference}")

    from huggingface_hub import hf_hub_download, snapshot_download  # noqa: PLC0415
    from huggingface_hub.utils import tqdm  # noqa: PLC0415

    class _DownloadOnly(tqdm):
        """`snapshot_download`'s bars, shown only for bytes actually downloaded.

        On a cache hit it still prints a "Fetching N files" counter and, in a
        terminal, two empty byte bars ("Download complete 0.00B"), once per
        repository and per process. The counter is dropped; a byte bar stays
        silent while its total is zero, which it is until a download starts.
        """

        def __init__(self, *args, **kwargs):
            if str(kwargs.get("desc", "")).startswith("Fetching"):
                kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def display(self, *args, **kwargs):
            return bool(self.total) and super().display(*args, **kwargs)

        def close(self):
            if not self.total:
                self.leave = False  # close() would otherwise end the empty bar with a newline
            super().close()

    owner, name, *rest = reference.split("/")
    repo_id = f"{owner}/{name}"
    inside = "/".join(rest)
    if not inside:
        # Nothing here loads TensorFlow or Flax, and the CLIP repository carries a
        # mirror of its weights in both — three gigabytes that would never be read.
        return snapshot_download(repo_id, ignore_patterns=["*.msgpack", "*.h5"], tqdm_class=_DownloadOnly)
    # A file is fetched alone; a folder brings only itself, not the whole repository,
    # which matters when the sibling folders are tens of gigabytes.
    if Path(inside).suffix:
        return hf_hub_download(repo_id, inside)
    return str(Path(snapshot_download(repo_id, allow_patterns=f"{inside}/*", tqdm_class=_DownloadOnly)) / inside)
