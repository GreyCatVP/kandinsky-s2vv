"""Lay out a Diffusers bundle for one S2V configuration.

    PYTHONPATH=src .venv/bin/python dev/bench/build_diffusers_bundle.py \\
        --config src/kandinsky_s2v/configs/s2v_lite.yaml \\
        --out /path/to/Kandinsky-S2V-1.0-Lite-Diffusers

The result is one folder per component and `modular_model_index.json`, which is
what `ModularPipeline.from_pretrained` reads — Diffusers carries S2V as a modular
pipeline only. It names the pipeline and blocks, base or distilled, with every
component pointing at `--repo-id` (the Hub id the bundle is uploaded under;
defaults to the folder itself).

Big weights are symlinked, never copied. The transformer is symlinked straight
at the converted checkpoint — the port keeps the training parameter names, so no
remapping is needed and the file is used as it is. The two text encoders are
*exported* once into `--shared` and symlinked from every bundle: the Qwen
checkpoint on disk is the full Omni model including the Talker, which this
pipeline never runs, and the CLIP checkpoint is a dual-encoder from which only
the text tower is used.

`ModularPipeline.from_pretrained` resolves the class names against the `diffusers`
package, so it loads the bundle only once the port has been injected there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from kandinsky_s2v.pipeline.weights import resolve

TRANSFORMER_CLASS = "KandinskyS2VTransformer3DModel"
PISCHEDULER_CLASS = "KandinskyS2VPiflowScheduler"


def components(distilled: bool) -> dict[str, tuple[str, str]]:
    """(library, class) of each component, as the modular index names them."""
    return {
        "transformer": ("diffusers", TRANSFORMER_CLASS),
        "vae": ("diffusers", "AutoencoderKLHunyuanVideo"),
        "text_encoder": ("transformers", "Qwen2_5OmniThinkerForConditionalGeneration"),
        "tokenizer": ("transformers", "Qwen2_5OmniProcessor"),
        "text_encoder_2": ("transformers", "CLIPTextModel"),
        "tokenizer_2": ("transformers", "CLIPTokenizer"),
        "scheduler": ("diffusers", PISCHEDULER_CLASS if distilled else "FlowMatchEulerDiscreteScheduler"),
    }


def write_modular_index(root: Path, repo_id: str, distilled: bool) -> None:
    """`modular_model_index.json`: the pipeline and blocks to build, and where each component loads from."""
    import diffusers  # noqa: PLC0415

    prefix = "KandinskyS2VDistilled" if distilled else "KandinskyS2V"
    index = {
        "_class_name": f"{prefix}ModularPipeline",
        "_diffusers_version": diffusers.__version__,
        "_blocks_class_name": f"{prefix}AutoBlocks",
    }
    for name, (library, class_name) in components(distilled).items():
        spec = {
            "type_hint": [library, class_name],
            "pretrained_model_name_or_path": repo_id,
            "subfolder": name,
            "variant": None,
            "revision": None,
        }
        index[name] = [library, class_name, spec]
    (root / "modular_model_index.json").write_text(json.dumps(index, indent=2) + "\n")


WEIGHT_NAME = "diffusion_pytorch_model.safetensors"


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    # Absolute: a relative link resolves against the bundle folder, not the working directory.
    target.symlink_to(source.absolute())


def export_thinker(qwen_omni_path: str, out: Path) -> None:
    """Save the Thinker alone; the Talker in the Omni checkpoint is dead weight here."""
    if (out / "config.json").is_file():
        print(f"[bundle] thinker already exported to {out}")
        return
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration  # noqa: PLC0415

    from kandinsky_s2v.core.components.omni_embedder import _DTYPE_KWARG  # noqa: PLC0415

    print(f"[bundle] exporting the Thinker from {qwen_omni_path} (a few minutes)...", flush=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(qwen_omni_path, **{_DTYPE_KWARG: torch.bfloat16})
    model.save_pretrained(out)
    del model


def export_clip_text(clip_path: str, out: Path) -> None:
    """Save the CLIP text tower alone; the checkpoint on disk is a dual encoder."""
    if (out / "config.json").is_file():
        print(f"[bundle] CLIP text tower already exported to {out}")
        return
    from kandinsky_s2v.core.components.omni_embedder import load_clip_text_model  # noqa: PLC0415

    print(f"[bundle] exporting the CLIP text tower from {clip_path}...", flush=True)
    load_clip_text_model(clip_path).save_pretrained(out)


def save_tokenizers(qwen_omni_path: str, clip_path: str, root: Path) -> None:
    """Re-save both tokenizers so the bundle carries exactly the files they need."""
    from transformers import CLIPTokenizer, Qwen2_5OmniProcessor  # noqa: PLC0415

    Qwen2_5OmniProcessor.from_pretrained(qwen_omni_path).save_pretrained(root / "tokenizer")
    CLIPTokenizer.from_pretrained(clip_path).save_pretrained(root / "tokenizer_2")


def link_folder(source: Path, target: Path) -> None:
    """Symlink every file of a component folder (not the folder itself, for clarity)."""
    for path in sorted(source.iterdir()):
        if path.is_file():
            link(path, target / path.name)


def scheduler_config(cfg: dict) -> tuple[str, dict]:
    """The scheduler class and config this configuration needs."""
    piflow = cfg.get("piflow") or {}
    if not piflow.get("enabled"):
        return "FlowMatchEulerDiscreteScheduler", {
            "num_train_timesteps": 1000,
            "shift": cfg["generation"]["scheduler_scale"],
        }
    return PISCHEDULER_CLASS, {
        "num_train_timesteps": 1000,
        "shift": piflow.get("shift") or cfg["generation"]["scheduler_scale"],
        "n_grid": piflow["n_grid"],
        "eps": piflow["eps"],
        "final_step_size_scale": piflow["final_step_size_scale"],
        "num_policy_substeps": piflow["num_policy_substeps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="bundle root")
    parser.add_argument("--shared", required=True, help="where the exported encoders live, shared by all bundles")
    parser.add_argument("--repo-id", help="what the modular index points the components at; defaults to --out")
    args = parser.parse_args()

    import diffusers  # noqa: PLC0415

    cfg = yaml.safe_load(Path(args.config).read_text())
    # The configs name Hub references; a local path passes through unchanged.
    paths = {name: resolve(reference) for name, reference in cfg["paths"].items()}
    root = Path(args.out)
    shared = Path(args.shared)
    version = diffusers.__version__

    # transformer
    dit = dict(cfg["dit"])
    piflow = cfg.get("piflow") or {}
    if piflow.get("enabled"):
        # The distilled head is widened n_grid times. The native DiT applies that
        # in a subclass; in the port the scheduler splits the grid, so the widening
        # has to be visible in the transformer config.
        dit["out_visual_dim"] = dit["out_visual_dim"] * piflow["n_grid"]
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": TRANSFORMER_CLASS, "_diffusers_version": version, **dit}, indent=2) + "\n"
    )
    link(Path(paths["dit"]), root / "transformer" / WEIGHT_NAME)

    # vae
    link_folder(Path(paths["vae"]), root / "vae")

    # encoders
    export_thinker(paths["qwen_omni"], shared / "thinker")
    export_clip_text(paths["clip"], shared / "clip_text")
    link_folder(shared / "thinker", root / "text_encoder")
    link_folder(shared / "clip_text", root / "text_encoder_2")
    save_tokenizers(paths["qwen_omni"], paths["clip"], root)

    # scheduler
    scheduler_class, scheduler_cfg = scheduler_config(cfg)
    (root / "scheduler").mkdir(parents=True, exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": scheduler_class, "_diffusers_version": version, **scheduler_cfg}, indent=2) + "\n"
    )

    # index
    distilled = scheduler_class == PISCHEDULER_CLASS
    write_modular_index(root, args.repo_id or str(root.resolve()), distilled)

    print(f"[bundle] {root}")
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = list(folder.iterdir())
        size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"[bundle]   {folder.name + '/':<16} {len(files):>3} files  {size / 2**30:6.2f} GiB")


if __name__ == "__main__":
    main()
