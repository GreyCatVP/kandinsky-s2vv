"""Build a ready-to-run S2V pipeline from a YAML config."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from ..core.components.attention import resolve_attention_engine
from ..core.components.dit import DiffusionTransformer3DS2V
from ..core.components.omni_embedder import S2VTextAudioEmbedder
from ..core.components.piflow_dit import PiFlowDiffusionTransformer3DS2V
from ..core.parallel import SequenceParallel
from ..core.utils.compile import compile_visual_blocks
from .config import S2VPipelineConfig, load_config
from .pipeline import KandinskyS2VPipeline
from .weights import resolve


def build_vae(vae_path: str, device: str | torch.device = "cuda", tiling: bool = False):
    """HunyuanVideo VAE, as the S2V reference loads it."""
    from diffusers import AutoencoderKLHunyuanVideo  # noqa: PLC0415

    # `torch_dtype`, not a cast afterwards: diffusers warns on `.to(dtype)` even
    # when, as here, the model has no float32 modules to protect.
    vae = AutoencoderKLHunyuanVideo.from_pretrained(vae_path, torch_dtype=torch.float16).to(device).eval()
    if tiling:
        vae.enable_tiling()
    return vae


def _materialize_meta_buffers(module: torch.nn.Module, device: torch.device) -> None:
    """Allocate ``persistent=False`` buffers left on ``meta`` after an assign load.

    RoPE tables and time-embedding frequencies are never stored in checkpoints,
    so after building on ``meta`` they must be recomputed.

    Modules opt in by defining ``materialize_buffers()``. Deliberately *not*
    ``reset_parameters()``: torch's default re-initialises weights, so calling it
    here once silently replaced the loaded time-embedding weights with random
    ones. Any module carrying a non-persistent buffer must implement
    ``materialize_buffers`` and touch nothing but that buffer.
    """
    for mod in module.modules():
        meta_buffers = [
            name
            for name, buf in mod._buffers.items()  # noqa: SLF001
            if buf is not None and buf.device.type == "meta"
        ]
        if not meta_buffers:
            continue
        for name in meta_buffers:
            buf = mod._buffers[name]  # noqa: SLF001
            mod._buffers[name] = torch.empty(buf.shape, dtype=buf.dtype, device=device)  # noqa: SLF001
        if not hasattr(mod, "materialize_buffers"):
            raise TypeError(
                f"{type(mod).__name__} has non-persistent buffer(s) {meta_buffers} but no "
                "materialize_buffers(); add one that rebuilds only those buffers"
            )
        mod.materialize_buffers()


def create_bare_dit(
    cfg: S2VPipelineConfig,
    device: str | torch.device,
    attention_engine: str,
) -> DiffusionTransformer3DS2V:
    """Construct the DiT on ``meta`` and load its weights in place."""
    device = torch.device(device)
    dit_cfg = cfg.dit
    kwargs = dit_cfg.model_dump()
    kwargs["attention_engine"] = attention_engine

    with torch.device("meta"):
        dit: DiffusionTransformer3DS2V = (
            PiFlowDiffusionTransformer3DS2V(n_grid=cfg.piflow.n_grid, **kwargs)
            if cfg.piflow.enabled
            else DiffusionTransformer3DS2V(**kwargs)
        )

    # Loaded as stored: the converter already decides per tensor (bf16 in bulk,
    # fp32 for the time embedding and every norm). Casting here would undo that.
    state_dict = load_file(resolve(cfg.paths.dit), device="cpu")
    dit.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict

    _materialize_meta_buffers(dit, torch.device("cpu"))
    return dit.to(device).eval()


def get_s2v_pipeline(
    conf_path: str | Path,
    device: str | torch.device = "cuda",
    attention_engine: str | None = None,
) -> KandinskyS2VPipeline:
    cfg = load_config(conf_path)
    return build_pipeline(cfg, device, attention_engine)


def build_pipeline(
    cfg: S2VPipelineConfig,
    device: str | torch.device = "cuda",
    attention_engine: str | None = None,
) -> KandinskyS2VPipeline:
    device = torch.device(device)
    torch.set_float32_matmul_precision("high")

    # Under torchrun every process is one shard of the visual sequence; see core/parallel.py.
    sequence_parallel = SequenceParallel.from_env()
    if sequence_parallel is not None:
        sequence_parallel.check_heads(cfg.dit.model_dim // sum(cfg.dit.axes_dims))
        if device.type == "cuda":
            device = sequence_parallel.device

    if cfg.piflow.enabled and cfg.generation.guidance_weight != 1.0:
        raise ValueError("piflow.enabled expects generation.guidance_weight=1.0 (CFG is distilled into the student)")

    requested = attention_engine if attention_engine is not None else cfg.attention.engine
    engine = resolve_attention_engine(requested)

    # With module offload the weights stay on CPU until their stage runs.
    load_device = torch.device("cpu") if cfg.offload.strategy == "module" else device

    embedder = S2VTextAudioEmbedder(
        resolve(cfg.paths.qwen_omni),
        resolve(cfg.paths.clip),
        device=load_device,
        clip_max_length=cfg.text.clip_max_length,
    )
    vae = build_vae(resolve(cfg.paths.vae), load_device, tiling=cfg.offload.vae_tiling)
    dit = create_bare_dit(cfg, load_device, engine)
    dit.sequence_parallel = sequence_parallel
    if cfg.compile.strategy == "torch":
        dit = compile_visual_blocks(dit)

    return KandinskyS2VPipeline(
        cfg=cfg, dit=dit, vae=vae, embedder=embedder, device=device, sequence_parallel=sequence_parallel
    )
