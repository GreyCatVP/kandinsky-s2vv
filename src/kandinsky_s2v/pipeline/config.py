"""Hierarchical typed S2V pipeline configuration (Pydantic v2).

Each subsection is consumed by one layer:
  paths      → factory weight loading
  dit        → DiffusionTransformer3DS2V(...)
  generation → KandinskyS2VPipeline defaults
  piflow     → distilled DX-policy sampler
  offload    → ModuleOffload
  attention  → self-attention kernel selection
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Cfg):
    dit: str
    vae: str
    qwen_omni: str
    clip: str


class DiTConfig(_Cfg):
    """Maps 1:1 onto the K5 ``dit.params`` section of ``configs/model/dit_s2v*.yaml``."""

    in_visual_dim: int = 16
    out_visual_dim: int = 16
    in_text_dim: int = 3584
    in_text_dim2: int = 768
    in_audio_dim: int = 3584
    time_dim: int = 512
    patch_size: tuple[int, int, int] = (1, 2, 2)
    # FramePack memory: latent frames kept at 1x / 2x / 4x temporal downsampling.
    frame_buckets: tuple[int, int, int] = (2, 4, 24)
    enable_framepack: bool = True
    index_rope_ref_image: int = 40
    patch_proj1x_coef: tuple[int, int, int] = (1, 1, 1)
    patch_proj2x_coef: tuple[int, int, int] = (2, 2, 2)
    patch_proj4x_coef: tuple[int, int, int] = (4, 4, 4)
    model_dim: int = 1792
    ff_dim: int = 7168
    num_text_blocks: int = 2
    num_audio_blocks: int = 2
    num_visual_blocks: int = 32
    axes_dims: tuple[int, int, int] = (16, 24, 24)
    # Indices of visual blocks carrying the audio cross-attention adapter.
    audio_inject_layers: tuple[int, ...] = ()
    text_rope_max_pos: int = 4096

    @field_validator(
        "patch_size",
        "frame_buckets",
        "patch_proj1x_coef",
        "patch_proj2x_coef",
        "patch_proj4x_coef",
        "axes_dims",
        "audio_inject_layers",
        mode="before",
    )
    @classmethod
    def _as_tuple(cls, v):
        return tuple(v)


class GenerationConfig(_Cfg):
    fps: int = 25
    # Latent frames per chunk (K5 ``duration``); 32 → (32-1)*4+1 = 125 pixel frames.
    latent_frames: int = 32
    allowed_resolutions: tuple[tuple[int, int], ...] = ((512, 512), (768, 512), (512, 768))
    num_steps: int = 50
    guidance_weight: float = 3.0
    scheduler_scale: float = 10.0
    scale_factor: tuple[float, float, float] = (1.0, 2.0, 2.0)
    seed: int = 5
    prompt: str
    negative_prompt: str

    @field_validator("allowed_resolutions", mode="before")
    @classmethod
    def _as_resolutions(cls, v):
        return tuple(tuple(item) for item in v)

    @field_validator("scale_factor", mode="before")
    @classmethod
    def _as_tuple(cls, v):
        return tuple(v)


class PiFlowConfig(_Cfg):
    """π-Flow DX-policy settings, embedded next to a distilled checkpoint."""

    enabled: bool = False
    n_grid: int = Field(default=5, ge=2)
    num_policy_substeps: int = Field(default=128, ge=1)
    final_step_size_scale: float = Field(default=0.5, gt=0, le=1)
    shift: float | None = None  # None → generation.scheduler_scale
    eps: float = Field(default=1e-6, gt=0)


class TextConfig(_Cfg):
    clip_max_length: int = 77


class OffloadConfig(_Cfg):
    strategy: Literal["none", "module"] = "none"
    # Decode the chunk in tiles instead of one pass. Cuts the decode peak from
    # ~36 GiB to ~7 GiB at 125x512x768, but tile seams shift individual pixels
    # (measured: max 57/255, mean 0.59/255), so the reference behaviour is off.
    vae_tiling: bool = False


class CompileConfig(_Cfg):
    """``torch.compile`` of the visual blocks.

    ``torch`` buys ~1.31x per step at the cost of ~26 s of warm-up on Lite
    (longer on Pro, which has 60 blocks), so it pays off on multi-chunk runs
    and batches rather than on a single short clip. Off by default.
    """

    strategy: Literal["none", "torch"] = "none"


class AttentionConfig(_Cfg):
    engine: Literal[
        "flash_attention_3",
        "flash_attention_2",
        "sdpa",
        "auto",
    ] = "flash_attention_3"


class S2VPipelineConfig(_Cfg):
    paths: PathsConfig
    dit: DiTConfig = Field(default_factory=DiTConfig)
    generation: GenerationConfig
    piflow: PiFlowConfig = Field(default_factory=PiFlowConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    offload: OffloadConfig = Field(default_factory=OffloadConfig)
    compile: CompileConfig = Field(default_factory=CompileConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)

    @model_validator(mode="after")
    def _check_chunk_covers_memory(self) -> S2VPipelineConfig:
        """A chunk must be long enough to fill the FramePack memory of the next one.

        ``roll_memory`` slices the tail of a chunk into the 1x/2x/4x buffers, and
        the RoPE grids for those buffers are sized from ``frame_buckets``. A
        shorter chunk would silently produce undersized buffers that no longer
        match their position grids.
        """
        needed = sum(self.dit.frame_buckets)
        if self.generation.latent_frames < needed:
            raise ValueError(
                f"generation.latent_frames={self.generation.latent_frames} is shorter than the "
                f"FramePack memory window sum(dit.frame_buckets)={needed}"
            )
        return self


def load_config(path: str | Path) -> S2VPipelineConfig:
    """Load and validate a hierarchical YAML S2V pipeline config."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return S2VPipelineConfig.model_validate(raw)
