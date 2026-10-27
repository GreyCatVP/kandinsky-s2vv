"""Kandinsky S2V 1.0 in Modular Diffusers form: the blocks and their pipelines."""

from .modular_blocks_kandinsky_s2v import KandinskyS2VAutoBlocks
from .modular_blocks_kandinsky_s2v_distilled import KandinskyS2VDistilledAutoBlocks
from .modular_pipeline import KandinskyS2VDistilledModularPipeline, KandinskyS2VModularPipeline

__all__ = [
    "KandinskyS2VAutoBlocks",
    "KandinskyS2VDistilledAutoBlocks",
    "KandinskyS2VDistilledModularPipeline",
    "KandinskyS2VModularPipeline",
]
