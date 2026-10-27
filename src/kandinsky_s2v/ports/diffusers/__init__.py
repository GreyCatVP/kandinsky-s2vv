"""Diffusers port of the Kandinsky S2V 1.0 model, scheduler and modular pipeline.

The files here follow upstream Diffusers conventions — self-contained modules,
``KandinskyS2V``-prefixed classes, absolute ``diffusers.*`` imports — so they
can be dropped into a Diffusers checkout with the imports rewritten to relative
ones. The pipeline is the ``modular`` package: Diffusers carries S2V as a
modular pipeline only.
"""

from .scheduling_piflow import KandinskyS2VPiflowScheduler
from .transformer_kandinsky_s2v import KandinskyS2VTransformer3DModel

__all__ = [
    "KandinskyS2VPiflowScheduler",
    "KandinskyS2VTransformer3DModel",
]
