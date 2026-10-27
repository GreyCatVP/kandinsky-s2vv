"""Grid-output DiT used by distilled π-Flow S2V checkpoints."""

from __future__ import annotations

from torch import Tensor

from .dit import DiffusionTransformer3DS2V

_MIN_GRID_POINTS = 2


class PiFlowDiffusionTransformer3DS2V(DiffusionTransformer3DS2V):
    """Widens the output head ``n_grid`` times and reshapes it into a grid axis."""

    def __init__(self, *, n_grid: int, out_visual_dim: int = 16, **kwargs):
        if n_grid < _MIN_GRID_POINTS:
            raise ValueError(f"PiFlow n_grid must be >= {_MIN_GRID_POINTS}, got {n_grid}")
        super().__init__(out_visual_dim=out_visual_dim * n_grid, **kwargs)
        self.n_grid = int(n_grid)
        self.base_out_visual_dim = int(out_visual_dim)

    def forward(self, *args, **kwargs) -> Tensor:
        velocity = super().forward(*args, **kwargs)
        # (T, H, W, n_grid * C) → (T, n_grid, H, W, C).
        # Grid-major split, byte-for-byte as the distillation fork emits it
        # (kandinsky/model/dit_s2v_dx.py). Do not "tidy" the axis order: the
        # distilled checkpoints were trained against exactly this layout..
        # Grid-major, matching the distillation fork (dit_s2v_dx.py) and the way
        # the student head is initialised there (the teacher row tiled n_grid
        # times). Do not "fix" to channel-major: it would silently shuffle the
        # policy grid while every shape still checks out.
        grid = velocity.view(*velocity.shape[:-1], self.n_grid, self.base_out_visual_dim)
        return grid.permute(0, 3, 1, 2, 4)
