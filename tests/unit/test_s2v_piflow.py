"""π-Flow sampler: evaluation count, grid handling and the Euler fallback."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from kandinsky_s2v.core.algo.denoise_loop import ChunkContext, TextContext, denoise_chunk
from kandinsky_s2v.core.algo.piflow_math import DXPolicy
from kandinsky_s2v.core.algo.piflow_sampler import piflow_denoise_chunk

DURATION, HEIGHT, WIDTH, CHANNELS = 4, 2, 2, 3


class _CountingDiT(nn.Module):
    """Returns a constant velocity (optionally as a DX grid) and counts calls."""

    def __init__(self, n_grid: int | None = None):
        super().__init__()
        self.calls = 0
        self.n_grid = n_grid
        self.audio_scales: list[float] = []

    def forward(self, latent: Tensor, _ref, _m4, _m2, _m1, _text, audio, *_args):
        self.calls += 1
        self.audio_scales.append(float(audio.abs().sum()))
        velocity = torch.full_like(latent, 0.1)
        if self.n_grid is None:
            return velocity
        return velocity.unsqueeze(1).expand(-1, self.n_grid, -1, -1, -1).contiguous()


def _context() -> ChunkContext:
    return ChunkContext(
        ref=torch.zeros(1, HEIGHT, WIDTH, CHANNELS),
        memory=[torch.zeros(1, HEIGHT, WIDTH, CHANNELS)] * 3,
        audio_embeds=torch.ones(DURATION, 8),
        visual_rope_pos=[],
        ref_rope_pos=[],
        memory_rope_pos=[[], [], []],
        audio_rope_pos=torch.zeros(DURATION, dtype=torch.int32),
        duration=DURATION,
    )


def _text() -> TextContext:
    return TextContext(torch.zeros(2, 8), torch.zeros(1, 4), torch.zeros(2, dtype=torch.int32))


@pytest.fixture
def latent() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(DURATION, HEIGHT, WIDTH, CHANNELS)


def test_piflow_uses_one_forward_per_step(latent: Tensor):
    dit = _CountingDiT(n_grid=5)
    piflow_denoise_chunk(
        latent,
        dit,
        _context(),
        _text(),
        _text(),
        num_steps=16,
        scheduler_scale=10.0,
        guidance_weight=1.0,
        num_policy_substeps=8,
    )
    assert dit.calls == 16  # noqa: PLR2004


def test_piflow_with_cfg_costs_two_forwards_per_step(latent: Tensor):
    dit = _CountingDiT(n_grid=5)
    piflow_denoise_chunk(
        latent,
        dit,
        _context(),
        _text(),
        _text(),
        num_steps=4,
        scheduler_scale=10.0,
        guidance_weight=3.0,
        num_policy_substeps=8,
    )
    assert dit.calls == 8  # noqa: PLR2004
    # The unconditional pass zeroes the audio features.
    assert dit.audio_scales[1] == 0.0


def test_euler_loop_uses_num_steps_minus_one_updates(latent: Tensor):
    dit = _CountingDiT()
    denoise_chunk(
        latent,
        dit,
        _context(),
        _text(),
        _text(),
        num_steps=10,
        guidance_weight=1.0,
        scheduler_scale=10.0,
    )
    assert dit.calls == 9  # noqa: PLR2004


def test_euler_cfg_zeroes_audio_in_uncond_pass(latent: Tensor):
    dit = _CountingDiT()
    denoise_chunk(
        latent,
        dit,
        _context(),
        _text(),
        _text(),
        num_steps=3,
        guidance_weight=3.0,
        scheduler_scale=10.0,
    )
    assert dit.calls == 4  # noqa: PLR2004
    assert dit.audio_scales[1] == 0.0


def test_piflow_rejects_zero_steps(latent: Tensor):
    with pytest.raises(ValueError):
        piflow_denoise_chunk(
            latent,
            _CountingDiT(n_grid=5),
            _context(),
            _text(),
            _text(),
            num_steps=0,
            scheduler_scale=10.0,
        )


def test_dx_policy_interpolates_between_grid_points():
    x_t = torch.zeros(2, 1, 1, 1)
    sigma = torch.full((2, 1, 1, 1), 0.5)
    grid = torch.stack([torch.full_like(x_t, -1.0), torch.full_like(x_t, 1.0)], dim=1)
    policy = DXPolicy(grid, x_t, sigma, torch.full_like(sigma, 0.2), shift=1.0, eps=1e-6)
    velocity = policy.pi(x_t, sigma.squeeze())
    assert torch.isfinite(velocity).all()
    assert velocity.shape == x_t.shape
