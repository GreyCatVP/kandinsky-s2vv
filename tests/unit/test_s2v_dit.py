"""S2V DiT construction and a full forward pass on CPU."""

from __future__ import annotations

import pytest
import torch

from kandinsky_s2v.core.algo.denoise_loop import ChunkContext, TextContext, dit_forward
from kandinsky_s2v.core.algo.prepare_latents import init_memory
from kandinsky_s2v.core.algo.prepare_ropes import (
    audio_rope_pos,
    memory_rope_pos,
    ref_rope_pos,
    text_rope_pos,
    visual_rope_pos,
)
from kandinsky_s2v.core.components.dit import DiffusionTransformer3DS2V
from kandinsky_s2v.core.components.piflow_dit import PiFlowDiffusionTransformer3DS2V
from kandinsky_s2v.pipeline.factory import _materialize_meta_buffers

CPU = torch.device("cpu")
PATCH = (1, 2, 2)
FRAME_BUCKETS = (2, 4, 24)
COEFS = ((1, 1, 1), (2, 2, 2), (4, 4, 4))
DURATION, HEIGHT, WIDTH, CHANNELS = 4, 16, 24, 16
AUDIO_TOKENS, TEXT_TOKENS = 13, 7

TINY = dict(
    in_visual_dim=CHANNELS,
    out_visual_dim=CHANNELS,
    in_text_dim=32,
    in_text_dim2=16,
    in_audio_dim=32,
    time_dim=16,
    patch_size=PATCH,
    frame_buckets=FRAME_BUCKETS,
    enable_framepack=True,
    index_rope_ref_image=40,
    patch_proj1x_coef=COEFS[0],
    patch_proj2x_coef=COEFS[1],
    patch_proj4x_coef=COEFS[2],
    model_dim=32,
    ff_dim=64,
    num_text_blocks=1,
    num_audio_blocks=1,
    num_visual_blocks=3,
    axes_dims=(4, 6, 6),
    audio_inject_layers=(0, 2),
    attention_engine="sdpa",
)


def _inputs():
    memory = init_memory(FRAME_BUCKETS, COEFS, PATCH, HEIGHT, WIDTH, CHANNELS, CPU)
    ctx = ChunkContext(
        ref=torch.randn(1, HEIGHT, WIDTH, CHANNELS),
        memory=memory,
        audio_embeds=torch.randn(AUDIO_TOKENS, 32),
        visual_rope_pos=visual_rope_pos(DURATION, HEIGHT, WIDTH, PATCH, CPU),
        ref_rope_pos=ref_rope_pos(40, HEIGHT, WIDTH, PATCH, CPU),
        memory_rope_pos=memory_rope_pos(memory, FRAME_BUCKETS, COEFS, PATCH, CPU),
        audio_rope_pos=audio_rope_pos(AUDIO_TOKENS, DURATION, CPU),
        duration=DURATION,
        scale_factor=(1.0, 2.0, 2.0),
    )
    text = TextContext(torch.randn(TEXT_TOKENS, 32), torch.randn(1, 16), text_rope_pos(TEXT_TOKENS, CPU))
    latent = torch.randn(DURATION, HEIGHT, WIDTH, CHANNELS)
    return latent, ctx, text


def _randomize_modulations(model: torch.nn.Module) -> None:
    """Modulation heads are zero-initialised, which makes every block an identity."""
    generator = torch.Generator().manual_seed(1234)
    for name, module in model.named_modules():
        if name.endswith(("modulation", "visual_modulation", "text_modulation")):
            for param in module.parameters():
                param.data = torch.randn(param.shape, generator=generator) * 0.1


def test_audio_adapters_only_on_injected_layers():
    dit = DiffusionTransformer3DS2V(**TINY)
    assert [b.enable_audio for b in dit.visual_transformer_blocks] == [True, False, True]


def test_forward_preserves_latent_shape():
    torch.manual_seed(0)
    dit = DiffusionTransformer3DS2V(**TINY).eval()
    _randomize_modulations(dit)
    latent, ctx, text = _inputs()
    with torch.no_grad():
        out = dit_forward(dit, latent, torch.tensor([0.7]), ctx, text)
    assert out.shape == latent.shape
    assert torch.isfinite(out).all()


def test_piflow_wrapper_adds_a_grid_axis():
    torch.manual_seed(0)
    n_grid = 5
    dit = PiFlowDiffusionTransformer3DS2V(n_grid=n_grid, **TINY).eval()
    latent, ctx, text = _inputs()
    with torch.no_grad():
        out = dit_forward(dit, latent, torch.tensor([0.7]), ctx, text)
    assert out.shape == (DURATION, n_grid, HEIGHT, WIDTH, CHANNELS)


def test_memory_changes_the_prediction():
    """Memory tokens must actually reach the visual stream."""
    torch.manual_seed(0)
    dit = DiffusionTransformer3DS2V(**TINY).eval()
    _randomize_modulations(dit)
    latent, ctx, text = _inputs()
    with torch.no_grad():
        zero_memory = dit_forward(dit, latent, torch.tensor([0.7]), ctx, text)
        ctx.memory = [torch.randn_like(m) for m in ctx.memory]
        filled_memory = dit_forward(dit, latent, torch.tensor([0.7]), ctx, text)
    assert not torch.allclose(zero_memory, filled_memory)


def test_audio_changes_the_prediction():
    """The frame-aligned audio adapter must reach the visual stream."""
    torch.manual_seed(0)
    dit = DiffusionTransformer3DS2V(**TINY).eval()
    _randomize_modulations(dit)
    latent, ctx, text = _inputs()
    with torch.no_grad():
        with_audio = dit_forward(dit, latent, torch.tensor([0.7]), ctx, text)
        without_audio = dit_forward(dit, latent, torch.tensor([0.7]), ctx, text, audio_scale=0.0)
    assert not torch.allclose(with_audio, without_audio)


def test_materializing_meta_buffers_keeps_loaded_weights():
    """Buffer materialisation must not re-initialise loaded parameters.

    The DiT is built on ``meta`` and filled with ``load_state_dict(assign=True)``,
    which leaves non-persistent buffers (RoPE tables, time frequencies) on meta.
    Rebuilding those must touch buffers only — an over-eager ``reset_parameters``
    silently replaces the loaded time-embedding weights with random ones.
    """
    torch.manual_seed(0)
    reference = DiffusionTransformer3DS2V(**TINY)
    _randomize_modulations(reference)
    # `assign=True` hands these very tensors to the model, so keep an untouched
    # copy to compare against — comparing with `state` would be vacuous, since
    # an in-place re-init would mutate both sides.
    expected = {k: v.clone() for k, v in reference.state_dict().items()}
    state = {k: v.clone() for k, v in reference.state_dict().items()}

    with torch.device("meta"):
        loaded = DiffusionTransformer3DS2V(**TINY)
    loaded.load_state_dict(state, strict=True, assign=True)
    _materialize_meta_buffers(loaded, CPU)

    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, expected[name]), f"{name} changed during materialisation"

    # And the rebuilt buffers must be real, not meta.
    for name, buf in loaded.named_buffers():
        assert buf.device.type != "meta", f"{name} left on meta"


def test_buffer_materialisation_requires_explicit_optin():
    """A module with a non-persistent buffer must declare materialize_buffers().

    Guards the contract: relying on torch's default ``reset_parameters`` here
    would re-initialise loaded weights instead of just rebuilding buffers.
    """

    class Sneaky(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("table", torch.zeros(2), persistent=False)

    with torch.device("meta"):
        module = Sneaky()
    with pytest.raises(TypeError, match="materialize_buffers"):
        _materialize_meta_buffers(module, CPU)
