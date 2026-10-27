"""The Diffusers port must reproduce the native implementation exactly.

Both checks run on CPU in float32 with a tiny model, so they need no GPU and no
weights: at that precision the two implementations perform the same operations
in the same order, and anything other than a bit-identical result is a real
behavioural difference, not numerical noise.
"""

from __future__ import annotations

import torch

from kandinsky_s2v.core.algo.denoise_loop import ChunkContext, TextContext
from kandinsky_s2v.core.algo.piflow_sampler import piflow_denoise_chunk
from kandinsky_s2v.core.components.dit import DiffusionTransformer3DS2V
from kandinsky_s2v.core.components.piflow_dit import PiFlowDiffusionTransformer3DS2V
from kandinsky_s2v.ports.diffusers.scheduling_piflow import KandinskyS2VPiflowScheduler
from kandinsky_s2v.ports.diffusers.transformer_kandinsky_s2v import KandinskyS2VTransformer3DModel

CONFIG = {
    "in_visual_dim": 4,
    "out_visual_dim": 4,
    "in_text_dim": 8,
    "in_text_dim2": 6,
    "in_audio_dim": 10,
    "time_dim": 16,
    "patch_size": (1, 2, 2),
    "frame_buckets": (2, 4, 24),
    "enable_framepack": True,
    "index_rope_ref_image": 40,
    "patch_proj1x_coef": (1, 1, 1),
    "patch_proj2x_coef": (2, 2, 2),
    "patch_proj4x_coef": (4, 4, 4),
    "model_dim": 32,
    "ff_dim": 64,
    "num_text_blocks": 2,
    "num_audio_blocks": 2,
    "num_visual_blocks": 3,
    "axes_dims": (4, 6, 6),
    "audio_inject_layers": (0, 2),
    "text_rope_max_pos": 128,
}

DURATION, HEIGHT, WIDTH = 4, 16, 16
NUM_TEXT_TOKENS, NUM_AUDIO_TOKENS = 7, 13

N_GRID = 5
SHIFT = 10.0
EPS = 1e-6
FINAL_STEP_SIZE_SCALE = 0.5
NUM_POLICY_SUBSTEPS = 128


def rope_pos(duration: int, height: int, width: int, t_start: int = 0) -> list[torch.Tensor]:
    return [torch.arange(t_start, t_start + duration), torch.arange(height), torch.arange(width)]


def randomize_modulations(model: torch.nn.Module) -> None:
    """Modulation layers are zero-initialised, which would make every block the identity."""
    for name, parameter in model.named_parameters():
        if "modulation" in name:
            parameter.data.normal_(0.0, 0.02)


def build_inputs() -> dict:
    generator = torch.Generator().manual_seed(1234)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    channels = CONFIG["in_visual_dim"]
    return {
        "x": randn(DURATION, HEIGHT, WIDTH, channels),
        "ref": randn(1, HEIGHT, WIDTH, channels),
        # 1x keeps the visual patch (1, 2, 2); 2x uses (2, 4, 4) and 4x (4, 8, 8),
        # so their frame counts must divide by the coarser temporal patch.
        "memory_1x": randn(2, HEIGHT, WIDTH, channels),
        "memory_2x": randn(4, HEIGHT, WIDTH, channels),
        "memory_4x": randn(4, HEIGHT, WIDTH, channels),
        "text": randn(NUM_TEXT_TOKENS, CONFIG["in_text_dim"]),
        "audio": randn(NUM_AUDIO_TOKENS, CONFIG["in_audio_dim"]),
        "pooled": randn(1, CONFIG["in_text_dim2"]),
        "time": torch.tensor([0.7]),
        "visual_rope_pos": rope_pos(DURATION, HEIGHT // 2, WIDTH // 2),
        "ref_rope_pos": rope_pos(1, HEIGHT // 2, WIDTH // 2, t_start=CONFIG["index_rope_ref_image"]),
        "memory_rope_pos_1x": rope_pos(2, HEIGHT // 2, WIDTH // 2, t_start=-2),
        "memory_rope_pos_2x": rope_pos(2, HEIGHT // 4, WIDTH // 4, t_start=-4),
        "memory_rope_pos_4x": rope_pos(1, HEIGHT // 8, WIDTH // 8, t_start=-8),
        "text_rope_pos": torch.arange(NUM_TEXT_TOKENS),
        "audio_rope_pos": torch.arange(NUM_AUDIO_TOKENS),
    }


def call_ported(model, inputs: dict, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    return model(
        hidden_states=latents.unsqueeze(0),
        encoder_hidden_states=inputs["text"].unsqueeze(0),
        audio_hidden_states=inputs["audio"].unsqueeze(0),
        reference_hidden_states=inputs["ref"].unsqueeze(0),
        memory_hidden_states=(
            inputs["memory_4x"].unsqueeze(0),
            inputs["memory_2x"].unsqueeze(0),
            inputs["memory_1x"].unsqueeze(0),
        ),
        timestep=timestep,
        pooled_projections=inputs["pooled"],
        visual_rope_pos=inputs["visual_rope_pos"],
        reference_rope_pos=inputs["ref_rope_pos"],
        memory_rope_pos=(
            inputs["memory_rope_pos_4x"],
            inputs["memory_rope_pos_2x"],
            inputs["memory_rope_pos_1x"],
        ),
        text_rope_pos=inputs["text_rope_pos"],
        audio_rope_pos=inputs["audio_rope_pos"],
        scale_factor=(1.0, 1.0, 1.0),
        duration=DURATION,
        return_dict=False,
    )[0].squeeze(0)


def test_transformer_matches_native_dit():
    """One forward, and the state dict must load without a single key renamed."""
    torch.manual_seed(0)
    native = DiffusionTransformer3DS2V(**CONFIG, attention_engine="sdpa").eval()
    randomize_modulations(native)

    ported = KandinskyS2VTransformer3DModel(**CONFIG).eval()
    ported.load_state_dict(native.state_dict(), strict=True)

    inputs = build_inputs()
    with torch.no_grad():
        expected = native(
            inputs["x"],
            inputs["ref"],
            inputs["memory_4x"],
            inputs["memory_2x"],
            inputs["memory_1x"],
            inputs["text"],
            inputs["audio"],
            inputs["pooled"],
            inputs["time"],
            inputs["visual_rope_pos"],
            inputs["ref_rope_pos"],
            inputs["memory_rope_pos_4x"],
            inputs["memory_rope_pos_2x"],
            inputs["memory_rope_pos_1x"],
            inputs["text_rope_pos"],
            inputs["audio_rope_pos"],
            (1.0, 1.0, 1.0),
            DURATION,
        )
        actual = call_ported(ported, inputs, inputs["x"], inputs["time"])

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_piflow_scheduler_matches_native_sampler():
    """Four segments of the policy sampler, including the grid split and the rollout."""
    torch.manual_seed(0)
    num_steps = 4
    base_out_dim = CONFIG["out_visual_dim"]
    native_kwargs = {key: value for key, value in CONFIG.items() if key != "out_visual_dim"}

    native = PiFlowDiffusionTransformer3DS2V(
        n_grid=N_GRID, out_visual_dim=base_out_dim, attention_engine="sdpa", **native_kwargs
    ).eval()
    randomize_modulations(native)
    ported = KandinskyS2VTransformer3DModel(**{**CONFIG, "out_visual_dim": base_out_dim * N_GRID}).eval()
    ported.load_state_dict(native.state_dict(), strict=True)

    inputs = build_inputs()
    ctx = ChunkContext(
        ref=inputs["ref"],
        memory=[inputs["memory_1x"], inputs["memory_2x"], inputs["memory_4x"]],
        audio_embeds=inputs["audio"],
        visual_rope_pos=inputs["visual_rope_pos"],
        ref_rope_pos=inputs["ref_rope_pos"],
        memory_rope_pos=[
            inputs["memory_rope_pos_1x"],
            inputs["memory_rope_pos_2x"],
            inputs["memory_rope_pos_4x"],
        ],
        audio_rope_pos=inputs["audio_rope_pos"],
        duration=DURATION,
        scale_factor=(1.0, 1.0, 1.0),
    )
    text = TextContext(text_embeds=inputs["text"], pooled_embed=inputs["pooled"], rope_pos=inputs["text_rope_pos"])

    expected = piflow_denoise_chunk(
        inputs["x"],
        native,
        ctx,
        text,
        text,
        num_steps=num_steps,
        scheduler_scale=SHIFT,
        guidance_weight=1.0,
        num_policy_substeps=NUM_POLICY_SUBSTEPS,
        final_step_size_scale=FINAL_STEP_SIZE_SCALE,
        shift=SHIFT,
        eps=EPS,
    )

    scheduler = KandinskyS2VPiflowScheduler(
        shift=SHIFT,
        n_grid=N_GRID,
        eps=EPS,
        final_step_size_scale=FINAL_STEP_SIZE_SCALE,
        num_policy_substeps=NUM_POLICY_SUBSTEPS,
    )
    scheduler.set_timesteps(num_steps)

    actual = inputs["x"].clone()
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            model_output = call_ported(ported, inputs, actual, timestep.expand(1))
            actual = scheduler.step(model_output.float(), timestep, actual, return_dict=False)[0]

    assert torch.equal(actual, expected)
