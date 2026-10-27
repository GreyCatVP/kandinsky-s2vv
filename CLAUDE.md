# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Kandinsky S2V 1.0 — inference for **speech-to-video**: an avatar video is
generated from a **reference frame plus a speech track**. The whole figure is
animated, not only the head. Audio is a *condition*, not an output.

Six shipped configurations, all in [src/kandinsky_s2v/configs/](src/kandinsky_s2v/configs/):

| Config | Model | Sampler |
|---|---|---|
| `s2v_lite.yaml` | Lite 2B | Euler, 50 steps, CFG 3.0 |
| `s2v_pro.yaml` | Pro 21B | Euler, 50 steps, CFG 3.0, module offload |
| `s2v_lite_distill.yaml` | Lite 2B distilled | π-Flow, 16 steps, CFG baked in |
| `s2v_pro_distill.yaml` | Pro 21B distilled | π-Flow, 16 steps, CFG baked in, module offload |
| `s2v_pro_hd.yaml` | Pro 21B, HD grid | as `s2v_pro.yaml`, plus `vae_tiling` |
| `s2v_pro_distill_hd.yaml` | Pro 21B distilled, HD grid | as `s2v_pro_distill.yaml`, plus `vae_tiling` |

## Common commands

Package manager is **uv**; task runner is **just**.

```bash
just setup # install deps + dev group; torch resolved from the pytorch-cu130 index
just lint # ruff check + format --check
just fmt # ruff format + ruff check --fix
just test # pytest; pass args, e.g. `just test tests/unit/test_s2v_piflow.py`

just generate --audio speech.wav --image reference.png --out out.mp4 \
 --config src/kandinsky_s2v/configs/s2v_lite.yaml
just generate-parallel 4 --audio speech.wav --image reference.png --out out.mp4 \
 --config src/kandinsky_s2v/configs/s2v_pro_hd.yaml # torchrun, 4 GPUs of one node
just convert --checkpoint /path/to/ema_model --out /path/to/model.safetensors
```

`just test` needs neither a GPU nor weights. A few tests skip themselves when
what they need is absent: a Diffusers checkout for the injection test, two GPUs
for the NCCL variant of the sequence-parallel test.

`just typecheck` (mypy on `src`) exists but is not clean yet; most of the errors
are in `ports/diffusers/`, where the Diffusers base classes are loosely typed.
Do not treat a mypy failure there as a regression you introduced — compare the
error count before and after your change.

The JUSTFILE also has `parity` and `parity-pipeline`. They compare this code
against the original kandinsky-5 training code, which is not part of this
repository, and fail without a checkout of it.

New runtime deps go in `[project.dependencies]`, dev tools in
`[dependency-groups].dev`. Add new recipes to [JUSTFILE](JUSTFILE) rather than
documenting raw commands ad-hoc.

### Running on GPU

`torch` is routed to the `pytorch-cu130` index, which needs an NVIDIA driver
≥ 580 (or a CUDA forward-compatibility layer in the image). With an older
driver `torch.cuda.is_available()` returns False even though the GPUs are
present, and the venv is CPU-only. Everything that does not need a GPU — tests,
lint, checkpoint conversion — runs regardless.

For GPU work on such a machine, use a CUDA environment matching the driver; the
code supports it (loaders pick `torch_dtype` vs `dtype` and the CLIP key prefix
by transformers version, and the attention build is detected rather than
hardcoded), and [dev/bench/run_e2e.py](dev/bench/run_e2e.py) runs a generation
without the CLI for environments that lack cyclopts.

## CLI

`kandy-s2v` → [src/kandinsky_s2v/cli.py](src/kandinsky_s2v/cli.py), built with
**cyclopts**, not argparse. Subcommands are plain typed functions; parameters
are inferred from type hints. Do NOT introduce a second CLI framework.
Two subcommands: `generate` and `convert`.

## Architecture

```
src/kandinsky_s2v/
 core/
 components/ # objects with weights: dit, piflow_dit, omni_embedder, rope, attention, nn
 algo/ # pure functions: denoise_loop, piflow_{math,sampler}, prepare_{latents,ropes},
 # chunks, audio_io, postprocess, mux
 parallel.py # sequence parallelism under torchrun
 pipeline/ # config (pydantic) + factory + KandinskyS2VPipeline + weights (Hub download)
 configs/ # the six YAML configurations
 ckpt/ # FSDP shards → single safetensors
 ports/
 diffusers/ # the same model in Diffusers form; see "Diffusers port"
 modular/ # and as Modular Diffusers blocks
cli.py # kandy-s2v
```

### How the model is built

- Text and speech conditioning come from **Qwen2.5-Omni-7B** — Thinker hidden
 states for text, audio-tower features (3584-dim, 25 tokens/s) for speech.
 CLIP supplies a pooled text embedding.
- Audio enters through per-layer `cross_attention_v2a` adapters on a subset of
 visual blocks (13 of 32 on Lite, 18 of 60 on Pro), frame-aligned so each
 latent frame attends only to its own audio tokens. The split is uneven: the
 first frame gets one token, the rest divide the remainder (125 = 1 + 31×4).
- The visual stream is one packed sequence `[memory | ref | visual]` with
 FramePack memory at three temporal scales, carried between 5-second chunks.
- `memory` and `ref` are clean, so time modulation is selected **per token**.

### Invariants

1. **One process is one device, batch size 1.** A plain run never touches
 `torch.distributed`. Under `torchrun` the factory attaches a
 `SequenceParallel` to the DiT ([core/parallel.py](src/kandinsky_s2v/core/parallel.py))
 and the visual sequence is sharded across the processes with plain
 collectives. Every process runs the whole pipeline on rank 0's inputs
 (broadcast); only the visual blocks are split; rank 0 saves. One process
 group generates one video: never fold a second video into the same launch as
 a data-parallel dimension — when a long clip shares a launch with a short one,
 the short one's ranks move on and the long one dies of a collective timeout.
 `tests/unit/test_sequence_parallel.py` holds the sharded forward to the
 single-process one. Measured: 2 and 4 processes are bit-identical to one GPU;
 8 differ by kernel noise (PSNR > 40 dB), because cuBLAS and the attention
 kernels choose algorithms by the per-process shape.

2. **Module and parameter names match the training checkpoints exactly.** That
 is why `kandy-s2v convert` is a pure shard gather with **no key remapping**,
 and why the published `.safetensors` load into both the native DiT and the
 Diffusers port unchanged. Keep it that way — renaming a module silently
 breaks every checkpoint.

3. **π-Flow code is a literal port of the code the distilled models were
 trained with.** Two spots look like defects and are not: the grid-major
 output reshape in `piflow_dit.py`, and `clamp(min=0)` in `piflow_math.py`
 against `clamp(min=eps)` in `piflow_sampler.py`. Both are marked with
 comments. Do not "align" them — the distilled checkpoints were trained
 against exactly this behaviour.

4. **Non-persistent buffers are rebuilt by `materialize_buffers`, and it touches
 nothing else.** The DiT is built on `meta` and filled with
 `load_state_dict(assign=True)`, then `_materialize_meta_buffers` calls
 `materialize_buffers()` on modules that still hold meta buffers, and refuses a
 module that has such a buffer but no such method. It is deliberately not
 `reset_parameters`: torch's default re-initialises Linear weights and would
 silently destroy the loaded checkpoint. `test_s2v_dit.py` guards this.

5. **Performance claims come from end-to-end runs, not micro-benchmarks.**
 Compiling the visual blocks is 1.31x on a single step but only 1.03x on a
 three-chunk Lite distilled clip, because warm-up and the VAE decode dilute
 it. Quote the end-to-end number, and say which config it is.

6. **Defaults reproduce the original model's output, and cost nothing at
 start-up.** Anything that diverges (`offload.vae_tiling`, which cuts the
 decode peak but shifts pixels by up to 57/255 at tile seams) or that trades
 warm-up for throughput (`compile.strategy: torch`) is off by default.

## Performance knobs

Both are opt-in, both measured at 512x768 on one H100:

- `compile.strategy: torch` — `torch.compile(dynamic=True)` on the visual
 blocks only, which is where a step profile put 99.8% of the time. On a cold
 inductor and Triton cache compilation costs ~20 s, then 1.22–1.28x per chunk
 ([dev/bench/time_configs.py](dev/bench/time_configs.py)): it pays off on the
 first chunk on Lite, Pro and Pro distilled, and from the third on Lite
 distilled, whose 16-step chunks save only ~8 s each. A warm cache hides the
 compilation, so measure cold (fresh `TORCHINDUCTOR_CACHE_DIR` and
 `TRITON_CACHE_DIR`).
- `offload.vae_tiling: true` — tiles the VAE decode, which owns the memory
 peak (57 GiB on Lite, of which the transformer is ~7 GiB).

Sequence parallelism (`just generate-parallel N`) is the other lever;
[dev/bench/time_parallel.py](dev/bench/time_parallel.py) measures it. Two and
four GPUs are the sweet spot at 512; eight pays in HD and on Pro.

## Weights

The configs name Hub references (`owner/repo/path`), which
[pipeline/weights.py](src/kandinsky_s2v/pipeline/weights.py) downloads on first
use; a local path in their place is used as it is. The DiT weights are
published as `kandinskylab/Kandinsky-S2V-1.0-{Lite,Pro,Lite-distill,Pro-distill}`
(4.2 GiB for Lite, 39.8 GiB for Pro). The encoders come from upstream:
`Qwen/Qwen2.5-Omni-7B`, `openai/clip-vit-large-patch14` and
`hunyuanvideo-community/HunyuanVideo/vae`.

### Dtypes

The bulk is bf16, but the time embedding and every normalisation stay **fp32**
(206 of 1013 tensors on Lite, 354 of 1805 on Pro) — they feed the AdaLN path,
which runs in fp32 anyway. RoPE tables need no rule: they are non-persistent
buffers, absent from the checkpoint, rebuilt in fp32 at load time.

Two places enforce this and must stay in step: `_FP32_KEY_PARTS` in
[ckpt/convert.py](src/kandinsky_s2v/ckpt/convert.py) decides what is written,
and `create_bare_dit` loads weights **as stored** — casting fp32 down to bf16
on load would silently undo the whole thing.

### Converting a checkpoint

`kandy-s2v convert` gathers an FSDP-sharded training checkpoint (any shard
count) into one safetensors file. After converting, run an end-to-end
generation with the matching config: nothing else checks that the file and the
config still belong together.

## Diffusers port

[src/kandinsky_s2v/ports/diffusers/](src/kandinsky_s2v/ports/diffusers/) is this
model written to upstream Diffusers conventions: self-contained modules,
`KandinskyS2V`-prefixed classes, attention behind a processor. It holds the
transformer, the π-Flow scheduler, and a `modular/` package with the pipeline as
[Modular Diffusers](https://huggingface.co/docs/diffusers/modular_diffusers/overview)
blocks. There is no standard `DiffusionPipeline`: Diffusers prefers modular for
new models, and one implementation to maintain beats two.

Every parameter name is kept, so a converted checkpoint loads into the port
unchanged. The distilled configs differ in one field only: their head is
widened `n_grid` times, which the native DiT does in a subclass and the port
does by having the **scheduler** split the grid — hence `out_visual_dim: 80`,
and one transformer class for both.

### Modular pipeline

Laid out as upstream families are — `encoders` (input planning, text and
speech, the reference frame), `before_denoise`, `denoise`, `decoders`, a blocks
file per checkpoint kind, and `modular_pipeline.py` with the `ModularPipeline`
classes, the constants and the small helpers. It imports nothing from
`diffusers.pipelines`.

Three things are shaped by the framework and look odd without the reason:

- **The timestep loop is a plain `for` inside one block.** The chunk loop is a
 `LoopSequentialPipelineBlocks`, and loop blocks cannot nest.
- **Base and distilled are two block sets**, not a switch: a component spec is
 deduplicated by name and the first declaration wins, so the distilled guider
 (`guidance_scale=1.0`), scheduler and 16-step default need blocks of their own.
- **The FramePack memory goes to the transformer beside the guider, not through
 it.** `guider.prepare_inputs` reads any tuple as a (cond, uncond) pair, and the
 memory and its RoPE positions are tuples of three.

A Diffusers checkpoint names its pipeline and blocks in
`modular_model_index.json` (`_class_name`, `_blocks_class_name`), so
`ModularPipeline.from_pretrained` picks base or distilled from the checkpoint
alone. The classes are not in a released Diffusers version yet, so
`from_pretrained` resolves them only against a Diffusers tree that contains the
port; [dev/bench/load_diffusers_bundle.py](dev/bench/load_diffusers_bundle.py)
builds the pipeline from a bundle with the port's own classes instead, which
works with stock Diffusers.

[dev/diffusers_tests/](dev/diffusers_tests/) holds the Diffusers-side tests in
Diffusers form (relative imports into its test suite); they run inside a
Diffusers tree, not here, and this repo's ruff excludes them.

Two rules the Diffusers tests enforce and this repo's tests cannot, because
everything here runs on one device in one dtype: random draws go through
`randn_tensor` (a CPU generator driving a GPU pipeline is how Diffusers gets
cross-device reproducibility), and `decode_latents` returns the dtype the caller
expects, not the bf16 its autocast region produced.

### Offloading Pro

Lite and Lite distilled run fully on an 80 GB GPU (56.8 GiB peak). Pro does
not: beside the 7B Omni encoder, the VAE decode of a chunk has nowhere to go.
Offload through a `ComponentsManager`, and two things matter:

- The default `AutoOffloadStrategy` is not enough — it moves models aside only
 when the next one's weights do not fit, so the VAE stays resident before the
 decode that needs the room. Pass an `offload_strategy` that returns every
 resident hook (one model on the GPU at a time): `lambda hooks, **_: hooks`.
 This brings Pro distilled to a 45.9 GiB peak.
- Enable the offload *after* `load_components`: adding a component re-enables
 it with the default strategy, silently dropping the one passed earlier.

### Comparing against the native pipeline

[dev/bench/s2v_parity_diffusers_pipeline.py](dev/bench/s2v_parity_diffusers_pipeline.py)
diffs the modular pipeline's decoded frames against the native pipeline's on
GPU. Three things must be controlled or the number means nothing:

1. **Pin and seed the reference latent.** `encode_reference_image` samples the
 VAE posterior; drawn without a generator it differs from run to run, and the
 deviation between two implementations is so sensitive to its input that a
 5e-4 change in it moved the result threefold.
2. **Match the attention kernel** with `--attention sdpa`. The native pipeline
 takes FA3 from its config while the port uses the Diffusers dispatcher.
3. **Judge per chunk.** With classifier-free guidance the deviation grows from
 one chunk to the next, so an average over the clip flatters a long run.

The frame-level comparison is a regression check, not a debugging tool: it
returns one number in which every link is summed. When it moves, compare links
individually — a single transformer forward
([dev/bench/s2v_forward_parity.py](dev/bench/s2v_forward_parity.py)) and the
conditioning ([dev/bench/s2v_conditioning_parity.py](dev/bench/s2v_conditioning_parity.py))
are bit-identical between the two, and either returns zero or it does not.
Two past mismatches were found that way: an Euler sigma schedule built with
`np.linspace` (float64, one ULP off in the step size) instead of
`torch.linspace` in float32, and a VAE decode run in fp16 where the native
pipeline's autocast region runs it in bf16.

## Known limitations

- One prompt, one video per call: no batching, no `num_images_per_prompt`.
- No fp16 inference: casting the whole model bypasses the fp32 normalisations
 (see "Dtypes").
- Only Diffusers is ported; vLLM, SGLang and ComfyUI are not.
