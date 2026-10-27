## What this adds

Kandinsky S2V 1.0 — speech-to-video. Given a reference frame and a speech track, the model
animates the person in the frame to match the speech: the whole figure, not only the head.
Audio is a condition, not an output.

It is versioned apart from the other Kandinsky models, so it gets files of its own, and it is
integrated as a modular pipeline only — no `DiffusionPipeline` class, nothing in `pipelines/`:

- `KandinskyS2VTransformer3DModel` — `models/transformers/transformer_kandinsky_s2v.py`
- `KandinskyS2VPiflowScheduler` — `schedulers/scheduling_kandinsky_s2v_piflow.py`, the few-step
  pi-Flow scheduler of the distilled checkpoints
- `KandinskyS2VAutoBlocks` / `KandinskyS2VModularPipeline` and
  `KandinskyS2VDistilledAutoBlocks` / `KandinskyS2VDistilledModularPipeline` —
  `modular_pipelines/kandinsky_s2v/`

A clip is generated chunk by chunk (5 s each) with FramePack memory carried between chunks.
The chunk loop is a `LoopSequentialPipelineBlocks`; the timestep loop is a plain loop inside
one block, as in Helios, since loop blocks cannot nest. Base and distilled are separate block
sets rather than a switch, because the distilled guider (`guidance_scale=1.0`), scheduler and
16-step default need component specs of their own. Each checkpoint names its blocks in
`modular_model_index.json`, so `ModularPipeline.from_pretrained` picks the variant from the
checkpoint alone.

Docs: a pipeline page, a transformer page and a scheduler page.

## Usage

```python
import torch
from diffusers import ModularPipeline
from diffusers.utils import export_to_video

pipe = ModularPipeline.from_pretrained("kandinskylab/Kandinsky-S2V-1.0-Lite-distill-Diffusers")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

video = pipe(image="reference.png", audio="speech.wav", output="videos")[0]
export_to_video(video, "avatar.mp4", fps=25)  # frames only; mux the speech track separately
```

Pro does not fit an 80 GB card fully resident; the pipeline page shows offloading through a
`ComponentsManager` with one model on the GPU at a time (45.9 GiB peak on Pro distilled).

## Checkpoints

Four bundles, one per configuration, each with a `modular_model_index.json`:
`kandinskylab/Kandinsky-S2V-1.0-{Lite,Pro,Lite-distill,Pro-distill}-Diffusers`.

They are private until release: set `HF_TOKEN` to a token with access to `kandinskylab` to run
the usage example above. The tests do not need it — the two tiny repositories they load are
public.

## Tests

- the transformer (`ModelTesterMixin`-based);
- for base and distilled, a tester config with the fast, loading and memory mixins, each
  loading a tiny public repository — `kandinskylab/tiny-kandinsky-s2v-modular-pipe` and
  `kandinskylab/tiny-kandinsky-s2v-distilled-modular-pipe` (happy to move them under
  `hf-internal-testing`);
- that `ModularPipeline.from_pretrained` picks the variant from the checkpoint, that memory
  carries over two chunks, that speech past the encoder limit is rejected, and that guidance
  is rejected on the distilled blocks.

Four mixin tests are skipped for each of base and distilled, each for a property of the model:
batching and `num_images_per_prompt` (one prompt, one video per call), and fp16 inference
(casting the whole model bypasses `_keep_in_fp32_modules`).

Locally on current `main`: 60 passed, 8 skipped; `make quality`,
`check_copies`, `check_dummies`, `check_doc_toc`, `check_support_list`,
`check_forward_call_docstrings`, `check_return_annotations` and `modular_auto_docstring` pass.
