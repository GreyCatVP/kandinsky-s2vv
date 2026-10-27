---
title: Kandinsky S2V 1.0
emoji: 🗣️
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12"
app_file: app.py
pinned: true
suggested_hardware: zero-a10g
license: mit
models:
- kandinskylab/Kandinsky-S2V-1.0-Lite-distill
- Qwen/Qwen2.5-Omni-7B
- openai/clip-vit-large-patch14
- hunyuanvideo-community/HunyuanVideo
short_description: Speech-driven avatar video from a reference frame
---

# Kandinsky S2V 1.0

Gradio demo of the Kandinsky S2V 1.0 pipeline: an avatar video from a reference
frame and a speech track, generated chunk by chunk (5 s each).

## Setup

- **Hardware**: ZeroGPU. The pipeline is built on the emulated CUDA device at
  startup; each request runs in `@spaces.GPU` on an `xlarge` slot (96 GB). The
  reference-settings peak is 57 GiB on Lite, owned by the VAE decode, so the
  48 GB `large` slot does not fit. A dedicated GPU Space works too:
  `@spaces.GPU` is a no-op there.
- **Secrets**: `HF_TOKEN` with read access to the model repositories while they
  are private.
- **Code**: the `kandinsky_s2v` package is not installed, it is uploaded next to
  `app.py`, so the Space runs exactly the code it was given. From the root of
  this repository:

  ```bash
  hf upload $SPACE dev/space . --repo-type space --exclude "**/__pycache__/**"
  hf upload $SPACE src/kandinsky_s2v kandinsky_s2v --repo-type space --exclude "**/__pycache__/**"
  ```

  Re-upload the package after changing it.
- **Variables** (optional): `S2V_CONFIG` picks one of the package configs
  (`s2v_lite_distill.yaml` by default); `S2V_MAX_SECONDS` caps the speech
  track (10 s); `S2V_GPU_DURATION` is the GPU time requested per call (180 s);
  `S2V_ZEROGPU_SIZE` selects the slot (`xlarge`).
- **Examples**: `examples/<name>.png` with `examples/<name>.wav` beside it.

Attention runs through SDPA: FA3 is not available on Spaces, so generation is
slower than the H100 numbers in the main repository.
