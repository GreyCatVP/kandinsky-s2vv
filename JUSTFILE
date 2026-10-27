lock:
    @uv lock

setup:
    @uv sync --group dev

lint:
    @uv run ruff check .
    @uv run ruff format --check .

fmt:
    @uv run ruff format .
    @uv run ruff check --fix .

typecheck:
    @uv run mypy src

test *ARGS:
    @uv run pytest {{ARGS}}

# Generate an avatar video from a speech track and a reference frame.
# just generate --audio speech.wav --image reference.png --out out.mp4 \
#     --config src/kandinsky_s2v/configs/s2v_lite.yaml
generate *ARGS:
    @uv run kandy-s2v generate {{ARGS}}

# The same generation split across N GPUs of one node (sequence parallel, see
# core/parallel.py). N must divide the head count: 2 or 4 on Lite, 2/4/8 on Pro.
# just generate-parallel 4 --audio speech.wav --image reference.png --out out.mp4 \
#     --config src/kandinsky_s2v/configs/s2v_pro_hd.yaml
# OMP_NUM_THREADS=1 is torchrun's own default, set here only so it stops saying
# so; TORCH_CPP_LOG_LEVEL=ERROR hides c10d trying IPv6 first (errno 97) on hosts
# that list ::1 for localhost but run without IPv6, as these pods do.
generate-parallel N *ARGS:
    @OMP_NUM_THREADS=1 TORCH_CPP_LOG_LEVEL=ERROR uv run torchrun --standalone --nproc-per-node {{N}} -m kandinsky_s2v.cli generate {{ARGS}}

# Gather an FSDP-sharded S2V training checkpoint into one bf16 safetensors file.
# just convert --checkpoint /path/to/step_14000/ema_model --out /path/to/model.safetensors
convert *ARGS:
    @uv run kandy-s2v convert {{ARGS}}

# CPU parity check of the S2V DiT against the kandinsky-5 reference (no GPU, no weights).
# Needs a kandinsky-5 checkout: dev/ref/kandinsky-5, or KANDINSKY5_REPO=/path/to/kandinsky-5.
parity:
    @TORCHDYNAMO_DISABLE=1 uv run python dev/bench/s2v_parity_cpu.py

# GPU parity of the whole generation loop against the reference pipeline.
# Needs weights and a checkout; distilled configs need KANDINSKY5_PIFLOW_REPO.
# just parity-pipeline --chunks 2 --steps 8
parity-pipeline *ARGS:
    @TORCHDYNAMO_DISABLE=1 uv run python dev/bench/s2v_parity_pipeline.py {{ARGS}}

# Remove generated Python, test and lint artifacts. Keeps the venv and outputs.
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    find . -type d \( -name '__pycache__' -o -name '.ruff_cache' -o -name '.pytest_cache' -o -name '.mypy_cache' \) -prune -exec rm -rf {} +
    find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
