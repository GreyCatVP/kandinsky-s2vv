"""Text + speech conditioning for S2V.

Two encoders, matching K5 ``embedders_s2v.py``:

* **Qwen2.5-Omni-7B Thinker** — produces both the dense text tokens (last hidden
  layer, sliced to the user text span) and, via a forward hook on the audio
  tower, the raw speech features that condition the DiT.  At 16 kHz the tower
  emits 25 tokens/s, i.e. one token per video frame at fps 25.
* **CLIP ViT-L/14 text tower** — pooled embedding feeding the AdaLN time branch.

Batch size is always 1, so packed ``cu_seqlens`` are not returned.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import Tensor
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers.utils import logging as hf_logging

_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
AUDIO_SAMPLE_RATE = 16000

# transformers renamed the load-time dtype kwarg in 5.0; loading straight into
# bf16 halves both peak host RAM and load time for the 7B Thinker.
_TRANSFORMERS_DTYPE_KWARG_VERSION = 5
_DTYPE_KWARG = (
    "dtype" if int(transformers.__version__.split(".")[0]) >= _TRANSFORMERS_DTYPE_KWARG_VERSION else "torch_dtype"
)


@contextmanager
def _quiet_transformers() -> Iterator[None]:
    """Silence transformers' warnings and progress bars for one load."""
    verbosity, bars = hf_logging.get_verbosity(), hf_logging.is_progress_bar_enabled()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.set_verbosity(verbosity)
        if bars:
            hf_logging.enable_progress_bar()


def load_thinker(qwen_omni_path: str, attn_implementation: str) -> Qwen2_5OmniThinkerForConditionalGeneration:
    """The Thinker alone, out of the full Omni checkpoint.

    Loading it that way makes transformers report every Talker and token2wav
    weight as unexpected, a page of noise on every run. The report is silenced;
    what would matter, a Thinker weight missing from the checkpoint, is checked
    here instead.
    """
    with _quiet_transformers():
        # With `output_loading_info` this is a (model, info) pair, which the stubs do not know.
        loaded: Any = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            qwen_omni_path,
            attn_implementation=attn_implementation,
            output_loading_info=True,
            **{_DTYPE_KWARG: torch.bfloat16},  # type: ignore[arg-type]
        )
    thinker, info = loaded
    if info["missing_keys"]:
        raise RuntimeError(f"Thinker weights missing from {qwen_omni_path}: {sorted(info['missing_keys'])[:5]}")
    return thinker


def load_clip_text_model(clip_path: str | Path) -> CLIPTextModel:
    """Load the CLIP text tower, tolerating a dual-encoder (``CLIPModel``) save."""
    root = Path(clip_path)
    model = CLIPTextModel(CLIPTextConfig.from_pretrained(root))

    safetensors_path = root / "model.safetensors"
    bin_path = root / "pytorch_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file  # noqa: PLC0415

        state = load_file(str(safetensors_path))
    elif bin_path.is_file():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No CLIP weights under {root}")

    prefix = "text_model."
    text_state = {k: v for k, v in state.items() if k.startswith(prefix) and not k.endswith("position_ids")}
    if not text_state:
        raise RuntimeError(f"No text_model.* keys in CLIP checkpoint at {root}")

    # transformers 5.x drops the `text_model.` prefix from CLIPTextModel's own
    # state dict, 4.x keeps it — match whichever this install expects.
    if not any(k.startswith(prefix) for k in model.state_dict()):
        text_state = {k[len(prefix) :]: v for k, v in text_state.items()}
    model.load_state_dict(text_state, strict=True)
    return model


def load_audio(path: str | Path, start: float = 0.0, end: float | None = None):
    """Mono 16 kHz waveform slice, as K5 ``process_mm_info`` loads it."""
    import librosa  # noqa: PLC0415

    return librosa.load(
        str(path),
        sr=AUDIO_SAMPLE_RATE,
        offset=start,
        duration=None if end is None else end - start,
    )[0]


def resolve_thinker_attention() -> str:
    """Pick the best flash-attention build transformers can actually use.

    The project depends on FA3, the CUDA 12.8 conda environment ships FA2, and a
    plain install may have neither — so this must be detected, not hardcoded.
    ``is_flash_attn_3_available`` only exists in newer transformers, hence getattr.
    """
    from transformers.utils import import_utils  # noqa: PLC0415

    for probe, name in (
        ("is_flash_attn_3_available", "flash_attention_3"),
        ("is_flash_attn_2_available", "flash_attention_2"),
    ):
        checker = getattr(import_utils, probe, None)
        if checker is not None and checker():
            return name
    return "sdpa"


class S2VTextAudioEmbedder:
    """Qwen2.5-Omni (text tokens + audio-tower features) and CLIP (pooled)."""

    def __init__(
        self,
        qwen_omni_path: str,
        clip_path: str,
        device: str | torch.device = "cuda",
        clip_max_length: int = 77,
        attn_implementation: str | None = None,
    ):
        """``attn_implementation=None`` picks the best available build."""
        self.device = torch.device(device)
        attn_implementation = attn_implementation or resolve_thinker_attention()
        self.thinker = load_thinker(qwen_omni_path, attn_implementation).to(self.device).eval()
        # The processor parses the whole Omni config, whose Talker carries the text
        # model's token ids and fails transformers' check against its own small vocabulary.
        with _quiet_transformers():
            self.processor = Qwen2_5OmniProcessor.from_pretrained(qwen_omni_path)

        self.clip = load_clip_text_model(clip_path).to(self.device).eval()
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
        self.clip_max_length = clip_max_length

    def to(self, device: str | torch.device) -> S2VTextAudioEmbedder:
        self.device = torch.device(device)
        self.thinker.to(self.device)
        self.clip.to(self.device)
        return self

    def _conversation(self, text: str, audio_path: str | None, start: float, end: float | None):
        user_content: list[dict] = [{"type": "text", "text": text}]
        if audio_path is not None:
            audio_item: dict = {"type": "audio", "audio": audio_path}
            if end is not None:
                audio_item |= {"audio_start": start, "audio_end": end}
            user_content.append(audio_item)
        return [
            {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

    @torch.no_grad()
    def encode(
        self,
        text: str,
        audio_path: str | None = None,
        audio_start: float = 0.0,
        audio_end: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Returns ``(text_embeds, pooled_embed, audio_embeds)``.

        ``text_embeds``  — (S_text, 3584) Thinker hidden states of the user text
        ``pooled_embed`` — (1, 768) CLIP pooled
        ``audio_embeds`` — (S_audio, 3584) audio-tower features, None without audio
        """
        conversation = self._conversation(text, audio_path, audio_start, audio_end)
        chat_text = self.processor.apply_chat_template([conversation], add_generation_prompt=False, tokenize=False)
        audios = None if audio_path is None else [load_audio(audio_path, audio_start, audio_end)]
        inputs = self.processor(
            text=chat_text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self.device).to(self.thinker.dtype)

        captured: dict[str, Tensor] = {}

        def hook(_module, _inputs, output):
            captured["features"] = getattr(output, "last_hidden_state", output)

        handle = self.thinker.audio_tower.register_forward_hook(hook)
        try:
            output = self.thinker(**inputs, output_hidden_states=True, return_dict=True)
        finally:
            handle.remove()

        text_embeds = self._slice_user_text(output.hidden_states[-1], inputs)
        pooled_embed = self._encode_clip(text)
        return text_embeds, pooled_embed, captured.get("features")

    def _slice_user_text(self, hidden_states: Tensor, inputs) -> Tensor:
        """Take the hidden states of the user's text span only (K5 parity)."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        seq_len = input_ids.shape[1]
        device = input_ids.device

        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        audio_start_token_id = self.thinker.config.audio_start_token_id
        user_role_len = len(self.processor.tokenizer("user\n", add_special_tokens=False).input_ids)

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        last_im_start = torch.where(input_ids == im_start_id, positions, -1).max(dim=1).values
        start = (last_im_start + 1 + user_role_len).unsqueeze(1)
        end = torch.where(input_ids == audio_start_token_id, positions, seq_len - 2).min(dim=1).values.unsqueeze(1)
        mask = (positions >= start) & (positions < end) & (attention_mask == 1)
        return hidden_states[mask]

    @torch.no_grad()
    def _encode_clip(self, text: str) -> Tensor:
        inputs = self.clip_tokenizer(
            [text],
            max_length=self.clip_max_length,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        return self.clip(**inputs)["pooler_output"]
