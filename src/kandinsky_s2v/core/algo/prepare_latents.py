"""Reference-frame encoding, noise and FramePack memory buffers."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor


def select_best_resolution(image_path: str | Path, allowed_resolutions: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """Pick the ``(height, width)`` bucket closest in aspect ratio to the image."""
    with Image.open(image_path) as image:
        width, height = image.size
    aspect = width / height
    return min(allowed_resolutions, key=lambda res: abs(aspect - res[1] / res[0]))


def crop_and_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Resize to cover, then center-crop."""
    from torchvision import transforms  # noqa: PLC0415

    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = transforms.functional.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    return transforms.functional.center_crop(image, (target_height, target_width))


def encode_reference_image(
    image_path: str | Path,
    resolution: tuple[int, int],
    vae,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Reference frame → scaled VAE latent ``(1, H/8, W/8, C)``."""
    from torchvision import transforms  # noqa: PLC0415

    with Image.open(image_path) as raw:
        image = crop_and_resize(raw.convert("RGB"), resolution[0], resolution[1])

    tensor = transforms.ToTensor()(image).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float16) * 2.0 - 1.0
    tensor = tensor.unsqueeze(2)  # (1, 3, 1, H, W)
    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.sample(generator).permute(0, 2, 3, 4, 1)
    return (latent * vae.config.scaling_factor).squeeze(1)


def _align(size: int, multiple: int) -> int:
    return (size // multiple) * multiple


def crop_reference_latent(ref: Tensor, patch_size: tuple[int, int, int]) -> Tensor:
    """Crop spatial dims to a whole number of patches."""
    return ref[:, : _align(ref.shape[1], patch_size[1]), : _align(ref.shape[2], patch_size[2]), :]


def init_noise(  # noqa: PLR0913
    duration: int,
    height: int,
    width: int,
    dim: int,
    patch_size: tuple[int, int, int],
    generator: torch.Generator,
    device: str | torch.device,
) -> Tensor:
    """Fresh Gaussian latent for one chunk, cropped to whole patches."""
    return torch.randn(
        duration,
        _align(height, patch_size[1]),
        _align(width, patch_size[2]),
        dim,
        device=device,
        generator=generator,
    )


def init_memory(  # noqa: PLR0913
    frame_buckets: tuple[int, int, int],
    coefs: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    patch_size: tuple[int, int, int],
    height: int,
    width: int,
    dim: int,
    device: str | torch.device,
) -> list[Tensor]:
    """Zero memory buffers for the first chunk, in ``[1x, 2x, 4x]`` order."""
    buffers = []
    for bucket, coef in zip(frame_buckets, coefs, strict=True):
        h_mult = patch_size[1] * coef[1]
        w_mult = patch_size[2] * coef[2]
        buffers.append(torch.zeros(bucket, _align(height, h_mult), _align(width, w_mult), dim, device=device))
    return buffers


def roll_memory(
    latent: Tensor,
    frame_buckets: tuple[int, int, int],
    coefs: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    patch_size: tuple[int, int, int],
) -> list[Tensor]:
    """FramePack carry-over: the tail of this chunk becomes the next chunk's memory.

    Returns ``[proj1x, proj2x, proj4x]`` — the most recent ``frame_buckets[0]``
    frames at full patch resolution, the preceding ``frame_buckets[1]`` at 2x and
    the ``frame_buckets[2]`` before those at 4x.

    Requires ``latent.shape[0] >= sum(frame_buckets)``: a shorter chunk cannot
    fill the window, and the undersized buffers would not match the position
    grids built from ``frame_buckets`` — the failure would surface later, in
    RoPE. The pipeline enforces this through the config, but this function is
    pure algo and may be called directly by a port, so it checks too.
    """
    height, width = latent.shape[1], latent.shape[2]
    needed = sum(frame_buckets)
    if latent.shape[0] < needed:
        raise ValueError(
            f"latent has {latent.shape[0]} frames but the FramePack memory window "
            f"sum(frame_buckets)={needed} requires at least that many"
        )
    offsets = [0]
    for bucket in frame_buckets:
        offsets.append(offsets[-1] + bucket)

    buffers = []
    for i, coef in enumerate(coefs):
        start, stop = offsets[i + 1], offsets[i]
        window = latent[latent.shape[0] - start : latent.shape[0] - stop]
        h_mult = patch_size[1] * coef[1]
        w_mult = patch_size[2] * coef[2]
        buffers.append(window[:, : _align(height, h_mult), : _align(width, w_mult), :].clone())
    return buffers
