"""Image loading, normalization, and tensor conversion helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image, ImageOps

ImageSource = str | Path | bytes | bytearray | BinaryIO


def open_image(source: ImageSource) -> Image.Image:
    """Open an image and normalize it to an EXIF-correct RGB PIL image.

    Transparent images are flattened onto a white background. Metadata is kept
    in memory but is not written back when we save outputs.
    """
    if isinstance(source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(source)

    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    return image


def limit_image(image: Image.Image, max_side: int | None) -> Image.Image:
    """Downscale an image so its longest side is at most ``max_side``."""
    if not max_side or max(image.size) <= max_side:
        return image
    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def to_tensor(image: Image.Image, device: str | torch.device) -> torch.Tensor:
    """Convert a PIL RGB image to a float32 tensor in [0, 1], shape (1, 3, H, W)."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a float tensor in [0, 1] to a PIL RGB image."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array)


def image_to_bytes(
    image: Image.Image, image_format: str = "PNG", quality: int = 95
) -> bytes:
    """Encode an image to bytes without EXIF metadata."""
    buffer = io.BytesIO()
    save_kwargs: dict = {}
    if image_format.upper() in {"JPG", "JPEG"}:
        save_kwargs = {"quality": quality, "subsampling": 0}
    image.save(buffer, format=image_format, **save_kwargs)
    return buffer.getvalue()


def estimate_sharpness(image: Image.Image) -> float:
    """Laplacian variance; higher means sharper. Useful for UI diagnostics."""
    import cv2

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
