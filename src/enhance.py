"""Classical color and contrast correction for faded photographs.

These operations are deliberately conservative: they correct a detected color
cast, stretch clipped tones, and add a little local contrast without changing
image content. They are not a replacement for learned colorization.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _channel_cast(rgb: np.ndarray) -> float:
    means = rgb.reshape(-1, 3).mean(axis=0)
    return float((means.max() - means.min()) / (means.mean() + 1e-6))


def auto_enhance(image: Image.Image, strength: float = 0.65) -> Image.Image:
    """Neutralize a color cast and improve contrast/saturation mildly."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    original = rgb.copy()

    # 1. Gray-world white balance, only when a cast is actually present.
    cast = _channel_cast(rgb)
    if cast > 0.035:
        means = rgb.reshape(-1, 3).mean(axis=0)
        target = means.mean()
        gains = np.clip(target / (means + 1e-6), 0.80, 1.30)
        gains = 1.0 + (gains - 1.0) * strength
        rgb = np.clip(rgb * gains, 0, 1)

    # 2. Percentile contrast stretch.
    low, high = np.percentile(rgb, [1.0, 99.0])
    if high - low > 0.05:
        stretched = np.clip((rgb - low) / (high - low + 1e-6), 0, 1)
        rgb = rgb * (1 - 0.45 * strength) + stretched * (0.45 * strength)

    # 3. CLAHE on luminance and a small saturation lift.
    uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel.astype(np.uint8)).astype(np.float32)

    # Saturation: push a/b away from the neutral point of 128.
    a_channel = (a_channel - 128.0) * (1.0 + 0.20 * strength) + 128.0
    b_channel = (b_channel - 128.0) * (1.0 + 0.20 * strength) + 128.0

    lab = cv2.merge(
        [
            np.clip(l_channel, 0, 255),
            np.clip(a_channel, 0, 255),
            np.clip(b_channel, 0, 255),
        ]
    ).astype(np.uint8)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Blend with the original so the result does not look over-processed.
    result = original * (1 - strength) + (enhanced.astype(np.float32) / 255.0) * strength
    return Image.fromarray((np.clip(result, 0, 1) * 255).round().astype(np.uint8))


def is_low_saturation(image: Image.Image, threshold: float = 0.16) -> bool:
    hsv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean() / 255.0) < threshold
