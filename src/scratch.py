"""Conservative dust and scratch reduction.

Long scratches and dust specks are detected as thin, high-frequency bright or
dark structures, then removed with OpenCV inpainting. This is a classical
fallback: it works best on scans with thin damage and is deliberately optional
because aggressive masks can erase real detail.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _line_mask(responses: np.ndarray, threshold: int) -> np.ndarray:
    _, binary = cv2.threshold(responses, threshold, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=22,
        minLineLength=25,
        maxLineGap=2,
    )
    mask = np.zeros_like(responses)
    if lines is None:
        return mask

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = float(np.hypot(x2 - x1, y2 - y1))
        samples = max(2, int(length))
        xs = np.linspace(x1, x2, samples).astype(int)
        ys = np.linspace(y1, y2, samples).astype(int)
        if responses[ys, xs].mean() < threshold:
            continue
        cv2.line(mask, (x1, y1), (x2, y2), 255, 5)
    return mask


def _dust_mask(responses: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(responses)
    mask = np.zeros_like(responses)
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        if 4 <= area <= 80 and max(width, height) <= 14:
            mask[labels == index] = 255
    return mask


def reduce_scratches(
    image: Image.Image, threshold: int = 20, inpaint_radius: int = 4
) -> Image.Image:
    """Detect thin bright/dark damage and inpaint it."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    responses = np.maximum(top_hat, black_hat)

    mask = _line_mask(responses, threshold) | _dust_mask(responses)
    if not mask.any():
        return image

    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    repaired = cv2.inpaint(rgb, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return Image.fromarray(repaired)
