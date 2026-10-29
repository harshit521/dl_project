"""Image quality metrics for restoration evaluation."""

from __future__ import annotations

import numpy as np
from PIL import Image


def to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def psnr(reference: Image.Image, candidate: Image.Image) -> float:
    from skimage.metrics import peak_signal_noise_ratio

    return float(
        peak_signal_noise_ratio(
            to_numpy(reference), to_numpy(candidate), data_range=1.0
        )
    )


def ssim(reference: Image.Image, candidate: Image.Image) -> float:
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            to_numpy(reference),
            to_numpy(candidate),
            data_range=1.0,
            channel_axis=2,
        )
    )


class LPIPSMetric:
    """Lazy LPIPS wrapper (uses AlexNet weights on CPU)."""

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            import lpips

            self._model = lpips.LPIPS(net="alex", verbose=False).eval()
        return self._model

    def __call__(self, reference: Image.Image, candidate: Image.Image) -> float:
        import torch

        model = self._load()
        a = torch.from_numpy(to_numpy(reference)).permute(2, 0, 1)
        b = torch.from_numpy(to_numpy(candidate)).permute(2, 0, 1)
        # LPIPS expects values in [-1, 1].
        a = (a * 2 - 1).unsqueeze(0)
        b = (b * 2 - 1).unsqueeze(0)
        with torch.inference_mode():
            return float(model(a, b).item())
