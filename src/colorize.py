"""Optional colorization for black-and-white / faded photographs.

Uses the SIGGRAPH 2017 colorization model (richzhang/colorization). The model
predicts Lab ``ab`` channels from the ``L`` channel at 256x256 and the result is
upsampled back to the original resolution.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.config import WEIGHTS_DIR
from src.models import ModelManager, get_model_manager

COLORIZER_PATH = WEIGHTS_DIR / "siggraph17-df00044c.pth"


class Colorizer:
    def __init__(
        self,
        manager: ModelManager | None = None,
        weights_path: str | Path | None = None,
    ) -> None:
        self.manager = manager or get_model_manager()
        self.weights_path = Path(weights_path or COLORIZER_PATH)
        self._model: torch.nn.Module | None = None

    def _load(self) -> torch.nn.Module:
        if self._model is None:
            if not self.weights_path.exists():
                raise FileNotFoundError(
                    f"Colorizer weights not found at {self.weights_path}. "
                    "Run: python scripts/download_weights.py --with-colorizer"
                )
            from src.colorizers.siggraph17 import SIGGRAPHGenerator

            model = SIGGRAPHGenerator()
            state = torch.load(
                str(self.weights_path), map_location="cpu", weights_only=True
            )
            model.load_state_dict(state)
            self._model = model.eval().to(self.manager.device)
        return self._model

    @torch.inference_mode()
    def colorize(
        self, image: Image.Image, strength: float = 1.0, size: int = 256
    ) -> Image.Image:
        """Colorize an image while preserving its original luminance."""
        model = self._load()
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_original = lab[:, :, 0] * (100.0 / 255.0)

        small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        lab_small = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_small = lab_small[:, :, 0] * (100.0 / 255.0)

        tensor = (
            torch.from_numpy(l_small)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.manager.device)
        )
        ab = model(tensor)
        ab = torch.nn.functional.interpolate(
            ab, size=(height, width), mode="bilinear", align_corners=False
        )
        ab = ab[0].float().cpu().numpy().transpose(1, 2, 0)

        lab_out = np.stack(
            [
                l_original * (255.0 / 100.0),
                np.clip(ab[:, :, 0] + 128.0, 0, 255),
                np.clip(ab[:, :, 1] + 128.0, 0, 255),
            ],
            axis=2,
        ).astype(np.uint8)
        colorized = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

        if strength < 1.0:
            # Blend in Lab space so luminance stays consistent.
            lab_color = cv2.cvtColor(colorized, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab_gray = np.stack(
                [
                    lab[:, :, 0],
                    np.full_like(lab[:, :, 0], 128.0),
                    np.full_like(lab[:, :, 0], 128.0),
                ],
                axis=2,
            )
            blended = lab_gray * (1 - strength) + lab_color * strength
            colorized = cv2.cvtColor(
                np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB
            )

        return Image.fromarray(colorized)


def is_effectively_grayscale(image: Image.Image, saturation_threshold: float = 18.0) -> bool:
    """Heuristic used by the UI to suggest colorization."""
    hsv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean()) < saturation_threshold
