"""Project-wide paths and runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
ASSETS_DIR = ROOT / "assets"
CACHE_DIR = ROOT / ".cache"
HF_HOME = CACHE_DIR / "huggingface"

for _directory in (WEIGHTS_DIR, DATA_DIR, OUTPUTS_DIR, CACHE_DIR, HF_HOME):
    _directory.mkdir(parents=True, exist_ok=True)

# Keep Hugging Face downloads inside the project so disk usage stays predictable.
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
RESTORER_PATH = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"
GFPGAN_PATH = WEIGHTS_DIR / "GFPGANv1.4.pth"
FACE_DETECTOR_PATH = WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

DEFAULT_MAX_SIDE = 1024
DEFAULT_MAX_OUTPUT_SIDE = 4096
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_OVERLAP = 32


def get_device(preferred: str | None = None) -> str:
    """Return the best available PyTorch device."""
    if preferred:
        return preferred
    env_device = os.getenv("PHOTO_DEVICE")
    if env_device:
        return env_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()
