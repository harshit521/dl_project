"""Lazy model loading and device management."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import (
    CAPTION_MODEL,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    DEVICE,
    GFPGAN_PATH,
    RESTORER_PATH,
    get_device,
)


@dataclass
class RestorerBundle:
    model: torch.nn.Module
    scale: int
    path: str


@dataclass
class CaptionerBundle:
    processor: object
    model: torch.nn.Module
    name: str


class ModelManager:
    """Loads each model once and keeps it resident.

    Models are loaded lazily so importing the package (and running unit tests)
    does not download or initialize multi-hundred-megabyte weights.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or get_device()
        self._restorer: RestorerBundle | None = None
        self._captioner: CaptionerBundle | None = None
        self._gfpgan = None
        self._lock = threading.RLock()

    def _to_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move a model to the selected device, falling back to CPU if needed."""
        try:
            return model.to(self.device)
        except Exception:
            if self.device != "cpu":
                self.device = "cpu"
                return model.to("cpu")
            raise

    # ------------------------------------------------------------------
    # Super-resolution / restoration
    # ------------------------------------------------------------------
    def load_restorer(self, path: str | Path | None = None) -> RestorerBundle:
        with self._lock:
            if self._restorer is None:
                from spandrel import ModelLoader

                model_path = Path(path or RESTORER_PATH)
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Restorer weights not found at {model_path}. "
                        "Run: python scripts/download_weights.py"
                    )

                descriptor = ModelLoader().load_from_file(str(model_path))
                model = self._to_device(descriptor.model.eval())
                self._restorer = RestorerBundle(
                    model=model,
                    scale=int(descriptor.scale),
                    path=str(model_path),
                )
        return self._restorer

    @property
    def restorer(self) -> RestorerBundle:
        return self.load_restorer()

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------
    def load_captioner(self, name: str | None = None) -> CaptionerBundle:
        with self._lock:
            if self._captioner is None:
                from transformers import BlipForConditionalGeneration, BlipProcessor

                model_name = name or CAPTION_MODEL
                processor = BlipProcessor.from_pretrained(model_name)
                model = self._to_device(
                    BlipForConditionalGeneration.from_pretrained(
                        model_name, torch_dtype=torch.float32
                    ).eval()
                )
                self._captioner = CaptionerBundle(
                    processor=processor, model=model, name=model_name
                )
        return self._captioner

    @property
    def captioner(self) -> CaptionerBundle:
        return self.load_captioner()

    # ------------------------------------------------------------------
    # Optional GFPGAN face restoration
    # ------------------------------------------------------------------
    def load_gfpgan(self, path: str | Path | None = None):
        """Return a GFPGANer, or None when the optional package is unavailable."""
        with self._lock:
            if self._gfpgan is not None:
                return self._gfpgan

            model_path = Path(path or GFPGAN_PATH)
            if not model_path.exists():
                return None

            try:
                from gfpgan import GFPGANer
            except Exception:
                return None

            try:
                self._gfpgan = GFPGANer(
                    model_path=str(model_path),
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    device=self.device,
                )
            except Exception:
                self._gfpgan = None
            return self._gfpgan

    # ------------------------------------------------------------------
    # Introspection / cleanup
    # ------------------------------------------------------------------
    def describe(self) -> dict:
        restorer = self._restorer
        captioner = self._captioner
        return {
            "device": self.device,
            "restorer": Path(restorer.path).name if restorer else None,
            "restorer_scale": restorer.scale if restorer else None,
            "captioner": captioner.name if captioner else None,
            "gfpgan": self._gfpgan is not None,
            "tile_size": DEFAULT_TILE_SIZE,
            "tile_overlap": DEFAULT_TILE_OVERLAP,
        }

    def unload_captioner(self) -> None:
        with self._lock:
            self._captioner = None
            if self.device == "mps":
                torch.mps.empty_cache()

    def unload_restorer(self) -> None:
        with self._lock:
            self._restorer = None
            if self.device == "mps":
                torch.mps.empty_cache()


_shared_manager: ModelManager | None = None


def get_model_manager(device: str | None = None) -> ModelManager:
    """Return a process-wide manager (useful for Streamlit and the CLI)."""
    global _shared_manager
    if _shared_manager is None or (device and device != _shared_manager.device):
        _shared_manager = ModelManager(device=device or DEVICE)
    return _shared_manager
