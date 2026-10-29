"""End-to-end restoration + captioning pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from src.caption import Captioner
from src.colorize import Colorizer
from src.config import (
    DEFAULT_MAX_OUTPUT_SIDE,
    DEFAULT_MAX_SIDE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
)
from src.enhance import auto_enhance as auto_enhance_image
from src.faces import FaceEnhancementResult, enhance_faces
from src.models import ModelManager, get_model_manager
from src.preprocess import limit_image, open_image
from src.restore import Restorer
from src.scratch import reduce_scratches


@dataclass
class PipelineOptions:
    scale: int = 4
    face_enhance: bool = True
    caption: bool = True
    auto_enhance: bool = True
    scratch_removal: bool = False
    colorize: bool = False
    colorize_strength: float = 1.0
    max_side: int = DEFAULT_MAX_SIDE
    max_output_side: int = DEFAULT_MAX_OUTPUT_SIDE
    tile_size: int = DEFAULT_TILE_SIZE
    overlap: int = DEFAULT_TILE_OVERLAP
    prompt: str | None = "a restored old photograph of"


@dataclass
class PipelineResult:
    before: Image.Image
    after: Image.Image
    caption: str
    meta: dict = field(default_factory=dict)


class Pipeline:
    """Compose restoration, optional face enhancement, and captioning."""

    def __init__(
        self,
        manager: ModelManager | None = None,
        restorer: Restorer | None = None,
        captioner: Captioner | None = None,
        colorizer: Colorizer | None = None,
        face_enhancer=enhance_faces,
    ) -> None:
        self.manager = manager or get_model_manager()
        self._restorer = restorer
        self._captioner = captioner
        self._colorizer = colorizer
        self._face_enhancer = face_enhancer

    @property
    def colorizer(self) -> Colorizer:
        if self._colorizer is None:
            self._colorizer = Colorizer(manager=self.manager)
        return self._colorizer

    @property
    def restorer(self) -> Restorer:
        if self._restorer is None:
            self._restorer = Restorer(manager=self.manager)
        return self._restorer

    @property
    def captioner(self) -> Captioner:
        if self._captioner is None:
            self._captioner = Captioner(manager=self.manager)
        return self._captioner

    def process(
        self,
        source,
        options: PipelineOptions | None = None,
    ) -> PipelineResult:
        options = options or PipelineOptions()
        if options.scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")

        t_start = time.perf_counter()
        image = open_image(source)
        original_size = image.size

        input_limit = options.max_side
        if options.max_output_side:
            input_limit = min(input_limit, options.max_output_side // options.scale)
        image = limit_image(image, max(16, input_limit))
        before = image.copy()

        timings: dict[str, float] = {}
        warnings: list[str] = []

        if options.scratch_removal:
            t0 = time.perf_counter()
            try:
                image = reduce_scratches(image)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"Scratch reduction skipped: {exc}")
            timings["scratch_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        after = self.restorer.upscale(image, scale=options.scale)
        timings["restore_s"] = round(time.perf_counter() - t0, 3)

        face_count = 0
        face_backend = "none"
        if options.face_enhance:
            t0 = time.perf_counter()
            try:
                face_result: FaceEnhancementResult = self._face_enhancer(
                    after, restorer=self.restorer, manager=self.manager
                )
                after = face_result.image
                face_count = face_result.faces
                face_backend = face_result.backend
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"Face enhancement skipped: {exc}")
            timings["faces_s"] = round(time.perf_counter() - t0, 3)

        if options.auto_enhance:
            t0 = time.perf_counter()
            try:
                after = auto_enhance_image(after)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"Auto color correction skipped: {exc}")
            timings["enhance_s"] = round(time.perf_counter() - t0, 3)

        if options.colorize:
            t0 = time.perf_counter()
            try:
                after = self.colorizer.colorize(
                    after, strength=options.colorize_strength
                )
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(
                    "Colorization skipped: "
                    f"{exc}. Run scripts/download_weights.py --with-colorizer."
                )
            timings["colorize_s"] = round(time.perf_counter() - t0, 3)

        caption = ""
        if options.caption:
            t0 = time.perf_counter()
            try:
                caption = self.captioner.caption(after, prompt=options.prompt)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"Captioning skipped: {exc}")
            timings["caption_s"] = round(time.perf_counter() - t0, 3)

        timings["total_s"] = round(time.perf_counter() - t_start, 3)

        if after.size != (before.width * options.scale, before.height * options.scale):
            warnings.append(
                "Output size differs from the expected upscaled size; "
                "the result was resized for consistency."
            )
            after = after.resize(
                (before.width * options.scale, before.height * options.scale),
                Image.Resampling.LANCZOS,
            )

        meta = {
            "device": self.manager.device,
            "scale": options.scale,
            "input_size": list(before.size),
            "source_size": list(original_size),
            "output_size": list(after.size),
            "face_count": face_count,
            "face_backend": face_backend,
            "auto_enhance": options.auto_enhance,
            "scratch_removal": options.scratch_removal,
            "colorized": options.colorize,
            "caption_model": self.manager.describe().get("captioner"),
            "restorer": self.manager.describe().get("restorer"),
            "timings": timings,
            "warnings": warnings,
        }

        return PipelineResult(before=before, after=after, caption=caption, meta=meta)

    def save(
        self,
        result: PipelineResult,
        output_path: str | Path,
        image_format: str | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = image_format or (
            "JPEG" if output_path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
        )
        result.after.save(output_path, format=fmt, quality=95, subsampling=0)
        return output_path


def result_to_dict(result: PipelineResult) -> dict:
    """Serializable subset of a result (images excluded)."""
    return {"caption": result.caption, "meta": result.meta}
