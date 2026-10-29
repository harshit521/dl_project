from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from src.caption import clean_caption
from src.faces import FaceEnhancementResult
from src.pipeline import Pipeline, PipelineOptions


@dataclass
class FakeManager:
    device: str = "cpu"

    def describe(self) -> dict:
        return {"captioner": "fake-blip", "restorer": "fake-esrgan"}


class FakeRestorer:
    def upscale(self, image: Image.Image, scale: int = 4) -> Image.Image:
        return image.resize(
            (image.width * scale, image.height * scale),
            Image.Resampling.NEAREST,
        )


class FakeCaptioner:
    def caption(self, image: Image.Image, prompt: str | None = None) -> str:
        return "A test photograph."


def fake_face_enhancer(image, restorer=None, manager=None):
    return FaceEnhancementResult(image=image, faces=0, backend="none")


def make_pipeline() -> Pipeline:
    return Pipeline(
        manager=FakeManager(),
        restorer=FakeRestorer(),
        captioner=FakeCaptioner(),
        face_enhancer=fake_face_enhancer,
    )


def test_pipeline_end_to_end_with_fakes():
    image = Image.new("RGB", (40, 30), (120, 100, 80))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    result = make_pipeline().process(
        buffer.getvalue(),
        PipelineOptions(scale=2, caption=True, auto_enhance=False),
    )

    assert result.after.size == (80, 60)
    assert result.caption == "A test photograph."
    assert result.meta["input_size"] == [40, 30]
    assert result.meta["output_size"] == [80, 60]
    assert "timings" in result.meta


def test_pipeline_respects_max_output_side():
    image = Image.new("RGB", (800, 600), "gray")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    result = make_pipeline().process(
        buffer.getvalue(),
        PipelineOptions(
            scale=4,
            caption=False,
            auto_enhance=False,
            max_side=1024,
            max_output_side=1024,
        ),
    )
    # Input is capped at 256 px, then upscaled 4x.
    assert result.after.size == (1024, 768)


def test_clean_caption_removes_prompt_and_adds_period():
    text = "a restored old photograph of a woman in a hat"
    assert clean_caption(text, prompt="a restored old photograph of") == (
        "A woman in a hat."
    )


def test_clean_caption_empty_fallback():
    assert clean_caption("   ", prompt=None) == "A restored photograph."
