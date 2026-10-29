"""BLIP-based image captioning."""

from __future__ import annotations

import re

import torch
from PIL import Image

from src.models import ModelManager, get_model_manager


class Captioner:
    """Generate a short natural-language description of an image."""

    def __init__(
        self,
        manager: ModelManager | None = None,
        prompt: str | None = "a restored old photograph of",
    ) -> None:
        self.manager = manager or get_model_manager()
        self.prompt = prompt

    def caption(self, image: Image.Image, prompt: str | None = None) -> str:
        bundle = self.manager.captioner
        use_prompt = self.prompt if prompt is None else prompt

        # BLIP's processor resizes internally; capping here just avoids
        # feeding unnecessarily large arrays to the vision encoder.
        if max(image.size) > 768:
            image = image.copy()
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)

        kwargs = {"images": image, "return_tensors": "pt"}
        if use_prompt:
            kwargs["text"] = use_prompt

        inputs = bundle.processor(**kwargs).to(self.manager.device)
        with torch.inference_mode():
            output = bundle.model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=3,
                no_repeat_ngram_size=2,
            )
        text = bundle.processor.decode(output[0], skip_special_tokens=True)
        return clean_caption(text, prompt=use_prompt)


def clean_caption(text: str, prompt: str | None = None) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if prompt:
        prefix = prompt.strip()
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].lstrip(" ,:;-")
    if not text:
        return "A restored photograph."
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text
