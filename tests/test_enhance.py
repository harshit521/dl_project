from __future__ import annotations

from PIL import Image

from src.enhance import auto_enhance, is_low_saturation


def test_auto_enhance_preserves_shape():
    image = Image.new("RGB", (64, 48), (180, 150, 110))
    output = auto_enhance(image)
    assert output.size == (64, 48)
    assert output.mode == "RGB"


def test_low_saturation_detection():
    gray = Image.new("RGB", (32, 32), (128, 128, 128))
    colorful = Image.new("RGB", (32, 32), (220, 40, 40))
    assert is_low_saturation(gray)
    assert not is_low_saturation(colorful)
