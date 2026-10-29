from __future__ import annotations

import io

from PIL import Image

from src.preprocess import image_to_bytes, limit_image, open_image, to_pil, to_tensor


def test_open_image_flattens_alpha():
    image = Image.new("RGBA", (32, 24), (255, 0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    loaded = open_image(buffer.getvalue())
    assert loaded.mode == "RGB"
    assert loaded.size == (32, 24)
    # Fully transparent pixels become white.
    assert loaded.getpixel((0, 0)) == (255, 255, 255)


def test_open_image_handles_grayscale():
    image = Image.new("L", (10, 12), 128)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    loaded = open_image(buffer.getvalue())
    assert loaded.mode == "RGB"


def test_limit_image_keeps_aspect_ratio():
    image = Image.new("RGB", (4000, 2000), "blue")
    resized = limit_image(image, 1000)
    assert resized.size == (1000, 500)


def test_tensor_round_trip():
    image = Image.new("RGB", (16, 16), (10, 20, 30))
    tensor = to_tensor(image, "cpu")
    assert tensor.shape == (1, 3, 16, 16)
    restored = to_pil(tensor)
    assert restored.size == (16, 16)
    assert restored.getpixel((0, 0)) == (10, 20, 30)


def test_image_to_bytes_strips_metadata():
    image = Image.new("RGB", (8, 8), "red")
    payload = image_to_bytes(image, "PNG")
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
