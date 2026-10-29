from __future__ import annotations

import numpy as np
from PIL import Image

from src.scratch import reduce_scratches


def test_scratch_reduction_preserves_shape():
    array = np.full((64, 64, 3), 120, dtype=np.uint8)
    array[10:30, 32] = 250  # a bright vertical scratch
    image = Image.fromarray(array)
    output = reduce_scratches(image)
    assert output.size == image.size
    assert output.mode == "RGB"


def test_scratch_reduction_noop_on_flat_image():
    image = Image.new("RGB", (32, 32), (90, 90, 90))
    output = reduce_scratches(image)
    assert output.size == image.size
