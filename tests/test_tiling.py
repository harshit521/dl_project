from __future__ import annotations

import torch
import torch.nn.functional as F

from src.tiling import tiled_inference


class FakeUpscaler(torch.nn.Module):
    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="nearest")


def test_tiled_matches_direct_for_identity_upscaler():
    model = FakeUpscaler(4)
    tensor = torch.rand(1, 3, 100, 120)
    direct = model(tensor)
    tiled = tiled_inference(model, tensor, scale=4, tile_size=48, overlap=8)
    assert tiled.shape == direct.shape
    assert torch.allclose(tiled, direct, atol=1e-4)


def test_small_images_use_direct_path():
    model = FakeUpscaler(2)
    tensor = torch.rand(1, 3, 20, 20)
    output = tiled_inference(model, tensor, scale=2, tile_size=64, overlap=8)
    assert output.shape == (1, 3, 40, 40)


def test_invalid_shape_raises():
    model = FakeUpscaler(2)
    try:
        tiled_inference(model, torch.rand(3, 10, 10), scale=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an unbatched tensor")
