"""Overlapping tiled inference for memory-safe super-resolution."""

from __future__ import annotations

import torch


def _feather_mask(
    height: int, width: int, overlap: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create a 2D weighting mask that fades at tile edges."""
    overlap = max(0, min(overlap, height // 2 - 1, width // 2 - 1))
    wy = torch.ones(height, device=device, dtype=dtype)
    wx = torch.ones(width, device=device, dtype=dtype)

    if overlap > 0:
        ramp = torch.linspace(
            1.0 / (overlap + 1), 1.0, overlap, device=device, dtype=dtype
        )
        wy[:overlap] *= ramp
        wy[-overlap:] *= ramp.flip(0)
        wx[:overlap] *= ramp
        wx[-overlap:] *= ramp.flip(0)

    return wy[:, None] * wx[None, :]


def _tile_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.inference_mode()
def tiled_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    scale: int,
    tile_size: int = 256,
    overlap: int = 32,
) -> torch.Tensor:
    """Run a super-resolution model on overlapping tiles and blend the result.

    ``tensor`` must have shape (1, C, H, W). The return value has shape
    (1, C, H * scale, W * scale).
    """
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("tiled_inference expects a tensor shaped (1, C, H, W)")

    _, channels, height, width = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    if height <= tile_size and width <= tile_size:
        return model(tensor)

    tile_size = max(16, min(tile_size, height, width))
    overlap = max(0, min(overlap, tile_size // 2 - 1))
    stride = max(1, tile_size - overlap)

    ys = _tile_positions(height, tile_size, stride)
    xs = _tile_positions(width, tile_size, stride)

    out_height, out_width = height * scale, width * scale
    canvas = torch.zeros(
        (1, channels, out_height, out_width), device=device, dtype=dtype
    )
    weights = torch.zeros(
        (1, 1, out_height, out_width), device=device, dtype=dtype
    )

    for y in ys:
        for x in xs:
            tile = tensor[:, :, y : y + tile_size, x : x + tile_size]
            output = model(tile)
            actual_h = min(tile_size, height - y) * scale
            actual_w = min(tile_size, width - x) * scale
            output = output[:, :, :actual_h, :actual_w]

            mask = _feather_mask(
                actual_h, actual_w, overlap * scale, device, dtype
            )
            mask = mask.unsqueeze(0).unsqueeze(0)

            canvas[
                :, :, y * scale : y * scale + actual_h, x * scale : x * scale + actual_w
            ] += output * mask
            weights[
                :, :, y * scale : y * scale + actual_h, x * scale : x * scale + actual_w
            ] += mask

    return canvas / weights.clamp_min(1e-6)


def suggest_tile_size(height: int, width: int, max_tile: int = 512) -> int:
    """Pick a reasonable tile size for an image."""
    if max(height, width) <= max_tile:
        return max(height, width)
    return max_tile
