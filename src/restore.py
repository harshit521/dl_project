"""Super-resolution restoration using Real-ESRGAN."""

from __future__ import annotations

from PIL import Image

from src.config import DEFAULT_TILE_OVERLAP, DEFAULT_TILE_SIZE
from src.models import ModelManager, get_model_manager
from src.preprocess import limit_image, to_pil, to_tensor
from src.tiling import tiled_inference


class Restorer:
    """Thin wrapper around the Real-ESRGAN model with x2/x4 support and tiling."""

    def __init__(
        self,
        manager: ModelManager | None = None,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_TILE_OVERLAP,
    ) -> None:
        self.manager = manager or get_model_manager()
        self.tile_size = tile_size
        self.overlap = overlap

    def upscale(self, image: Image.Image, scale: int = 4) -> Image.Image:
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")

        bundle = self.manager.restorer
        tensor = to_tensor(image, self.manager.device)
        output = tiled_inference(
            bundle.model,
            tensor,
            scale=bundle.scale,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )
        restored = to_pil(output)

        # Real-ESRGAN's native scale is 4x. For a 2x request, downsample the
        # 4x result rather than loading a second model.
        if scale != bundle.scale:
            target = (image.width * scale, image.height * scale)
            restored = restored.resize(target, Image.Resampling.LANCZOS)

        return restored

    def restore(
        self, image: Image.Image, scale: int = 4, max_output_side: int | None = 4096
    ) -> Image.Image:
        """Restore an image, optionally guarding the output dimensions."""
        if max_output_side:
            max_input = max(16, max_output_side // scale)
            image = limit_image(image, max_input)
        return self.upscale(image, scale=scale)
