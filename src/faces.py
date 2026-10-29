"""Face detection and enhancement.

Primary backend: GFPGAN when the optional ``gfpgan`` package is installed.
Fallback backend: crop each detected face and run the Real-ESRGAN restorer on
it, then feather it back into the image. The fallback is not identity-aware,
but it sharpens faces and keeps the feature usable without extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.config import FACE_DETECTOR_PATH
from src.models import ModelManager, get_model_manager
from src.restore import Restorer


@dataclass
class FaceEnhancementResult:
    image: Image.Image
    faces: int
    backend: str


def _yunet_detect(
    image: Image.Image, max_faces: int
) -> list[tuple[int, int, int, int]]:
    if not FACE_DETECTOR_PATH.exists():
        return []
    if not hasattr(cv2, "FaceDetectorYN_create"):
        return []

    # YuNet becomes unreliable on very large canvases, so detect at <= 1024 px
    # and map the boxes back to the original resolution.
    original_width, original_height = image.size
    detect_scale = min(1.0, 1024.0 / max(original_width, original_height))
    if detect_scale < 1.0:
        work = image.resize(
            (
                max(16, int(round(original_width * detect_scale))),
                max(16, int(round(original_height * detect_scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        work = image

    bgr = cv2.cvtColor(np.asarray(work), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    detector = cv2.FaceDetectorYN_create(
        str(FACE_DETECTOR_PATH), "", (width, height), 0.6, 0.3, 5000
    )
    detector.setInputSize((width, height))
    _, faces = detector.detect(bgr)

    if faces is None:
        return []

    inverse_scale = 1.0 / detect_scale
    boxes = []
    for face in faces:
        x, y, w, h = (float(v) for v in face[:4])
        score = float(face[-1])
        if score < 0.6 or w < 16 or h < 16:
            continue
        x = int(round(x * inverse_scale))
        y = int(round(y * inverse_scale))
        w = int(round(w * inverse_scale))
        h = int(round(h * inverse_scale))
        x, y = max(0, x), max(0, y)
        w, h = min(w, original_width - x), min(h, original_height - y)
        if w >= 16 and h >= 16:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    return boxes[:max_faces]


def _haar_detect(
    image: Image.Image, max_faces: int
) -> list[tuple[int, int, int, int]]:
    if not hasattr(cv2, "CascadeClassifier"):
        return []
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return []
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
    )
    boxes = [tuple(int(v) for v in face) for face in faces]
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    return boxes[:max_faces]


def detect_faces(image: Image.Image, max_faces: int = 8) -> list[tuple[int, int, int, int]]:
    """Return face boxes as (x, y, w, h) in RGB image coordinates.

    YuNet (OpenCV's DNN detector) is used when its ONNX weights are present;
    the legacy Haar cascade is kept as a fallback.
    """
    boxes = _yunet_detect(image, max_faces)
    if boxes:
        return boxes
    return _haar_detect(image, max_faces)


def _feather_mask(size: tuple[int, int], feather: int = 12) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    margin = max(1, feather // 2)
    draw.ellipse(
        (margin, margin, size[0] - margin, size[1] - margin), fill=255
    )
    return mask.filter(ImageFilter.GaussianBlur(feather))


def _enhance_face_crops(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    restorer: Restorer,
) -> Image.Image:
    result = image.copy()
    width, height = image.size

    for x, y, w, h in boxes:
        margin_x, margin_y = int(w * 0.25), int(h * 0.25)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(width, x + w + margin_x)
        bottom = min(height, y + h + margin_y)
        if right - left < 24 or bottom - top < 24:
            continue

        crop = image.crop((left, top, right, bottom))
        # Keep face-crop restoration bounded: large crops add latency without
        # improving the pasted result, which is resized back to the crop size.
        work = crop.copy()
        if max(work.size) > 320:
            work.thumbnail((320, 320), Image.Resampling.LANCZOS)
        enhanced = restorer.upscale(work, scale=2).resize(
            crop.size, Image.Resampling.LANCZOS
        )
        mask = _feather_mask(crop.size)
        result.paste(enhanced, (left, top), mask)

    return result


def enhance_faces(
    image: Image.Image,
    restorer: Restorer | None = None,
    manager: ModelManager | None = None,
) -> FaceEnhancementResult:
    """Enhance detected faces with GFPGAN if available, else Real-ESRGAN crops."""
    manager = manager or get_model_manager()
    restorer = restorer or Restorer(manager=manager)
    boxes = detect_faces(image)
    if not boxes:
        return FaceEnhancementResult(image=image, faces=0, backend="none")

    gfpgan = manager.load_gfpgan()
    if gfpgan is not None:
        try:
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            _, _, output = gfpgan.enhance(
                bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            if output is not None:
                rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                return FaceEnhancementResult(
                    image=Image.fromarray(rgb), faces=len(boxes), backend="gfpgan"
                )
        except Exception:
            # Fall through to the Real-ESRGAN crop fallback.
            pass

    enhanced = _enhance_face_crops(image, boxes, restorer)
    return FaceEnhancementResult(
        image=enhanced, faces=len(boxes), backend="realesrgan-crops"
    )
