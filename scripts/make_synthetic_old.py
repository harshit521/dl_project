"""Generate synthetic old-photo degradations from modern images.

This gives us a repeatable demo/evaluation set without needing a historical
photo dataset. The same effects are used for qualitative demos and for paired
super-resolution evaluation.

Examples
--------
python scripts/make_synthetic_old.py --input assets/sample_scene.jpg --out data/demo
python scripts/make_synthetic_old.py --input assets --out data/demo --all
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def add_fade(image: Image.Image, rng: random.Random) -> Image.Image:
    """Desaturate, lower contrast, add a warm/sepia cast, and lift blacks."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    gray = array.mean(axis=2, keepdims=True)

    saturation = rng.uniform(0.25, 0.65)
    array = gray + (array - gray) * saturation

    contrast = rng.uniform(0.6, 0.85)
    array = (array - 0.5) * contrast + 0.5
    array = array + rng.uniform(0.03, 0.10)

    sepia_strength = rng.uniform(0.15, 0.45)
    sepia = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        dtype=np.float32,
    )
    sepia_image = array @ sepia.T
    array = array * (1 - sepia_strength) + sepia_image * sepia_strength

    tint = np.array(
        [rng.uniform(1.0, 1.10), rng.uniform(0.97, 1.02), rng.uniform(0.85, 0.98)],
        dtype=np.float32,
    )
    array *= tint
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))


def add_vignette(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image, dtype=np.float32) / 255.0
    h, w = array.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(cx**2 + cy**2)
    strength = rng.uniform(0.15, 0.45)
    mask = np.clip(1.0 - strength * radius**2, 0, 1)[..., None]
    array *= mask
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))


def add_scratches(image: Image.Image, rng: random.Random) -> Image.Image:
    """Draw thin lines and dust specks to mimic physical print damage."""
    damaged = image.copy()
    draw = ImageDraw.Draw(damaged)
    w, h = damaged.size

    for _ in range(rng.randint(3, 9)):
        x1, y1 = rng.randint(0, w - 1), rng.randint(0, h - 1)
        length = rng.randint(max(12, w // 20), max(24, w // 3))
        angle = rng.uniform(-0.5, 0.5)
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
        color = rng.choice([(245, 240, 225), (220, 210, 190), (90, 80, 70)])
        draw.line((x1, y1, x2, y2), fill=color, width=rng.randint(1, 3))

    for _ in range(rng.randint(20, 80)):
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        radius = rng.randint(1, 3)
        color = rng.choice([(230, 225, 210), (120, 110, 95)])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    return damaged.filter(ImageFilter.GaussianBlur(radius=0.4))


def add_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image, dtype=np.float32) / 255.0
    sigma = rng.uniform(0.01, 0.045)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(
        0, sigma, array.shape
    )
    array = np.clip(array + noise, 0, 1)
    return Image.fromarray((array * 255).astype(np.uint8))


def jpeg_artifacts(image: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(18, 55)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def make_old_photo(
    image: Image.Image, severity: float = 1.0, seed: int = 42
) -> Image.Image:
    """Apply a randomized, deterministic old-photo degradation chain."""
    rng = random.Random(seed)
    result = image.copy()

    # Resolution loss first, like a low-quality scan.
    if severity > 0.25 and max(result.size) > 256:
        scale = rng.uniform(0.55, 0.85)
        result = result.resize(
            (max(16, int(result.width * scale)), max(16, int(result.height * scale))),
            Image.Resampling.BICUBIC,
        )
        result = result.resize(image.size, Image.Resampling.BICUBIC)

    result = add_fade(result, rng)
    if rng.random() < 0.8:
        result = result.filter(
            ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.6) * severity)
        )
    result = add_vignette(result, rng)
    if rng.random() < 0.9:
        result = add_scratches(result, rng)
    if rng.random() < 0.85:
        result = add_noise(result, rng)
    if rng.random() < 0.85:
        result = jpeg_artifacts(result, rng)

    # A little extra contrast/saturation loss on top.
    result = ImageEnhance.Color(result).enhance(rng.uniform(0.6, 0.9))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.8, 1.0))
    return result


def make_sr_pair(
    image: Image.Image, scale: int = 4, seed: int = 42
) -> tuple[Image.Image, Image.Image]:
    """Return (high_resolution, low_resolution) pair for SR evaluation."""
    rng = random.Random(seed)
    hr = image.copy()
    lr_size = (max(16, hr.width // scale), max(16, hr.height // scale))
    lr = hr.resize(lr_size, Image.Resampling.BICUBIC)
    lr = make_old_photo(lr, severity=0.7, seed=rng.randint(0, 2**31 - 1))
    return hr, lr


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=4, choices=(2, 4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all", action="store_true", help="Process every supported image in a directory"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pairs_dir = args.out / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(args.input)
    if not images:
        print(f"No supported images found in {args.input}")
        return 1
    if not args.all:
        images = images[:1]

    manifest_path = args.out / "manifest.csv"
    rows: list[dict] = []

    for index, path in enumerate(images):
        with Image.open(path) as opened:
            image = opened.convert("RGB")

        # Keep the demo image manageable.
        if max(image.size) > 768:
            image = image.copy()
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)

        old = make_old_photo(image, severity=1.0, seed=args.seed + index)
        old_name = f"{path.stem}_old.jpg"
        old.save(args.out / old_name, quality=92)

        hr, lr = make_sr_pair(image, scale=args.scale, seed=args.seed + index)
        stem = f"{index:03d}_{path.stem}"
        hr_path = pairs_dir / f"{stem}_hr.png"
        lr_path = pairs_dir / f"{stem}_lr.png"
        hr.save(hr_path)
        lr.save(lr_path)

        rows.append(
            {
                "name": stem,
                "hr": str(hr_path),
                "lr": str(lr_path),
                "old": str(args.out / old_name),
                "scale": args.scale,
            }
        )
        print(f"Generated: {old_name} and SR pair {stem}")

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} sample(s) to {args.out}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
