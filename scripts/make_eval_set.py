"""Build a small, deterministic super-resolution evaluation set.

Uses the public sample images shipped with scikit-image, so no external
download or personal dataset is required. Each source image is split into
256x256 crops, downscaled 4x, and lightly degraded to mimic a scan.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from make_synthetic_old import make_old_photo
from PIL import Image
from skimage import data as skdata

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "test"


def load_sources() -> list[tuple[str, Image.Image]]:
    sources = []
    for name in ("astronaut", "chelsea", "coffee", "rocket", "horse", "camera"):
        array = getattr(skdata, name)()
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        sources.append((name, Image.fromarray(array.astype(np.uint8)).convert("RGB")))
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--crops", type=int, default=4, help="Crops per source image")
    parser.add_argument("--size", type=int, default=256, help="Crop size")
    parser.add_argument("--scale", type=int, default=4, choices=(2, 4))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pairs_dir = args.out / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name, image in load_sources():
        # Make sure there is enough room for the requested crops.
        if min(image.size) < args.size:
            factor = args.size / min(image.size)
            image = image.resize(
                (int(image.width * factor), int(image.height * factor)),
                Image.Resampling.LANCZOS,
            )

        for index in range(args.crops):
            left = rng.randint(0, image.width - args.size)
            top = rng.randint(0, image.height - args.size)
            crop = image.crop((left, top, left + args.size, top + args.size))

            lr_size = (
                max(16, crop.width // args.scale),
                max(16, crop.height // args.scale),
            )
            lr_clean = crop.resize(lr_size, Image.Resampling.BICUBIC)
            lr_old = make_old_photo(
                lr_clean.copy(), severity=0.55, seed=args.seed + len(rows)
            )

            stem = f"{name}_{index:02d}"
            hr_path = pairs_dir / f"{stem}_hr.png"
            crop.save(hr_path)
            for kind, lr in (("clean", lr_clean), ("old", lr_old)):
                lr_path = pairs_dir / f"{stem}_lr_{kind}.png"
                lr.save(lr_path)
                rows.append(
                    {
                        "name": stem,
                        "kind": kind,
                        "hr": str(hr_path.relative_to(ROOT)),
                        "lr": str(lr_path.relative_to(ROOT)),
                        "scale": args.scale,
                    }
                )

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} pairs to {pairs_dir}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
