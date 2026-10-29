"""Download the pretrained weights used by the app.

Real-ESRGAN is fetched from the Hugging Face mirror of the official weights
(the original GitHub release asset is no longer available). BLIP is downloaded
lazily by Transformers, but this script can prefetch it.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --with-gfpgan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)

RESTORER_URL = (
    "https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth"
)
GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)
FACE_DETECTOR_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
COLORIZER_URL = "https://colorizers.s3.amazonaws.com/siggraph17-df00044c.pth"


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Already present: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"Downloading {url}\n  -> {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    print(f"\r  {percent:5.1f}% ({downloaded / 1e6:.1f} MB)", end="")
    print()
    temporary.replace(destination)
    print(f"Saved: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument(
        "--with-gfpgan",
        action="store_true",
        help="Also download GFPGAN v1.4 (optional face restoration)",
    )
    parser.add_argument(
        "--with-colorizer",
        action="store_true",
        help="Download the SIGGRAPH 2017 colorizer (~130 MB)",
    )
    parser.add_argument(
        "--with-caption",
        action="store_true",
        help="Prefetch the BLIP captioning model from Hugging Face",
    )
    args = parser.parse_args()

    download(RESTORER_URL, WEIGHTS / "RealESRGAN_x4plus.pth", force=args.force)
    download(
        FACE_DETECTOR_URL,
        WEIGHTS / "face_detection_yunet_2023mar.onnx",
        force=args.force,
    )

    if args.with_colorizer:
        try:
            download(COLORIZER_URL, WEIGHTS / "siggraph17-df00044c.pth", force=args.force)
        except Exception as exc:
            print(f"Colorizer download failed: {exc}", file=sys.stderr)

    if args.with_gfpgan:
        try:
            download(GFPGAN_URL, WEIGHTS / "GFPGANv1.4.pth", force=args.force)
        except Exception as exc:
            print(f"GFPGAN download failed: {exc}", file=sys.stderr)
            print(
                "Face enhancement will fall back to Real-ESRGAN face crops.",
                file=sys.stderr,
            )

    if args.with_caption:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        name = "Salesforce/blip-image-captioning-base"
        print(f"Prefetching {name} ...")
        BlipProcessor.from_pretrained(name)
        BlipForConditionalGeneration.from_pretrained(name)
        print("Caption model cached.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
