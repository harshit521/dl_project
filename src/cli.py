"""Command-line entry point.

Examples
--------
python -m src.cli process assets/sample_scene.jpg --scale 4
python -m src.cli process assets/sample_portrait.jpg -o outputs/portrait.png --no-face
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import DEFAULT_MAX_SIDE, OUTPUTS_DIR
from src.models import ModelManager
from src.pipeline import Pipeline, PipelineOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo-restore",
        description="AI photo restoration and captioning (pretrained models, no training).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Restore and caption an image")
    process.add_argument("input", type=Path, help="Input image path")
    process.add_argument(
        "-o", "--output", type=Path, default=None, help="Output image path"
    )
    process.add_argument(
        "--scale", type=int, choices=(2, 4), default=4, help="Upscale factor"
    )
    process.add_argument(
        "--no-face", action="store_true", help="Disable face enhancement"
    )
    process.add_argument(
        "--no-caption", action="store_true", help="Disable captioning"
    )
    process.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable automatic color/contrast correction",
    )
    process.add_argument(
        "--scratch",
        action="store_true",
        help="Experimental dust/scratch reduction (pre-upscale)",
    )
    process.add_argument(
        "--colorize",
        action="store_true",
        help="Colorize a grayscale or faded photo",
    )
    process.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_MAX_SIDE,
        help="Maximum input side before upscaling",
    )
    process.add_argument(
        "--tile-size", type=int, default=256, help="Tile size for large images"
    )
    process.add_argument(
        "--overlap", type=int, default=32, help="Tile overlap in pixels"
    )
    process.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default=None,
        help="Force a compute device",
    )
    process.add_argument(
        "--json", action="store_true", help="Print metadata as JSON"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "process":
        return 2

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    manager = ModelManager(device=args.device)
    pipeline = Pipeline(manager=manager)
    options = PipelineOptions(
        scale=args.scale,
        face_enhance=not args.no_face,
        caption=not args.no_caption,
        auto_enhance=not args.no_enhance,
        scratch_removal=args.scratch,
        colorize=args.colorize,
        max_side=args.max_side,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    try:
        result = pipeline.process(args.input, options=options)
    except Exception as exc:
        print(f"Processing failed: {exc}", file=sys.stderr)
        return 1

    output = args.output
    if output is None:
        output = OUTPUTS_DIR / f"{args.input.stem}_restored.png"
    pipeline.save(result, output)

    print(f"Saved: {output}")
    if result.caption:
        print(f"Caption: {result.caption}")
    if args.json:
        print(json.dumps(result.meta, indent=2))
    else:
        timings = result.meta.get("timings", {})
        print(
            f"Time: {timings.get('total_s', 0):.2f}s | "
            f"Device: {result.meta.get('device')} | "
            f"Faces: {result.meta.get('face_count')}"
        )
        for warning in result.meta.get("warnings", []):
            print(f"Warning: {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
