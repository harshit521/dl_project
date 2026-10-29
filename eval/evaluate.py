"""Evaluate Real-ESRGAN restoration against a bicubic baseline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import LPIPSMetric, psnr, ssim  # noqa: E402
from src.models import ModelManager  # noqa: E402
from src.preprocess import open_image  # noqa: E402
from src.restore import Restorer  # noqa: E402


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data" / "test" / "manifest.csv"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results.csv")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--device", default=None, choices=("mps", "cuda", "cpu"))
    args = parser.parse_args()

    if not args.manifest.exists():
        print(
            f"Manifest not found: {args.manifest}\n"
            "Run: python scripts/make_eval_set.py",
            file=sys.stderr,
        )
        return 1

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    manager = ModelManager(device=args.device)
    restorer = Restorer(manager=manager)
    lpips_metric = None if args.no_lpips else LPIPSMetric()

    outputs_dir = ROOT / "outputs" / "eval"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for index, row in enumerate(rows, start=1):
        hr = open_image(ROOT / row["hr"])
        lr = open_image(ROOT / row["lr"])
        scale = int(row.get("scale", 4))

        restored = restorer.upscale(lr, scale=scale)
        if restored.size != hr.size:
            restored = restored.resize(hr.size, Image.Resampling.LANCZOS)
        bicubic = lr.resize(hr.size, Image.Resampling.BICUBIC)

        result = {
            "name": row["name"],
            "kind": row.get("kind", "mixed"),
            "scale": scale,
            "input_size": f"{lr.width}x{lr.height}",
            "output_size": f"{hr.width}x{hr.height}",
            "bicubic_psnr": round(psnr(hr, bicubic), 3),
            "bicubic_ssim": round(ssim(hr, bicubic), 4),
            "restored_psnr": round(psnr(hr, restored), 3),
            "restored_ssim": round(ssim(hr, restored), 4),
        }
        if lpips_metric is not None:
            result["bicubic_lpips"] = round(lpips_metric(hr, bicubic), 4)
            result["restored_lpips"] = round(lpips_metric(hr, restored), 4)
        results.append(result)

        if args.save_visuals:
            width = hr.width
            canvas = Image.new("RGB", (width * 3, hr.height), "white")
            canvas.paste(lr.resize(hr.size, Image.Resampling.NEAREST), (0, 0))
            canvas.paste(bicubic, (width, 0))
            canvas.paste(restored, (width * 2, 0))
            canvas.save(outputs_dir / f"{row['name']}_comparison.jpg", quality=90)

        print(
            f"[{index}/{len(rows)}] {row['name']} ({result['kind']}): "
            f"PSNR {result['bicubic_psnr']:.2f} -> {result['restored_psnr']:.2f} dB"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    groups: dict[str, list[dict]] = {}
    for result in results:
        groups.setdefault(result["kind"], []).append(result)

    summaries: list[dict] = []
    for kind, group in groups.items():
        summary = {
            "group": kind,
            "n": len(group),
            "bicubic_psnr": mean([r["bicubic_psnr"] for r in group]),
            "restored_psnr": mean([r["restored_psnr"] for r in group]),
            "bicubic_ssim": mean([r["bicubic_ssim"] for r in group]),
            "restored_ssim": mean([r["restored_ssim"] for r in group]),
        }
        if lpips_metric is not None:
            summary["bicubic_lpips"] = mean([r["bicubic_lpips"] for r in group])
            summary["restored_lpips"] = mean([r["restored_lpips"] for r in group])
        summaries.append(summary)

    summary_path = args.output.with_name("summary.csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\nSummary")
    for summary in summaries:
        print(f"  {summary['group']} ({summary['n']} images)")
        print(
            f"    PSNR:  {summary['bicubic_psnr']:.2f} dB (bicubic) -> "
            f"{summary['restored_psnr']:.2f} dB (restored)"
        )
        print(
            f"    SSIM:  {summary['bicubic_ssim']:.4f} (bicubic) -> "
            f"{summary['restored_ssim']:.4f} (restored)"
        )
        if lpips_metric is not None:
            print(
                f"    LPIPS: {summary['bicubic_lpips']:.4f} (bicubic) -> "
                f"{summary['restored_lpips']:.4f} (restored; lower is better)"
            )
    print(f"\nResults: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
