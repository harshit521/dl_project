# Evaluation Methodology and Results

## Goal

Measure whether the restoration pipeline improves image quality relative to a
bicubic baseline, and report metrics honestly instead of relying on visual
impressions alone.

## Test set

`scripts/make_eval_set.py` builds a deterministic set from public sample images
shipped with scikit-image (`astronaut`, `chelsea`, `coffee`, `rocket`, `horse`,
`camera`). For each source:

- 4 random 256×256 crops (seed 7) → 24 high-resolution crops.
- **clean** variant: bicubic downscale to 64×64 (x4).
- **old** variant: bicubic downscale followed by synthetic fade, blur, noise,
  JPEG compression, scratches/dust, and vignetting.

Total: 48 LR/HR pairs. No personal or historical dataset is required.

## Metrics

| Metric | Direction | What it captures |
|---|---|---|
| PSNR | Higher is better | Pixel-wise fidelity |
| SSIM | Higher is better | Structural similarity |
| LPIPS (AlexNet) | Lower is better | Perceptual similarity |

LPIPS uses a pretrained AlexNet feature extractor. It is an **evaluation
metric**, not part of the restoration model.

## Results

Command:

```bash
python scripts/make_eval_set.py
python eval/evaluate.py
```

Outputs: `eval/results.csv`, `eval/summary.csv`.

| Group | N | PSNR (bicubic) | PSNR (restored) | SSIM (bicubic) | SSIM (restored) | LPIPS (bicubic) | LPIPS (restored) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 24 | **33.74** | 29.87 | **0.8436** | 0.8000 | 0.2835 | **0.1371** |
| Old | 24 | **12.67** | 12.10 | 0.3420 | **0.3657** | 0.8726 | **0.6783** |

## Interpretation

1. **Perceptual quality improves substantially.** LPIPS drops by 52% on clean
   pairs and 22% on old-photo pairs. This matches the visual result: sharper
   edges, less blur, and more natural texture.
2. **PSNR decreases.** Real-ESRGAN is a GAN-based perceptual model; it does not
   optimize mean squared error. It may invent plausible texture and shift
   contrast, which increases pixel-wise error even when the image looks better.
   Reporting only PSNR would unfairly penalize the model.
3. **SSIM is mixed.** It drops slightly on clean downscales but improves on the
   old-photo set, where the baseline is very degraded.
4. **The old-photo PSNR is low for both methods** because the synthetic
   degradation includes color fading, which Real-ESRGAN does not undo. The app's
   separate auto color correction and colorizer address that failure mode, but
   they are not part of this benchmark.

## Caveats

- The test set is small (24 source crops) and synthetic. It is a sanity check,
  not a benchmark paper.
- Real historical prints contain damage types not modeled here (creases, mold,
  missing regions, halftone patterns).
- Face restoration is evaluated qualitatively; identity preservation would
  require an ArcFace-style metric and a face dataset.
- Captioning is not quantitatively evaluated here. BLIP captions should be
  checked manually for hallucination.
- LPIPS depends on pretrained AlexNet weights; results are comparable only when
  using the same metric implementation.

## Reproducing

```bash
python scripts/make_eval_set.py
python eval/evaluate.py                 # includes LPIPS
python eval/evaluate.py --no-lpips      # faster, PSNR/SSIM only
python eval/evaluate.py --save-visuals  # writes side-by-side comparisons
```

Visual comparisons are written to `outputs/eval/`.
