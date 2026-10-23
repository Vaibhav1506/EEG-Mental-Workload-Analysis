# STEW v3 — optimised preprocessing and training

Updated 2026-10-22T20:55:54+00:00. Device: NVIDIA GeForce RTX 4060 Laptop GPU.

Evaluation identical to v1/v2: 45 rated subjects, 3 classes (ratings 1-3 / 4-6 / 7-9), five subject-disjoint test folds (seed 42), 3,330 test windows of 4 s. Values are mean ± SD over the five test folds.

Training-majority baseline accuracy: 46.67%.

## Variant `filtered-ea-sag`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| STFT-ViT | 5 | 53.63 ± 7.47 | 48.85 ± 7.43 | 0.473 | 60.00 ± 4.65 | 53.39 | +0.24 |
| STFT-Swin | 5 | 56.34 ± 3.55 | 52.00 ± 4.66 | 0.499 | 62.22 ± 4.65 | 51.14 | +5.20 |

## Variant `filtered`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| STFT-ViT | 5 | 52.07 ± 9.99 | 49.82 ± 10.08 | 0.481 | 52.22 ± 15.01 | 53.39 | -1.32 |
| STFT-Swin | 5 | 53.96 ± 8.23 | 50.37 ± 7.67 | 0.492 | 60.00 ± 9.94 | 51.14 | +2.82 |

## Variant `selected`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| STFT-ViT | 5 | 50.54 ± 9.67 | 47.64 ± 10.08 | 0.459 | 51.11 ± 14.38 | 53.39 | -2.85 |
| STFT-Swin | 5 | 56.34 ± 3.55 | 52.00 ± 4.66 | 0.499 | 62.22 ± 4.65 | 51.14 | +5.20 |

