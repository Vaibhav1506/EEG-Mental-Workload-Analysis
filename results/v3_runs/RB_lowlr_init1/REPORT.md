# STEW v3 — optimised preprocessing and training

Updated 2026-10-22T22:25:53+00:00. Device: NVIDIA GeForce RTX 4060 Laptop GPU.

Evaluation identical to v1/v2: 45 rated subjects, 3 classes (ratings 1-3 / 4-6 / 7-9), five subject-disjoint test folds (seed 42), 3,330 test windows of 4 s. Values are mean ± SD over the five test folds.

Training-majority baseline accuracy: 46.67%.

## Variant `filtered-ea-sag`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| STFT-ViT | 5 | 55.35 ± 6.12 | 49.74 ± 5.97 | 0.486 | 58.89 ± 6.33 | 53.39 | +1.95 |
| STFT-Swin | 5 | 56.61 ± 5.62 | 52.14 ± 6.81 | 0.505 | 62.22 ± 4.65 | 51.14 | +5.47 |

## Variant `filtered`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| STFT-ViT | 5 | 53.75 ± 7.98 | 50.67 ± 7.89 | 0.497 | 58.89 ± 12.17 | 53.39 | +0.36 |
| STFT-Swin | 5 | 54.80 ± 6.14 | 51.74 ± 6.08 | 0.504 | 62.22 ± 9.13 | 51.14 | +3.66 |

## Variant `selected`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| STFT-ViT | 5 | 52.04 ± 6.41 | 48.03 ± 5.82 | 0.470 | 55.56 ± 11.79 | 53.39 | -1.35 |
| STFT-Swin | 5 | 55.68 ± 4.32 | 51.76 ± 4.73 | 0.501 | 64.44 ± 3.04 | 51.14 | +4.53 |

