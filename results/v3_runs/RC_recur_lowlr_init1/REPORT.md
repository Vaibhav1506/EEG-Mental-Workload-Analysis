# STEW v3 — optimised preprocessing and training

Updated 2026-10-23T10:09:43+00:00. Device: NVIDIA GeForce RTX 4060 Laptop GPU.

Evaluation identical to v1/v2: 45 rated subjects, 3 classes (ratings 1-3 / 4-6 / 7-9), five subject-disjoint test folds (seed 42), 3,330 test windows of 4 s. Values are mean ± SD over the five test folds.

Training-majority baseline accuracy: 46.67%.

## Variant `filtered-ea-sag`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 5 | 56.91 ± 3.52 | 52.95 ± 1.69 | 0.523 | 67.78 ± 8.24 | 48.20 | +8.71 |
| BiLSTM | 5 | 56.82 ± 3.86 | 53.03 ± 3.70 | 0.517 | 64.44 ± 3.04 | 45.08 | +11.74 |

