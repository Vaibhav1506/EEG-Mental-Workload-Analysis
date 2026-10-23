# STEW v3 — optimised preprocessing and training

Updated 2026-10-23T10:08:12+00:00. Device: NVIDIA GeForce RTX 4060 Laptop GPU.

Evaluation identical to v1/v2: 45 rated subjects, 3 classes (ratings 1-3 / 4-6 / 7-9), five subject-disjoint test folds (seed 42), 3,330 test windows of 4 s. Values are mean ± SD over the five test folds.

Training-majority baseline accuracy: 46.67%.

## Variant `filtered-ea-sag`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 5 | 55.29 ± 4.81 | 51.46 ± 4.40 | 0.502 | 61.11 ± 7.86 | 48.20 | +7.09 |
| BiLSTM | 5 | 55.89 ± 3.84 | 52.78 ± 3.58 | 0.511 | 65.56 ± 8.24 | 45.08 | +10.81 |

