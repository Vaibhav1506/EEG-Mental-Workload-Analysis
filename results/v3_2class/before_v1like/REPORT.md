# STEW v3 — optimised preprocessing and training

Updated 2026-10-23T01:11:39+00:00. Device: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel.

Task: 2class (rest / task). Evaluation identical to v1/v2: 45 rated subjects, 3 classes (ratings 1-3 / 4-6 / 7-9), five subject-disjoint test folds (seed 42), 3,330 test windows of 4 s. Values are mean ± SD over the five test folds.

Training-majority baseline accuracy: 50.00%.

## Variant `raw`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D-CNN | 5 | 62.34 ± 4.87 | 62.34 ± 4.87 | 0.609 | 66.67 ± 8.78 | — | — |
| RNN | 5 | 56.25 ± 4.85 | 56.25 ± 4.85 | 0.552 | 62.22 ± 12.04 | — | — |
| LSTM | 5 | 66.10 ± 3.71 | 66.10 ± 3.71 | 0.656 | 72.22 ± 3.93 | — | — |
| BiLSTM | 5 | 65.98 ± 3.38 | 65.98 ± 3.38 | 0.649 | 74.44 ± 8.43 | — | — |
| GRU | 5 | 58.08 ± 4.06 | 58.08 ± 4.06 | 0.560 | 60.00 ± 7.24 | — | — |
| GNN | 5 | 60.06 ± 7.36 | 60.06 ± 7.36 | 0.592 | 65.56 ± 18.17 | — | — |
| STFT-ViT | 5 | 70.60 ± 5.59 | 70.60 ± 5.59 | 0.701 | 76.67 ± 7.24 | — | — |
| STFT-Swin | 5 | 69.34 ± 5.06 | 69.34 ± 5.06 | 0.681 | 73.33 ± 9.94 | — | — |
| EEGNet Adapted | 5 | 52.55 ± 6.33 | 52.55 ± 6.33 | 0.518 | 54.44 ± 7.24 | — | — |
| DeepConvNet Adapted | 5 | 50.78 ± 0.96 | 50.78 ± 0.96 | 0.389 | 50.00 ± 0.00 | — | — |
| ShallowConvNet Adapted | 5 | 68.05 ± 6.37 | 68.05 ± 6.37 | 0.667 | 71.11 ± 8.24 | — | — |
| TemporalCNN | 5 | 67.09 ± 5.52 | 67.09 ± 5.52 | 0.666 | 77.78 ± 8.78 | — | — |
| Top-3 Validation Ensemble | 5 | 71.23 ± 7.08 | 71.23 ± 7.08 | 0.704 | 78.89 ± 12.67 | — | — |

