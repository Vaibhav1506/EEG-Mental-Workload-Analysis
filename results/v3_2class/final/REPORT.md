# STEW v3 — optimised preprocessing and training

Updated 2026-10-23T10:15:06+00:00. Device: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel.

Task: 2class (rest / task). Evaluation identical to v1/v2: 45 rated subjects, 3 classes (ratings 1-3 / 4-6 / 7-9), five subject-disjoint test folds (seed 42), 3,330 test windows of 4 s. Values are mean ± SD over the five test folds.

Training-majority baseline accuracy: 50.00%.

## Variant `filtered-ea-sag`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D-CNN | 5 | 87.78 ± 3.22 | 87.78 ± 3.22 | 0.878 | 94.44 ± 6.80 | — | — |
| RNN | 5 | 86.28 ± 2.85 | 86.28 ± 2.85 | 0.863 | 97.78 ± 3.04 | — | — |
| LSTM | 5 | 84.59 ± 3.79 | 84.59 ± 3.79 | 0.845 | 94.44 ± 7.86 | — | — |
| BiLSTM | 5 | 84.11 ± 3.57 | 84.11 ± 3.57 | 0.841 | 94.44 ± 7.86 | — | — |
| GRU | 5 | 85.20 ± 3.92 | 85.20 ± 3.92 | 0.852 | 95.56 ± 6.09 | — | — |
| GNN | 5 | 89.04 ± 2.14 | 89.04 ± 2.14 | 0.890 | 98.89 ± 2.48 | — | — |
| STFT-ViT | 5 | 84.17 ± 3.16 | 84.17 ± 3.16 | 0.842 | 94.44 ± 7.86 | — | — |
| STFT-Swin | 5 | 87.21 ± 3.45 | 87.21 ± 3.45 | 0.872 | 96.67 ± 4.97 | — | — |
| EEGNet Adapted | 5 | 86.85 ± 1.92 | 86.85 ± 1.92 | 0.868 | 96.67 ± 3.04 | — | — |
| DeepConvNet Adapted | 5 | 86.46 ± 1.86 | 86.46 ± 1.86 | 0.864 | 96.67 ± 3.04 | — | — |
| ShallowConvNet Adapted | 5 | 88.92 ± 3.48 | 88.92 ± 3.48 | 0.889 | 94.44 ± 5.56 | — | — |
| TemporalCNN | 5 | 88.20 ± 3.24 | 88.20 ± 3.24 | 0.882 | 98.89 ± 2.48 | — | — |
| Top-3 Validation Ensemble | 5 | 90.39 ± 2.61 | 90.39 ± 2.61 | 0.904 | 97.78 ± 3.04 | — | — |

