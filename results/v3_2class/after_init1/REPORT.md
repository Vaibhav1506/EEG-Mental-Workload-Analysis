# STEW v3 — optimised preprocessing and training

Updated 2026-10-23T09:55:41+00:00. Device: NVIDIA GeForce RTX 4060 Laptop GPU.

Task: 2class (rest / task). Evaluation identical to v1/v2: 45 rated subjects, 3 classes (ratings 1-3 / 4-6 / 7-9), five subject-disjoint test folds (seed 42), 3,330 test windows of 4 s. Values are mean ± SD over the five test folds.

Training-majority baseline accuracy: 50.00%.

## Variant `filtered-ea-sag`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D-CNN | 5 | 87.93 ± 3.66 | 87.93 ± 3.66 | 0.879 | 94.44 ± 6.80 | — | — |
| RNN | 5 | 86.10 ± 2.67 | 86.10 ± 2.67 | 0.861 | 97.78 ± 3.04 | — | — |
| LSTM | 5 | 83.81 ± 3.95 | 83.81 ± 3.95 | 0.838 | 95.56 ± 6.09 | — | — |
| BiLSTM | 5 | 84.32 ± 3.92 | 84.32 ± 3.92 | 0.843 | 95.56 ± 6.09 | — | — |
| GRU | 5 | 84.62 ± 3.91 | 84.62 ± 3.91 | 0.846 | 94.44 ± 7.86 | — | — |
| GNN | 5 | 88.47 ± 2.00 | 88.47 ± 2.00 | 0.884 | 96.67 ± 3.04 | — | — |
| STFT-ViT | 5 | 83.69 ± 4.16 | 83.69 ± 4.16 | 0.837 | 94.44 ± 7.86 | — | — |
| STFT-Swin | 5 | 87.39 ± 2.88 | 87.39 ± 2.88 | 0.874 | 95.56 ± 6.09 | — | — |
| EEGNet Adapted | 5 | 86.31 ± 2.07 | 86.31 ± 2.07 | 0.863 | 96.67 ± 3.04 | — | — |
| DeepConvNet Adapted | 5 | 85.62 ± 1.92 | 85.62 ± 1.92 | 0.856 | 95.56 ± 2.48 | — | — |
| TemporalCNN | 5 | 87.63 ± 3.48 | 87.63 ± 3.48 | 0.876 | 98.89 ± 2.48 | — | — |
| ShallowConvNet Adapted | 5 | 89.04 ± 3.14 | 89.04 ± 3.14 | 0.890 | 96.67 ± 4.97 | — | — |
| Top-3 Validation Ensemble | 5 | 90.36 ± 2.64 | 90.36 ± 2.64 | 0.904 | 97.78 ± 3.04 | — | — |

