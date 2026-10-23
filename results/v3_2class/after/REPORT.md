# STEW v3 — optimised preprocessing and training

Updated 2026-10-23T01:11:30+00:00. Device: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel.

Task: 2class (rest / task). Evaluation identical to v1/v2: 45 rated subjects, 3 classes (ratings 1-3 / 4-6 / 7-9), five subject-disjoint test folds (seed 42), 3,330 test windows of 4 s. Values are mean ± SD over the five test folds.

Training-majority baseline accuracy: 50.00%.

## Variant `filtered-ea-sag`

| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D-CNN | 5 | 87.21 ± 3.29 | 87.21 ± 3.29 | 0.872 | 95.56 ± 4.65 | — | — |
| RNN | 5 | 86.13 ± 2.77 | 86.13 ± 2.77 | 0.861 | 96.67 ± 4.97 | — | — |
| LSTM | 5 | 83.72 ± 3.41 | 83.72 ± 3.41 | 0.837 | 94.44 ± 7.86 | — | — |
| BiLSTM | 5 | 83.72 ± 3.58 | 83.72 ± 3.58 | 0.837 | 94.44 ± 7.86 | — | — |
| GRU | 5 | 84.89 ± 3.86 | 84.89 ± 3.86 | 0.849 | 96.67 ± 4.97 | — | — |
| GNN | 5 | 88.53 ± 2.38 | 88.53 ± 2.38 | 0.885 | 98.89 ± 2.48 | — | — |
| STFT-ViT | 5 | 83.87 ± 4.06 | 83.87 ± 4.06 | 0.839 | 93.33 ± 7.24 | — | — |
| STFT-Swin | 5 | 87.21 ± 4.10 | 87.21 ± 4.10 | 0.872 | 96.67 ± 4.97 | — | — |
| EEGNet Adapted | 5 | 86.97 ± 2.00 | 86.97 ± 2.00 | 0.869 | 97.78 ± 3.04 | — | — |
| DeepConvNet Adapted | 5 | 86.34 ± 2.17 | 86.34 ± 2.17 | 0.863 | 95.56 ± 4.65 | — | — |
| ShallowConvNet Adapted | 5 | 88.77 ± 3.38 | 88.77 ± 3.38 | 0.888 | 94.44 ± 6.80 | — | — |
| TemporalCNN | 5 | 88.53 ± 2.67 | 88.53 ± 2.67 | 0.885 | 98.89 ± 2.48 | — | — |
| Top-3 Validation Ensemble | 5 | 90.63 ± 2.99 | 90.63 ± 2.99 | 0.906 | 96.67 ± 3.04 | — | — |

