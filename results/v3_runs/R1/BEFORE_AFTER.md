# Before vs after — STEW 3-class workload, unseen-subject 5-fold CV

| Model | Before: accuracy % | After: accuracy % | Change (pp) | After: macro-F1 | After: recording accuracy % |
|---|---:|---:|---:|---:|---:|
| 1D-CNN | 44.86 ± 8.76 | 59.40 ± 3.20 | +14.53 | 0.531 | 64.44 ± 4.97 |
| RNN | 37.42 ± 4.07 | 60.06 ± 6.13 | +22.64 | 0.532 | 66.67 ± 6.80 |
| LSTM | 48.20 ± 5.98 | 57.30 ± 5.46 | +9.10 | 0.518 | 67.78 ± 12.04 |
| BiLSTM | 45.08 ± 6.94 | 55.71 ± 5.41 | +10.63 | 0.506 | 64.44 ± 7.45 |
| GRU | 36.04 ± 6.79 | 57.81 ± 5.53 | +21.77 | 0.522 | 66.67 ± 14.16 |
| GNN | 46.43 ± 6.04 | 58.26 ± 5.08 | +11.83 | 0.512 | 63.33 ± 4.97 |
| STFT-ViT | 53.39 ± 7.52 | 52.64 ± 10.01 | -0.75 | 0.476 | 54.44 ± 16.39 |
| STFT-Swin | 51.14 ± 6.56 | 57.93 ± 6.16 | +6.79 | 0.511 | 66.67 ± 6.80 |
| EEGNet Adapted | 45.95 ± 6.69 | 60.51 ± 4.34 | +14.56 | 0.532 | 67.78 ± 9.13 |
| DeepConvNet Adapted | 36.94 ± 6.64 | 57.60 ± 7.20 | +20.66 | 0.506 | 60.00 ± 9.13 |
| ShallowConvNet Adapted | 43.87 ± 8.49 | 58.20 ± 5.00 | +14.32 | 0.520 | 62.22 ± 8.24 |
| TemporalCNN | 41.20 ± 8.88 | 58.74 ± 2.19 | +17.54 | 0.518 | 63.33 ± 7.45 |
| Top-3 Validation Ensemble | — | 57.33 ± 6.04 | — | 0.503 | 65.56 ± 4.65 |

Accuracy = 4-second window accuracy, mean ± SD over the 5 subject-disjoint test folds. Recording accuracy = one decision per 150-second recording (average of its window probabilities).
