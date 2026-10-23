# Before vs after — STEW 3-class workload, unseen-subject 5-fold CV

| Model | Before: accuracy % | After: accuracy % | Change (pp) | After: macro-F1 | After: recording accuracy % |
|---|---:|---:|---:|---:|---:|
| 1D-CNN | 44.86 ± 8.76 | 58.35 ± 2.35 | +13.48 | 0.522 | 64.44 ± 4.97 |
| RNN | 37.42 ± 4.07 | 59.43 ± 4.82 | +22.01 | 0.533 | 70.00 ± 10.83 |
| LSTM | 48.20 ± 5.98 | 56.70 ± 4.15 | +8.50 | 0.514 | 65.56 ± 12.04 |
| BiLSTM | 45.08 ± 6.94 | 56.82 ± 3.77 | +11.74 | 0.513 | 61.11 ± 7.86 |
| GRU | 36.04 ± 6.79 | 57.81 ± 4.35 | +21.77 | 0.523 | 68.89 ± 10.09 |
| GNN | 46.43 ± 6.04 | 58.92 ± 5.05 | +12.49 | 0.514 | 66.67 ± 5.56 |
| STFT-ViT | 53.39 ± 7.52 | 54.95 ± 6.38 | +1.56 | 0.483 | 60.00 ± 10.69 |
| STFT-Swin | 51.14 ± 6.56 | 57.51 ± 4.56 | +6.37 | 0.506 | 61.11 ± 7.86 |
| EEGNet Adapted | 45.95 ± 6.69 | 61.80 ± 3.08 | +15.86 | 0.541 | 73.33 ± 7.24 |
| DeepConvNet Adapted | 36.94 ± 6.64 | 58.74 ± 4.79 | +21.80 | 0.521 | 66.67 ± 5.56 |
| ShallowConvNet Adapted | 43.87 ± 8.49 | 60.63 ± 2.47 | +16.76 | 0.546 | 66.67 ± 5.56 |
| TemporalCNN | 41.20 ± 8.88 | 59.43 ± 2.75 | +18.23 | 0.523 | 64.44 ± 4.97 |
| Top-3 Validation Ensemble | — | 58.59 ± 3.89 | — | 0.515 | 65.56 ± 2.48 |

Accuracy = 4-second window accuracy, mean ± SD over the 5 subject-disjoint test folds. Recording accuracy = one decision per 150-second recording (average of its window probabilities).
