# Before vs after — STEW 2-class rest vs task, unseen-subject 5-fold CV

| Model | Before: accuracy % | After: accuracy % | Change (pp) | After: macro-F1 | After: recording accuracy % |
|---|---:|---:|---:|---:|---:|
| 1D-CNN | 62.34 ± 4.87 | 87.78 ± 3.22 | +25.44 | 0.878 | 94.44 ± 6.80 |
| RNN | 56.25 ± 4.85 | 86.28 ± 2.85 | +30.03 | 0.863 | 97.78 ± 3.04 |
| LSTM | 66.10 ± 3.71 | 84.59 ± 3.79 | +18.50 | 0.845 | 94.44 ± 7.86 |
| BiLSTM | 65.98 ± 3.38 | 84.11 ± 3.57 | +18.14 | 0.841 | 94.44 ± 7.86 |
| GRU | 58.08 ± 4.06 | 85.20 ± 3.92 | +27.12 | 0.852 | 95.56 ± 6.09 |
| GNN | 60.06 ± 7.36 | 89.04 ± 2.14 | +28.98 | 0.890 | 98.89 ± 2.48 |
| STFT-ViT | 70.60 ± 5.59 | 84.17 ± 3.16 | +13.57 | 0.842 | 94.44 ± 7.86 |
| STFT-Swin | 69.34 ± 5.06 | 87.21 ± 3.45 | +17.87 | 0.872 | 96.67 ± 4.97 |
| EEGNet Adapted | 52.55 ± 6.33 | 86.85 ± 1.92 | +34.29 | 0.868 | 96.67 ± 3.04 |
| DeepConvNet Adapted | 50.78 ± 0.96 | 86.46 ± 1.86 | +35.68 | 0.864 | 96.67 ± 3.04 |
| ShallowConvNet Adapted | 68.05 ± 6.37 | 88.92 ± 3.48 | +20.87 | 0.889 | 94.44 ± 5.56 |
| TemporalCNN | 67.09 ± 5.52 | 88.20 ± 3.24 | +21.11 | 0.882 | 98.89 ± 2.48 |
| Top-3 Validation Ensemble | 71.23 ± 7.08 | 90.39 ± 2.61 | +19.16 | 0.904 | 97.78 ± 3.04 |

Accuracy = 4-second window accuracy, mean ± SD over the 5 subject-disjoint test folds. Recording accuracy = one decision per 150-second recording (average of its window probabilities).
