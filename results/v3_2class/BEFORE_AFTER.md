# Before vs after — STEW 2-class rest vs task, unseen-subject 5-fold CV

| Model | Before: accuracy % | After: accuracy % | Change (pp) | After: macro-F1 | After: recording accuracy % |
|---|---:|---:|---:|---:|---:|
| 1D-CNN | 62.34 ± 4.87 | 87.21 ± 3.29 | +24.86 | 0.872 | 95.56 ± 4.65 |
| RNN | 56.25 ± 4.85 | 86.13 ± 2.77 | +29.88 | 0.861 | 96.67 ± 4.97 |
| LSTM | 66.10 ± 3.71 | 83.72 ± 3.41 | +17.63 | 0.837 | 94.44 ± 7.86 |
| BiLSTM | 65.98 ± 3.38 | 83.72 ± 3.58 | +17.75 | 0.837 | 94.44 ± 7.86 |
| GRU | 58.08 ± 4.06 | 84.89 ± 3.86 | +26.82 | 0.849 | 96.67 ± 4.97 |
| GNN | 60.06 ± 7.36 | 88.53 ± 2.38 | +28.47 | 0.885 | 98.89 ± 2.48 |
| STFT-ViT | 70.60 ± 5.59 | 83.87 ± 4.06 | +13.27 | 0.839 | 93.33 ± 7.24 |
| STFT-Swin | 69.34 ± 5.06 | 87.21 ± 4.10 | +17.87 | 0.872 | 96.67 ± 4.97 |
| EEGNet Adapted | 52.55 ± 6.33 | 86.97 ± 2.00 | +34.41 | 0.869 | 97.78 ± 3.04 |
| DeepConvNet Adapted | 50.78 ± 0.96 | 86.34 ± 2.17 | +35.56 | 0.863 | 95.56 ± 4.65 |
| ShallowConvNet Adapted | 68.05 ± 6.37 | 88.77 ± 3.38 | +20.72 | 0.888 | 94.44 ± 6.80 |
| TemporalCNN | 67.09 ± 5.52 | 88.53 ± 2.67 | +21.44 | 0.885 | 98.89 ± 2.48 |
| Top-3 Validation Ensemble | 71.23 ± 7.08 | 90.63 ± 2.99 | +19.40 | 0.906 | 96.67 ± 3.04 |

Accuracy = 4-second window accuracy, mean ± SD over the 5 subject-disjoint test folds. Recording accuracy = one decision per 150-second recording (average of its window probabilities).
