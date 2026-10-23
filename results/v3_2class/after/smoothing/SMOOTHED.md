# Causal temporal smoothing — 2class

A decision every 4 s from the average of the last k windows of the same recording (past only). k chosen per outer fold from development out-of-fold accuracy pooled over the 12 models; test subjects play no part.

| Fold | Chosen k | Context | Dev accuracy at chosen k |
|---:|---:|---:|---:|
| 1 | 15 | 60 s | 95.38% |
| 2 | 15 | 60 s | 95.09% |
| 3 | 15 | 60 s | 95.02% |
| 4 | 15 | 60 s | 93.76% |
| 5 | 15 | 60 s | 94.09% |

| Model | Before: window % | Before: smoothed % | After: window % | After: smoothed % | After smoothed: balanced acc. % | After smoothed: macro-F1 | After: recording % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D-CNN | 62.34 ± 4.87 | 66.55 ± 8.54 | 87.21 ± 3.29 | 93.93 ± 4.16 | 93.93 | 0.939 | 95.56 ± 4.65 |
| RNN | 56.25 ± 4.85 | 59.07 ± 6.53 | 86.13 ± 2.77 | 95.29 ± 4.44 | 95.29 | 0.953 | 96.67 ± 4.97 |
| LSTM | 66.10 ± 3.71 | 74.05 ± 6.72 | 83.72 ± 3.41 | 92.19 ± 4.46 | 92.19 | 0.921 | 94.44 ± 7.86 |
| BiLSTM | 65.98 ± 3.38 | 75.08 ± 7.29 | 83.72 ± 3.58 | 92.31 ± 4.30 | 92.31 | 0.923 | 94.44 ± 7.86 |
| GRU | 58.08 ± 4.06 | 61.41 ± 5.43 | 84.89 ± 3.86 | 94.02 ± 4.23 | 94.02 | 0.940 | 96.67 ± 4.97 |
| GNN | 60.06 ± 7.36 | 63.81 ± 13.40 | 88.53 ± 2.38 | 96.82 ± 2.00 | 96.82 | 0.968 | 98.89 ± 2.48 |
| STFT-ViT | 70.60 ± 5.59 | 74.11 ± 9.68 | 83.87 ± 4.06 | 92.10 ± 5.05 | 92.10 | 0.921 | 93.33 ± 7.24 |
| STFT-Swin | 69.34 ± 5.06 | 75.02 ± 8.74 | 87.21 ± 4.10 | 95.32 ± 5.05 | 95.32 | 0.953 | 96.67 ± 4.97 |
| EEGNet Adapted | 52.55 ± 6.33 | 52.64 ± 6.92 | 86.97 ± 2.00 | 95.26 ± 2.15 | 95.26 | 0.953 | 97.78 ± 3.04 |
| DeepConvNet Adapted | 50.78 ± 0.96 | 50.09 ± 0.47 | 86.34 ± 2.17 | 94.32 ± 2.50 | 94.32 | 0.943 | 95.56 ± 4.65 |
| ShallowConvNet Adapted | 68.05 ± 6.37 | 71.89 ± 8.06 | 88.77 ± 3.38 | 94.08 ± 4.66 | 94.08 | 0.941 | 94.44 ± 6.80 |
| TemporalCNN | 67.09 ± 5.52 | 76.85 ± 7.74 | 88.53 ± 2.67 | 97.00 ± 2.65 | 97.00 | 0.970 | 98.89 ± 2.48 |
| Top-3 Validation Ensemble | 71.23 ± 7.08 | 78.11 ± 9.64 | 90.63 ± 2.99 | 97.24 ± 1.92 | 97.24 | 0.972 | 96.67 ± 3.04 |

Test accuracy at every k (mean over all single models and 5 folds; descriptive only, not used for any choice):

| k (windows) | Context | Mean test accuracy % |
|---:|---:|---:|
| 1 | 4 s | 86.32 |
| 2 | 8 s | 89.28 |
| 4 | 16 s | 91.51 |
| 8 | 32 s | 93.19 |
| 15 | 60 s | 94.39 |
