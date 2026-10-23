# Causal temporal smoothing — 2class

A decision every 4 s from the average of the last k windows of the same recording (past only). k chosen per outer fold from development out-of-fold accuracy pooled over the 12 models; test subjects play no part.

| Fold | Chosen k | Context | Dev accuracy at chosen k |
|---:|---:|---:|---:|
| 1 | 15 | 60 s | 95.97% |
| 2 | 15 | 60 s | 95.79% |
| 3 | 15 | 60 s | 95.65% |
| 4 | 15 | 60 s | 94.10% |
| 5 | 15 | 60 s | 94.77% |

| Model | Before: window % | Before: smoothed % | After: window % | After: smoothed % | After smoothed: balanced acc. % | After smoothed: macro-F1 | After: recording % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D-CNN | 62.34 ± 4.87 | 66.55 ± 8.54 | 87.78 ± 3.22 | 94.20 ± 4.26 | 94.20 | 0.942 | 94.44 ± 6.80 |
| RNN | 56.25 ± 4.85 | 59.07 ± 6.53 | 86.28 ± 2.85 | 95.80 ± 3.73 | 95.80 | 0.958 | 97.78 ± 3.04 |
| LSTM | 66.10 ± 3.71 | 74.05 ± 6.72 | 84.59 ± 3.79 | 93.63 ± 4.74 | 93.63 | 0.936 | 94.44 ± 7.86 |
| BiLSTM | 65.98 ± 3.38 | 75.08 ± 7.29 | 84.11 ± 3.57 | 92.34 ± 4.08 | 92.34 | 0.923 | 94.44 ± 7.86 |
| GRU | 58.08 ± 4.06 | 61.41 ± 5.43 | 85.20 ± 3.92 | 94.26 ± 4.69 | 94.26 | 0.942 | 95.56 ± 6.09 |
| GNN | 60.06 ± 7.36 | 63.81 ± 13.40 | 89.04 ± 2.14 | 96.70 ± 1.75 | 96.70 | 0.967 | 98.89 ± 2.48 |
| STFT-ViT | 70.60 ± 5.59 | 74.11 ± 9.68 | 84.17 ± 3.16 | 92.22 ± 5.39 | 92.22 | 0.922 | 94.44 ± 7.86 |
| STFT-Swin | 69.34 ± 5.06 | 75.02 ± 8.74 | 87.21 ± 3.45 | 96.07 ± 4.21 | 96.07 | 0.961 | 96.67 ± 4.97 |
| EEGNet Adapted | 52.55 ± 6.33 | 52.64 ± 6.92 | 86.85 ± 1.92 | 95.86 ± 1.24 | 95.86 | 0.959 | 96.67 ± 3.04 |
| DeepConvNet Adapted | 50.78 ± 0.96 | 50.09 ± 0.47 | 86.46 ± 1.86 | 94.86 ± 2.28 | 94.86 | 0.949 | 96.67 ± 3.04 |
| ShallowConvNet Adapted | 68.05 ± 6.37 | 71.89 ± 8.06 | 88.92 ± 3.48 | 95.08 ± 3.60 | 95.08 | 0.951 | 94.44 ± 5.56 |
| TemporalCNN | 67.09 ± 5.52 | 76.85 ± 7.74 | 88.20 ± 3.24 | 96.94 ± 2.91 | 96.94 | 0.969 | 98.89 ± 2.48 |
| Top-3 Validation Ensemble | 71.23 ± 7.08 | 78.11 ± 9.64 | 90.39 ± 2.61 | 97.36 ± 1.93 | 97.36 | 0.974 | 97.78 ± 3.04 |

Test accuracy at every k (mean over all single models and 5 folds; descriptive only, not used for any choice):

| k (windows) | Context | Mean test accuracy % |
|---:|---:|---:|
| 1 | 4 s | 86.57 |
| 2 | 8 s | 89.66 |
| 4 | 16 s | 91.76 |
| 8 | 32 s | 93.45 |
| 15 | 60 s | 94.83 |
