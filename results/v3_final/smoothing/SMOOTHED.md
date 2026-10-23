# Causal temporal smoothing — 3class

A decision every 4 s from the average of the last k windows of the same recording (past only). k chosen per outer fold from development out-of-fold accuracy pooled over the 12 models; test subjects play no part.

| Fold | Chosen k | Context | Dev accuracy at chosen k |
|---:|---:|---:|---:|
| 1 | 15 | 60 s | 71.65% |
| 2 | 15 | 60 s | 67.42% |
| 3 | 15 | 60 s | 69.12% |
| 4 | 15 | 60 s | 65.15% |
| 5 | 15 | 60 s | 69.06% |

| Model | Before: window % | Before: smoothed % | After: window % | After: smoothed % | After smoothed: balanced acc. % | After smoothed: macro-F1 | After: recording % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D-CNN | 44.86 ± 8.76 | 47.72 ± 11.32 | 58.35 ± 2.35 | 62.10 ± 3.98 | 55.89 | 0.548 | 64.44 ± 4.97 |
| RNN | 37.42 ± 4.07 | 39.64 ± 6.32 | 59.43 ± 4.82 | 66.37 ± 10.81 | 58.66 | 0.581 | 70.00 ± 10.83 |
| LSTM | 48.20 ± 5.98 | 54.05 ± 9.34 | 56.70 ± 4.15 | 63.84 ± 8.82 | 59.26 | 0.565 | 65.56 ± 12.04 |
| BiLSTM | 45.08 ± 6.94 | 49.34 ± 9.08 | 56.82 ± 3.77 | 60.57 ± 5.81 | 55.69 | 0.524 | 61.11 ± 7.86 |
| GRU | 36.04 ± 6.79 | 37.48 ± 8.56 | 57.81 ± 4.35 | 64.41 ± 9.25 | 60.35 | 0.568 | 68.89 ± 10.09 |
| GNN | 46.43 ± 6.04 | 50.15 ± 7.48 | 58.92 ± 5.05 | 64.29 ± 3.85 | 57.66 | 0.553 | 66.67 ± 5.56 |
| STFT-ViT | 53.39 ± 7.52 | 54.08 ± 9.92 | 54.95 ± 6.38 | 58.74 ± 7.32 | 52.74 | 0.512 | 60.00 ± 10.69 |
| STFT-Swin | 51.14 ± 6.56 | 55.56 ± 9.93 | 57.51 ± 4.56 | 62.07 ± 4.55 | 55.42 | 0.538 | 61.11 ± 7.86 |
| EEGNet Adapted | 45.95 ± 6.69 | 50.03 ± 8.06 | 61.80 ± 3.08 | 68.53 ± 4.66 | 61.41 | 0.602 | 73.33 ± 7.24 |
| DeepConvNet Adapted | 36.94 ± 6.64 | 39.70 ± 9.36 | 58.74 ± 4.79 | 65.32 ± 5.14 | 60.20 | 0.580 | 66.67 ± 5.56 |
| ShallowConvNet Adapted | 43.87 ± 8.49 | 49.46 ± 10.24 | 60.63 ± 2.47 | 66.19 ± 2.29 | 60.20 | 0.588 | 66.67 ± 5.56 |
| TemporalCNN | 41.20 ± 8.88 | 43.12 ± 11.20 | 59.43 ± 2.75 | 64.74 ± 4.50 | 58.98 | 0.558 | 64.44 ± 4.97 |
| Top-3 Validation Ensemble | — | — | 58.59 ± 3.89 | 63.78 ± 5.07 | 57.87 | 0.552 | 65.56 ± 2.48 |

Test accuracy at every k (mean over all single models and 5 folds; descriptive only, not used for any choice):

| k (windows) | Context | Mean test accuracy % |
|---:|---:|---:|
| 1 | 4 s | 58.42 |
| 2 | 8 s | 60.47 |
| 4 | 16 s | 62.02 |
| 8 | 32 s | 62.99 |
| 15 | 60 s | 63.93 |
