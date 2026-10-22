# STEW initial benchmark

Status: COMPLETE. 120 model/condition/fold runs completed.

This is an initial comparison of compact, explicitly defined model adaptations. It does not reproduce the screenshot's unidentified experiment.

45 rated subjects; 90 recordings; 3330 windows; three rating classes (1-3, 4-6, 7-9).
5 subject-disjoint test folds; separate validation subjects within each development fold; one initialization per fold.
Four-second windows, stride 4 seconds. Maximum 20 epochs; early stopping after 5 non-improving validation macro-F1 epochs.
Raw means no additional signal filtering. Both conditions use normalization fitted exclusively on the training subjects.
Filtered means 50 Hz notch (Q=30), fourth-order 1-40 Hz Butterworth bandpass, and average rereferencing on each continuous recording. Offline zero-phase filtering; ASR/ICA is not included.

| Model | Raw accuracy | Filtered accuracy | Change (pp) | Filtered macro-F1 |
|---|---:|---:|---:|---:|
| 1D-CNN | 44.86% +/- 8.76 | 52.22% +/- 6.88 | +7.36 | 0.429 |
| RNN | 37.42% +/- 4.07 | 41.95% +/- 9.69 | +4.53 | 0.372 |
| LSTM | 48.20% +/- 5.98 | 48.95% +/- 6.92 | +0.75 | 0.428 |
| BiLSTM | 45.08% +/- 6.94 | 49.76% +/- 9.38 | +4.68 | 0.434 |
| GRU | 36.04% +/- 6.79 | 50.84% +/- 7.16 | +14.80 | 0.456 |
| GNN | 46.43% +/- 6.04 | 44.38% +/- 9.63 | -2.04 | 0.368 |
| STFT-ViT | 53.39% +/- 7.52 | 55.17% +/- 5.49 | +1.77 | 0.497 |
| STFT-Swin | 51.14% +/- 6.56 | 55.65% +/- 7.44 | +4.50 | 0.490 |
| EEGNet Adapted | 45.95% +/- 6.69 | 48.41% +/- 8.41 | +2.46 | 0.407 |
| DeepConvNet Adapted | 36.94% +/- 6.64 | 50.63% +/- 9.84 | +13.69 | 0.395 |
| ShallowConvNet Adapted | 43.87% +/- 8.49 | 57.75% +/- 4.07 | +13.87 | 0.508 |
| TemporalCNN | 41.20% +/- 8.88 | 50.93% +/- 3.73 | +9.73 | 0.442 |

## Interpretation

Accuracy is measured on windows. The +/- value is sample standard deviation across test folds, not a confidence interval or variation across random initializations.
Subjects are disjoint across train/validation/test within each fold; every rated subject is tested once per model and condition.
Folds share training data, so their results are not independent replications. No statistical significance claim is made.
Recording results average the predicted class probabilities across windows in each recording. They do not measure instantaneous workload labels.
The majority-class baseline uses only the training class counts for each fold. A naive class-majority baseline is necessary because labels are imbalanced.
Test folds are reserved for evaluation. Checkpoint selection uses validation macro-F1; no test tuning or architecture search was performed.
The fixed small training budget, compact architectures, and single seed per fold limit claims about the best attainable model accuracy. More training and nested tuning may change rankings.
STEW labels are recording-level self-ratings inherited by each window. Many windows do not constitute many independent subjects.
The original STEW paper uses SVR and NCA, overlapping windows, and a particular subject holdout. This benchmark uses direct three-class neural classifiers and a different split protocol; scores are not directly interchangeable.
The photograph does not reveal its dataset, split, preprocessing, adaptations, or meaning of +/-; its numbers are reference text, not measured benchmark results.

See docs/BENCHMARK.md in the project folder for model definitions, reproduction commands, and source links.

## Verification and observed results

All audit checks passed: 120 fits, 79,920 window predictions, 2,160 recording predictions, and 90 input file hashes. Metrics and fold statistics were independently recomputed from the saved predictions.

Training-majority baseline accuracy: 46.67% +/- 3.04 percentage points across folds.

Highest mean window accuracy in this run: ShallowConvNet Adapted (filtered), 57.75% +/- 4.07 percentage points. Highest mean filtered accuracy: ShallowConvNet Adapted (filtered), 57.75% +/- 4.07 percentage points. These rankings are descriptive and do not establish a statistically significant advantage.

30 of 120 fits reached the 20-epoch cap; 8 selected their best checkpoint at that cap. This supports treating the run as an initial comparison with a limited training budget.
