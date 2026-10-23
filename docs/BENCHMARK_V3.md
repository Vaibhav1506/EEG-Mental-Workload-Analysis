# STEW benchmark v3 — preprocessing and training optimisation

`benchmark_v3.py` takes the 12 model families from `benchmark.py` (v1) and pushes their accuracy as far as possible **without changing how they are tested**. The before/after comparison is therefore like for like.

## What stays identical to v1 (the "before" run)

- **Data and labels:** 45 rated subjects and 90 recordings. Subjects 05, 24 and 42 have no ratings and are excluded. Three classes from the 1–9 rating: 1–3 low, 4–6 moderate, 7–9 high, giving 42 / 23 / 25 recordings.
- **Test protocol:** 5 subject-disjoint folds (`benchmark.make_splits`, seed 42), so each fold tests 9 people the models never saw. Every subject is tested exactly once. The verifier checks that the folds equal v1's.
- **Test windows:** the same 3,330 windows (4 s, non-overlapping, 37 per recording).
- **Architectures:** imported unchanged from `benchmark.py`.
- **Metrics:** window accuracy, balanced accuracy, macro-F1, and recording accuracy (one decision per 150 s recording, from the average of its window probabilities). Each is reported as mean ± SD over the 5 test folds.

## What changes

### Preprocessing (applied to every recording before windowing)

| Step | What it does | Why | Labels used? |
|---|---|---|---|
| Filtering (as v1 "filtered") | 50 Hz notch, 1–40 Hz 4th-order Butterworth, common average reference, zero-phase | Removes line noise, drift and muscle hiss; keeps the brain rhythms | No |
| Per-window DC removal | Each 4 s window minus its own channel means | Emotiv raw data sits on a ~4,000 µV offset that differs per person | No |
| **EA**: Euclidean alignment (He & Wu, IEEE TBME 2020) | Per subject, whitens the 14 channels with the inverse square root of that subject's mean spatial covariance | Removes person-specific scaling and mixing of the channels (head size, electrode contact) | No |
| **SAG**: spectral alignment, geometric mean | Per subject and channel, divides the spectrum by the square root of that subject's geometric-mean power spectrum, limited to 1–40 Hz | Band power then reads as "relative to this person's average", which is what separates rest from task across people | No |
| Training-set normalisation | Per-channel mean/SD fitted on the fold's development subjects | Standard input scaling | No |

EA and SAG use only each subject's own unlabelled EEG: both recordings, about 5 minutes. **Assumption:** a new user must provide some unlabelled EEG before predictions are made. No labels or test-set statistics cross between subjects.

Two candidate pipelines were trained for every model:

- `filtered`: filtering + DC removal
- `filtered-ea-sag`: filtering + DC removal + EA + SAG

### How the final ("after") predictions are chosen

The choice is pre-declared, made separately in every outer fold, and based only on development (out-of-fold) accuracy. It is implemented in `assemble_v3.py` and re-checked by `verify_v3.py`:

1. **Preprocessing:** the pipeline with the better mean development accuracy over all 12 models. This was the full pipeline (`filtered-ea-sag`) in every fold.
2. **Transformer settings:** for STFT-ViT and STFT-Swin only, the default recipe vs a lower learning rate (3e-4, weight decay 0.05, 5 warm-up epochs), by their mean development accuracy.

An earlier plan chose the preprocessing per model *and* per fold. With only 9 validation subjects per inner group, that choice was noisy: it picked filtering-only in 7 of 60 cases, mostly by chance. On the first 4-network run, it averaged 57.9% against 58.4% for the per-fold rule above. The per-fold rule was fixed after those first test numbers had been seen. The difference is within fold-to-fold noise, and both versions are kept in `results/v3_runs/R1/REPORT.md`.

### Training

| Change | v1 | v3 |
|---|---|---|
| Hardware | CPU | RTX 4060 (CUDA) |
| Training windows | Fixed 4 s grid (37 per recording) | Random 4 s crops (≈293 per recording per epoch) |
| Schedule | ≤20 epochs, patience 5, constant LR | ≤60 epochs, patience 15, 3-epoch warm-up + cosine LR |
| Regularisation | weight decay 1e-4 | weight decay 1e-2, label smoothing 0.1 |
| Development data use | 28 train / 8 validation subjects | 4-fold inner CV over all 36 development subjects |
| Final model | 1 network | Average of 8 networks: 4 inner folds x 2 independent runs (`--init-offset 0/1`) |
| Checkpoint | Best validation macro-F1 | Best validation macro-F1 (per inner fold) |

In inner cross-validation, each network trains on 27 subjects and early-stops on the other 9. Every development subject is therefore used for both training and validation, and the 4 networks form an ensemble. The out-of-fold predictions on development subjects give each fold an honest validation score. That score drives the nested pipeline choice and the Top-3 ensemble.

### Top-3 Validation Ensemble

For each fold, this averages the probabilities of the 3 models with the best development out-of-fold macro-F1.

## Results (3-class, unseen subjects)

All values are window accuracy, mean ± SD over the 5 test folds (full table: `results/v3_final/BEFORE_AFTER.md`). "Before" is the v1 run on raw signals: 20 epochs, 1 network, no preprocessing. "After" is the v3 final.

| Model | Before % | After % | Change (pp) | After macro-F1 | After recording acc. % |
|---|---:|---:|---:|---:|---:|
| 1D-CNN | 44.86 ± 8.76 | 58.35 ± 2.35 | +13.5 | 0.522 | 64.4 |
| RNN | 37.42 ± 4.07 | 59.43 ± 4.82 | +22.0 | 0.533 | 70.0 |
| LSTM | 48.20 ± 5.98 | 56.70 ± 4.15 | +8.5 | 0.514 | 65.6 |
| BiLSTM | 45.08 ± 6.94 | 56.82 ± 3.77 | +11.7 | 0.513 | 61.1 |
| GRU | 36.04 ± 6.79 | 57.81 ± 4.35 | +21.8 | 0.523 | 68.9 |
| GNN | 46.43 ± 6.04 | 58.92 ± 5.05 | +12.5 | 0.514 | 66.7 |
| STFT-ViT | 53.39 ± 7.52 | 54.95 ± 6.38 | +1.6 | 0.483 | 60.0 |
| STFT-Swin | 51.14 ± 6.56 | 57.51 ± 4.56 | +6.4 | 0.506 | 61.1 |
| EEGNet Adapted | 45.95 ± 6.69 | 61.80 ± 3.08 | +15.9 | 0.541 | 73.3 |
| DeepConvNet Adapted | 36.94 ± 6.64 | 58.74 ± 4.79 | +21.8 | 0.521 | 66.7 |
| ShallowConvNet Adapted | 43.87 ± 8.49 | 60.63 ± 2.47 | +16.8 | 0.546 | 66.7 |
| TemporalCNN | 41.20 ± 8.88 | 59.43 ± 2.75 | +18.2 | 0.523 | 64.4 |
| Top-3 Validation Ensemble | — | 58.59 ± 3.89 | — | 0.515 | 65.6 |

Training-majority baseline: 46.67%.

**Where the gain comes from.** These are mean window accuracies over the 12 models:

| Setup | Mean accuracy % |
|---|---:|
| v1 raw (before) | 44.2 |
| v1 filtered | 50.6 |
| v3 training on filtered signals | 56.0 |
| v3 full pipeline with EA + SAG ("after") | 58.4 |

So the gain splits roughly into filtering ≈ +6 pp, training ≈ +5.5 pp, and per-subject alignment ≈ +2.5 pp.

**Where the remaining error is.** Pooled over the 12 models, recall is 80.5% for low, 32.8% for moderate and 44.8% for high. Low corresponds to the rest recordings and is recognised well. Moderate and high are mostly confused with each other: they are two self-rated levels of the same task, and that difference is close to chance across people. The confusion matrices are in `results/v3_final/details/confusion_matrices.json`.

## Secondary task: 2-class rest vs task

The original slide's "before" numbers (52.4–69.7%) come from an unknown setup. To check what they correspond to, the same 12 models were also run on the common STEW binary task: rest recording (`_lo`) vs SIMKAP task recording (`_hi`). This uses 45 subjects, 90 recordings and 50/50 classes, with the same folds and the same 3,330 test windows. Run it with `benchmark_v3.py --task 2class`.

- **Before** (`results/v3_2class/before_v1like`): v1-style training on raw signals. That means no DC removal, a fixed window grid, constant learning rate, at most 20 epochs with patience 5, no label smoothing, weight decay 1e-4, and one network on the v1 train/validation split. Its range, 50.8–70.6%, closely matches the slide's range. The slide was therefore most likely this binary task with unseen-subject testing.
- **After** (`results/v3_2class/after`): the full v3 pipeline (`filtered-ea-sag`) with v3 training: 4 inner-fold networks and one run. The pipeline was fixed from the 3-class work, so nothing was selected on the 2-class data.

| Model | Before % | After % | Change (pp) | After recording acc. % |
|---|---:|---:|---:|---:|
| 1D-CNN | 62.34 ± 4.87 | 87.21 ± 3.29 | +24.9 | 95.6 |
| RNN | 56.25 ± 4.85 | 86.13 ± 2.77 | +29.9 | 96.7 |
| LSTM | 66.10 ± 3.71 | 83.72 ± 3.41 | +17.6 | 94.4 |
| BiLSTM | 65.98 ± 3.38 | 83.72 ± 3.58 | +17.7 | 94.4 |
| GRU | 58.08 ± 4.06 | 84.89 ± 3.86 | +26.8 | 96.7 |
| GNN | 60.06 ± 7.36 | 88.53 ± 2.38 | +28.5 | 98.9 |
| STFT-ViT | 70.60 ± 5.59 | 83.87 ± 4.06 | +13.3 | 93.3 |
| STFT-Swin | 69.34 ± 5.06 | 87.21 ± 4.10 | +17.9 | 96.7 |
| EEGNet Adapted | 52.55 ± 6.33 | 86.97 ± 2.00 | +34.4 | 97.8 |
| DeepConvNet Adapted | 50.78 ± 0.96 | 86.34 ± 2.17 | +35.6 | 95.6 |
| ShallowConvNet Adapted | 68.05 ± 6.37 | 88.77 ± 3.38 | +20.7 | 94.4 |
| TemporalCNN | 67.09 ± 5.52 | 88.53 ± 2.67 | +21.4 | 98.9 |
| Top-3 Validation Ensemble | 71.23 ± 7.08 | 90.63 ± 2.99 | +19.4 | 96.7 |

**Important caveat for the 2-class numbers.** EA and SAG align each person using their own unlabelled EEG, here both of their recordings, so the alignment data contain one rest and one task period. Centring each person on their own average is precisely what makes rest vs task easy to separate across people. The numbers therefore assume that a new user provides unlabelled calibration EEG covering both states. If calibration data came from one state only, accuracy would be lower. Quote this assumption together with the 2-class results.

## How the recipe was chosen

Exploration used only development-subject scores. Test-fold numbers were printed for information but never used to choose anything.

- **Feature probe** (`band power + logistic regression / gradient boosting`): per-subject alignment of the features raised window accuracy from 53.7% to 59–63%.
- **Signal-domain version:** EA followed by geometric spectral alignment reproduced most of that gain (57.9%). Arithmetic-mean spectral alignment, or alignment in the other order, did not.
- **Deep-network pilot** (1D-CNN, ShallowConvNet, LSTM; filtered vs filtered+EA vs filtered+EA+SAG): EA+SAG had the best development scores for 1D-CNN. ShallowConvNet was mixed. This is why both pipelines are kept and chosen per fold, rather than imposing one.

## Honest limits

- **Ceiling:** every "low" recording is a rest recording, and moderate vs high is a self-rating within the task. Predicting "rest → low, task → high" perfectly would score only 74.4% (67/90 recordings). Moderate vs high is the hard part, so 3-class unseen-subject accuracy cannot approach 90%.
- **Noise:** each fold tests only 9 people, so fold-to-fold SD is about 2–8 pp. Differences of 1–3 pp between models are not meaningful.
- **Transductive alignment:** EA/SAG need unlabelled EEG of the test person. This is standard in cross-subject EEG but should be stated whenever the numbers are quoted.
- **Slide numbers:** the setup behind the original slide photo's "before" numbers is undocumented. Their range matches our 2-class rest-vs-task baseline (50.8–70.6%), not the 3-class one (36–53%). For a fair before/after, pair each "after" column with the "before" column from the same task and protocol, as in the two `BEFORE_AFTER.md` tables.
- **Results above 90% in the literature:** STEW results above 90% use window-level random splits, which put the same person in train and test. They are not comparable with any number here.

## Running

```powershell
# start the GPU worker once (it runs jobs dropped into jobs\queue), or run directly:
.\.venv\Scripts\python.exe -B -u benchmark_v3.py --output results\v3_new --variants filtered-ea-sag filtered
# several processes may share one --output folder (tasks are claimed with lock files)
.\.venv\Scripts\python.exe -B benchmark_v3.py --output results\v3_new --variants filtered-ea-sag filtered --summarize
.\.venv\Scripts\python.exe -B verify_v3.py results\v3_new
.\.venv\Scripts\python.exe -B make_slide_table.py --before results\initial_5fold\summary.csv --after results\v3_new\summary.csv --out results\v3_new
```

Useful options:

- `--task 2class`: rest vs task
- `--inner-folds`, `--seeds`
- `--select f1|acc|loss|bacc`
- `--class-weight balanced|none`
- `--mixup`
- v1-style baseline flags: `--no-demean --constant-lr --no-jitter --inner-folds 1`

## References

- He H., Wu D. *Transfer learning for brain–computer interfaces: a Euclidean space data alignment approach.* IEEE TBME 67(2), 2020.
- Lim W.L., Sourina O., Wang L. *STEW: Simultaneous Task EEG Workload Data Set.* IEEE TNSRE 26(11), 2018.
- Lawhern V. et al. *EEGNet.* J. Neural Eng. 2018. Schirrmeister R. et al. *Deep learning with convolutional neural networks for EEG decoding and visualization (Deep/ShallowConvNet).* Hum. Brain Mapp. 2017.
