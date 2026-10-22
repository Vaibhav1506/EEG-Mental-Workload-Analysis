# STEW benchmark

This project now contains a runnable initial comparison of the 12 model families visible in the supplied photograph. These are compact adaptations defined in `benchmark.py`, not reproductions of unknown implementations in the photograph. Results should be described as an initial benchmark, not a tuned state-of-the-art comparison.

## Data and labels

- Dataset: local `data/stew`, sampled at 128 Hz, 14 channels, 19,200 samples (150 seconds) per recording.
- Use the 45 subjects in `ratings.txt`; exclude subjects 05, 24, and 42 because ratings are absent. There are 90 labeled recordings.
- Use the rating in column 2 for the `lo` recording and column 3 for the `hi` recording. Classes are ratings 1-3 = low, 4-6 = moderate, 7-9 = high. The filenames identify recording conditions, not the three-class labels.
- There are 42 low, 23 moderate, and 25 high recordings.
- Use four-second windows (512 samples), stride four seconds for this first run: 37 windows per recording, 3,330 windows total. The remaining two seconds of each recording are unused, identically in both conditions.
- The original paper uses a one-second stride; this runner supports `--stride-seconds 1` as a separate experiment. Do not mix these configurations in a results table.

## Evaluation

Five shuffled subject folds, seed 42: nine test subjects and 36 development subjects in each fold. Hold out eight of the 36 development subjects for validation; train on the other 28. The same identities are used for all models and both conditions. Each subject appears in a test fold exactly once.

Keep complete recordings and all windows from one subject in the same partition within a fold. Normalization uses only training subjects, separately for each condition: subtract per-channel training mean and divide by per-channel training standard deviation. Validation and test data never fit the normalizer. Filtering is fixed, nonlearned, and applied independently to each recording. Overlapping windows, when enabled, stay inside their subject partition.

Use direct three-class cross-entropy, class weights from training counts only, AdamW (learning rate 0.001; weight decay 0.0001), batch 128, gradient norm limit 1, at most 20 epochs, and early stopping after five epochs without improvement in validation macro-F1. Restore the best validation checkpoint before predicting the test data. All models receive the same cap and stopping rule, but no model-specific hyperparameter tuning. One initialization per fold; more seeds and nested tuning are needed before making strong model-ranking claims.

Report window accuracy, balanced accuracy, macro-F1, per-recording accuracy, confusion matrices, parameter count, training time, and inference time. Report mean and sample SD across folds. SD across overlapping training folds is descriptive, not a confidence interval. Recording predictions average probabilities across the recording's windows. The training-majority baseline is computed for every fold.

## Paired signal conditions

- **Raw:** supplied recording samples with training-fitted normalization; no additional filtering.
- **Filtered:** 50 Hz notch with Q=30, fourth-order Butterworth 1-40 Hz bandpass, and rereferencing to the instantaneous average of the 14 channels. Filter continuous recordings using offline zero-phase filtering, then create matching windows and apply training-fitted normalization.

The filtered condition does not include ASR or ICA and does not claim to reproduce the original paper's full preprocessing pipeline. The original STEW paper includes ASR. Neither condition implies that the supplied recordings are untouched by the acquisition hardware. Filtering is offline; inference timings cover the network prediction pass, not preprocessing or acquisition latency.

## Model definitions

| Family | Definition in this benchmark |
|---|---|
| 1D-CNN | Temporal convolutions 32/64/64, kernels 9/7/5; batch normalization, ELU, average pooling, dropout, dense head. |
| RNN | Average pool time by four, one tanh RNN layer with 32 hidden units; mean and final timestep features. |
| LSTM | Same temporal pooling, one 32-unit LSTM; mean and final timestep features. |
| BiLSTM | Same pooling, 32 hidden units per direction; mean and final timestep features. |
| GRU | Same pooling, one 32-unit GRU; mean and final timestep features. |
| GNN | Each EEG channel is a graph node. Shared temporal convolutions encode nodes; two four-head graph attention layers on a complete graph learn edge weights; learned channel embeddings; mean node readout. No anatomical graph is claimed. |
| STFT-ViT | Hann STFT: FFT 64, hop 16, no centered padding; log(1+power), bilinear resizing 33x29 to 32x32; 4x4 patches, embedding 32, 64 tokens, learned positions, two transformer layers, four heads, feedforward width 64. |
| STFT-Swin | Same STFT input; 4x4 patch embedding; an 8x8 stage with regular and shifted 4x4 windows, relative position bias and cyclic boundary mask; 2x2 patch merging; 4x4 stage with 64 features; mean readout. |
| EEGNet Adapted | 8 temporal filters of length 63; depth multiplier 2 and channel-spanning spatial convolution; depthwise length-15 and pointwise separable convolution; pooling 4 then 8; dropout 0.5. No original max-norm constraints. |
| DeepConvNet Adapted | Temporal/spatial block and three temporal blocks, widths 16/32/64/128, length-10 temporal kernels, max pooling 3, ELU and dropout 0.5. Reduced widths relative to the original architecture. |
| ShallowConvNet Adapted | 20 length-25 temporal filters and spatial filters, batch normalization, square, average pool 75 with stride 15, clipped log and dropout 0.5. Reduced width relative to the original architecture. |
| TemporalCNN | Pool time by four; 32 channels; residual causal convolution blocks with dilations 1/2/4, two length-5 convolutions per block; mean readout. |

STFT generation is deterministic per window and computed from the corresponding condition's normalized signal. It does not fit global spectrum statistics. The model families have different parameter counts and inductive biases; this is an equal training-budget comparison, not an equal-capacity comparison.

## Run on this computer

From this project directory, in PowerShell:

```powershell
& '..\VE\Scripts\python.exe' -B -m unittest discover -s tests -v
& '..\VE\Scripts\python.exe' -B -u benchmark.py --output results/new_experiment
& '..\VE\Scripts\python.exe' -B verify_benchmark.py
```

The existing shared Python environment already contains the needed packages. Tested benchmark versions are in the root `requirements.txt`; the earlier dependency list is preserved in `References/requirements_original.txt`.

To run only the three recommended starting models, use a new output folder:

```powershell
& '..\VE\Scripts\python.exe' -B -u benchmark.py --models '1D-CNN' 'EEGNet Adapted' 'LSTM' --output results/three_models
```

For a longer experiment with overlapping windows, use a distinct output folder:

```powershell
& '..\VE\Scripts\python.exe' -B -u benchmark.py --stride-seconds 1 --epochs 100 --patience 15 --output results/longer_5fold
```

This longer setting is an example and is not part of the initial results. The runner refuses nonempty output folders to protect earlier experiments. It does not support resuming partially completed runs; use a new folder after interruption.

## Retained results

The readable report, chart, and compact summary are directly inside `results/initial_5fold/`. Detailed fold metrics, predictions, training histories, and the baseline table are in `details/`. JSON records are grouped in `audit/`, together with the exact executed Python source snapshot. The snapshot preserves the original experiment's source fingerprint after the working code is reorganized or updated.

The verifier checks input/source hashes, rating labels, subject coverage and separation, prediction probabilities, validation checkpoint selection, and independently recomputes metrics, recording aggregation, confusion matrices, and fold statistics. It writes `audit/verification.json` only after all checks pass. Historical run arguments retain their original paths; a project-relative dataset location lets verification find the moved data without rewriting experimental settings.

Training seconds include the training loop and validation passes. Network inference timing is a warm-start CPU prediction pass with batches up to 128. Timings depend on hardware and load.

There is one dataset copy in `data/stew/`. No saved model checkpoint, intermediate spectrogram, persistent array cache, or duplicate JSON summary is needed. Keep the retained audit records with the results; the original reference paper, notes, and earlier dependency list are in `References/`.

## References and comparability

- STEW: Lim, Sourina and Wang, *STEW: Simultaneous Task EEG Workload Data Set* (2018), [DOI](https://doi.org/10.1109/TNSRE.2018.2872924). The local reference PDF describes ASR preprocessing, an SVR/NCA baseline, overlapping windows, and a particular 36/9 subject holdout. This benchmark uses direct classifiers and five test folds, so it does not reproduce that numerical baseline.
- Lawhern et al., *EEGNet* (2018), [paper](https://arxiv.org/abs/1611.08024), and the [authors' EEG model implementations](https://github.com/vlawhern/arl-eegmodels), which also link the original DeepConvNet/ShallowConvNet publication. The implementations here are independently written compact PyTorch adaptations.
- Lee et al., *Risk of data leakage in estimating the diagnostic performance of a deep-learning-based computer-aided system for psychiatric disorders* (2023), [paper](https://doi.org/10.1038/s41598-023-43542-8). This motivates retaining subject separation when segmenting EEG.
- Liu et al., *Swin Transformer* (2021), [paper](https://arxiv.org/abs/2103.14030). The STFT-Swin baseline uses a small two-stage adaptation, not the original image-scale configuration or pretrained weights.

The photograph supplies model names and visible before-preprocessing percentages only. Its experiment identity and settings are unknown. No number from that photograph is inserted into measured results, and differences from those numbers are not used as evidence of improvement.
