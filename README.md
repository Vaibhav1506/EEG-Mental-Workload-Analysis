# EEG Mental Workload Analysis

Classify STEW EEG recordings into low, moderate, and high workload, and compare 12 compact model families on raw and filtered signals.

## Folder guide

```text
benchmark.py                 Run a new benchmark
verify_benchmark.py          Check saved experiment results
requirements.txt             Tested benchmark dependencies
data/stew/                   EEG recordings and ratings
docs/BENCHMARK.md            Full protocol and model definitions
References/                  Research paper, notes, original dependency list
tests/                       Checks for labels, subject separation, and models
results/initial_5fold/
    REPORT.md                Read the measured results here
    comparison.png           Model comparison chart
    summary.csv              Compact results table
    details/                 Predictions, training history, and fold metrics
    audit/                   Experiment records and executed source snapshot
```

## Read the completed benchmark

Open [the results report](results/initial_5fold/REPORT.md) or [the comparison chart](results/initial_5fold/comparison.png). The initial run covers 45 rated subjects, five subject-disjoint test folds, 12 model families, and both signal conditions. These are initial results with a maximum of 20 epochs per fit.

## Run from the project folder

The existing Python environment is in the parent folder's `VE` directory. In PowerShell:

```powershell
& '..\VE\Scripts\python.exe' -B -m unittest discover -s tests -v
& '..\VE\Scripts\python.exe' -B verify_benchmark.py
& '..\VE\Scripts\python.exe' -B -u benchmark.py --output results/new_experiment
```

Choose a new output folder for each experiment. Existing results are protected against overwriting. See [the full protocol](docs/BENCHMARK.md) for options and examples. For a different Python environment, install `requirements.txt`; the dataset must also be present in `data/stew/`.

## Are the JSON files needed?

Python files contain the program. The recordings and `requirements.txt` supply its data and dependencies. Training generates experiment records as JSON; they are not Python code and are not needed to start a fresh run.

Keep the records in each result folder's `audit/` directory when retaining an experiment:

- `run_metadata.json`: settings, software versions, timing, and source fingerprint.
- `subject_splits.json`: which subjects were used for training, validation, and testing.
- `dataset_manifest.json`: recording labels and data fingerprints.
- `training_normalization.json`: fitted training means and standard deviations.
- `confusion_matrices.json`: class-by-class prediction counts.
- `verification.json`: the latest verification outcome; it can be regenerated.

The verifier reads the settings, subject splits, manifest, and confusion matrices to check the predictions. Deleting those files would leave the headline scores without their audit trail. `audit/benchmark_source.py` preserves the exact code used for that experiment, so later code changes do not erase its provenance.

The original paper and notes are in `References/`. The earlier, broader dependency list is preserved there as `requirements_original.txt`; the root `requirements.txt` contains the versions tested for this benchmark.
