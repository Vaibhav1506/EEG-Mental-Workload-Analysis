# EEG Mental Workload Classification (STEW)

This project classifies STEW EEG recordings into low, moderate and high mental workload with 12 deep-learning model families. Every model is tested on people it has never seen.

## Results to read first

Round 2 results are final. Round 1 is kept for reference in `results/v3_final/` and `results/v3_2class/BEFORE_AFTER.md`.

| What | Where |
|---|---|
| **3-class before vs after, per 4-second window** | `results/v3_final2/BEFORE_AFTER.md`, `results/v3_final2/before_after.png` |
| **3-class before vs after, decision from the last 60 s** | `results/v3_final2/smoothing/SMOOTHED.md`, `results/v3_final2/smoothing/before_after_smoothed.png` |
| **2-class (rest vs task) before vs after, per window** | `results/v3_2class/final/BEFORE_AFTER.md`, `results/v3_2class/final/before_after.png` |
| **2-class, decision from the last 60 s** | `results/v3_2class/final/smoothing/SMOOTHED.md`, `results/v3_2class/final/smoothing/before_after_smoothed.png` |
| Full results, every pipeline and metric | `results/v3_final2/REPORT.md`, `results/v3_2class/final/REPORT.md` |
| Baseline ("before") runs | `results/initial_5fold/` (3-class), `results/v3_2class/before_v1like/` (2-class) |
| What was changed and why | `Docs/BENCHMARK_V3.md` |
| Baseline protocol and model definitions | `Docs/BENCHMARK.md` |

## Folder guide

```text
Dataset/                 STEW recordings (subXX_lo/hi.txt) and ratings.txt
benchmark.py             v1: data, labels, folds, the 12 architectures (baseline run)
benchmark_v3.py          v3: preprocessing (filtering, EA, spectral alignment) + optimised training
verify_benchmark.py      audits a v1 result folder
verify_v3.py             audits a v3 result folder (recomputes every metric from saved predictions)
make_slide_table.py      builds the before/after table and chart
jobs/gpu_worker.py       runs queued training jobs on the GPU (see below)
results/initial_5fold/   v1 baseline results
results/v3_final2/       FINAL 3-class results, round 2 (assembled from v3_runs; REPORT, tables, charts, smoothing/, audit)
results/v3_final/        round-1 3-class results (kept for reference)
results/v3_runs/         the v3 training runs the finals are built from (default x2, transformer low-LR x2, LSTM/BiLSTM low-LR x2)
results/v3_2class/       2-class rest vs task: before_v1like (baseline), after + after_init1 (runs), final (FINAL, 8 networks)
assemble_v3.py           averages replicate runs and applies the pre-declared final choice
smooth_v3.py             causal temporal smoothing: decision every 4 s from the last k windows (k chosen on development data)
outputs/cache/           cached recordings (rebuilt automatically if deleted)
Docs/                    protocol documents and the STEW dataset paper
References/              original paper and notes
tests/                   unit tests for labels, subject separation and models
DELETE/                  superseded files kept only until you delete them
```

## Run on this computer (PowerShell, from this folder)

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -u benchmark_v3.py --output results\v3_new --variants filtered-ea-sag filtered
.\.venv\Scripts\python.exe -B benchmark_v3.py   --output results\v3_new --variants filtered-ea-sag filtered --summarize
.\.venv\Scripts\python.exe -B verify_v3.py results\v3_new
```

Long runs can be split across several processes that share one output folder; each model/fold task is claimed with a lock file. `jobs\gpu_worker.py` does this automatically. Start it with `.\.venv\Scripts\python.exe -u jobs\gpu_worker.py` and leave the window open. It runs the JSON job files placed in `jobs\queue` and writes their logs to `jobs\logs`.

## Evaluation in one paragraph

There are 45 rated subjects and 90 recordings, with classes from the 1–9 rating (1–3 low, 4–6 moderate, 7–9 high). Testing uses 5 subject-disjoint folds of 9 test subjects each, the same folds in every version. Test data are 4-second windows (3,330 in total); recording accuracy averages each 150 s recording's windows into one decision. Test subjects are never used for training, normalisation fitting, early stopping or any choice of method. The alignment steps (EA, SAG) use each subject's own *unlabelled* EEG. See `Docs/BENCHMARK_V3.md` for this assumption and the realistic ceiling of about 74% for this 3-class task.
