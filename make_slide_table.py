"""Before/after table and chart for the slides, from two summary.csv files.

    python make_slide_table.py --before results/initial_5fold/summary.csv --before-variant raw \
        --after results/v3_final/summary.csv --after-variant selected --out results/v3_final
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["1D-CNN", "RNN", "LSTM", "BiLSTM", "GRU", "GNN", "STFT-ViT", "STFT-Swin", "EEGNet Adapted",
         "DeepConvNet Adapted", "ShallowConvNet Adapted", "TemporalCNN", "Top-3 Validation Ensemble"]


def read(path, variant):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["model"]: r for r in csv.DictReader(f) if r["variant"] == variant}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--before-variant", default="raw")
    p.add_argument("--after", required=True)
    p.add_argument("--after-variant", default="selected")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="STEW 3-class workload, unseen-subject 5-fold CV")
    a = p.parse_args()
    before, after = read(a.before, a.before_variant), read(a.after, a.after_variant)
    out = Path(a.out)
    rows = []
    for m in ORDER:
        if m not in after:
            continue
        b = before.get(m)
        f = lambda r, k: float(r[k]) * 100 if r and r.get(k) not in (None, "") else float("nan")
        rows.append({"model": m,
                     "before_accuracy": f(b, "accuracy_mean"), "before_sd": f(b, "accuracy_sd"),
                     "after_accuracy": f(after[m], "accuracy_mean"), "after_sd": f(after[m], "accuracy_sd"),
                     "after_macro_f1": float(after[m]["macro_f1_mean"]),
                     "after_balanced_accuracy": f(after[m], "balanced_accuracy_mean"),
                     "after_recording_accuracy": f(after[m], "recording_accuracy_mean"),
                     "after_recording_sd": f(after[m], "recording_accuracy_sd"),
                     "folds": int(after[m]["folds_completed"])})
    with open(out / "before_after.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    fmt = lambda m, s: "—" if np.isnan(m) else f"{m:.2f} ± {s:.2f}"
    lines = [f"# Before vs after — {a.title}", "",
             "| Model | Before: accuracy % | After: accuracy % | Change (pp) | After: macro-F1 | After: recording accuracy % |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        ch = "—" if np.isnan(r["before_accuracy"]) else f"{r['after_accuracy'] - r['before_accuracy']:+.2f}"
        lines.append(f"| {r['model']} | {fmt(r['before_accuracy'], r['before_sd'])} | {fmt(r['after_accuracy'], r['after_sd'])} | "
                     f"{ch} | {r['after_macro_f1']:.3f} | {fmt(r['after_recording_accuracy'], r['after_recording_sd'])} |")
    lines += ["", "Accuracy = 4-second window accuracy, mean ± SD over the 5 subject-disjoint test folds. "
              "Recording accuracy = one decision per 150-second recording (average of its window probabilities)."]
    (out / "BEFORE_AFTER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    models = [r["model"] for r in rows]
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(models) + 1.5), layout="constrained")
    ax.barh(y - 0.2, [r["before_accuracy"] for r in rows], 0.4, xerr=[r["before_sd"] for r in rows],
            color="#9ca3af", label="Before (raw signals, baseline training)", capsize=2)
    ax.barh(y + 0.2, [r["after_accuracy"] for r in rows], 0.4, xerr=[r["after_sd"] for r in rows],
            color="#2563eb", label="After (preprocessing + optimised training)", capsize=2)
    for yi, r in zip(y, rows):
        ax.text(r["after_accuracy"] + r["after_sd"] + 1, yi + 0.2, f"{r['after_accuracy']:.1f}", va="center", fontsize=8)
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Window accuracy (%)")
    ax.set_title(a.title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8, frameon=False)
    fig.savefig(out / "before_after.png", dpi=180)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
