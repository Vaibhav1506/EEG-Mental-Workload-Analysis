"""Causal temporal smoothing: a decision every 4 s from the last k windows of the same recording.

Each window's probabilities are averaged with those of the k-1 windows just before it (only the past,
never the future; fewer at the start of a recording). k is chosen separately in every outer fold from
the development (out-of-fold) predictions, pooled over all models: the candidate with the best mean
development accuracy, ties -> the shorter context. Test subjects play no part in the choice.

    python smooth_v3.py --folder results/v3_final --variant after --task 3class \
        --before-v1 results/initial_5fold/details/window_predictions.csv
    python smooth_v3.py --folder results/v3_2class/after --variant filtered-ea-sag --task 2class \
        --before-parts results/v3_2class/before_v1like --before-variant raw
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
KS = [1, 2, 4, 8, 15]  # context: 4, 8, 16, 32, 60 seconds
ENSEMBLE = "Top-3 Validation Ensemble"
ORDER = ["1D-CNN", "RNN", "LSTM", "BiLSTM", "GRU", "GNN", "STFT-ViT", "STFT-Swin", "EEGNet Adapted",
         "DeepConvNet Adapted", "ShallowConvNet Adapted", "TemporalCNN", ENSEMBLE]


def labels(task):
    ratings = np.loadtxt(ROOT / "Dataset" / "ratings.txt", delimiter=",", dtype=int)
    names = np.array([f"sub{s:02d}_{c}" for s, _, _ in ratings for c in ("lo", "hi")])
    if task == "2class":
        y = np.array([0 if n.endswith("_lo") else 1 for n in names])
    else:
        y = np.array([(r - 1) // 3 for _, lo, hi in ratings for r in (lo, hi)])
    return names, y


def smooth(prob, rec, start, k):
    """Causal moving average over the last k windows of each recording (fewer at its start)."""
    if k == 1:
        return prob
    out = np.empty_like(prob)
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        idx = idx[np.argsort(start[idx], kind="stable")]
        c = np.cumsum(prob[idx], axis=0)
        for j, i in enumerate(idx):
            back = j - k
            out[i] = (c[j] - (c[back] if back >= 0 else 0)) / (j - max(back, -1))
    return out


def scores(y, p, n_classes):
    labels_ = list(range(n_classes))
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, labels=labels_, average="macro", zero_division=0))


def load_parts(folder, variant):
    parts = {}
    for js in (folder / "parts").glob("*.json"):
        meta = json.loads(js.read_text(encoding="utf-8"))
        if meta["variant"] != variant:
            continue
        z = np.load(str(js)[:-5] + ".npz")
        parts[(meta["model"], meta["fold"])] = {k: z[k] for k in z.files}
    fm = folder / "details" / "fold_metrics.csv"
    if fm.exists():  # rebuild the Top-3 ensemble from its recorded members
        for row in csv.DictReader(fm.open(encoding="utf-8")):
            if row["model"] == ENSEMBLE and row["variant"] == variant:
                members = row["members"].split(" + ")
                fold = int(row["fold"])
                arrs = [parts[(m, fold)] for m in members]
                parts[(ENSEMBLE, fold)] = {**arrs[0],
                                           "test_probability": np.mean([a["test_probability"] for a in arrs], 0),
                                           "dev_probability": np.mean([a["dev_probability"] for a in arrs], 0)}
    return parts


def load_v1(path, names):
    """v1 window_predictions.csv (raw variant) -> {(model, fold): {test_probability, test_rec, test_start}}."""
    index = {n: i for i, n in enumerate(names)}
    tmp = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["variant"] != "raw":
            continue
        key = (r["model"], int(r["fold"]))
        d = tmp.setdefault(key, ([], [], []))
        d[0].append([float(r["probability_low"]), float(r["probability_moderate"]), float(r["probability_high"])])
        d[1].append(index[r["recording"]])
        d[2].append(int(r["start_sample"]))
    return {k: {"test_probability": np.array(a), "test_rec": np.array(b), "test_start": np.array(c)} for k, (a, b, c) in tmp.items()}


def plot(out, csv_rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in csv_rows]
    names = [r["model"] for r in rows]
    y = np.arange(len(rows))
    val = lambda r, k: 100 * float(r[k]) if r[k] != "" else np.nan
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.6), layout="constrained")
    ax.barh(y - 0.2, [val(r, "before_smoothed") for r in rows], 0.4,
            xerr=[val(r, "before_smoothed_sd") for r in rows], color="#9ca3af", capsize=2,
            label="Before (raw signals, baseline training), smoothed the same way")
    ax.barh(y + 0.2, [val(r, "after_smoothed") for r in rows], 0.4,
            xerr=[val(r, "after_smoothed_sd") for r in rows], color="#2563eb", capsize=2, ecolor="#111827",
            label="After (preprocessing + optimised training), smoothed")
    for yi, r in zip(y, rows):
        ax.text(1.5, yi + 0.2, f"{val(r, 'after_smoothed'):.1f}",
                va="center", ha="left", fontsize=8, color="white", fontweight="bold")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy of the decision made every 4 s (%)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1, fontsize=8, frameon=False)
    fig.savefig(out / "before_after_smoothed.png", dpi=180)
    plt.close(fig)


def fmt(v):
    return f"{100 * np.mean(v):.2f} ± {100 * np.std(v, ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--task", choices=["3class", "2class"], default="3class")
    ap.add_argument("--before-v1", help="v1 window_predictions.csv for the 'before' column (3-class)")
    ap.add_argument("--before-parts", help="a v3-style folder with parts for the 'before' column")
    ap.add_argument("--before-variant", default="raw")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    folder = ROOT / a.folder
    names, y_rec = labels(a.task)
    nc = 2 if a.task == "2class" else 3
    parts = load_parts(folder, a.variant)
    models = [m for m in ORDER if any(k[0] == m for k in parts)]
    folds = sorted({k[1] for k in parts})
    base = [m for m in models if m != ENSEMBLE]
    # 1. choose k per fold from development predictions only (pooled over the 12 models)
    choice = {}
    for f in folds:
        dev = {k: float(np.mean([accuracy_score(y_rec[parts[(m, f)]["dev_rec"]],
                                                smooth(parts[(m, f)]["dev_probability"], parts[(m, f)]["dev_rec"],
                                                       parts[(m, f)]["dev_start"], k).argmax(1)) for m in base]))
               for k in KS}
        best = max(KS, key=lambda k: (round(dev[k], 12), -k))
        choice[f] = {"k": best, "context_seconds": 4 * best, "dev_accuracy_by_k": dev}
    # 2. score the test windows with the chosen k (and, descriptively, every k)
    out = folder / "smoothing"
    out.mkdir(exist_ok=True)
    rows, curve = [], {k: [] for k in KS}
    for m in models:
        per = {"win": [], "sm": [], "bacc": [], "f1": [], "rec": []}
        for f in folds:
            d = parts[(m, f)]
            y = y_rec[d["test_rec"]]
            base_acc = accuracy_score(y, d["test_probability"].argmax(1))
            sm = smooth(d["test_probability"], d["test_rec"], d["test_start"], choice[f]["k"]).argmax(1)
            acc, bacc, f1 = scores(y, sm, nc)
            rt, rp = [], []
            for r in np.unique(d["test_rec"]):
                sel = d["test_rec"] == r
                rt.append(y_rec[r]); rp.append(d["test_probability"][sel].mean(0).argmax())
            per["win"].append(base_acc); per["sm"].append(acc); per["bacc"].append(bacc); per["f1"].append(f1)
            per["rec"].append(accuracy_score(rt, rp))
            if m != ENSEMBLE:
                for k in KS:
                    curve[k].append(accuracy_score(y, smooth(d["test_probability"], d["test_rec"], d["test_start"], k).argmax(1)))
        rows.append({"model": m, **{k: v for k, v in per.items()}})
    before = {}
    if a.before_v1:
        before = load_v1(ROOT / a.before_v1, names)
    elif a.before_parts:
        before = load_parts(ROOT / a.before_parts, a.before_variant)
    lines = [f"# Causal temporal smoothing — {a.task}", "",
             "A decision every 4 s from the average of the last k windows of the same recording (past only). "
             "k chosen per outer fold from development out-of-fold accuracy pooled over the 12 models; test subjects play no part.", "",
             "| Fold | Chosen k | Context | Dev accuracy at chosen k |", "|---:|---:|---:|---:|"]
    for f in folds:
        c = choice[f]
        lines.append(f"| {f} | {c['k']} | {c['context_seconds']} s | {100 * c['dev_accuracy_by_k'][c['k']]:.2f}% |")
    lines += ["", "| Model | Before: window % | Before: smoothed % | After: window % | After: smoothed % | After smoothed: balanced acc. % | After smoothed: macro-F1 | After: recording % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    csv_rows = []
    for r in rows:
        m = r["model"]
        bw, bs = "—", "—"
        if before and all((m, f) in before for f in folds):
            bwin, bsm = [], []
            for f in folds:
                d = before[(m, f)]
                y = y_rec[d["test_rec"]]
                bwin.append(accuracy_score(y, d["test_probability"].argmax(1)))
                bsm.append(accuracy_score(y, smooth(d["test_probability"], d["test_rec"], d["test_start"], choice[f]["k"]).argmax(1)))
            bw, bs = fmt(bwin), fmt(bsm)
            csv_rows.append({"model": m, "before_window": np.mean(bwin), "before_window_sd": np.std(bwin, ddof=1),
                             "before_smoothed": np.mean(bsm), "before_smoothed_sd": np.std(bsm, ddof=1)})
        else:
            csv_rows.append({"model": m, "before_window": "", "before_window_sd": "", "before_smoothed": "",
                             "before_smoothed_sd": ""})
        csv_rows[-1].update({"after_window": np.mean(r["win"]), "after_window_sd": np.std(r["win"], ddof=1),
                             "after_smoothed": np.mean(r["sm"]), "after_smoothed_sd": np.std(r["sm"], ddof=1),
                             "after_smoothed_balanced_accuracy": np.mean(r["bacc"]), "after_smoothed_macro_f1": np.mean(r["f1"]),
                             "after_recording": np.mean(r["rec"])})
        lines.append(f"| {m} | {bw} | {bs} | {fmt(r['win'])} | {fmt(r['sm'])} | {100 * np.mean(r['bacc']):.2f} | "
                     f"{np.mean(r['f1']):.3f} | {fmt(r['rec'])} |")
    lines += ["", "Test accuracy at every k (mean over all single models and 5 folds; descriptive only, not used for any choice):", "",
              "| k (windows) | Context | Mean test accuracy % |", "|---:|---:|---:|"]
    for k in KS:
        lines.append(f"| {k} | {4 * k} s | {100 * np.mean(curve[k]):.2f} |")
    (out / "SMOOTHED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "smoothed_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    plot(out, csv_rows, a.title or f"STEW {a.task}, unseen subjects, decision from the last {4 * max(c['k'] for c in choice.values())} s")
    (out / "smoothing_choice.json").write_text(json.dumps({"candidates_k": KS, "rule": __doc__.split("\n\n")[1],
                                                           "folds": choice}, indent=2), encoding="utf-8")
    # self-check: k = 1 must reproduce the stored window accuracy
    if (folder / "details" / "fold_metrics.csv").exists():
        stored = {(r["model"], int(r["fold"])): float(r["accuracy"])
                  for r in csv.DictReader((folder / "details" / "fold_metrics.csv").open(encoding="utf-8")) if r["variant"] == a.variant}
        for (m, f), d in parts.items():
            if (m, f) in stored:
                assert abs(accuracy_score(y_rec[d["test_rec"]], d["test_probability"].argmax(1)) - stored[(m, f)]) < 1e-9, (m, f)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
