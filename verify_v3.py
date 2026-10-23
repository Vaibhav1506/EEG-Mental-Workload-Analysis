"""Independent audit of a benchmark_v3 result folder. Recomputes everything from the saved
probabilities and fails loudly on any mismatch.

    python verify_v3.py results/v3_final
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, KFold

ROOT = Path(__file__).resolve().parent
ENSEMBLE = "Top-3 Validation Ensemble"
SELECTED = "selected"


def check(cond, msg):
    if not cond:
        raise SystemExit(f"FAILED: {msg}")


def splits_like_v1(subjects, folds=5, seed=42):
    """Independent re-implementation of benchmark.make_splits."""
    unique = np.unique(subjects)
    out = []
    for fold, (dev, test) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(unique)):
        d, t = unique[dev], unique[test]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold).split(d, groups=d))
        out.append({"fold": fold + 1, "train_subjects": d[a].tolist(), "validation_subjects": d[b].tolist(),
                    "test_subjects": t.tolist()})
    return out


LABELS = [0, 1, 2]


def metrics(y, p):
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def main(folder):
    folder = Path(folder).resolve()
    ratings = np.loadtxt(ROOT / "Dataset" / "ratings.txt", delimiter=",", dtype=int)
    check(set(ratings[:, 0]) == set(range(1, 49)) - {5, 24, 42}, "45 rated subjects expected (05, 24, 42 excluded)")
    names, y_rec, subj_rec = [], [], []
    for s, lo, hi in ratings:
        for c, r in [("lo", lo), ("hi", hi)]:
            names.append(f"sub{s:02d}_{c}"); y_rec.append((r - 1) // 3); subj_rec.append(s)
    names, y_rec, subj_rec = np.array(names), np.array(y_rec), np.array(subj_rec)
    check(np.bincount(y_rec).tolist() == [42, 23, 25], "class counts 42/23/25")
    task = json.loads((folder / "audit" / "settings.json").read_text(encoding="utf-8"))["settings"].get("task", "3class")
    global LABELS
    if task == "2class":
        y_rec = np.array([0 if n.endswith("_lo") else 1 for n in names])
        LABELS = [0, 1]
    splits = splits_like_v1(np.repeat(subj_rec, 37))
    saved = json.loads((folder / "audit" / "subject_splits.json").read_text(encoding="utf-8"))
    check(saved == splits, "saved subject splits equal an independent re-derivation")
    v1_splits = ROOT / "results" / "initial_5fold" / "audit" / "subject_splits.json"
    if v1_splits.exists():
        check(json.loads(v1_splits.read_text(encoding="utf-8")) == splits, "folds identical to v1")
    tested = sorted(s for sp in splits for s in sp["test_subjects"])
    check(tested == sorted(ratings[:, 0].tolist()), "every subject tested exactly once")
    parts = {}
    for js in sorted((folder / "parts").glob("*.json")):
        meta = json.loads(js.read_text(encoding="utf-8"))
        z = np.load(str(js)[:-5] + ".npz")
        parts[(meta["model"], meta["variant"], meta["fold"])] = (meta, {k: z[k] for k in z.files})
    check(parts, "no finished parts")
    # recompute development out-of-fold scores from the saved probabilities (do not trust stored values)
    for key, (meta, d) in list(parts.items()):
        dy = y_rec[d["dev_rec"]]
        acc_, bacc_, f1_ = metrics(dy, d["dev_probability"].argmax(1))
        stored = meta.get("dev_oof") or {}
        check(abs(stored.get("accuracy", -1) - acc_) < 1e-9 and abs(stored.get("macro_f1", -1) - f1_) < 1e-9,
              f"{key}: stored development scores match recomputation")
    assembly = folder / "audit" / "assembly.json"
    if assembly.exists():  # assembled folder: each part must be the mean of its replicate runs
        for key, (meta, d) in parts.items():
            reps = [np.load(ROOT / (r + ".npz")) for r in meta["replicates"]]
            for k in ["test_probability", "dev_probability"]:
                check(np.allclose(np.mean([r[k] for r in reps], 0), d[k], atol=1e-7), f"{key}: {k} = mean of replicates")
            for k in ["test_rec", "test_start", "dev_rec", "dev_start"]:
                check(all(np.array_equal(r[k], d[k]) for r in reps), f"{key}: {k} identical across replicates")
    recomputed = {}
    for (model, variant, fold), (meta, d) in parts.items():
        sp = splits[fold - 1]
        dev = set(sp["train_subjects"] + sp["validation_subjects"])
        test_subj = set(subj_rec[d["test_rec"]].tolist())
        check(test_subj == set(sp["test_subjects"]), f"{model}/{variant}/{fold}: test windows = fold test subjects")
        check(not (set(subj_rec[d["dev_rec"]].tolist()) & set(sp["test_subjects"])),
              f"{model}/{variant}/{fold}: no test subject in development predictions")
        check(set(subj_rec[d["dev_rec"]].tolist()) <= dev, f"{model}/{variant}/{fold}: dev predictions only for dev subjects")
        # evaluation windows identical to v1: every test recording, starts 0..18432 step 512
        ids = np.flatnonzero(np.isin(subj_rec, sp["test_subjects"]))
        exp_rec = np.repeat(ids, 37)
        exp_start = np.tile(np.arange(0, 19200 - 511, 512), len(ids))
        check(np.array_equal(d["test_rec"], exp_rec) and np.array_equal(d["test_start"], exp_start),
              f"{model}/{variant}/{fold}: evaluation windows identical to v1")
        p = d["test_probability"]
        check(np.all(p >= -1e-6) and np.all(p <= 1 + 1e-6) and np.allclose(p.sum(1), 1, atol=1e-4),
              f"{model}/{variant}/{fold}: valid probabilities")
        y = y_rec[d["test_rec"]]
        acc, bacc, f1 = metrics(y, p.argmax(1))
        rt, rp = [], []
        for r in np.unique(d["test_rec"]):
            sel = d["test_rec"] == r
            rt.append(y_rec[r]); rp.append(p[sel].mean(0).argmax())
        recomputed[(model, variant, fold)] = (acc, bacc, f1, accuracy_score(rt, rp))
        fits = meta["fits"]
        for f in fits:
            check(1 <= f["best_epoch"] <= f["epochs_run"], f"{model}/{variant}/{fold}: checkpoint within run")
    rows = list(csv.DictReader((folder / "details" / "fold_metrics.csv").open(encoding="utf-8")))
    # nested preprocessing choice: the chosen variant has the best development out-of-fold accuracy
    sel_file = folder / "audit" / "selection.json"
    real_variants = json.loads(sel_file.read_text(encoding="utf-8"))["candidate_variants"] if sel_file.exists() else \
        sorted({r["variant"] for r in rows})
    rule_file = folder / "audit" / "final_rule.json"
    if rule_file.exists():  # assembled folder: re-derive the pre-declared two-level choice from dev scores
        rule = json.loads(rule_file.read_text(encoding="utf-8"))
        dacc = lambda c, m, f: metrics(y_rec[parts[(m, c, f)][1]["dev_rec"]], parts[(m, c, f)][1]["dev_probability"].argmax(1))[0]
        for f in range(1, 6):
            pre_s = {c: np.mean([dacc(c, m, f) for m in rule["models"]]) for c in rule["preprocessing_candidates"]}
            pre = max(rule["preprocessing_candidates"], key=lambda c: (pre_s[c], -rule["preprocessing_candidates"].index(c)))
            choice = {m: pre for m in rule["models"]}
            for g in rule.get("recipe_groups", []):
                alt = g["alternatives"][pre]
                if np.mean([dacc(alt, m, f) for m in g["models"]]) > np.mean([dacc(pre, m, f) for m in g["models"]]):
                    for m in g["models"]:
                        choice[m] = alt
            for m in rule["models"]:
                meta, d = parts[(m, rule["name"], f)]
                check(meta["chosen_variant"] == choice[m], f"{m} fold {f}: final choice follows the declared rule")
                check(np.array_equal(d["test_probability"], parts[(m, choice[m], f)][1]["test_probability"]),
                      f"{m} fold {f}: final part is the chosen candidate")
    check(all(r["variant"] in real_variants + [SELECTED] for r in rows), "only candidate variants reported")
    for r in [r for r in rows if r["variant"] == SELECTED and r["model"] != ENSEMBLE]:
        model, fold = r["model"], int(r["fold"])
        configured = [v for v in real_variants if any(k[0] == model and k[1] == v for k in parts)]
        cands = [v for v in configured if (model, v, fold) in parts]
        check(len(cands) == len(configured), f"{model} fold {fold}: all candidates present")
        best = max(cands, key=lambda v: (parts[(model, v, fold)][0]["dev_oof"]["accuracy"], -real_variants.index(v)))
        check(r["chosen_variant"] == best, f"{model} fold {fold}: selected variant is the best by dev accuracy")
        parts[(model, SELECTED, fold)] = parts[(model, best, fold)]
        recomputed[(model, SELECTED, fold)] = recomputed[(model, best, fold)]
    # ensemble check: members are the top-3 by development out-of-fold macro-F1 and probs are their mean
    for (model, variant, fold) in list(parts):
        if variant == SELECTED:
            meta, d = parts[(model, variant, fold)]
            dev_y = y_rec[d["dev_rec"]]
            parts[(model, variant, fold)] = ({**meta, "dev_oof": {"macro_f1": metrics(dev_y, d["dev_probability"].argmax(1))[2]}}, d)
    ens_rows = [r for r in rows if r["model"] == ENSEMBLE]
    for r in ens_rows:
        fold, variant = int(r["fold"]), r["variant"]
        cands = [(m, meta["dev_oof"]["macro_f1"]) for (m, v, f), (meta, _) in parts.items() if v == variant and f == fold]
        top = [m for m, _ in sorted(cands, key=lambda t: -t[1])[:3]]
        check(r["members"] == " + ".join(top), f"ensemble fold {fold}: members are top-3 by dev macro-F1")
        p = np.mean([parts[(m, variant, fold)][1]["test_probability"] for m in top], 0)
        d0 = parts[(top[0], variant, fold)][1]
        y = y_rec[d0["test_rec"]]
        acc, bacc, f1 = metrics(y, p.argmax(1))
        rt, rp = [], []
        for rr in np.unique(d0["test_rec"]):
            sel = d0["test_rec"] == rr
            rt.append(y_rec[rr]); rp.append(p[sel].mean(0).argmax())
        recomputed[(ENSEMBLE, variant, fold)] = (acc, bacc, f1, accuracy_score(rt, rp))
    for r in rows:
        key = (r["model"], r["variant"], int(r["fold"]))
        check(key in recomputed, f"row {key} has saved probabilities")
        acc, bacc, f1, racc = recomputed[key]
        for name, val in [("accuracy", acc), ("balanced_accuracy", bacc), ("macro_f1", f1), ("recording_accuracy", racc)]:
            check(abs(float(r[name]) - val) < 1e-9, f"{key} {name} recomputed {val} vs saved {r[name]}")
    summary = list(csv.DictReader((folder / "summary.csv").open(encoding="utf-8")))
    for s in summary:
        vals = [recomputed[k] for k in recomputed if k[0] == s["model"] and k[1] == s["variant"]]
        check(int(s["folds_completed"]) == len(vals), f"{s['model']}/{s['variant']} fold count")
        check(abs(float(s["accuracy_mean"]) - np.mean([v[0] for v in vals])) < 1e-9, f"{s['model']}/{s['variant']} mean acc")
        if len(vals) > 1:
            check(abs(float(s["accuracy_sd"]) - np.std([v[0] for v in vals], ddof=1)) < 1e-9, f"{s['model']}/{s['variant']} SD")
    result = {"status": "PASSED", "parts": len(parts), "rows": len(rows), "summary_rows": len(summary),
              "test_windows_checked": int(sum(len(d["test_rec"]) for _, d in parts.values()))}
    (folder / "audit" / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/v3_final")
