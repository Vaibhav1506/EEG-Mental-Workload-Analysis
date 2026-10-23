"""Assemble the final v3 result folder from several benchmark_v3 run folders.

Each *candidate* is one preprocessing variant trained with one recipe. A candidate may have several
replicate runs (same folds, same inner groups, different initialisations: --init-offset). Replicates are
averaged into one ensemble (test and out-of-fold development probabilities). The result folder then
holds one part per candidate/model/fold, and `benchmark_v3.py --summarize` performs the nested choice
(per model and fold, the candidate with the best development out-of-fold accuracy).

    python assemble_v3.py assemble_config.json
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KEYS = ["test_probability", "test_rec", "test_start", "dev_probability", "dev_rec", "dev_start"]


def slug(text):
    return "".join(c if c.isalnum() else "_" for c in text)


DEFAULTS = {"task": "3class", "no_demean": False, "constant_lr": False, "no_jitter": False, "init_offset": 0}


def load_settings(folder):
    """Run settings, with defaults for options added after the folder was created; summarize-only flags dropped."""
    s = json.loads((folder / "audit" / "settings.json").read_text(encoding="utf-8"))["settings"]
    s = {**DEFAULTS, **s}
    s.pop("no_select", None)
    return s


def dev_acc(out, cand, model, fold):
    meta = json.loads((out / "parts" / f"{slug(cand)}__{slug(model)}__fold{fold}.json").read_text(encoding="utf-8"))
    return meta["dev_oof"]["accuracy"]


def apply_final_rule(out, rule):
    """Pre-declared two-level choice, made separately in every outer fold from development (out-of-fold)
    accuracy only:
      1. preprocessing: the candidate with the best mean development accuracy over ALL models;
      2. recipe groups (e.g. the two transformers): default recipe vs an alternative recipe, by the
         mean development accuracy over the group's models, using the preprocessing chosen in step 1.
    The chosen candidate's part is copied as variant rule["name"]."""
    models = rule["models"]
    log = {"rule": apply_final_rule.__doc__, "folds": {}}
    for fold in range(1, 6):
        pre_scores = {c: float(np.mean([dev_acc(out, c, m, fold) for m in models])) for c in rule["preprocessing_candidates"]}
        pre = max(rule["preprocessing_candidates"], key=lambda c: (pre_scores[c], -rule["preprocessing_candidates"].index(c)))
        choice = {m: pre for m in models}
        groups = []
        for g in rule.get("recipe_groups", []):
            alt = g["alternatives"][pre]
            base_s = float(np.mean([dev_acc(out, pre, m, fold) for m in g["models"]]))
            alt_s = float(np.mean([dev_acc(out, alt, m, fold) for m in g["models"]]))
            pick = alt if alt_s > base_s else pre
            for m in g["models"]:
                choice[m] = pick
            groups.append({"models": g["models"], "default": pre, "default_dev_acc": base_s,
                           "alternative": alt, "alternative_dev_acc": alt_s, "chosen": pick})
        for m in models:
            src = out / "parts" / f"{slug(choice[m])}__{slug(m)}__fold{fold}"
            meta = json.loads(Path(str(src) + ".json").read_text(encoding="utf-8"))
            meta = {**meta, "variant": rule["name"], "chosen_variant": choice[m]}
            dest = out / "parts" / f"{slug(rule['name'])}__{slug(m)}__fold{fold}"
            shutil.copy(str(src) + ".npz", str(dest) + ".npz")
            Path(str(dest) + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        log["folds"][fold] = {"preprocessing_dev_acc": pre_scores, "preprocessing": pre, "recipe_groups": groups,
                              "chosen": choice}
        print(f"fold {fold}: preprocessing {pre} {pre_scores}; groups {[(g['chosen'], round(g['default_dev_acc'], 4), round(g['alternative_dev_acc'], 4)) for g in groups]}")
    (out / "audit" / "final_rule.json").write_text(json.dumps({**rule, **log}, indent=2), encoding="utf-8")
    return log


def main(config_path):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out = ROOT / cfg["output"]
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new output folder")
    (out / "parts").mkdir(parents=True)
    (out / "audit").mkdir()
    folds = range(1, 6)
    ratings = np.loadtxt(ROOT / "Dataset" / "ratings.txt", delimiter=",", dtype=int)
    names = [f"sub{s:02d}_{c}" for s, _, _ in ratings for c in ["lo", "hi"]]
    record = {"candidates": [], "rule": "replicates averaged; nested choice is made by benchmark_v3.py --summarize"}
    first_settings = None
    for cand in cfg["candidates"]:
        runs = [ROOT / r for r in cand["runs"]]
        settings = [load_settings(r) for r in runs]
        base = [{k: v for k, v in s.items() if k != "init_offset"} for s in settings]
        if any(b != base[0] for b in base):
            raise SystemExit(f"candidate {cand['name']}: replicate runs must differ only in init_offset")
        if len({s.get("init_offset", 0) for s in settings}) != len(settings):
            raise SystemExit(f"candidate {cand['name']}: replicates need distinct init_offset")
        first_settings = first_settings or settings[0]
        if settings[0].get("task", "3class") != first_settings.get("task", "3class") or settings[0]["seed"] != first_settings["seed"]:
            raise SystemExit("all candidates must share task and fold seed")
        written = []
        for model in cand["models"]:
            for fold in folds:
                srcs = [r / "parts" / f"{slug(cand['variant'])}__{slug(model)}__fold{fold}" for r in runs]
                if not all(Path(str(s) + ".json").exists() for s in srcs):
                    raise SystemExit(f"missing part for {cand['name']} {model} fold {fold}: {srcs}")
                metas = [json.loads(Path(str(s) + ".json").read_text(encoding="utf-8")) for s in srcs]
                arrs = [np.load(str(s) + ".npz") for s in srcs]
                for k in ["test_rec", "test_start", "dev_rec", "dev_start"]:
                    if any(not np.array_equal(a[k], arrs[0][k]) for a in arrs):
                        raise SystemExit(f"{cand['name']} {model} fold {fold}: replicate windows differ ({k})")
                data = {k: arrs[0][k] for k in ["test_rec", "test_start", "dev_rec", "dev_start"]}
                data["test_probability"] = np.mean([a["test_probability"] for a in arrs], axis=0)
                data["dev_probability"] = np.mean([a["dev_probability"] for a in arrs], axis=0)
                dest = out / "parts" / f"{slug(cand['name'])}__{slug(model)}__fold{fold}"
                np.savez(str(dest) + ".npz", **data)
                if first_settings.get("task", "3class") == "2class":
                    y_rec = np.array([0 if n.endswith("_lo") else 1 for n in names])
                else:
                    y_rec = np.array([(r - 1) // 3 for _, lo, hi in ratings for r in (lo, hi)])
                dy = y_rec[data["dev_rec"]]
                dp = data["dev_probability"].argmax(1)
                labels = np.unique(y_rec)
                f1 = np.mean([2 * np.sum((dp == c) & (dy == c)) / max(1, np.sum(dp == c) + np.sum(dy == c)) for c in labels])
                recall = np.mean([np.mean(dp[dy == c] == c) for c in labels])
                fits = [f for m in metas for f in m["fits"]]
                meta = {"model": model, "variant": cand["name"], "fold": fold, "fits": fits,
                        "replicates": [str(Path(cand["runs"][i]) / "parts" / srcs[i].name) for i in range(len(srcs))],
                        "parameters": metas[0]["parameters"],
                        "training_seconds": float(sum(m["training_seconds"] for m in metas)),
                        "inference_ms_per_window": float(sum(m["inference_ms_per_window"] for m in metas)),
                        "dev_oof": {"accuracy": float(np.mean(dp == dy)), "balanced_accuracy": float(recall),
                                    "macro_f1": float(f1)}}
                (Path(str(dest) + ".json")).write_text(json.dumps(meta, indent=1), encoding="utf-8")
                written.append((model, fold))
        record["candidates"].append({"name": cand["name"], "variant": cand["variant"], "models": cand["models"],
                                     "runs": cand["runs"], "settings": [load_settings(r) for r in runs]})
        print(f"candidate {cand['name']}: {len(written)} parts from {len(runs)} replicate run(s)")
    if "final" in cfg:
        record["final_rule"] = apply_final_rule(out, cfg["final"])
    settings = dict(first_settings)
    settings["init_offset"] = "assembled"
    (out / "audit" / "settings.json").write_text(json.dumps({"settings": settings, "assembled": True,
                                                            "code_sha256": hashlib.sha256((ROOT / "benchmark_v3.py").read_bytes()).hexdigest()},
                                                           indent=2), encoding="utf-8")
    (out / "audit" / "assembly.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    shutil.copy(ROOT / "benchmark_v3.py", out / "audit" / "benchmark_v3_source.py")
    shutil.copy(Path(config_path), out / "audit" / "assemble_config.json")
    print(f"assembled into {out}; now run benchmark_v3.py --output {cfg['output']} --summarize --variants "
          + " ".join(c["name"] for c in cfg["candidates"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assemble_config.json")
