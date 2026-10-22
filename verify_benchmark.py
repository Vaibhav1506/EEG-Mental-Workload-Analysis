"""Audit retained predictions, subject splits, metrics, and input provenance."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    root = Path(__file__).resolve().parent
    out = (Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results" / "initial_5fold").resolve()
    audit = out / "audit" if (out / "audit").is_dir() else out
    details = out / "details" if (out / "details").is_dir() else out
    metadata = json.loads((audit / "run_metadata.json").read_text())
    manifest = json.loads((audit / "dataset_manifest.json").read_text())
    splits = json.loads((audit / "subject_splits.json").read_text())
    metrics, summary = read_csv(details / "fold_metrics.csv"), read_csv(out / "summary.csv")
    windows, recordings = read_csv(details / "window_predictions.csv"), read_csv(details / "recording_predictions.csv")
    history = read_csv(details / "training_history.csv")
    baseline = read_csv(details / "majority_baseline.csv")
    matrices = json.loads((audit / "confusion_matrices.json").read_text())
    args = metadata["arguments"]
    assert metadata.get("completed_utc"), "Run has not completed"
    source_name = metadata.get("source_file", "benchmark_source.py")
    assert Path(source_name).name == source_name, "Source archive must be inside the audit folder"
    source = audit / source_name
    if not source.exists():
        source = root / "benchmark.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["code_sha256"], "Executed source hash does not match"
    data = root / metadata["data_relative_to_project"] if metadata.get("data_relative_to_project") else Path(args["data"])
    assert len(metrics) == len(args["models"]) * 2 * args["folds"]
    assert len(summary) == len(args["models"]) * 2
    lookup = {Path(row["file"]).stem: row for row in manifest}
    assert len(lookup) == len(manifest) == 90
    for row in manifest:
        path = data / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], f"Data changed: {path}"
        assert row["class"] == (row["rating"] - 1) // 3
    all_subjects = {r["subject"] for r in manifest}
    rating_file = data / "ratings.txt"
    current_ratings = np.loadtxt(rating_file, delimiter=",", dtype=int)
    rating_lookup = {int(subject): (int(lo), int(hi)) for subject, lo, hi in current_ratings}
    assert len(rating_lookup) == len(current_ratings) and set(rating_lookup) == all_subjects
    for row in manifest:
        assert row["rating"] == rating_lookup[row["subject"]][0 if row["condition"] == "lo" else 1]
    covered = []
    for split in splits:
        a, b, c = (set(split[k]) for k in ["train_subjects", "validation_subjects", "test_subjects"])
        assert not (a & b or a & c or b & c)
        assert a | b | c == all_subjects
        covered.extend(c)
    assert len(covered) == len(set(covered)) == 45
    assert set(covered) == all_subjects
    probabilities = ["probability_low", "probability_moderate", "probability_high"]
    metric_groups = {}
    for row in metrics:
        key = row["model"], row["variant"], int(row["fold"])
        assert key not in metric_groups
        metric_groups[key] = row
    prediction_groups = {key: [] for key in metric_groups}
    recording_groups = {key: [] for key in metric_groups}
    history_groups = {key: [] for key in metric_groups}
    for row in history:
        key = row["model"], row["variant"], int(row["fold"])
        assert key in history_groups
        history_groups[key].append(row)
    for key, entries in history_groups.items():
        result = metric_groups[key]
        assert [int(r["epoch"]) for r in entries] == list(range(1, int(result["epochs_run"]) + 1))
        best = next(r for r in entries if int(r["epoch"]) == int(result["best_epoch"]))
        np.testing.assert_allclose(float(result["validation_macro_f1"]), float(best["macro_f1"]), atol=1e-12)
        assert max(float(r["macro_f1"]) for r in entries) <= float(best["macro_f1"]) + 2e-6
        assert all(np.isfinite(float(r["training_loss"])) for r in entries)
    assert len(baseline) == args["folds"]
    for row in baseline:
        split = next(s for s in splits if s["fold"] == int(row["fold"]))
        train_labels = [r["class"] for r in manifest if r["subject"] in split["train_subjects"]]
        test_labels = [r["class"] for r in manifest if r["subject"] in split["test_subjects"]]
        majority = int(np.bincount(train_labels, minlength=3).argmax())
        assert row["training_majority_class"] == ["low", "moderate", "high"][majority]
        np.testing.assert_allclose(float(row["accuracy"]), np.mean(np.array(test_labels) == majority), atol=1e-12)
    for row in windows:
        key = row["model"], row["variant"], int(row["fold"])
        assert key in prediction_groups
        prediction_groups[key].append(row)
    for row in recordings:
        key = row["model"], row["variant"], int(row["fold"])
        assert key in recording_groups
        recording_groups[key].append(row)
    expected_starts = list(range(0, 19200 - 512 + 1, round(args["stride_seconds"] * 128)))
    for key, rows in prediction_groups.items():
        model, variant, fold = key
        split = next(s for s in splits if s["fold"] == fold)
        expected = {name for name, row in lookup.items() if row["subject"] in split["test_subjects"]}
        grouped = {}
        for row in rows:
            name = row["recording"]
            assert name in expected
            assert int(row["subject"]) == lookup[name]["subject"]
            assert int(row["true_class"]) == lookup[name]["class"]
            probability = np.array([float(row[p]) for p in probabilities])
            assert np.isfinite(probability).all() and (probability >= 0).all()
            np.testing.assert_allclose(probability.sum(), 1, atol=2e-6)
            assert probability.argmax() == int(row["predicted_class"])
            grouped.setdefault(name, []).append(row)
        assert set(grouped) == expected
        for name, group in grouped.items():
            assert sorted(int(r["start_sample"]) for r in group) == expected_starts
        truth = [int(r["true_class"]) for r in rows]
        pred = [int(r["predicted_class"]) for r in rows]
        recomputed = {"accuracy": accuracy_score(truth, pred),
                      "balanced_accuracy": balanced_accuracy_score(truth, pred),
                      "macro_f1": f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0)}
        for metric, value in recomputed.items():
            np.testing.assert_allclose(float(metric_groups[key][metric]), value, atol=1e-12)
        recs = recording_groups[key]
        assert len(recs) == len(expected) and {r["recording"] for r in recs} == expected
        for record in recs:
            averaged = np.mean([[float(w[p]) for p in probabilities] for w in grouped[record["recording"]]], axis=0)
            np.testing.assert_allclose([float(record[p]) for p in probabilities], averaged, atol=2e-6)
            assert int(record["true_class"]) == lookup[record["recording"]]["class"]
            assert int(record["predicted_class"]) == averaged.argmax()
        np.testing.assert_allclose(float(metric_groups[key]["recording_accuracy"]),
                                  accuracy_score([int(r["true_class"]) for r in recs],
                                                 [int(r["predicted_class"]) for r in recs]), atol=1e-12)
    for row in summary:
        matching = [r for r in metrics if r["model"] == row["model"] and r["variant"] == row["variant"]]
        assert len(matching) == args["folds"]
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "recording_accuracy"]:
            values = [float(r[metric]) for r in matching]
            np.testing.assert_allclose(float(row[metric + "_mean"]), np.mean(values), atol=1e-12)
            np.testing.assert_allclose(float(row[metric + "_sd"]), np.std(values, ddof=1), atol=1e-12)
    for matrix in matrices:
        rows = [r for r in windows if r["model"] == matrix["model"] and r["variant"] == matrix["variant"]]
        assert len(rows) == metadata["windows"]
        np.testing.assert_array_equal(matrix["matrix"], confusion_matrix(
            [int(r["true_class"]) for r in rows], [int(r["predicted_class"]) for r in rows], labels=[0, 1, 2]))
    report = {"passed": True, "model_condition_fold_runs": len(metrics), "window_predictions_checked": len(windows),
              "recording_predictions_checked": len(recordings), "input_file_hashes_checked": len(manifest),
              "ratings_sha256": hashlib.sha256(rating_file.read_bytes()).hexdigest(),
              "runs_reaching_epoch_cap": sum(int(r["epochs_run"]) == args["epochs"] for r in metrics),
              "runs_with_best_checkpoint_at_epoch_cap": sum(int(r["best_epoch"]) == args["epochs"] for r in metrics),
              "checks": ["input and source hashes", "rating labels", "disjoint subject splits", "complete test coverage",
                         "validation checkpoint selection", "training-majority baseline", "probability normalization",
                         "window and recording metrics recomputed", "mean and SD recomputed",
                         "confusion matrices recomputed"]}
    (audit / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    best = max(summary, key=lambda r: float(r["accuracy_mean"]))
    best_filtered = max((r for r in summary if r["variant"] == "filtered"), key=lambda r: float(r["accuracy_mean"]))
    baseline_values = [100 * float(r["accuracy"]) for r in baseline]
    format_result = lambda r: (f"{r['model']} ({r['variant']}), {100*float(r['accuracy_mean']):.2f}% "
                                f"+/- {100*float(r['accuracy_sd']):.2f} percentage points")
    section = ("\n## Verification and observed results\n\n"
               f"All audit checks passed: {len(metrics)} fits, {len(windows):,} window predictions, "
               f"{len(recordings):,} recording predictions, and 90 input file hashes. "
               "Metrics and fold statistics were independently recomputed from the saved predictions.\n\n"
               f"Training-majority baseline accuracy: {np.mean(baseline_values):.2f}% "
               f"+/- {np.std(baseline_values, ddof=1):.2f} percentage points across folds.\n\n"
               f"Highest mean window accuracy in this run: {format_result(best)}. "
               f"Highest mean filtered accuracy: {format_result(best_filtered)}. "
               "These rankings are descriptive and do not establish a statistically significant advantage.\n\n"
               f"{report['runs_reaching_epoch_cap']} of {len(metrics)} fits reached the {args['epochs']}-epoch cap; "
               f"{report['runs_with_best_checkpoint_at_epoch_cap']} selected their best checkpoint at that cap. "
               "This supports treating the run as an initial comparison with a limited training budget.\n")
    report_path = out / "REPORT.md"
    original = report_path.read_text(encoding="utf-8").split("\n## Verification and observed results")[0]
    report_path.write_text(original.rstrip() + "\n" + section, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
