"""Subject-disjoint, paired raw/filtered STEW benchmark. See BENCHMARK.md."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, KFold
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
NAMES = ["1D-CNN", "RNN", "LSTM", "BiLSTM", "GRU", "GNN", "STFT-ViT", "STFT-Swin",
         "EEGNet Adapted", "DeepConvNet Adapted", "ShallowConvNet Adapted", "TemporalCNN"]
CLASSES = ["low", "moderate", "high"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rating_class(rating):
    rating = np.asarray(rating)
    if not np.all((rating >= 1) & (rating <= 9) & (rating == rating.astype(int))):
        raise ValueError("Ratings must be integers from 1 to 9")
    return ((rating.astype(int) - 1) // 3).astype(np.int64)


def prepare_data(folder, window_seconds=4, stride_seconds=4):
    ratings = np.loadtxt(folder / "ratings.txt", delimiter=",", dtype=int)
    if ratings.ndim != 2 or ratings.shape[1] != 3 or len(np.unique(ratings[:, 0])) != len(ratings):
        raise ValueError("Expected unique subject,rating_lo,rating_hi rows")
    window, stride = round(window_seconds * 128), round(stride_seconds * 128)
    if window != 512 or stride < 1:
        raise ValueError("Models use four-second windows (512 samples); stride must be positive")
    raw, processed, ys, subjects, recordings, starts, manifest = [], [], [], [], [], [], []
    sos = signal.butter(4, [1, 40], btype="bandpass", fs=128, output="sos")
    notch_b, notch_a = signal.iirnotch(50, 30, fs=128)
    for subject, low_rating, high_rating in ratings:
        for condition, rating in [("lo", low_rating), ("hi", high_rating)]:
            file = folder / f"sub{subject:02d}_{condition}.txt"
            data = np.loadtxt(file, dtype=np.float64)
            if data.shape != (19200, 14) or not np.isfinite(data).all():
                raise ValueError(f"Invalid recording: {file.name}, shape {data.shape}")
            clean = signal.filtfilt(notch_b, notch_a, data, axis=0)
            clean = signal.sosfiltfilt(sos, clean, axis=0)
            clean -= clean.mean(axis=1, keepdims=True)
            manifest.append({"file": file.name, "subject": int(subject), "condition": condition,
                             "rating": int(rating), "class": int(rating_class(rating)),
                             "samples": len(data), "channels": 14,
                             "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
            for start in range(0, len(data) - window + 1, stride):
                raw.append(data[start:start + window].T.astype(np.float32))
                processed.append(clean[start:start + window].T.astype(np.float32))
                ys.append(int(rating_class(rating)))
                subjects.append(int(subject))
                recordings.append(file.stem)
                starts.append(start)
    return ({"raw": np.stack(raw), "filtered": np.stack(processed)}, np.array(ys),
            np.array(subjects), np.array(recordings), np.array(starts), manifest)


def make_splits(subjects, folds, seed):
    unique = np.unique(subjects)
    splits = []
    for fold, (development, test) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(unique)):
        dev_subjects, test_subjects = unique[development], unique[test]
        inner_train, inner_val = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                      random_state=seed + fold).split(dev_subjects, groups=dev_subjects))
        train_subjects, val_subjects = dev_subjects[inner_train], dev_subjects[inner_val]
        assert not (set(train_subjects) & set(val_subjects) or set(train_subjects) & set(test_subjects)
                    or set(val_subjects) & set(test_subjects))
        splits.append({"fold": fold + 1, "train_subjects": train_subjects.tolist(),
                       "validation_subjects": val_subjects.tolist(), "test_subjects": test_subjects.tolist()})
    return splits


def normalize_from_training(x, train):
    mean = x[train].mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = x[train].std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    normalized = (x - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite input after normalization")
    return normalized, mean, std


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout, kernel in [(14, 32, 9), (32, 64, 7), (64, 64, 5)]:
            layers.extend([nn.Conv1d(cin, cout, kernel, padding=kernel // 2), nn.BatchNorm1d(cout),
                           nn.ELU(), nn.AvgPool1d(4), nn.Dropout(0.25)])
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(256, 3))

    def forward(self, x):
        return self.net(x)


class Recurrent(nn.Module):
    def __init__(self, kind, bidirectional=False):
        super().__init__()
        self.rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind](
            14, 32, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(64 * (2 if bidirectional else 1), 3))

    def forward(self, x):
        sequence, _ = self.rnn(F.avg_pool1d(x, 4).transpose(1, 2))
        return self.head(torch.cat([sequence.mean(dim=1), sequence[:, -1]], dim=1))


class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 63), padding=(0, 31), bias=False), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (14, 1), groups=8, bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.5), nn.Flatten(), nn.Linear(16 * 16, 3))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class DeepConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(1, 16, (1, 10)), nn.Conv2d(16, 16, (14, 1), bias=False),
                  nn.BatchNorm2d(16), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(0.5)]
        for cin, cout in [(16, 32), (32, 64), (64, 128)]:
            layers.extend([nn.Conv2d(cin, cout, (1, 10), bias=False), nn.BatchNorm2d(cout),
                           nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(0.5)])
        self.net = nn.Sequential(*layers, nn.Flatten(), nn.Linear(128, 3))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class ShallowConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 20, (1, 25)),
                                  nn.Conv2d(20, 20, (14, 1), bias=False), nn.BatchNorm2d(20))
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Flatten(), nn.Linear(20 * 28, 3))

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).square()
        x = F.avg_pool2d(x, (1, 75), (1, 15)).clamp_min(1e-6).log()
        return self.head(x)


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.convs = nn.ModuleList([nn.Conv1d(32, 32, 5, dilation=dilation) for _ in range(2)])
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        original = x
        for conv in self.convs:
            x = self.dropout(F.elu(conv(F.pad(x, (4 * self.dilation, 0)))))
        return F.elu(x + original)


class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.AvgPool1d(4), nn.Conv1d(14, 32, 1),
                                 CausalBlock(1), CausalBlock(2), CausalBlock(4),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)


class GraphAttention(nn.Module):
    """14 channel nodes; a complete graph with learned attention edge weights."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(1, 16, 15, stride=4, padding=7), nn.ELU(),
                                      nn.Conv1d(16, 16, 7, stride=4, padding=3), nn.ELU(),
                                      nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(64, 32))
        self.channels = nn.Parameter(torch.zeros(1, 14, 32))
        self.attention = nn.ModuleList([nn.MultiheadAttention(32, 4, dropout=0.25, batch_first=True)
                                       for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(32) for _ in range(2)])
        self.head = nn.Linear(32, 3)

    def forward(self, x):
        batch = len(x)
        x = self.features(x.reshape(batch * 14, 1, 512)).reshape(batch, 14, 32) + self.channels
        for attention, norm in zip(self.attention, self.norms):
            x = norm(x + attention(x, x, x, need_weights=False)[0])
        return self.head(x.mean(dim=1))


def spectrogram(x):
    window = torch.hann_window(64)
    chunks = []
    with torch.no_grad():
        for batch in x.split(256):
            spec = torch.stft(batch.reshape(-1, 512), n_fft=64, hop_length=16,
                              win_length=64, window=window, center=False, return_complex=True)
            spec = torch.log1p(spec.abs().square()).reshape(len(batch), 14, 33, 29)
            chunks.append(F.interpolate(spec, size=(32, 32), mode="bilinear", align_corners=False))
    return torch.cat(chunks)


class STFTViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(14, 32, 4, stride=4)
        self.position = nn.Parameter(torch.zeros(1, 64, 32))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.25, activation="gelu",
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 3))

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2) + self.position
        return self.head(self.encoder(x).mean(dim=1))


class WindowAttentionBlock(nn.Module):
    """Local attention with relative bias and a mask for cyclic shifted windows."""
    def __init__(self, dim, grid, shift):
        super().__init__()
        self.dim, self.grid, self.shift, self.window, self.heads = dim, grid, shift, 4, 4
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.25),
                                 nn.Linear(dim * 2, dim), nn.Dropout(0.25))
        self.relative_bias = nn.Parameter(torch.zeros(49, 4))
        coords = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")).flatten(1)
        delta = coords[:, :, None] - coords[:, None, :]
        self.register_buffer("relative_index", (delta[0] + 3) * 7 + delta[1] + 3)
        mask = torch.zeros(1, grid, grid, 1)
        if shift:
            slices = [slice(0, -4), slice(-4, -shift), slice(-shift, None)]
            count = 0
            for h in slices:
                for w in slices:
                    mask[:, h, w] = count
                    count += 1
        labels = self.partition(mask).squeeze(-1)
        mask = labels[:, :, None] - labels[:, None, :]
        self.register_buffer("mask", mask.ne(0).float() * -100.0)

    def partition(self, x):
        b, h, w, d = x.shape
        return x.reshape(b, h // 4, 4, w // 4, 4, d).permute(0, 1, 3, 2, 4, 5).reshape(-1, 16, d)

    def forward(self, x):
        b, h, w, d = x.shape
        y = self.norm1(x)
        if self.shift:
            y = torch.roll(y, shifts=(-self.shift, -self.shift), dims=(1, 2))
        y = self.partition(y)
        q, k, v = self.qkv(y).reshape(-1, 16, 3, 4, d // 4).permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1)) / (d // 4) ** 0.5
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)
        nw = (h // 4) * (w // 4)
        attention = attention.reshape(b, nw, 4, 16, 16) + bias[None, None] + self.mask[None, :, None]
        attention = F.dropout(attention.softmax(dim=-1), p=0.25, training=self.training)
        y = (attention.reshape(-1, 4, 16, 16) @ v).transpose(1, 2).reshape(-1, 16, d)
        y = self.proj(y).reshape(b, h // 4, w // 4, 4, 4, d).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, d)
        if self.shift:
            y = torch.roll(y, shifts=(self.shift, self.shift), dims=(1, 2))
        x = x + y
        return x + self.mlp(self.norm2(x))


class STFTSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(14, 32, 4, stride=4)
        self.stage1 = nn.Sequential(WindowAttentionBlock(32, 8, 0), WindowAttentionBlock(32, 8, 2))
        self.merge = nn.Conv2d(32, 64, 2, stride=2)
        self.stage2 = WindowAttentionBlock(64, 4, 0)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 3))

    def forward(self, x):
        x = self.stage1(self.patch(x).permute(0, 2, 3, 1))
        x = self.stage2(self.merge(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        return self.head(x.mean(dim=(1, 2)))


def make_model(name):
    factories = {"1D-CNN": ConvNet, "RNN": lambda: Recurrent("RNN"), "LSTM": lambda: Recurrent("LSTM"),
                 "BiLSTM": lambda: Recurrent("LSTM", True), "GRU": lambda: Recurrent("GRU"),
                 "GNN": GraphAttention, "STFT-ViT": STFTViT, "STFT-Swin": STFTSwin,
                 "EEGNet Adapted": EEGNet, "DeepConvNet Adapted": DeepConvNet,
                 "ShallowConvNet Adapted": ShallowConvNet, "TemporalCNN": TemporalCNN}
    return factories[name]()


def predict(model, x, indices, batch_size):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in np.array_split(indices, max(1, int(np.ceil(len(indices) / batch_size)))):
            output.append(model(x[batch]).softmax(dim=1).numpy())
    return np.concatenate(output)


def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))}


def train_and_evaluate(name, x, y, train, val, test, args, seed):
    seed_everything(seed)
    model = make_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    counts = np.bincount(y[train], minlength=3)
    if np.any(counts == 0):
        raise ValueError("Training split is missing a class")
    weights = torch.tensor(len(train) / (3 * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    labels = torch.from_numpy(y)
    best, best_state, best_epoch, history, stale = -1, None, 0, [], 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = train[torch.randperm(len(train)).numpy()]
        total_loss = 0
        for offset in range(0, len(permutation), args.batch_size):
            batch = permutation[offset:offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[batch]), labels[batch])
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss for {name}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        validation = scores(y[val], predict(model, x, val, args.batch_size).argmax(axis=1))
        history.append({"epoch": epoch, "training_loss": total_loss / len(train), **validation})
        if validation["macro_f1"] > best + 1e-6:
            best, best_state, best_epoch, stale = validation["macro_f1"], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    # Warm up the restored checkpoint, then measure one complete test-set prediction pass.
    predict(model, x, test[:min(16, len(test))], args.batch_size)
    start = time.perf_counter()
    probability = predict(model, x, test, args.batch_size)
    inference_seconds = time.perf_counter() - start
    result = {**scores(y[test], probability.argmax(axis=1)), "best_epoch": best_epoch,
              "epochs_run": len(history), "validation_macro_f1": best,
              "parameters": sum(p.numel() for p in model.parameters()),
              "training_seconds": training_seconds,
              "inference_ms_per_window": inference_seconds * 1000 / len(test)}
    return result, probability, history


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(out, results, predictions, histories, recording_predictions, args, metadata):
    write_csv(out / "fold_metrics.csv", results)
    write_csv(out / "window_predictions.csv", predictions)
    write_csv(out / "training_history.csv", histories)
    write_csv(out / "recording_predictions.csv", recording_predictions)
    summaries = []
    for name in args.models:
        for variant in ["raw", "filtered"]:
            rows = [r for r in results if r["model"] == name and r["variant"] == variant]
            if not rows:
                continue
            summary = {"model": name, "variant": variant, "folds_completed": len(rows)}
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "recording_accuracy"]:
                values = [r[metric] for r in rows]
                summary[metric + "_mean"] = float(np.mean(values))
                summary[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary["parameters"] = rows[0]["parameters"]
            summary["training_seconds_total"] = sum(r["training_seconds"] for r in rows)
            summaries.append(summary)
    write_csv(out / "summary.csv", summaries)
    (out / "summary.json").write_text(json.dumps({"metadata": metadata, "results": summaries}, indent=2), encoding="utf-8")
    table = ["| Model | Raw accuracy | Filtered accuracy | Change (pp) | Filtered macro-F1 |",
             "|---|---:|---:|---:|---:|"]
    completed_names = []
    for name in args.models:
        pairs = {r["variant"]: r for r in summaries if r["model"] == name}
        if set(pairs) != {"raw", "filtered"}:
            continue
        raw, clean = pairs["raw"], pairs["filtered"]
        fmt = lambda r: f"{100*r['accuracy_mean']:.2f}% +/- {100*(r['accuracy_sd'] or 0):.2f}"
        table.append(f"| {name} | {fmt(raw)} | {fmt(clean)} | "
                     f"{100*(clean['accuracy_mean']-raw['accuracy_mean']):+.2f} | {clean['macro_f1_mean']:.3f} |")
        completed_names.append(name)
    status = "COMPLETE" if len(results) == len(args.models) * 2 * args.folds else "IN PROGRESS"
    text = ["# STEW initial benchmark", "", f"Status: {status}. {len(results)} model/condition/fold runs completed.", "",
            "This is an initial comparison of compact, explicitly defined model adaptations. It does not reproduce the screenshot's unidentified experiment.", "",
            f"45 rated subjects; 90 recordings; {metadata['windows']} windows; three rating classes (1-3, 4-6, 7-9).",
            f"{args.folds} subject-disjoint test folds; separate validation subjects within each development fold; one initialization per fold.",
            f"Four-second windows, stride {args.stride_seconds:g} seconds. Maximum {args.epochs} epochs; early stopping after {args.patience} non-improving validation macro-F1 epochs.",
            "Raw means no additional signal filtering. Both conditions use normalization fitted exclusively on the training subjects.",
            "Filtered means 50 Hz notch (Q=30), fourth-order 1-40 Hz Butterworth bandpass, and average rereferencing on each continuous recording. Offline zero-phase filtering; ASR/ICA is not included.", "",
            *table, "", "## Interpretation", "",
            "Accuracy is measured on windows. The +/- value is sample standard deviation across test folds, not a confidence interval or variation across random initializations.",
            "Subjects are disjoint across train/validation/test within each fold; every rated subject is tested once per model and condition.",
            "Folds share training data, so their results are not independent replications. No statistical significance claim is made.",
            "Recording results average the predicted class probabilities across windows in each recording. They do not measure instantaneous workload labels.",
            "The majority-class baseline uses only the training class counts for each fold. A naive class-majority baseline is necessary because labels are imbalanced.",
            "Test folds are reserved for evaluation. Checkpoint selection uses validation macro-F1; no test tuning or architecture search was performed.",
            "The fixed small training budget, compact architectures, and single seed per fold limit claims about the best attainable model accuracy. More training and nested tuning may change rankings.",
            "STEW labels are recording-level self-ratings inherited by each window. Many windows do not constitute many independent subjects.",
            "The original STEW paper uses SVR and NCA, overlapping windows, and a particular subject holdout. This benchmark uses direct three-class neural classifiers and a different split protocol; scores are not directly interchangeable.",
            "The photograph does not reveal its dataset, split, preprocessing, adaptations, or meaning of +/-; its numbers are reference text, not measured benchmark results.", "",
            "See BENCHMARK.md for model definitions, reproduction commands, and source links."]
    (out / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    if not completed_names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(completed_names) * 0.5)), layout="constrained")
    positions = np.arange(len(completed_names))
    for variant, offset, color in [("raw", -0.18, "#64748b"), ("filtered", 0.18, "#0d9488")]:
        rows = [next(r for r in summaries if r["model"] == n and r["variant"] == variant) for n in completed_names]
        for ax, metric, scale in [(axes[0], "accuracy", 100), (axes[1], "macro_f1", 1)]:
            ax.barh(positions + offset, [scale*r[metric + "_mean"] for r in rows], height=0.34,
                    xerr=[scale*(r[metric + "_sd"] or 0) for r in rows], color=color, label=variant, capsize=2)
    axes[0].set_yticks(positions, completed_names)
    axes[1].set_yticks(positions, completed_names)
    axes[0].set_xlabel("Window accuracy (%)")
    axes[1].set_xlabel("Window macro-F1")
    axes[0].set_xlim(0, 100)
    axes[1].set_xlim(0, 1)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right")
    fig.suptitle("STEW: initial subject-disjoint benchmark\nMean and SD across held-out subject folds", fontsize=14)
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "STEW Dataset" / "STEW Dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results" / "initial_5fold")
    parser.add_argument("--models", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--stride-seconds", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty, to preserve existing results")
    if args.folds < 2 or args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("Require folds >= 2 and positive epochs, patience, batch size")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output
    start = time.perf_counter()
    print("Loading and validating STEW recordings...", flush=True)
    variants, y, subjects, recordings, starts, manifest = prepare_data(args.data, stride_seconds=args.stride_seconds)
    splits = make_splits(subjects, args.folds, args.seed)
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "subject_splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "sklearn": sklearn.__version__, "device": "CPU", "threads": args.threads,
                "subjects": len(np.unique(subjects)), "windows": len(y), "recordings": len(manifest),
                "class_window_counts": np.bincount(y, minlength=3).tolist(),
                "unrated_subjects_excluded": sorted({5, 24, 42}),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Validated {len(manifest)} recordings, {len(y)} windows, {len(np.unique(subjects))} subjects.", flush=True)
    results, predictions, histories, recording_predictions, baselines, normalizers = [], [], [], [], [], []
    total = len(args.models) * 2 * args.folds
    for split in splits:
        fold = split["fold"]
        train = np.flatnonzero(np.isin(subjects, split["train_subjects"]))
        val = np.flatnonzero(np.isin(subjects, split["validation_subjects"]))
        test = np.flatnonzero(np.isin(subjects, split["test_subjects"]))
        for indices in [train, val, test]:
            if set(np.unique(y[indices])) != {0, 1, 2}:
                raise ValueError("A split lacks a class; revise the split protocol before training")
        majority = int(np.bincount(y[train], minlength=3).argmax())
        baselines.append({"fold": fold, "training_majority_class": CLASSES[majority],
                          **scores(y[test], np.full(len(test), majority))})
        write_csv(out / "majority_baseline.csv", baselines)
        for variant, data in variants.items():
            normalized, mean, std = normalize_from_training(data, train)
            normalizers.append({"fold": fold, "variant": variant, "mean": mean.flatten().tolist(), "std": std.flatten().tolist()})
            (out / "training_normalization.json").write_text(json.dumps(normalizers, indent=2), encoding="utf-8")
            x = torch.from_numpy(normalized)
            spectral = None
            if any(n.startswith("STFT") for n in args.models):
                spectral = spectrogram(x)
            for name in args.models:
                print(f"[{len(results)+1}/{total}] fold={fold} {variant} {name}", flush=True)
                inputs = spectral if name.startswith("STFT") else x
                result, probability, history = train_and_evaluate(name, inputs, y, train, val, test,
                                                                  args, args.seed + fold * 100)
                key = {"model": name, "variant": variant, "fold": fold}
                test_recordings = recordings[test]
                true_recordings, predicted_recordings = [], []
                for recording in np.unique(test_recordings):
                    selected = test_recordings == recording
                    average = probability[selected].mean(axis=0)
                    truth = int(y[test[selected]][0])
                    predicted = int(average.argmax())
                    true_recordings.append(truth)
                    predicted_recordings.append(predicted)
                    recording_predictions.append({**key, "recording": recording,
                         "subject": int(subjects[test[selected]][0]), "true_class": truth,
                         "predicted_class": predicted, **{f"probability_{c}": float(p) for c,p in zip(CLASSES, average)}})
                result["recording_accuracy"] = float(accuracy_score(true_recordings, predicted_recordings))
                results.append({**key, **result})
                for idx, probs in zip(test, probability):
                    predictions.append({**key, "subject": int(subjects[idx]), "recording": recordings[idx],
                         "start_sample": int(starts[idx]), "true_class": int(y[idx]), "predicted_class": int(probs.argmax()),
                         **{f"probability_{c}": float(p) for c,p in zip(CLASSES, probs)}})
                histories.extend([{**key, **entry} for entry in history])
                print(f"  accuracy={result['accuracy']:.4f} macro-F1={result['macro_f1']:.4f} "
                      f"best epoch={result['best_epoch']}/{result['epochs_run']} time={result['training_seconds']:.1f}s", flush=True)
                build_report(out, results, predictions, histories, recording_predictions, args, metadata)
            del spectral, x, normalized
    matrices = []
    for name in args.models:
        for variant in variants:
            rows = [p for p in predictions if p["model"] == name and p["variant"] == variant]
            matrix = confusion_matrix([p["true_class"] for p in rows], [p["predicted_class"] for p in rows], labels=[0, 1, 2])
            matrices.append({"model": name, "variant": variant, "labels": CLASSES, "matrix": matrix.tolist()})
    (out / "confusion_matrices.json").write_text(json.dumps(matrices, indent=2), encoding="utf-8")
    metadata["wall_seconds"] = time.perf_counter() - start
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    build_report(out, results, predictions, histories, recording_predictions, args, metadata)
    print(f"Complete. {len(results)} fits in {metadata['wall_seconds']:.1f}s. Results: {out}", flush=True)


if __name__ == "__main__":
    main()
