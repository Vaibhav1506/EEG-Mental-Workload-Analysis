"""STEW benchmark v3: preprocessing + training optimisation for the 12 model families.

Evaluation is unchanged from v1/v2: the same 45 rated subjects, three rating classes, five
subject-disjoint test folds (benchmark.make_splits, seed 42) and the same 3,330 evaluation
windows (4 s, stride 4 s). Test subjects are never used for training, model selection or
normalisation fitting, and their labels are only read when scoring.

What v3 adds (see Docs/BENCHMARK_V3.md):
  * Preprocessing variants. "filtered" = v1 filtering (50 Hz notch, 1-40 Hz band-pass, common
    average reference). "-ea" = Euclidean alignment per subject (He & Wu, IEEE TBME 2020):
    each subject's signals are whitened by the inverse square root of that subject's mean
    spatial covariance. It is computed from the subject's own *unlabelled* EEG, so it needs no
    labels, but it does assume some unlabelled EEG of a new user is available before
    prediction (here: their two recordings).
  * Inner cross-validation ensembles: the 36 development subjects of each outer fold are split
    into K groups; K networks are trained, each early-stopped on a different held-out group,
    so every development subject is used for training and for validation. Test predictions are
    the mean of all K x seeds networks. Out-of-fold development predictions give an honest
    validation score per fold, used for the Top-3 ensemble.
  * Training recipe options (crops, label smoothing, class weighting, mixup, checkpoint metric).

Several processes may share one --output folder: each (variant, model, fold) task is claimed
with a lock file, results are written per task, and --summarize builds the tables.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from scipy import signal
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss)
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import benchmark as v1  # noqa: E402  (architectures, splits, labels)
from benchmark import make_splits, rating_class  # noqa: E402

DEEP = list(v1.NAMES)
ENSEMBLE = "Top-3 Validation Ensemble"
SELECTED = "selected"  # per model and fold: the candidate preprocessing with the best development (OOF) accuracy
CLASSES = ["low", "moderate", "high"]  # replaced by ["rest", "task"] with --task 2class
TASK_CLASSES = {"3class": ["low", "moderate", "high"], "2class": ["rest", "task"]}
WINDOW, T = 512, 19200
VARIANTS = ["raw", "filtered", "raw-ea", "filtered-ea", "raw-sa", "filtered-sa", "raw-sa-ea", "filtered-sa-ea"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slug(text):
    return "".join(c if c.isalnum() else "_" for c in text)


def scores(y, p):
    return {"accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=list(range(len(CLASSES))), average="macro", zero_division=0))}


# ----------------------------------------------------------------------------- data
def load_recordings(folder, cache):
    """Continuous recordings (90, 14, 19200) float32 for raw and filtered, labels, subjects."""
    ratings = np.loadtxt(folder / "ratings.txt", delimiter=",", dtype=int)
    if ratings.ndim != 2 or ratings.shape[1] != 3 or len(np.unique(ratings[:, 0])) != len(ratings):
        raise ValueError("Expected unique subject,rating_lo,rating_hi rows")
    files = [folder / f"sub{s:02d}_{c}.txt" for s, _, _ in ratings for c in ["lo", "hi"]]
    hashes = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
    key = hashlib.sha256(("".join(hashes) + ratings.tobytes().hex()).encode()).hexdigest()[:16]
    manifest = []
    for (s, lo, hi), c, f, h in zip(np.repeat(ratings, 2, axis=0), ["lo", "hi"] * len(ratings), files, hashes):
        rating = lo if c == "lo" else hi
        manifest.append({"file": f.name, "subject": int(s), "condition": c, "rating": int(rating),
                         "class": int(rating_class(rating)), "sha256": h})
    cache_file = cache / f"v3_recordings_{key}.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        return {"raw": z["raw"], "filtered": z["filtered"]}, z["y"], z["subjects"], z["names"], z["ratings"], manifest
    sos = signal.butter(4, [1, 40], btype="bandpass", fs=128, output="sos")
    notch_b, notch_a = signal.iirnotch(50, 30, fs=128)
    raw, filtered, y, subjects, names, rates = [], [], [], [], [], []
    for f, m in zip(files, manifest):
        data = np.loadtxt(f, dtype=np.float64)
        if data.shape != (T, 14) or not np.isfinite(data).all():
            raise ValueError(f"Invalid recording: {f.name}, shape {data.shape}")
        clean = signal.filtfilt(notch_b, notch_a, data, axis=0)
        clean = signal.sosfiltfilt(sos, clean, axis=0)
        clean -= clean.mean(axis=1, keepdims=True)
        raw.append(data.T.astype(np.float32))
        filtered.append(clean.T.astype(np.float32))
        y.append(m["class"]); subjects.append(m["subject"]); names.append(f.stem); rates.append(m["rating"])
    out = dict(raw=np.stack(raw), filtered=np.stack(filtered), y=np.array(y), subjects=np.array(subjects),
               names=np.array(names), ratings=np.array(rates))
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f"tmp_{os.getpid()}.npz"
    np.savez(tmp, **out)
    try:
        os.replace(tmp, cache_file)  # atomic; another process may have written it already
    except OSError:
        tmp.unlink(missing_ok=True)
    return {"raw": out["raw"], "filtered": out["filtered"]}, out["y"], out["subjects"], out["names"], out["ratings"], manifest


def euclidean_alignment(cont, subjects):
    """Per-subject whitening R_s = C_s^(-1/2), C_s = mean covariance of the subject's own
    (DC-removed, non-overlapping 4 s) windows from all of that subject's recordings. No labels used.
    Pseudo-inverse square root for rank-deficient data (average reference makes rank 13)."""
    out = np.empty_like(cont)
    mats = {}
    for s in np.unique(subjects):
        ids = np.flatnonzero(subjects == s)
        C = np.zeros((14, 14))
        n = 0
        for r in ids:
            for st in range(0, T - WINDOW + 1, WINDOW):
                w = cont[r, :, st:st + WINDOW].astype(np.float64)
                w = w - w.mean(axis=1, keepdims=True)
                C += w @ w.T / WINDOW
                n += 1
        C /= n
        evals, V = np.linalg.eigh(C)
        keep = evals > 1e-6 * evals.max()
        R = (V[:, keep] * evals[keep] ** -0.5) @ V[:, keep].T
        out[ids] = np.einsum("cd,rdt->rct", R, cont[ids].astype(np.float64)).astype(np.float32)
        mats[int(s)] = R
    return out, mats


def spectral_alignment(cont, subjects, smooth_hz=1.0, band=(1.0, 40.0)):
    """Per-subject, per-channel spectral whitening: each channel's spectrum is divided by the square
    root of that subject's own average power spectrum (Welch, 4 s segments, all of the subject's
    recordings, smoothed to ~1 Hz), and limited to 1-40 Hz. Zero-phase, label-free. A window's band
    power then measures power *relative to that person's average*, the signal-domain analogue of
    per-subject feature standardisation."""
    R_, C_, Tn = cont.shape
    out = np.empty_like(cont)
    fr = np.fft.rfftfreq(Tn, 1 / 128)
    for s in np.unique(subjects):
        ids = np.flatnonzero(subjects == s)
        f, P = signal.welch(cont[ids].astype(np.float64), fs=128, nperseg=512, noverlap=256, axis=-1)
        P = P.mean(0)  # (C, F)
        k = max(1, int(round(smooth_hz / (f[1] - f[0]))))
        P = np.stack([np.convolve(v, np.ones(k) / k, mode="same") for v in P])
        W = 1 / np.sqrt(np.maximum(P, 1e-12))
        Wf = np.stack([np.interp(fr, f, W[c]) for c in range(C_)])
        Wf[:, (fr < band[0]) | (fr > band[1])] = 0
        for r in ids:
            x = cont[r].astype(np.float64)
            spec = np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)
            out[r] = np.fft.irfft(spec * Wf, n=Tn, axis=1).astype(np.float32)
    return out


def geometric_spectral_alignment(cont, subjects, band=(1.0, 40.0)):
    """Per-subject, per-channel spectral whitening by the subject's *geometric-mean* power spectrum
    (mean of log Welch PSD over the subject's own non-overlapping 4 s windows, 1 Hz resolution),
    limited to 1-40 Hz, zero-phase. After it, a window's log band power is approximately its log band
    power minus that person's average log band power: subject-centred spectra. Label-free."""
    R_, C_, Tn = cont.shape
    out = np.empty_like(cont)
    fr = np.fft.rfftfreq(Tn, 1 / 128)
    for s in np.unique(subjects):
        ids = np.flatnonzero(subjects == s)
        segs = np.stack([cont[r, :, st:st + WINDOW] for r in ids for st in range(0, Tn - WINDOW + 1, WINDOW)])
        segs = segs.astype(np.float64)
        f, P = signal.welch(segs - segs.mean(axis=-1, keepdims=True), fs=128, nperseg=128, noverlap=64, axis=-1)
        G = np.exp(np.log(P + 1e-12).mean(axis=0))  # (C, F)
        W = 1 / np.sqrt(G)
        Wf = np.stack([np.interp(fr, f, W[c]) for c in range(C_)])
        Wf[:, (fr < band[0]) | (fr > band[1])] = 0
        for r in ids:
            x = cont[r].astype(np.float64)
            spec = np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)
            out[r] = np.fft.irfft(spec * Wf, n=Tn, axis=1).astype(np.float32)
    return out


def valid_variant(variant):
    parts = variant.split("-")
    return parts[0] in {"raw", "filtered"} and all(p in {"ea", "sa", "sag"} for p in parts[1:]) and \
        len(set(parts[1:])) == len(parts[1:])


def prepare_variant(variant, conts, subjects):
    """variant = <raw|filtered>[-step...]; steps (ea, sa, sag) applied in the order written."""
    parts = variant.split("-")
    cont = conts[parts[0]]
    for step in parts[1:]:
        if step == "ea":
            cont, _ = euclidean_alignment(cont, subjects)
        elif step == "sa":
            cont = spectral_alignment(cont, subjects)
        elif step == "sag":
            cont = geometric_spectral_alignment(cont, subjects)
    return cont


def window_grid(recording_ids, stride):
    starts = np.arange(0, T - WINDOW + 1, stride)
    return np.repeat(recording_ids, len(starts)), np.tile(starts, len(recording_ids))


class Windows:
    """Cuts (B,14,512) windows from continuous recordings kept on the device; per-window DC
    removal, then per-channel normalisation fitted on training windows only."""
    def __init__(self, continuous, device, demean=True):
        self.data = torch.from_numpy(continuous).to(device).transpose(1, 2).contiguous()  # R,T,C
        self.offsets = torch.arange(WINDOW, device=device)
        self.device, self.demean = device, demean
        self.mean = torch.zeros(1, 14, 1, device=device)
        self.std = torch.ones(1, 14, 1, device=device)

    def cut(self, rec, start, normalize=True):
        rec = torch.as_tensor(rec, device=self.device)
        start = torch.as_tensor(start, device=self.device)
        x = self.data[rec[:, None], start[:, None] + self.offsets[None]].transpose(1, 2)
        if self.demean:
            x = x - x.mean(dim=2, keepdim=True)
        return (x - self.mean) / self.std if normalize else x

    def fit_normalizer(self, rec, start):
        total, total_sq, count = 0, 0, 0
        for i in range(0, len(rec), 512):
            x = self.cut(rec[i:i + 512], start[i:i + 512], normalize=False).double()
            total = total + x.sum(dim=(0, 2))
            total_sq = total_sq + x.square().sum(dim=(0, 2))
            count += x.shape[0] * x.shape[2]
        mean = total / count
        std = (total_sq / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
        self.mean, self.std = mean.float().reshape(1, 14, 1), std.float().reshape(1, 14, 1)


def spectrogram(x):
    """Same transform as v1.spectrogram, on the input's device."""
    window = torch.hann_window(64, device=x.device)
    spec = torch.stft(x.reshape(-1, WINDOW), n_fft=64, hop_length=16, win_length=64, window=window,
                      center=False, return_complex=True)
    spec = torch.log1p(spec.abs().square()).reshape(len(x), 14, 33, 29)
    return F.interpolate(spec, size=(32, 32), mode="bilinear", align_corners=False)


class SpectralWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            spec = spectrogram(x)
        return self.model(spec)


def make_model(name):
    model = v1.make_model(name)
    if len(CLASSES) != 3:  # same architecture, output layer sized for the task
        last_name, last = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]  # output layer
        parent = model.get_submodule(last_name.rsplit(".", 1)[0]) if "." in last_name else model
        setattr(parent, last_name.rsplit(".", 1)[-1], nn.Linear(last.in_features, len(CLASSES)))
    return SpectralWrapper(model) if name.startswith("STFT") else model


# ----------------------------------------------------------------------------- training
def predict(model, windows, rec, start, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rec), batch_size):
            out.append(model(windows.cut(rec[i:i + batch_size], start[i:i + batch_size])).float().softmax(1).cpu())
    return torch.cat(out).numpy()


def selection_value(metric, y, prob):
    pred = prob.argmax(1)
    if metric == "loss":
        return -log_loss(y, np.clip(prob, 1e-7, 1), labels=list(range(len(CLASSES))))
    if metric == "acc":
        return accuracy_score(y, pred)
    if metric == "bacc":
        return balanced_accuracy_score(y, pred)
    if metric == "accf1":
        return 0.5 * accuracy_score(y, pred) + 0.5 * f1_score(y, pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
    return f1_score(y, pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0)


def augment(x, args):
    b = len(x)
    if args.scale_jitter > 0:
        x = x * torch.exp(args.scale_jitter * torch.randn(b, 1, 1, device=x.device))
    if args.noise > 0:
        x = x + args.noise * torch.randn_like(x)
    if args.channel_dropout > 0:
        x = x * (torch.rand(b, 14, 1, device=x.device) >= args.channel_dropout).float()
    return x


def train_one(name, windows, y_rec, train_rec, val_rec, args, seed, device, heartbeat=None):
    seed_everything(seed)
    model = make_model(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    grid_rec, grid_start = window_grid(train_rec, args.train_stride)
    nc = len(CLASSES)
    counts = np.bincount(y_rec[grid_rec], minlength=nc)
    if np.any(counts == 0):
        raise ValueError("Training split is missing a class")
    if args.class_weight == "balanced":
        weights = torch.tensor(len(grid_rec) / (nc * counts), dtype=torch.float32, device=device)
    else:
        weights = None
    labels = torch.from_numpy(y_rec).to(device)
    steps_per_epoch = math.ceil(len(grid_rec) / args.batch_size)
    total_steps, warmup = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    if args.constant_lr:  # v1-style baseline
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    else:
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, max(0, s - warmup) / max(1, total_steps - warmup)))))
    v_rec, v_start = window_grid(val_rec, WINDOW)
    v_y = y_rec[v_rec]
    best, best_state, best_epoch, history, stale = -np.inf, None, 0, [], 0
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(grid_rec))
        jitter = rng.integers(0, args.train_stride, len(order)) if args.train_stride > 1 and not args.no_jitter \
            else np.zeros(len(order), int)
        starts = np.minimum(grid_start[order] + jitter, T - WINDOW)
        recs = grid_rec[order]
        total_loss = 0.0
        for i in range(0, len(order), args.batch_size):
            r, s = recs[i:i + args.batch_size], starts[i:i + args.batch_size]
            x = augment(windows.cut(r, s), args)
            target = F.one_hot(labels[torch.as_tensor(r, device=device)], nc).float()
            target = target * (1 - args.label_smoothing) + args.label_smoothing / nc
            if args.mixup > 0:
                lam = float(rng.beta(args.mixup, args.mixup))
                perm = torch.randperm(len(x), device=device)
                x = lam * x + (1 - lam) * x[perm]
                target = lam * target + (1 - lam) * target[perm]
            optimizer.zero_grad(set_to_none=True)
            logp = F.log_softmax(model(x), dim=1)
            per = -(target * logp)
            if weights is not None:
                per = per * weights[None]
                loss = per.sum() / (target * weights[None]).sum()
            else:
                loss = per.sum(1).mean()
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss for {name}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total_loss += float(loss.detach()) * len(r)
        vp = predict(model, windows, v_rec, v_start)
        value = selection_value(args.select, v_y, vp)
        history.append({"epoch": epoch, "training_loss": total_loss / len(order), "selection": float(value),
                        **scores(v_y, vp.argmax(1))})
        if value > best + 1e-6:
            best, best_state, best_epoch, stale = value, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if heartbeat:
            heartbeat()
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": len(history), "selection_value": float(best),
                   "training_seconds": time.perf_counter() - started,
                   "parameters": sum(p.numel() for p in model.parameters())}, history


def inner_groups(dev_subjects, k, seed):
    dev_subjects = np.array(sorted(dev_subjects))
    return [dev_subjects[te] for _, te in KFold(k, shuffle=True, random_state=seed).split(dev_subjects)]


# ----------------------------------------------------------------------------- task runner
class Claim:
    """Lock file per task so several processes can share one output folder."""
    STALE = 900  # seconds without heartbeat

    def __init__(self, path):
        self.path = path

    def acquire(self):
        try:
            if self.path.exists() and time.time() - self.path.stat().st_mtime > self.STALE:
                self.path.unlink()
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {platform.node()} {now()}".encode())
            os.close(fd)
            return True
        except (FileExistsError, PermissionError):
            return False

    def beat(self):
        try:
            os.utime(self.path, None)
        except OSError:
            pass

    def release(self):
        try:
            self.path.unlink()
        except OSError:
            pass


def run_task(name, variant, split, windows, y_rec, subj_rec, args, device, claim):
    fold = split["fold"]
    dev = sorted(split["train_subjects"] + split["validation_subjects"])
    test_ids = np.flatnonzero(np.isin(subj_rec, split["test_subjects"]))
    test_rec, test_start = window_grid(test_ids, WINDOW)
    if args.inner_folds > 1:
        groups = inner_groups(dev, args.inner_folds, args.seed + fold)
    else:  # v1/v2 split: one validation group
        groups = [np.array(split["validation_subjects"])]
    dev_ids = np.flatnonzero(np.isin(subj_rec, dev))
    dev_rec, dev_start = window_grid(dev_ids, WINDOW)
    oof = np.zeros((len(dev_rec), len(CLASSES)))
    oof_n = np.zeros(len(dev_rec))
    test_probs, fits, history = [], [], []
    infer_seconds = 0.0
    for g, held in enumerate(groups):
        val_ids = np.flatnonzero(np.isin(subj_rec, held))
        train_ids = np.flatnonzero(np.isin(subj_rec, [s for s in dev if s not in set(held.tolist())]))
        if args.inner_folds <= 1:
            train_ids = np.flatnonzero(np.isin(subj_rec, split["train_subjects"]))
        for k in range(args.seeds):
            seed = args.seed + fold * 1000 + g * 10 + k + 100000 * args.init_offset
            model, info, hist = train_one(name, windows, y_rec, train_ids, val_ids, args, seed, device, claim.beat)
            sel = np.isin(dev_rec, val_ids)
            oof[sel] += predict(model, windows, dev_rec[sel], dev_start[sel])
            oof_n[sel] += 1
            predict(model, windows, test_rec[:16], test_start[:16])
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            test_probs.append(predict(model, windows, test_rec, test_start, 128))
            infer_seconds += time.perf_counter() - t0
            fits.append({"group": g, "seed": k, **info})
            history.extend([{"group": g, "seed": k, **h} for h in hist])
            print(f"    {name} {variant} fold {fold} group {g} seed {k}: {args.select}={info['selection_value']:.4f} "
                  f"best epoch {info['best_epoch']}/{info['epochs_run']} {info['training_seconds']:.1f}s", flush=True)
            del model
    keep = oof_n > 0
    oof_p = oof[keep] / oof_n[keep, None]
    return {"test_probability": np.mean(test_probs, 0), "test_rec": test_rec, "test_start": test_start,
            "dev_probability": oof_p, "dev_rec": dev_rec[keep], "dev_start": dev_start[keep],
            "fits": fits, "history": history, "inference_ms_per_window": infer_seconds * 1000 / len(test_rec)}


def save_result(out, name, variant, fold, res, y_rec, args):
    base = out / "parts" / f"{slug(variant)}__{slug(name)}__fold{fold}"
    np.savez(str(base) + ".npz", test_probability=res["test_probability"], test_rec=res["test_rec"],
             test_start=res["test_start"], dev_probability=res["dev_probability"], dev_rec=res["dev_rec"],
             dev_start=res["dev_start"])
    dev_y = y_rec[res["dev_rec"]]
    meta = {"model": name, "variant": variant, "fold": fold, "completed_utc": now(),
            "dev_oof": scores(dev_y, res["dev_probability"].argmax(1)),
            "fits": res["fits"], "inference_ms_per_window": res["inference_ms_per_window"],
            "parameters": res["fits"][0]["parameters"],
            "training_seconds": float(sum(f["training_seconds"] for f in res["fits"]))}
    tmp = Path(str(base) + ".json.tmp")
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(tmp, str(base) + ".json")
    with open(str(base) + "_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res["history"][0]))
        w.writeheader()
        w.writerows(res["history"])


# ----------------------------------------------------------------------------- summary
def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def summarize(out, y_rec, subj_rec, names, models, variants, splits, no_select=False):
    parts = out / "parts"
    loaded = {}
    for js in parts.glob("*.json"):
        meta = json.loads(js.read_text(encoding="utf-8"))
        z = np.load(str(js)[:-5] + ".npz")
        if meta["variant"] in variants and meta["model"] in models:
            loaded[(meta["model"], meta["variant"], meta["fold"])] = (meta, {k: z[k] for k in z.files})
    variants = list(variants)
    (out / "audit").mkdir(exist_ok=True)
    if not no_select:
        (out / "audit" / "selection.json").write_text(json.dumps(
            {"candidate_variants": variants, "models": models,
             "rule": "per model and outer fold, the candidate with the highest out-of-fold accuracy on that fold's "
                     "development subjects (ties -> earlier candidate)"}, indent=2), encoding="utf-8")
    if len(variants) > 1 and not no_select:
        # Nested preprocessing choice: for each model and outer fold, keep the candidate variant with the
        # highest out-of-fold accuracy on that fold's development subjects (test subjects play no part).
        for name in models:
            configured = [v for v in variants if any(k[0] == name and k[1] == v for k in loaded)]
            for sp in splits:
                fold = sp["fold"]
                cands = [v for v in configured if (name, v, fold) in loaded]
                if not cands or len(cands) < len(configured):
                    continue
                best = max(cands, key=lambda v: (loaded[(name, v, fold)][0]["dev_oof"]["accuracy"], -variants.index(v)))
                meta, data = loaded[(name, best, fold)]
                loaded[(name, SELECTED, fold)] = ({**meta, "variant": SELECTED, "chosen_variant": best}, data)
        variants = variants + [SELECTED]
    # Top-3 ensemble per (variant, fold): members ranked by out-of-fold development macro-F1.
    for variant in variants:
        for sp in splits:
            fold = sp["fold"]
            avail = [m for m in models if (m, variant, fold) in loaded]
            if len(avail) < len(models) or len(avail) < 3:
                continue
            ranked = sorted(avail, key=lambda m: -loaded[(m, variant, fold)][0]["dev_oof"]["macro_f1"])
            members = ranked[:3]
            arrs = [loaded[(m, variant, fold)][1] for m in members]
            meta = {"model": ENSEMBLE, "variant": variant, "fold": fold, "members": " + ".join(members),
                    "parameters": sum(loaded[(m, variant, fold)][0]["parameters"] for m in members),
                    "training_seconds": sum(loaded[(m, variant, fold)][0]["training_seconds"] for m in members),
                    "inference_ms_per_window": sum(loaded[(m, variant, fold)][0]["inference_ms_per_window"] for m in members)}
            data = {"test_probability": np.mean([a["test_probability"] for a in arrs], 0),
                    "test_rec": arrs[0]["test_rec"], "test_start": arrs[0]["test_start"],
                    "dev_probability": np.mean([a["dev_probability"] for a in arrs], 0), "dev_rec": arrs[0]["dev_rec"]}
            meta["dev_oof"] = scores(y_rec[data["dev_rec"]], data["dev_probability"].argmax(1))
            loaded[(ENSEMBLE, variant, fold)] = (meta, data)
    fold_rows, rec_rows, win_rows, matrices = [], [], [], []
    for (name, variant, fold), (meta, d) in sorted(loaded.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        sp = next(s for s in splits if s["fold"] == fold)
        assert set(subj_rec[d["test_rec"]]) == set(sp["test_subjects"]), "test windows must be the fold's test subjects"
        y = y_rec[d["test_rec"]]
        p = d["test_probability"]
        truth, pred = [], []
        for r in np.unique(d["test_rec"]):
            sel = d["test_rec"] == r
            avg = p[sel].mean(0)
            truth.append(int(y_rec[r])); pred.append(int(avg.argmax()))
            rec_rows.append({"model": name, "variant": variant, "fold": fold, "recording": names[r],
                             "subject": int(subj_rec[r]), "true_class": int(y_rec[r]), "predicted_class": int(avg.argmax()),
                             **{f"probability_{c}": round(float(v), 6) for c, v in zip(CLASSES, avg)}})
        for r, s, t, pr in zip(d["test_rec"], d["test_start"], y, p):
            win_rows.append({"model": name, "variant": variant, "fold": fold, "recording": names[r], "start_sample": int(s),
                             "true_class": int(t), "predicted_class": int(pr.argmax()),
                             **{f"probability_{c}": round(float(v), 6) for c, v in zip(CLASSES, pr)}})
        fold_rows.append({"model": name, "variant": variant, "fold": fold, **scores(y, p.argmax(1)),
                          "recording_accuracy": float(accuracy_score(truth, pred)),
                          "dev_oof_accuracy": meta["dev_oof"]["accuracy"], "dev_oof_macro_f1": meta["dev_oof"]["macro_f1"],
                          "members": meta.get("members", ""), "chosen_variant": meta.get("chosen_variant", ""),
                          "parameters": meta["parameters"],
                          "training_seconds": meta["training_seconds"],
                          "inference_ms_per_window": meta["inference_ms_per_window"]})
    summaries = []
    order = models + ([ENSEMBLE] if any(k[0] == ENSEMBLE for k in loaded) else [])
    for variant in variants:
        for name in order:
            rows = [r for r in fold_rows if r["model"] == name and r["variant"] == variant]
            if not rows:
                continue
            s = {"model": name, "variant": variant, "folds_completed": len(rows)}
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "recording_accuracy", "dev_oof_accuracy", "dev_oof_macro_f1"]:
                v = [r[metric] for r in rows]
                s[metric + "_mean"] = float(np.mean(v))
                s[metric + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
            s["parameters"] = rows[0]["parameters"]
            s["training_seconds_total"] = float(sum(r["training_seconds"] for r in rows))
            summaries.append(s)
            if len(rows) == len(splits):
                wr = [w for w in win_rows if w["model"] == name and w["variant"] == variant]
                m = confusion_matrix([w["true_class"] for w in wr], [w["predicted_class"] for w in wr], labels=list(range(len(CLASSES))))
                matrices.append({"model": name, "variant": variant, "labels": CLASSES, "matrix": m.tolist()})
    (out / "details").mkdir(exist_ok=True)
    write_csv(out / "details" / "fold_metrics.csv", fold_rows)
    write_csv(out / "details" / "recording_predictions.csv", rec_rows)
    write_csv(out / "details" / "window_predictions.csv", win_rows)
    (out / "details" / "confusion_matrices.json").write_text(json.dumps(matrices, indent=1), encoding="utf-8")
    write_csv(out / "summary.csv", summaries)
    return summaries, fold_rows


def report(out, summaries, args, metadata, baseline):
    v1_path = ROOT / "results" / "initial_5fold" / "summary.csv"
    before = {}
    if v1_path.exists() and args.task == "3class":
        with v1_path.open(newline="", encoding="utf-8") as f:
            before = {(r["model"], r["variant"]): r for r in csv.DictReader(f)}
    pct = lambda m, sd: f"{100*m:.2f} ± {100*(sd or 0):.2f}"
    lines = [f"# STEW v3 — optimised preprocessing and training", "",
             f"Updated {now()}. Device: {metadata.get('device_name', '?')}.", "",
             (f"Task: {args.task} ({' / '.join(CLASSES)}). " if args.task != "3class" else "") +
             "Evaluation identical to v1/v2: 45 rated subjects, 3 classes (ratings 1-3 / 4-6 / 7-9), five subject-disjoint "
             "test folds (seed 42), 3,330 test windows of 4 s. Values are mean ± SD over the five test folds.", "",
             f"Training-majority baseline accuracy: {100*np.mean(baseline):.2f}%.", ""]
    for variant in list(dict.fromkeys(s["variant"] for s in summaries)):
        rows = [s for s in summaries if s["variant"] == variant]
        if not rows:
            continue
        lines += [f"## Variant `{variant}`", "",
                  "| Model | Folds | Accuracy % | Balanced acc. % | Macro-F1 | Recording acc. % | v1 raw acc. % (before) | Change (pp) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s in rows:
            old = before.get((s["model"], "raw"))
            old_acc = f"{100*float(old['accuracy_mean']):.2f}" if old else "—"
            change = f"{100*(s['accuracy_mean']-float(old['accuracy_mean'])):+.2f}" if old else "—"
            lines.append(f"| {s['model']} | {s['folds_completed']} | {pct(s['accuracy_mean'], s['accuracy_sd'])} | "
                         f"{pct(s['balanced_accuracy_mean'], s['balanced_accuracy_sd'])} | {s['macro_f1_mean']:.3f} | "
                         f"{pct(s['recording_accuracy_mean'], s['recording_accuracy_sd'])} | {old_acc} | {change} |")
        lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "Dataset")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=DEEP, default=DEEP)
    p.add_argument("--variants", nargs="+", default=["filtered-ea-sag"],
                   help="<raw|filtered>[-ea][-sa|-sag] steps in order, e.g. filtered-ea-sag")
    p.add_argument("--task", choices=["3class", "2class"], default="3class",
                   help="3class = rating classes 1-3/4-6/7-9 (default); 2class = rest (lo) vs task (hi) recording")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--run-folds", type=int, nargs="+", default=None)
    p.add_argument("--inner-folds", type=int, default=4, help="K inner CV networks per outer fold (1 = v1 train/val split)")
    p.add_argument("--seeds", type=int, default=1, help="initialisations per inner fold")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--train-stride", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    p.add_argument("--select", choices=["f1", "acc", "bacc", "loss", "accf1"], default="f1")
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--scale-jitter", type=float, default=0.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--channel-dropout", type=float, default=0.0)
    p.add_argument("--no-demean", action="store_true", help="v1-style: keep each window's DC offset")
    p.add_argument("--constant-lr", action="store_true", help="v1-style: constant learning rate (no warm-up/cosine)")
    p.add_argument("--no-jitter", action="store_true", help="v1-style: fixed training windows (no random crop offsets)")
    p.add_argument("--seed", type=int, default=42, help="fold/split seed (keep 42 to match v1)")
    p.add_argument("--init-offset", type=int, default=0,
                   help="replicate run: different network initialisations and crop order, same folds and inner groups")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--mem-fraction", type=float, default=0.0, help="cap this process's share of GPU memory (0 = no cap)")
    p.add_argument("--summarize", action="store_true", help="only rebuild summary/report from finished parts")
    p.add_argument("--no-select", action="store_true", help="with --summarize: no per-model 'selected' rows "
                   "(assembled folders carry their own pre-declared choice, see assemble_v3.py)")
    args = p.parse_args()
    bad = [v for v in args.variants if not valid_variant(v)]
    if bad and not args.summarize:
        p.error(f"unknown variant(s): {bad}")
    args.data, args.output = args.data.resolve(), args.output.resolve()
    out = args.output
    (out / "parts").mkdir(parents=True, exist_ok=True)
    (out / "claims").mkdir(exist_ok=True)
    (out / "audit").mkdir(exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if device.type == "cuda" and args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device.index or 0)
    source = Path(__file__).read_bytes()
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                if k not in {"output", "run_folds", "threads", "summarize", "no_select", "mem_fraction", "device", "data",
                             "models", "variants"}}
    settings_file = out / "audit" / "settings.json"
    if settings_file.exists():
        old = json.loads(settings_file.read_text(encoding="utf-8"))
        old["settings"] = {"task": "3class", "no_demean": False, "constant_lr": False, "no_jitter": False,
                           "init_offset": 0, **old["settings"]}  # older folders
        old["settings"].pop("no_select", None)
        if old["settings"] != settings and not args.summarize:
            raise SystemExit(f"{out} was started with different settings: {old['settings']}")
        if old["code_sha256"] != hashlib.sha256(source).hexdigest() and not args.summarize:
            print("WARNING: benchmark_v3.py changed since this folder was started", flush=True)
    else:
        settings_file.write_text(json.dumps({"settings": settings, "code_sha256": hashlib.sha256(source).hexdigest(),
                                             "created_utc": now()}, indent=2), encoding="utf-8")
        (out / "audit" / "benchmark_v3_source.py").write_bytes(source)
    print(f"[{now()}] loading recordings...", flush=True)
    conts, y_rec, subj_rec, names, ratings, manifest = load_recordings(args.data, ROOT / "outputs" / "cache")
    global CLASSES
    CLASSES = TASK_CLASSES[args.task]
    if args.task == "2class":
        y_rec = np.array([0 if str(n).endswith("_lo") else 1 for n in names])
    eval_rec, _ = window_grid(np.arange(len(names)), WINDOW)
    splits = make_splits(subj_rec[eval_rec], args.folds, args.seed)
    (out / "audit" / "subject_splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (out / "audit" / "dataset_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else (platform.processor() or "CPU")
    metadata = {"device_name": device_name, "torch": torch.__version__, "python": sys.version.split()[0],
                "platform": platform.platform()}
    (out / "audit" / f"process_{platform.node()}_{os.getpid()}.json").write_text(json.dumps(
        {**metadata, "started_utc": now(), "run_folds": args.run_folds, "models": args.models,
         "variants": args.variants}, indent=2), encoding="utf-8")
    baseline = []
    for sp in splits:
        tr = np.isin(subj_rec, sp["train_subjects"] + sp["validation_subjects"])
        te = np.isin(subj_rec, sp["test_subjects"])
        majority = np.bincount(y_rec[tr], minlength=len(CLASSES)).argmax()
        baseline.append(float(np.mean(y_rec[te] == majority)))  # every recording has 37 windows
    if args.summarize:
        summaries, _ = summarize(out, y_rec, subj_rec, names, args.models, args.variants, splits, args.no_select)
        report(out, summaries, args, metadata, baseline)
        print((out / "REPORT.md").read_text(encoding="utf-8"))
        return
    prepared = {}
    # model-major order, so each model's folds (and variants) finish close together
    tasks = [(v, m, sp) for m in args.models for sp in splits for v in args.variants
             if args.run_folds is None or sp["fold"] in args.run_folds]
    done_now = 0
    for variant, name, sp in tasks:
        fold = sp["fold"]
        base = out / "parts" / f"{slug(variant)}__{slug(name)}__fold{fold}"
        if Path(str(base) + ".json").exists():
            continue
        claim = Claim(out / "claims" / f"{slug(variant)}__{slug(name)}__fold{fold}.lock")
        if not claim.acquire():
            continue
        try:
            if Path(str(base) + ".json").exists():
                continue
            if variant not in prepared:
                prepared[variant] = Windows(prepare_variant(variant, conts, subj_rec), device, demean=not args.no_demean)
            windows = prepared[variant]
            # normalisation fitted on this fold's development (training) subjects only
            dev_ids = np.flatnonzero(np.isin(subj_rec, sp["train_subjects"] + sp["validation_subjects"]))
            windows.fit_normalizer(*window_grid(dev_ids, WINDOW))
            print(f"[{now()}] start {variant} {name} fold {fold}", flush=True)
            t0 = time.perf_counter()
            res = run_task(name, variant, sp, windows, y_rec, subj_rec, args, device, claim)
            save_result(out, name, variant, fold, res, y_rec, args)
            y = y_rec[res["test_rec"]]
            sc = scores(y, res["test_probability"].argmax(1))
            dev_sc = scores(y_rec[res["dev_rec"]], res["dev_probability"].argmax(1))
            print(f"[{now()}] done  {variant} {name} fold {fold}: dev-OOF acc={dev_sc['accuracy']:.4f} "
                  f"F1={dev_sc['macro_f1']:.4f} | test acc={sc['accuracy']:.4f} F1={sc['macro_f1']:.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
            done_now += 1
            try:
                summaries, _ = summarize(out, y_rec, subj_rec, names, args.models, args.variants, splits)
                report(out, summaries, args, metadata, baseline)
            except Exception as e:  # noqa: BLE001  (another process may be writing)
                print(f"  (summary skipped: {e})", flush=True)
        finally:
            claim.release()
    print(f"[{now()}] process finished: {done_now} tasks completed here", flush=True)


if __name__ == "__main__":
    main()
