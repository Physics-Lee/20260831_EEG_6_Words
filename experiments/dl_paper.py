"""Train GRU / LSTM / ViT on the same 3-condition task and CV splits as the
paper baselines (ml_paper.py). Input: unified epochs (N, 38, 500), per-channel
z-scored with train-fold statistics.

Usage:  python dl_paper.py [--subjects russian/sub1 ...] [--models gru lstm vit]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dl_models import build_model, count_parameters

sys.path.insert(0, str(Path(__file__).parent))
from common import (CLASSES, RESULTS, leave_one_session_out_splits,
                    load_three_class, subject_list)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_EPOCHS = 60
PATIENCE = 8
BATCH = 64


def train_one_fold(Xtr, ytr, Xte, yte, model_name):
    torch.manual_seed(SEED)
    # per-channel z-score with train stats
    mu, sd = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    counts = np.bincount(ytr, minlength=3)
    w = torch.tensor(len(ytr) / (3 * np.maximum(counts, 1)), dtype=torch.float32,
                     device=DEVICE)

    # carve 15% of train folds for early stopping (stratified-ish by rounding)
    rng = np.random.RandomState(SEED)
    val_idx = np.concatenate([rng.choice(np.where(ytr == k)[0],
                                         max(1, int(0.15 * (ytr == k).sum())),
                                         replace=False)
                              for k in range(3)])
    tr_mask = np.ones(len(ytr), bool); tr_mask[val_idx] = False

    def ds(mask):
        return TensorDataset(torch.as_tensor(Xtr[mask], dtype=torch.float32),
                             torch.as_tensor(ytr[mask], dtype=torch.long))

    dl = DataLoader(ds(tr_mask), batch_size=BATCH, shuffle=True)
    dv = DataLoader(ds(val_idx), batch_size=256)

    model = build_model(model_name, n_channels=Xtr.shape[1], n_time=Xtr.shape[2],
                        device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)

    best_val, best_state, bad = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = [model(xb.to(DEVICE)).argmax(1).cpu() for xb, _ in dv]
            val_acc = (torch.cat(preds) == ytr[val_idx]).float().mean().item()
        if val_acc > best_val + 1e-4:
            best_val, best_state, bad = val_acc, \
                {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = [model(xb.to(DEVICE)).argmax(1).cpu()
                 for xb in torch.split(torch.as_tensor(Xte, dtype=torch.float32), 256)]
    pred = torch.cat(preds).numpy()
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    return (accuracy_score(yte, pred), balanced_accuracy_score(yte, pred),
            epoch + 1, count_parameters(model))


def run_subject(lang, sub, models):
    X, y, samples = load_three_class(lang, sub)
    X = X.astype(np.float32)
    splits = leave_one_session_out_splits(y, samples)
    rows = []
    for name in models:
        t0 = time.time()
        accs, baccs = [], []
        for tr, te in splits:
            acc, bacc, ep, n_par = train_one_fold(X[tr], y[tr], X[te], y[te], name)
            accs.append(acc); baccs.append(bacc)
        rows.append(dict(language=lang, subject=sub, model=name,
                         n_trials=len(y),
                         acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                         balanced_acc=float(np.mean(baccs)),
                         n_folds=len(splits),
                         params=n_par, secs=round(time.time() - t0, 1)))
        print(f"  [{lang}/{sub}] {name:5s} acc={np.mean(accs):.3f} "
              f"bacc={np.mean(baccs):.3f} ({rows[-1]['secs']}s, "
              f"{n_par/1e6:.2f}M params)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=["gru", "lstm", "vit"])
    args = ap.parse_args()
    subs = [a.split("/") for a in args.subjects] if args.subjects else subject_list()
    print(f"device={DEVICE}")

    all_rows = []
    for lang, sub in subs:
        print(f"== {lang}/{sub}", flush=True)
        all_rows += run_subject(lang, sub, args.models)

    df = pd.DataFrame(all_rows)
    out = RESULTS / "dl_results.csv"
    df.to_csv(out, index=False)
    print("\n=== per-model summary (mean over subjects) ===")
    print(df.groupby("model")[["acc_mean", "balanced_acc"]].agg(["mean", "std"]).round(3))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
