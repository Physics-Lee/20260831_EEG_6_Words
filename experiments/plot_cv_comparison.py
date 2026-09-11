"""Bar chart comparing random 5-fold (leaky) vs leave-one-session-out CV,
plus balanced accuracy and the paper-reported values. Saves to
results/cv_comparison.png (does NOT touch summary.png)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS

MODELS = [("lda", "LDA", 67), ("svm_linear", "SVM (linear)", 74),
          ("gru", "GRU", None), ("lstm", "LSTM", None),
          ("rf", "RandomForest", 78), ("vit", "ViT", None)]

# random 5-fold (leaky) accuracy, from results/*_stratified5.csv
old = pd.concat([pd.read_csv(RESULTS / "ml_results_stratified5.csv"),
                 pd.read_csv(RESULTS / "dl_results_stratified5.csv")])
o = old.groupby("model")["acc_mean"].mean()

# LOSO accuracy + SD across subjects, and balanced accuracy, from results/*.csv
new = pd.concat([pd.read_csv(RESULTS / "ml_results.csv"),
                 pd.read_csv(RESULTS / "dl_results.csv")])
g = new.groupby("model")
n_acc, n_std, n_bacc = g["acc_mean"].mean(), g["acc_mean"].std(), g["balanced_acc"].mean()

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(MODELS))
w = 0.35

b1 = ax.bar(x - w / 2, [o[m] * 100 for m, _, _ in MODELS], w,
            color="#8FB3DE", alpha=0.9,
            label="random 5-fold CV (leaky: adjacent trials shared across train/test)")
b2 = ax.bar(x + w / 2, [n_acc[m] * 100 for m, _, _ in MODELS], w,
            yerr=[n_std[m] * 100 for m, _, _ in MODELS],
            capsize=5, color="#4C72B0",
            error_kw=dict(elinewidth=1.4, ecolor="black", capthick=1.4),
            label="leave-one-session-out CV (leak-free) — bar: mean over 22 subjects\n"
                  "error bar: ±1 SD across subjects")

# balanced accuracy: small marker on top of LOSO bars
ax.scatter(x + w / 2, [n_bacc[m] * 100 for m, _, _ in MODELS],
           marker="D", s=45, color="#DD8452", zorder=4,
           label="orange diamond: balanced accuracy (LOSO)")

# paper-reported values
for i, (_, label, paper) in enumerate(MODELS):
    if paper is not None:
        ax.hlines(paper, i - w, i + w, colors="red", linestyles="--", lw=2,
                  label="red dash: value reported in the paper" if i == 0 else None)
        ax.text(i + w + 0.06, paper, f"{paper}%", color="red", fontsize=8.5,
                va="center")

# value labels on bars
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2,
                f"{r.get_height():.1f}", ha="center", fontsize=8.5)

ax.axhline(100 / 3, color="gray", ls="--", lw=1)
ax.text(len(MODELS) - 0.55, 100 / 3 + 1.2, "chance 33.3%", ha="right",
        color="gray", fontsize=9)

ax.set_xticks(x, [label for _, label, _ in MODELS])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("3-condition classification (rest / overt / inner), 22 subjects\n"
             "random 5-fold vs leave-one-session-out CV — same models & features, "
             "only the split differs")
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = RESULTS / "cv_comparison.png"
fig.savefig(out, dpi=150)
print(f"saved -> {out}")
