"""Summarize ML vs DL results: per-model mean +- std over 22 subjects, and the
paper's reported numbers for reference. Produces results/summary.csv and a
bar plot results/summary.png."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS

PAPER = {"rf": 78, "svm_linear": 74, "lda": 67}  # paper Fig./text, %, mean +- 4/5/6

MODEL_LABELS = {
    "rf": "RandomForest", "svm_linear": "SVM(linear)", "lda": "LDA",
    "gru": "GRU", "lstm": "LSTM", "vit": "ViT",
}
MODEL_ORDER = ["rf", "svm_linear", "lda", "gru", "lstm", "vit"]  # ML first, DL second


def main():
    ml = pd.read_csv(RESULTS / "ml_results.csv")
    dl = pd.read_csv(RESULTS / "dl_results.csv")
    df = pd.concat([ml, dl], ignore_index=True)

    agg = (df.groupby("model")
             .agg(acc_mean=("acc_mean", "mean"), acc_std_across_subjects=("acc_mean", "std"),
                  bacc_mean=("balanced_acc", "mean"), n_subjects=("subject", "count"),
                  secs=("secs", "mean"))
             .round(4))
    agg["paper_reported_%"] = agg.index.map(PAPER)
    agg["model_raw"] = agg.index
    agg = agg.reindex(MODEL_ORDER)  # fixed display order: ML baselines then deep
    agg.index = [MODEL_LABELS.get(m, m) for m in agg.index]
    agg = agg.reset_index().rename(columns={"index": "model"})

    out = RESULTS / "summary.csv"
    agg.to_csv(out, index=False)
    print(agg.to_string(index=False))

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.arange(len(agg))
    colors = ["#4C72B0"] * 3 + ["#DD8452"] * 3

    bars = ax.bar(xs, agg["acc_mean"] * 100,
                  yerr=agg["acc_std_across_subjects"] * 100,
                  capsize=5, color=colors, alpha=0.85,
                  error_kw=dict(elinewidth=1.4, ecolor="black", capthick=1.4),
                  label="bar: mean accuracy over 22 subjects\n"
                        "error bar: ±1 SD across subjects (per-subject CV accuracy)")

    # individual subject points (jittered) for transparency
    rng = np.random.RandomState(0)
    per_model = df.groupby("model")["acc_mean"]
    for i, m in enumerate(agg["model_raw"]):
        vals = per_model.get_group(m).values * 100
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.18, 0.18, len(vals)),
                   vals, s=12, color="black", alpha=0.35, zorder=3,
                   label="dot: individual subject" if i == 0 else None)

    # chance level
    ax.axhline(100 / 3, color="gray", ls="--", lw=1)
    ax.text(len(agg) - 0.4, 100 / 3 + 1, "chance 33.3%", ha="right",
            color="gray", fontsize=9)

    # paper-reported values as red dashes
    for i, row in agg.iterrows():
        if pd.notna(row["paper_reported_%"]):
            ax.hlines(row["paper_reported_%"], i - 0.35, i + 0.35,
                      colors="red", linestyles="--", lw=2,
                      label="red dash: value reported in the paper" if i == 0 else None)

    ax.set_xticks(list(xs), agg["model"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("3-condition classification (rest / overt / inner), 22 subjects\n"
                 "leave-one-session-out CV per subject — same splits for ML and deep models")
    ax.axvline(2.5, color="gray", ls=":", lw=1)
    ax.text(1, 2, "paper baselines (55-65 Hz SPM+coherence features)",
            ha="center", color="#4C72B0", fontsize=9)
    ax.text(4, 2, "deep models (raw epochs input)", ha="center",
            color="#DD8452", fontsize=9)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "summary.png", dpi=150)
    print(f"\nsaved -> {out} and {RESULTS/'summary.png'}")


if __name__ == "__main__":
    main()
