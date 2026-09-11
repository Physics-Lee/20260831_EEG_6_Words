"""Reproduce the paper's 3 baseline classifiers on 55-65 Hz SPM + coherence
features: Random Forest, linear SVM, LDA — per participant, 5-fold CV.

Usage:  python ml_paper.py [--subjects russian/sub1 spanish/sub0 ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
from common import (RESULTS, CLASSES, leave_one_session_out_splits,
                    load_three_class, spm_coherence_features, subject_list)


def build_clfs():
    return {
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        "svm_linear": make_pipeline(StandardScaler(),
                                    SVC(kernel="linear", C=1.0, cache_size=2000)),
        "lda": make_pipeline(StandardScaler(),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    }


def run_subject(lang, sub):
    X, y, samples = load_three_class(lang, sub)
    F = spm_coherence_features(X)
    splits = leave_one_session_out_splits(y, samples)
    rows = []
    for name, clf in build_clfs().items():
        t0 = time.time()
        accs, baccs = [], []
        for tr, te in splits:
            clf.fit(F[tr], y[tr])
            pred = clf.predict(F[te])
            accs.append(accuracy_score(y[te], pred))
            baccs.append(balanced_accuracy_score(y[te], pred))
        rows.append(dict(language=lang, subject=sub, model=name,
                         n_trials=len(y), n_rest=int((y == 0).sum()),
                         n_overt=int((y == 1).sum()), n_inner=int((y == 2).sum()),
                         acc_mean=np.mean(accs), acc_std=np.std(accs),
                         balanced_acc=np.mean(baccs),
                         n_folds=len(splits),
                         secs=round(time.time() - t0, 1)))
        print(f"  [{lang}/{sub}] {name:11s} acc={np.mean(accs):.3f} "
              f"bacc={np.mean(baccs):.3f} ({rows[-1]['secs']}s)", flush=True)
    return rows


def main():
    args = sys.argv[1:]
    subs = [a.split("/") for a in args] if args else subject_list()
    all_rows = []
    for lang, sub in subs:
        print(f"== {lang}/{sub}", flush=True)
        all_rows += run_subject(lang, sub)

    df = pd.DataFrame(all_rows)
    out = RESULTS / "ml_results.csv"
    df.to_csv(out, index=False)
    print("\n=== per-model summary (mean over subjects) ===")
    print(df.groupby("model")[["acc_mean", "balanced_acc"]].agg(["mean", "std"]).round(3))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
