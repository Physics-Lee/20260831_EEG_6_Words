"""Shared data loading + feature extraction for the 3-condition classification
reproduction (rest vs overt vs inner).

Paper protocol (Scientific Data s41597-026-07809-9, Technical Validation):
  - classifiers: Random Forest, SVM (linear kernel), LDA
  - features:    spectral power (SPM) + coherence in the 55-65 Hz band
  - task:        3-class per participant, 5-fold CV
  - reference:   RF 78+-4%, SVM 74+-5%, LDA 67+-6%

Unifies epochs across subjects to a common time axis (500 samples):
  - time:  typical subjects (750 samples, -0.5..1.0 s) keep the window
           -0.25..0.75 s around marker onset (samples 125:625);
           atypical subjects (500 samples, 0..1 s) keep all 500.
Channels stay native per subject (37, 38 or 40 depending on montage), except
37 -> zero-padded to 38 so the ViT patch grid divides evenly (38 = 2*19).
Models are trained per participant, so cross-subject channel identity does
not matter — only the per-subject input shape must be consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")

BASE = Path(r"D:/repositories/20260831_reproduce_20260817_kostulin/data/inner_speech_v2")
RESULTS = Path(r"D:/repositories/20260831_reproduce_20260817_kostulin/experiments/results")
RESULTS.mkdir(parents=True, exist_ok=True)

N_T = 500
SFREQ = 500.0
CLASSES = ["rest", "overt", "inner"]  # 0, 1, 2

BAND = (55.0, 65.0)


def subject_list():
    """(language, sub) pairs read from manifest.csv."""
    out = []
    lines = (BASE / "manifest.csv").read_text().strip().splitlines()[1:]
    for ln in lines:
        lang, sub = ln.strip().split(",")[:2]
        out.append((lang.lower(), sub))
    return out


def load_three_class(lang: str, sub: str):
    """Return X (N, 38, 500), y (N,), native epoch length (for reference)."""
    ep = mne.read_epochs(
        BASE / f"preprocessed/{lang}/{sub}/{sub}_epochs.fif", preload=True
    )
    codes = ep.events[:, 2]
    inv = {v: k for k, v in ep.event_id.items()}
    names = np.array([inv[c] for c in codes])

    # 6 directional words x {1: overt, 2: inner} + GZ: rest. Exclude GO/NEXT.
    is_overt = np.array([n.endswith("1") and n not in ("GO", "NEXT1", "GZ")
                         and not n.startswith(("GO", "NEXT")) for n in names])
    is_inner = np.array([n.endswith("2") and not n.startswith("NEXT") for n in names])
    is_rest = names == "GZ"

    sel = is_overt | is_inner | is_rest
    y = np.zeros(sel.sum(), dtype=np.int64)
    y[is_inner[sel]] = 2
    y[is_overt[sel]] = 1  # rest stays 0
    samples = ep.events[sel, 0]  # marker onset samples, for session grouping

    data = ep.get_data(copy=True)[sel]  # (N, C, T)
    n_ch, n_t = data.shape[1], data.shape[2]
    n_ch_out = n_ch if n_ch % 2 == 0 else n_ch + 1  # odd -> +1 for ViT patching

    # window centered on marker onset: [onset-0.25 s, onset+0.75 s), zero-padded
    # when the native epoch is shorter (russian/sub9 has only 375 samples).
    onset = max(0, int(round(-ep.tmin * ep.info["sfreq"])))
    lo, hi = onset - 125, onset + 375
    src_lo, src_hi = max(0, lo), min(n_t, hi)
    X = np.zeros((data.shape[0], n_ch_out, N_T), dtype=np.float64)
    X[:, :n_ch, src_lo - lo:src_hi - lo] = data[:, :, src_lo:src_hi]
    return X, y, samples


# ---------------------------------------------------------------
# 55-65 Hz SPM + coherence features (vectorized Welch cross-spectra)
# ---------------------------------------------------------------
def spm_coherence_features(X: np.ndarray, nperseg: int = 250, noverlap: int = 125):
    """X: (N, C, T) -> (N, C + C*(C-1)/2) features.

    SPM: log10 mean PSD in 55-65 Hz per channel.
    Coherence: band-averaged magnitude-squared coherence per channel pair.
    """
    from scipy.signal import get_window

    N, C, T = X.shape
    step = nperseg - noverlap
    n_seg = 1 + (T - nperseg) // step
    win = get_window("hann", nperseg)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / SFREQ)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])

    feats = []
    for i0 in range(0, N, 256):  # chunk to bound memory
        chunk = X[i0:i0 + 256]
        # strided segments (n, C, n_seg, nperseg)
        idx = np.arange(nperseg)[None, :] + step * np.arange(n_seg)[:, None]
        seg = chunk[:, :, idx] * win[None, None, None, :]  # (n, C, n_seg, nperseg)
        F = np.fft.rfft(seg, axis=-1)  # (n, C, n_seg, F)

        Sxx = (F.real**2 + F.imag**2).mean(axis=2)          # (n, C, F)
        # Sxy = mean over segments of F_c * conj(F_d)
        G = np.transpose(F, (0, 3, 1, 2))                    # (n, F, C, n_seg)
        csd = np.einsum("nfcs,nfds->ncdf", G, np.conj(G)) / n_seg  # (n,C,C,F)

        spm = np.log10(Sxx[:, :, band].mean(axis=-1) + 1e-20)  # (n, C)
        pxy_b = csd[:, :, :, band].mean(axis=-1)                # (n, C, C) complex
        pxx_b = Sxx[:, :, band].mean(axis=-1)                   # (n, C)
        coh = np.abs(pxy_b) ** 2 / (
            pxx_b[:, :, None] * pxx_b[:, None, :] + 1e-20)      # (n, C, C)

        iu = np.triu_indices(C, k=1)
        feats.append(np.hstack([spm, coh[:, iu[0], iu[1]]]))
    return np.vstack(feats)


def cv_splits(y: np.ndarray, n_splits: int = 5, seed: int = 42):
    """Deterministic stratified splits shared by ML and DL runs."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ---------------------------------------------------------------
# Leave-one-session-out: temporal segmentation of the marker stream.
# Random CV leaks because adjacent trials share drift/artifact state and
# the same word burst; here contiguous segments of trials (separated by
# long recording gaps) form the groups, and one segment is held out per fold.
# ---------------------------------------------------------------
def session_groups(samples: np.ndarray, y: np.ndarray,
                   min_segments: int = 6, max_segments: int = 10) -> np.ndarray:
    """Group marker onsets into chronological sessions.

    Adaptive gap threshold (60 -> 10 s) picks the first cut giving >= min_segments;
    then the smallest adjacent segments are merged until <= max_segments.
    If a class lives in a single segment (e.g. all rest trials recorded in one
    burst), that segment is split into 3 sub-segments so folds can still keep
    the class in train.
    """
    order = np.argsort(samples, kind="stable")
    s = samples[order] / SFREQ  # seconds
    gaps = np.diff(s)
    seg_bounds = []
    for thr in (60, 45, 30, 20, 15, 10):
        seg_bounds = np.flatnonzero(gaps > thr)
        if len(seg_bounds) + 1 >= min_segments:
            break
    seg_id = np.zeros(len(s), int)
    seg_id[seg_bounds + 1] = 1
    seg_id = np.cumsum(seg_id)  # segment index per trial (time-ordered)

    # merge smallest adjacent segments down to max_segments
    while seg_id.max() + 1 > max_segments:
        counts = np.bincount(seg_id)
        # cost of merging segment i with i+1 = total trials of the two
        costs = counts[:-1] + counts[1:]
        i = int(np.argmin(costs))
        seg_id[seg_id > i] -= 1

    # split any class-monolithic segment into 3 time-ordered sub-segments
    for k in np.unique(y):
        in_k = y == k
        if in_k.sum() < 9:
            continue
        segs_of_k = np.unique(seg_id[in_k])
        if len(segs_of_k) == 1:
            idx_k = np.flatnonzero(in_k)
            n_sub = min(3, len(idx_k) // 3)
            for j in range(1, n_sub):
                seg_id[idx_k[(len(idx_k) * j) // n_sub:]] += 1
    return seg_id


def leave_one_session_out_splits(y: np.ndarray, samples: np.ndarray):
    """(train, test) index pairs; folds whose train set lacks a class are
    skipped (degenerate)."""
    groups = session_groups(samples, y)
    splits = []
    for g in np.unique(groups):
        te = np.flatnonzero(groups == g)
        tr = np.flatnonzero(groups != g)
        if len(np.unique(y[tr])) == len(np.unique(y)):
            splits.append((tr, te))
    return splits
