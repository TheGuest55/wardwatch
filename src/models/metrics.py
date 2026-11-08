"""Fixed evaluation metrics shared by every model and experiment (./CLAUDE.md).

Headline = binary attack-vs-normal precision/recall/F1. Also PR-AUC, ROC-AUC, and FPR at a fixed
recall (operating-point comparison), plus a per-window latency helper for the real-time budget
(<50 ms/window for the deployed pair). Thresholds are chosen on validation elsewhere; these take a
threshold or a score and report.
"""
import time

import numpy as np


def fpr_at_recall(y_true, y_score, target_recall=0.90):
    """Lowest false-positive rate achievable while recall >= target_recall, by sweeping the score
    threshold. Returns (fpr, threshold, achieved_recall) or (nan, nan, nan) if unreachable."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    if P == 0 or N == 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    ys, sc = y_true[order], y_score[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    recall = tp / P
    fpr = fp / N
    ok = recall >= target_recall
    if not ok.any():
        return float("nan"), float("nan"), float("nan")
    i = int(np.argmax(ok))          # first threshold meeting the recall target
    return float(fpr[i]), float(sc[i]), float(recall[i])


def binary_metrics(y_true, y_score=None, y_pred=None, threshold=0.5, recall_targets=(0.90, 0.95)):
    """Binary attack-vs-normal metrics. Pass y_score for threshold/AUC/FPR metrics, or a precomputed
    y_pred for just precision/recall/F1."""
    from sklearn.metrics import (precision_recall_fscore_support, average_precision_score,
                                 roc_auc_score)
    y_true = np.asarray(y_true).astype(int)
    if y_pred is None:
        if y_score is None:
            raise ValueError("pass y_score or y_pred")
        y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    out = {"precision": float(p), "recall": float(r), "f1": float(f1),
           "n": int(len(y_true)), "n_positive": int((y_true == 1).sum())}
    if y_score is not None and (y_true == 1).any() and (y_true == 0).any():
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        for t in recall_targets:
            fpr, thr, rec = fpr_at_recall(y_true, y_score, t)
            out[f"fpr_at_recall_{int(t * 100)}"] = fpr
    return out


def per_window_latency_ms(score_fn, X, repeats=3):
    """Median per-window scoring latency in ms: time score_fn(X) over `repeats` passes and divide by
    the number of windows. score_fn should score the whole batch the way the deployed path would."""
    n = len(X)
    if n == 0:
        return float("nan")
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        score_fn(X)
        times.append((time.perf_counter() - t0) / n * 1000.0)
    return float(np.median(times))
