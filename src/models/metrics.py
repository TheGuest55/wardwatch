"""Fixed evaluation metrics shared by every model and experiment (./CLAUDE.md).

Headline = binary attack-vs-normal precision/recall/F1. Also PR-AUC, ROC-AUC, and FPR at a fixed
recall (operating-point comparison), plus a per-window latency helper for the real-time budget
(<50 ms/window for the deployed pair). Thresholds are chosen on validation elsewhere; these take a
threshold or a score and report.
"""
import time

import numpy as np


def fpr_at_recall(y_true, y_score, target_recall=0.90):
    """Lowest false-positive rate achievable while recall >= target_recall.

    Review N3: evaluated at the UNIQUE score thresholds only. A threshold t predicts positive every
    row with score >= t, so all rows that TIE at t count together — taking the last index of each
    tied run. Scoring per-sorted-row instead (the previous bug) could place the threshold in the
    middle of a tied block and under-count the false positives at that threshold. Returns
    (fpr, threshold, achieved_recall) or (nan, nan, nan) if unreachable."""
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
    # Last index of each unique score (descending): everything up to here has score >= that score.
    last = np.flatnonzero(np.diff(sc) != 0)
    last = np.append(last, len(sc) - 1)
    recall = tp[last] / P
    fpr = fp[last] / N
    thr = sc[last]
    ok = recall >= target_recall
    if not ok.any():
        return float("nan"), float("nan"), float("nan")
    # recall is non-decreasing as the threshold drops, so the first qualifying unique threshold is
    # the highest one — and thus the lowest fpr — that still meets the recall target.
    i = int(np.argmax(ok))
    return float(fpr[i]), float(thr[i]), float(recall[i])


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
    # accuracy is SECONDARY (the coursework brief names it): with ~98% benign windows an
    # all-benign predictor already scores ~0.98, so F1 stays the headline.
    out = {"precision": float(p), "recall": float(r), "f1": float(f1),
           "accuracy": float((y_pred == y_true).mean()) if len(y_true) else float("nan"),
           "n": int(len(y_true)), "n_positive": int((y_true == 1).sum())}
    if y_score is not None and (y_true == 1).any() and (y_true == 0).any():
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        for t in recall_targets:
            fpr, thr, rec = fpr_at_recall(y_true, y_score, t)
            out[f"fpr_at_recall_{int(t * 100)}"] = fpr
    return out


def per_window_latency_ms(score_fn, X, repeats=3):
    """Batch THROUGHPUT proxy: time score_fn(X) over `repeats` passes and divide by the number of
    windows. This is amortised cost, not single-window latency — use single_window_latency_ms for
    the deployed-path budget."""
    n = len(X)
    if n == 0:
        return float("nan")
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        score_fn(X)
        times.append((time.perf_counter() - t0) / n * 1000.0)
    return float(np.median(times))


def single_window_latency_ms(score_one, inputs, n_calls=200, warmup=20):
    """Deployed single-window latency (review N4): the real-time path scores ONE window at a time,
    so measure exactly that. `score_one(x)` takes one window's input and does the full per-window
    work — feature-row assembly AND model scoring — the way the live pipeline would. We cycle through
    `inputs`, run `warmup` untimed calls (JIT/caches/allocators settle), then time at least `n_calls`
    individual calls and report the median and p95 in ms (p95 is the tail the <50 ms budget must
    respect). Returns {median_ms, p95_ms, n, warmup}."""
    inputs = list(inputs)
    if not inputs:
        return {"median_ms": float("nan"), "p95_ms": float("nan"), "n": 0, "warmup": warmup}
    m = len(inputs)
    n_calls = max(int(n_calls), 200)          # the spec floor: >= 200 timed calls
    for i in range(max(0, int(warmup))):
        score_one(inputs[i % m])
    times = np.empty(n_calls, dtype=float)
    for i in range(n_calls):
        x = inputs[i % m]
        t0 = time.perf_counter()
        score_one(x)
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {"median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "n": int(n_calls), "warmup": int(max(0, warmup))}
