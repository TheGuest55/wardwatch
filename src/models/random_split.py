"""Random-split REFERENCE run for the tree classifiers (proposal section 3.4; never used for selection).

    python -m src.models.random_split --model xgboost [--seeds 1] [--for-report]
    python -m src.models.random_split --model random_forest [--seeds 1] [--for-report]

Question it answers: how much does the conventional protocol (shuffle every window, then split)
overstate detection performance compared with our time-based split (Experiment A)?

Protocol:
  * Same rows as Experiment A (victim/boundary rows excluded by the harness), same 51 features, same
    hyper-parameter grid, same rule (grid + threshold picked by validation max-F1, test touched once).
  * TIME split    = the `split` column (sessions r01-r03+n02 / r04,r06 / r05,r07), exactly as A.
  * RANDOM split  = all those rows pooled and shuffled into train/val/test of the SAME sizes,
    stratified by class (benign + each attack class) so class balance matches.
  * Both are run here with identical seeds, so the gap is caused only by how rows were split.
  * Default is ONE seed, as fixed in the proposal (a reference, not a result we select on).
    --seeds N repeats with split seed = model seed = 0..N-1 and reports mean +/- std.

Leakage diagnostics (explain WHY a random split inflates scores) for each test set:
  * neighbour_in_train: share of test windows whose immediate neighbour (same session, same host,
    the window 5 s before or after) is in train. Consecutive windows are near-identical, so the
    model has effectively seen the test window already.
  * attack_run_in_train: share of test ATTACK windows whose attack run (run_id) also has windows in
    train — the model has seen that exact attack instance, not just that attack type.
Under the time split both are 0 by construction (test sessions never appear in train).
"""
import argparse
import json
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
import pandas as pd

from . import data as mdata
from . import metrics as mmetrics
from . import tracking
from .train_tree import (DATASET, GRIDS, SPLIT, V2, best_threshold, grid, importances,
                         make_model, multiclass_targets)

OUT_DIR = V2 / "runs" / "random_split"
WINDOW_S = 5


def random_split_indices(strata, sizes, seed):
    """Shuffle row indices into (train, val, test) with exactly `sizes` rows each, stratified by
    `strata` (one label per row). Deterministic for a given seed. Returns three sorted int arrays."""
    from sklearn.model_selection import train_test_split
    strata = np.asarray(strata)
    n_tr, n_va, n_te = (int(s) for s in sizes)
    if n_tr + n_va + n_te != len(strata):
        raise ValueError(f"sizes {sizes} do not add up to {len(strata)} rows")
    idx = np.arange(len(strata))
    tr, rest = train_test_split(idx, train_size=n_tr, stratify=strata, random_state=seed)
    va, te = train_test_split(rest, train_size=n_va, stratify=strata[rest], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def neighbour_in_train(meta_train, meta_test, step=WINDOW_S):
    """Share of test windows whose adjacent window (same session + host, +/- step seconds) is in
    train."""
    if len(meta_test) == 0:
        return float("nan")
    key = lambda s, h, t: s.astype(str) + "|" + h.astype(str) + "|" + t.astype(str)
    t_tr = np.rint(meta_train["window_start_epoch"].to_numpy(dtype=float)).astype(np.int64)
    seen = set(key(meta_train["session"], meta_train["host_mac"], pd.Series(t_tr)))
    t_te = np.rint(meta_test["window_start_epoch"].to_numpy(dtype=float)).astype(np.int64)
    s, h = meta_test["session"].reset_index(drop=True), meta_test["host_mac"].reset_index(drop=True)
    before = key(s, h, pd.Series(t_te - step)).isin(seen).to_numpy()
    after = key(s, h, pd.Series(t_te + step)).isin(seen).to_numpy()
    return float((before | after).mean())


def attack_run_in_train(meta_train, meta_test):
    """Share of test ATTACK windows whose run_id also has attack windows in train."""
    att_te = meta_test[meta_test["label"] == "attack"]
    if len(att_te) == 0:
        return float("nan")
    runs_tr = set(meta_train.loc[meta_train["label"] == "attack", "run_id"].astype(str))
    return float(att_te["run_id"].astype(str).isin(runs_tr).mean())


def evaluate(name, Xtr, ytr, Xva, yva, Xte, yte, cls_te, seed, feats):
    """Grid on val (this seed), threshold on val, score test once. Same rule as Experiment A."""
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    search = []
    for params in grid(name):
        clf = make_model(name, params, seed, n_pos, n_neg).fit(Xtr, ytr)
        thr, f1 = best_threshold(yva, clf.predict_proba(Xva)[:, 1])
        search.append({"params": params, "val_f1": f1, "threshold": thr, "model": clf})
    best = max(search, key=lambda r: r["val_f1"])
    clf, thr = best["model"], best["threshold"]
    score = clf.predict_proba(Xte)[:, 1]
    m = mmetrics.binary_metrics(yte, y_score=score, threshold=thr)
    pred = score >= thr
    m["fpr"] = float((pred & (yte == 0)).sum() / max((yte == 0).sum(), 1))
    m["threshold"], m["val_f1"] = thr, best["val_f1"]
    for c in sorted(set(cls_te) - {"benign"}):
        m[f"recall_{c}"] = float(pred[cls_te == c].mean())
    return m, best["params"], importances(name, clf, feats)[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(GRIDS), required=True)
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of seeds (default 1, as fixed in the proposal)")
    ap.add_argument("--for-report", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    name = args.model

    df, fl = mdata.load_dataset(DATASET, V2 / "feature_list.json")
    ds = mdata.Dataset(df, fl, scale=False)
    feats = ds.feature_names
    splits = mdata.SPLITS
    X = np.vstack([ds.X(s) for s in splits])
    y = np.concatenate([ds.y(s) for s in splits])
    meta = pd.concat([ds.meta(s) for s in splits], ignore_index=True)
    cls = multiclass_targets(meta)
    sizes = [len(ds.y(s)) for s in splits]
    bounds = np.cumsum([0] + sizes)
    time_idx = [np.arange(bounds[i], bounds[i + 1]) for i in range(3)]
    print(f"[{name}] rows={len(y)} ({int(y.sum())} attack) | split sizes train/val/test = {sizes}")

    results = {"time": [], "random": []}
    details = {"time": [], "random": []}
    for seed in range(args.seeds):
        rand_idx = random_split_indices(cls, sizes, seed)
        for kind, (tr, va, te) in (("time", time_idx), ("random", rand_idx)):
            m, params, imp = evaluate(name, X[tr], y[tr], X[va], y[va], X[te], y[te], cls[te],
                                      seed, feats)
            m["neighbour_in_train"] = neighbour_in_train(meta.iloc[tr], meta.iloc[te])
            m["attack_run_in_train"] = attack_run_in_train(meta.iloc[tr], meta.iloc[te])
            results[kind].append(m)
            details[kind].append({"seed": seed, "params": params, "metrics": m,
                                  "top_features": imp})
            print(f"  seed {seed} {kind:6s} F1 {m['f1']:.4f} P {m['precision']:.4f} "
                  f"R {m['recall']:.4f} FPR {m['fpr']:.5f} PR-AUC {m.get('pr_auc', np.nan):.4f} | "
                  f"neighbour_in_train {m['neighbour_in_train']:.3f} "
                  f"attack_run_in_train {m['attack_run_in_train']:.3f} | {params}")

    agg = {}
    for kind, per_seed in results.items():
        for k in sorted({k for m in per_seed for k in m if isinstance(m[k], (int, float))}):
            v = np.array([m.get(k, np.nan) for m in per_seed], dtype=float)
            agg[f"{kind}_{k}_mean"], agg[f"{kind}_{k}_std"] = (float(np.nanmean(v)),
                                                              float(np.nanstd(v)))
    for k in ("f1", "precision", "recall", "pr_auc", "fpr"):
        agg[f"gap_{k}"] = agg[f"random_{k}_mean"] - agg[f"time_{k}_mean"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    det_path = OUT_DIR / f"{name}_details.json"
    det_path.write_text(json.dumps(details, indent=2, default=str))
    rec = tracking.build_run_record(
        experiment="random-split-reference", model=name, role="classifier",
        params={"grid": GRIDS[name], "seeds": list(range(args.seeds)),
                "threshold_rule": "val-max-F1", "random_split": "stratified by class, same sizes",
                "split_sizes": sizes},
        metrics=agg, tags={"dataset": "v2", "details": str(det_path),
                           "note": "reference only; never used for model selection"},
        seed=0, split_path=str(SPLIT), dataset_path=str(DATASET), for_report=args.for_report)
    tracking.log_run(rec, json_path=str(OUT_DIR / f"{name}.json"), use_mlflow=not args.no_mlflow)

    print(f"\n[{name}] random split vs time split ({args.seeds} seed(s))")
    print(f"  {'metric':22s} {'time':>8s} {'random':>8s} {'gap':>8s}")
    rows = ["f1", "precision", "recall", "pr_auc", "fpr", "neighbour_in_train",
            "attack_run_in_train"] + sorted(k[len("time_"):-len("_mean")] for k in agg
                                            if k.startswith("time_recall_") and k.endswith("_mean")
                                            and k != "time_recall_mean")
    for k in rows:
        t, r = agg.get(f"time_{k}_mean", np.nan), agg.get(f"random_{k}_mean", np.nan)
        print(f"  {k:22s} {t:8.4f} {r:8.4f} {r - t:+8.4f}")


if __name__ == "__main__":
    main()
