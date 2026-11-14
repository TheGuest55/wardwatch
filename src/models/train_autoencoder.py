"""Experiment A (time-based split) for the autoencoder, the unsupervised anomaly detector.

    python -m src.models.train_autoencoder --experiment A [--for-report] [--no-mlflow]

Mirrors src/models/train_tree.py so the numbers are comparable:
  * Inputs = feature_list.json features only (the harness rejects identifiers). Split = the `split`
    column (time-based). victim/boundary rows are excluded (harness).
  * UNSUPERVISED training: the model fits on BENIGN windows only (Dataset.normal_only), so no label
    reaches training. The architecture is chosen on VALIDATION (seed 0, small grid, by val binary F1
    at the val-optimal reconstruction-error threshold). The threshold is also fit on validation.
  * Test is touched once per seed with the frozen config + threshold. 5 seeds -> mean +/- std.
  * Binary attack-vs-normal metrics are the headline (same metric set as every model). No multiclass:
    an autoencoder scores one anomaly value, it does not name the attack type.
  * Per-feature mean reconstruction error (seed 0) is written next to the record: it is the
    autoencoder's plain-language explanation for triage AND the leakage sanity check (working rule 3:
    a single identifier dominating reconstruction would show here).
  * Seed-0 model + threshold are saved for the triage layer.

role = "anomaly" in the run record (the selection rule compares the best anomaly detector,
autoencoder vs Isolation Forest, against the best classifier).
"""
import argparse
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
import torch

from .. import common
from . import data as mdata
from . import metrics as mmetrics
from . import tracking
from . import autoencoder as ae

V2 = common.PROCESSED_DIR / "v2"
SPLIT = common.SESSIONS_DIR / "split_v2.json"
DATASET = V2 / "testbed_windows.parquet"
OUT_DIR = V2 / "runs" / "expA"
MODEL_DIR = V2 / "models" / "expA"
SEEDS = (0, 1, 2, 3, 4)

# Small validation grid (architecture). Kept deliberately small: a bigger net on this tabular data
# overfits normal traffic and the run cost (5 seeds x up to 50 epochs) must stay feasible.
GRID = {
    "hidden_bottleneck": [((32, 16), 8), ((32, 16), 4), ((64, 32), 16)],
}


def grid():
    for (hidden, bottleneck) in GRID["hidden_bottleneck"]:
        yield {"hidden": hidden, "bottleneck": bottleneck}


# Isolation Forest baseline (sklearn, already in the image). Unsupervised like the autoencoder:
# trained on benign-only traffic, it isolates outliers with random splits -- an attack window, being
# rare/odd, is isolated in fewer splits and so scores as more anomalous. Anomaly score = NEGATED
# sklearn score_samples (sklearn returns higher = more normal; we flip it so higher = more anomalous,
# matching the autoencoder's convention). Threshold on validation by binary F1, same as the AE.
IF_GRID = {"n_estimators": [200, 400], "max_samples": [0.5, 1.0]}


def if_grid():
    import itertools as _it
    keys = list(IF_GRID)
    for vals in _it.product(*(IF_GRID[k] for k in keys)):
        yield dict(zip(keys, vals))


def if_scores(model, X):
    """Anomaly score, higher = more anomalous (negated sklearn score_samples)."""
    return -model.score_samples(np.asarray(X))


def fit_isolation_forest(params, seed, X_train_normal):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=params["n_estimators"], max_samples=params["max_samples"],
                          contamination="auto", random_state=seed, n_jobs=1)
    clf.fit(np.asarray(X_train_normal))        # benign-only, like the autoencoder
    return clf


def _fit_one(params, seed, Xtr_normal, Xva_normal):
    """Train one autoencoder (one architecture, one seed) on benign-only data. Returns the model."""
    ae.set_determinism(seed, 1)
    cfg = ae.make_cfg(seed=seed, n_feat=Xtr_normal.shape[1], **params)
    model = ae.AutoEncoder(n_in=Xtr_normal.shape[1], hidden=params["hidden"],
                           bottleneck=params["bottleneck"], dropout=cfg["dropout"])
    info = ae.train(model, Xtr_normal, Xva_normal, cfg)
    return model, info


def per_feature_error(model, X, feats):
    """Mean per-feature reconstruction error (seed-0 explanation + leakage probe): for each feature,
    the average squared (input - reconstruction) over the given rows, normalised to sum to 1."""
    model.eval()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        recon = model(Xt).numpy()
    sq = (recon - np.asarray(X)) ** 2
    per = sq.mean(axis=0)
    total = per.sum() or 1.0
    return sorted(((f, float(v / total)) for f, v in zip(feats, per)), key=lambda kv: -kv[1])


def run_isolation_forest(ds, feats, Xtr_normal, Xva_all, yva, Xte, yte, args):
    """Experiment A for the Isolation Forest baseline: select on validation (seed 0), 5 seeds on
    test, same binary metric set and val-F1 threshold rule as the autoencoder."""
    search = []
    for params in if_grid():
        t0 = time.perf_counter()
        clf = fit_isolation_forest(params, 0, Xtr_normal)
        sva = if_scores(clf, Xva_all)
        thr = ae.pick_threshold(yva, sva)
        m = mmetrics.binary_metrics(yva, y_score=sva, y_pred=ae.predict(sva, thr))
        search.append({"params": params, "val_f1": m["f1"], "val_threshold": thr,
                       "fit_s": time.perf_counter() - t0})
        print(f"  grid {params} -> val F1 {m['f1']:.4f} @ thr {thr:.4f}")
    best = max(search, key=lambda r: r["val_f1"])
    params = best["params"]
    print(f"[isolation_forest] selected {params} (val F1 {best['val_f1']:.4f})")

    per_seed, saved = [], None
    for seed in SEEDS:
        t0 = time.perf_counter()
        clf = fit_isolation_forest(params, seed, Xtr_normal)
        fit_s = time.perf_counter() - t0
        thr = ae.pick_threshold(yva, if_scores(clf, Xva_all))
        score = if_scores(clf, Xte)
        pred = ae.predict(score, thr)
        m = mmetrics.binary_metrics(yte, y_score=score, y_pred=pred)
        m["fpr"] = float((pred.astype(bool) & (yte == 0)).sum() / max((yte == 0).sum(), 1))
        m.update({"threshold": thr, "train_s": fit_s})
        per_seed.append(m)
        print(f"  seed {seed}: test F1 {m['f1']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} "
              f"FPR {m['fpr']:.5f} PR-AUC {m.get('pr_auc', float('nan')):.4f}")
        if seed == 0:
            saved = clf
            lat = mmetrics.single_window_latency_ms(
                lambda x: if_scores(clf, x.reshape(1, -1)), Xte[:500])

    keys = sorted({k for m in per_seed for k in m if isinstance(m[k], (int, float))})
    agg = {}
    for k in keys:
        v = np.array([m[k] for m in per_seed if k in m], dtype=float)
        agg[f"{k}_mean"], agg[f"{k}_std"] = float(np.nanmean(v)), float(np.nanstd(v))
    agg["latency_median_ms"], agg["latency_p95_ms"] = lat["median_ms"], lat["p95_ms"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "isolation_forest_details.json").write_text(
        json.dumps({"grid_search": search, "per_seed": per_seed, "features": feats},
                   indent=2, default=str))
    rec = tracking.build_run_record(
        experiment="A", model="isolation_forest", role="anomaly",
        params={**params, "seeds": list(SEEDS), "threshold_rule": "val-max-F1",
                "trained_on": "benign-only"},
        metrics=agg, tags={"dataset": "v2",
                           "details": str(OUT_DIR / "isolation_forest_details.json")},
        seed=SEEDS[0], split_path=str(SPLIT), dataset_path=str(DATASET),
        for_report=args.for_report)
    tracking.log_run(rec, json_path=str(OUT_DIR / "isolation_forest.json"),
                     use_mlflow=not args.no_mlflow)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"model": saved, "threshold": float(per_seed[0]["threshold"]),
                 "feature_names": feats, "scaler": ds.scaler},
                MODEL_DIR / "isolation_forest_seed0.joblib")
    print(f"\n[isolation_forest] TEST binary F1 {agg['f1_mean']:.4f} +/- {agg['f1_std']:.4f} | "
          f"FPR {agg['fpr_mean']:.5f} | PR-AUC {agg['pr_auc_mean']:.4f} | "
          f"ROC-AUC {agg['roc_auc_mean']:.4f} | latency p95 {lat['p95_ms']:.2f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["autoencoder", "isolation_forest"], default="autoencoder",
                    help="which anomaly detector to train (the role has two: AE + IF baseline)")
    ap.add_argument("--experiment", choices=["A"], default="A",
                    help="A = time-based split (B and C added in later steps)")
    ap.add_argument("--for-report", action="store_true",
                    help="refuse to write the record unless provenance is complete")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    df, fl = mdata.load_dataset(DATASET, V2 / "feature_list.json")
    ds = mdata.Dataset(df, fl, scale=True)          # both detectors: scale (StandardScaler, train-only)
    feats = ds.feature_names
    Xtr_normal = ds.normal_only("train")            # benign only
    Xva_normal = ds.normal_only("val")              # benign only (AE early-stopping signal)
    Xva_all, yva = ds.X("val"), ds.y("val")         # full val (has attacks): threshold + selection
    Xte, yte = ds.X("test"), ds.y("test")           # untouched test
    print(f"[{args.model}] features={len(feats)} train_normal={len(Xtr_normal)} "
          f"val={len(yva)} ({int(yva.sum())} attack) test={len(yte)} ({int(yte.sum())} attack)")

    if args.model == "isolation_forest":
        return run_isolation_forest(ds, feats, Xtr_normal, Xva_all, yva, Xte, yte, args)

    # ===== autoencoder =====
    # ---- 1. architecture selection on validation (seed 0) ----
    search = []
    for params in grid():
        t0 = time.perf_counter()
        model, info = _fit_one(params, 0, Xtr_normal, Xva_normal)
        sva = ae.anomaly_scores(model, Xva_all)
        thr = ae.pick_threshold(yva, sva)
        m = mmetrics.binary_metrics(yva, y_score=sva, y_pred=ae.predict(sva, thr))
        search.append({"params": params, "val_f1": m["f1"], "val_threshold": thr,
                       "epochs": info["epochs_run"], "fit_s": time.perf_counter() - t0})
        print(f"  grid hidden={params['hidden']} bottleneck={params['bottleneck']} "
              f"-> val F1 {m['f1']:.4f} @ thr {thr:.4f} ({info['epochs_run']} epochs)")
    best = max(search, key=lambda r: r["val_f1"])
    params = best["params"]
    print(f"[autoencoder] selected hidden={params['hidden']} bottleneck={params['bottleneck']} "
          f"(val F1 {best['val_f1']:.4f})")

    # ---- 2. frozen architecture, 5 seeds: threshold on val, evaluate on test ----
    per_seed, pfe, saved = [], None, None
    for seed in SEEDS:
        t0 = time.perf_counter()
        model, info = _fit_one(params, seed, Xtr_normal, Xva_normal)
        fit_s = time.perf_counter() - t0
        thr = ae.pick_threshold(yva, ae.anomaly_scores(model, Xva_all))
        score = ae.anomaly_scores(model, Xte)
        pred = ae.predict(score, thr)
        m = mmetrics.binary_metrics(yte, y_score=score, y_pred=pred)
        m["fpr"] = float((pred.astype(bool) & (yte == 0)).sum() / max((yte == 0).sum(), 1))
        m.update({"threshold": thr, "train_s": fit_s, "epochs": info["epochs_run"]})
        per_seed.append(m)
        print(f"  seed {seed}: test F1 {m['f1']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} "
              f"FPR {m['fpr']:.5f} PR-AUC {m.get('pr_auc', float('nan')):.4f}")
        if seed == 0:
            saved = (model, thr)
            pfe = per_feature_error(model, Xte, feats)
            lat = mmetrics.single_window_latency_ms(
                lambda x: ae.anomaly_scores(model, x.reshape(1, -1)), Xte[:500])

    # ---- 3. aggregate + persist ----
    keys = sorted({k for m in per_seed for k in m if isinstance(m[k], (int, float))})
    agg = {}
    for k in keys:
        v = np.array([m[k] for m in per_seed if k in m], dtype=float)
        agg[f"{k}_mean"], agg[f"{k}_std"] = float(np.nanmean(v)), float(np.nanstd(v))
    agg["latency_median_ms"], agg["latency_p95_ms"] = lat["median_ms"], lat["p95_ms"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extra = {"grid_search": search, "per_seed": per_seed,
             "per_feature_error_seed0": pfe, "features": feats}
    (OUT_DIR / "autoencoder_details.json").write_text(json.dumps(extra, indent=2, default=str))

    rec = tracking.build_run_record(
        experiment="A", model="autoencoder", role="anomaly",
        params={"hidden": params["hidden"], "bottleneck": params["bottleneck"],
                "seeds": list(SEEDS), "threshold_rule": "val-max-F1",
                "trained_on": "benign-only"},
        metrics=agg, tags={"dataset": "v2", "details": str(OUT_DIR / "autoencoder_details.json")},
        seed=SEEDS[0], split_path=str(SPLIT), dataset_path=str(DATASET),
        for_report=args.for_report)
    tracking.log_run(rec, json_path=str(OUT_DIR / "autoencoder.json"), use_mlflow=not args.no_mlflow)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model0, thr0 = saved
    torch.save({"state_dict": model0.state_dict(),
                "model_args": {"n_in": model0.n_in, "hidden": list(model0.hidden),
                               "bottleneck": model0.bottleneck},
                "threshold": float(thr0), "feature_names": feats,
                "scaler_pickle": __import__("pickle").dumps(ds.scaler)},
               MODEL_DIR / "autoencoder_seed0.pt")

    print(f"\n[autoencoder] TEST binary F1 {agg['f1_mean']:.4f} +/- {agg['f1_std']:.4f} | "
          f"FPR {agg['fpr_mean']:.5f} | PR-AUC {agg['pr_auc_mean']:.4f} | "
          f"ROC-AUC {agg['roc_auc_mean']:.4f} | latency p95 {lat['p95_ms']:.2f} ms")
    print("  top reconstruction-error features:", [(f, round(v, 3)) for f, v in pfe[:10]])


if __name__ == "__main__":
    main()
