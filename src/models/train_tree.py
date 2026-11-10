"""Experiment A (time-based split) for the tree classifiers: XGBoost and its Random Forest baseline.

    python -m src.models.train_tree --model xgboost [--for-report]
    python -m src.models.train_tree --model random_forest [--for-report]

Protocol (./CLAUDE.md, fixed):
  * Inputs = feature_list.json features only (the harness rejects identifiers). Split = the `split`
    column (time-based). victim/boundary rows are excluded from training and metrics (harness).
  * Hyper-parameters are picked on VALIDATION (seed 0, small grid, by val binary F1 at the
    val-optimal threshold). The decision threshold is also fit on validation, never on test.
  * Test is touched once per seed with the frozen config + threshold. 5 seeds -> mean +/- std.
  * Binary attack-vs-normal metrics are the headline; Experiment A also reports multiclass macro-F1
    (benign + each attack class) and per-class binary recall, so the stolen-credential classes
    (spoof_vitals, pump_cmd) are visible separately.
  * Feature importances are written next to the record: a leaked identifier/timestamp would show up
    there first (working rule 3: suspiciously good results are bugs until proven otherwise).
  * Seed-0 binary model is saved for the triage layer (calibration happens there, on validation).
"""
import argparse
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np

from .. import common
from . import data as mdata
from . import metrics as mmetrics
from . import tracking

V2 = common.PROCESSED_DIR / "v2"
SPLIT = common.SESSIONS_DIR / "split_v2.json"
DATASET = V2 / "testbed_windows.parquet"
OUT_DIR = V2 / "runs" / "expA"
MODEL_DIR = V2 / "models" / "expA"
SEEDS = (0, 1, 2, 3, 4)
N_JOBS = int(os.environ.get("ANALYSIS_CPUS", os.cpu_count() or 1))

GRIDS = {
    "xgboost": {"max_depth": [4, 6, 8], "n_estimators": [200, 400], "learning_rate": [0.05, 0.1]},
    "random_forest": {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1, 5]},
}


def make_model(name, params, seed, n_pos, n_neg, multiclass=False):
    if name == "xgboost":
        from xgboost import XGBClassifier
        kw = dict(subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=N_JOBS,
                  random_state=seed, **params)
        if multiclass:
            return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **kw)
        # class imbalance: weight positives by neg/pos (fit on TRAIN counts only)
        return XGBClassifier(eval_metric="logloss", scale_pos_weight=n_neg / max(n_pos, 1), **kw)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=N_JOBS,
                                      random_state=seed, **params)
    raise ValueError(name)


def best_threshold(y, score):
    """Threshold maximising binary F1 on the given (validation) split."""
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])


def grid(name):
    g = GRIDS[name]
    keys = list(g)
    for vals in itertools.product(*(g[k] for k in keys)):
        yield dict(zip(keys, vals))


def multiclass_targets(meta):
    return np.where(meta["label"].to_numpy() == "attack", meta["attack_class"].astype(str), "benign")


def importances(name, model, feats):
    if name == "xgboost":
        gain = model.get_booster().get_score(importance_type="gain")
        vals = [gain.get(f"f{i}", 0.0) for i in range(len(feats))]
    else:
        vals = list(model.feature_importances_)
    total = sum(vals) or 1.0
    return sorted(((f, v / total) for f, v in zip(feats, vals)), key=lambda kv: -kv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(GRIDS), required=True)
    ap.add_argument("--for-report", action="store_true",
                    help="refuse to write the record unless provenance is complete")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    name = args.model

    df, fl = mdata.load_dataset(DATASET, V2 / "feature_list.json")
    ds = mdata.Dataset(df, fl, scale=False)          # trees: no scaling
    feats = ds.feature_names
    Xtr, ytr, Xva, yva, Xte, yte = (ds.X("train"), ds.y("train"), ds.X("val"), ds.y("val"),
                                    ds.X("test"), ds.y("test"))
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    print(f"[{name}] features={len(feats)} train={len(ytr)} ({n_pos} attack) "
          f"val={len(yva)} ({int(yva.sum())}) test={len(yte)} ({int(yte.sum())})")
    for s in mdata.SPLITS:
        m = ds.meta(s)
        cls = sorted(m.loc[m["label"] == "attack", "attack_class"].astype(str).unique())
        print(f"  {s} attack classes: {cls}")

    # ---- 1. model selection on validation (seed 0) ----
    search = []
    for params in grid(name):
        t0 = time.perf_counter()
        clf = make_model(name, params, 0, n_pos, n_neg).fit(Xtr, ytr)
        thr, f1 = best_threshold(yva, clf.predict_proba(Xva)[:, 1])
        search.append({"params": params, "val_f1": f1, "val_threshold": thr,
                       "fit_s": time.perf_counter() - t0})
        print(f"  grid {params} -> val F1 {f1:.4f} @ thr {thr:.3f}")
    best = max(search, key=lambda r: r["val_f1"])
    params = best["params"]
    print(f"[{name}] selected {params} (val F1 {best['val_f1']:.4f})")

    # ---- 2. frozen config, 5 seeds: threshold on val, evaluate on test ----
    meta_te = ds.meta("test")
    cls_te = meta_te["attack_class"].astype(str).to_numpy()
    lab_te = meta_te["label"].to_numpy()
    per_seed, imps, saved = [], None, None
    for seed in SEEDS:
        t0 = time.perf_counter()
        clf = make_model(name, params, seed, n_pos, n_neg).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        thr, val_f1 = best_threshold(yva, clf.predict_proba(Xva)[:, 1])
        score = clf.predict_proba(Xte)[:, 1]
        m = mmetrics.binary_metrics(yte, y_score=score, threshold=thr)
        m.update({"val_f1": val_f1, "threshold": thr, "train_s": fit_s})
        pred = score >= thr
        m["fpr"] = float((pred & (yte == 0)).sum() / max((yte == 0).sum(), 1))
        for c in sorted(set(cls_te[lab_te == "attack"])):
            idx = (lab_te == "attack") & (cls_te == c)
            m[f"recall_{c}"] = float(pred[idx].mean())
        per_seed.append(m)
        print(f"  seed {seed}: test F1 {m['f1']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} "
              f"FPR {m['fpr']:.5f} PR-AUC {m.get('pr_auc', float('nan')):.4f}")
        if seed == 0:
            imps = importances(name, clf, feats)
            saved = clf
            lat = mmetrics.single_window_latency_ms(
                lambda x: clf.predict_proba(x.reshape(1, -1))[:, 1], Xte[:500])

    # ---- 3. multiclass (Experiment A only): benign + named attack class ----
    from sklearn.metrics import f1_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    ytr_m, yte_m = multiclass_targets(ds.meta("train")), multiclass_targets(meta_te)
    enc = LabelEncoder().fit(ytr_m)
    known = np.isin(yte_m, enc.classes_)
    macro = []
    for seed in SEEDS:
        mc = make_model(name, params, seed, n_pos, n_neg, multiclass=True)
        mc.fit(Xtr, enc.transform(ytr_m), sample_weight=compute_sample_weight("balanced", ytr_m))
        pm = enc.inverse_transform(mc.predict(Xte[known]))
        macro.append(f1_score(yte_m[known], pm, average="macro", zero_division=0))
        if seed == 0:
            report = classification_report(yte_m[known], pm, zero_division=0, output_dict=True)
    print(f"[{name}] multiclass macro-F1 {np.mean(macro):.4f} +/- {np.std(macro):.4f}")

    # ---- 4. aggregate + persist ----
    keys = sorted({k for m in per_seed for k in m if isinstance(m[k], (int, float))})
    agg = {}
    for k in keys:
        v = np.array([m[k] for m in per_seed if k in m], dtype=float)
        agg[f"{k}_mean"], agg[f"{k}_std"] = float(np.nanmean(v)), float(np.nanstd(v))
    agg["macro_f1_mean"], agg["macro_f1_std"] = float(np.mean(macro)), float(np.std(macro))
    agg["latency_median_ms"], agg["latency_p95_ms"] = lat["median_ms"], lat["p95_ms"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extra = {"grid_search": search, "per_seed": per_seed, "multiclass_report_seed0": report,
             "feature_importance_seed0": imps, "features": feats}
    (OUT_DIR / f"{name}_details.json").write_text(json.dumps(extra, indent=2, default=str))

    rec = tracking.build_run_record(
        experiment="A", model=name, role="classifier",
        params={**params, "seeds": list(SEEDS), "threshold_rule": "val-max-F1"},
        metrics=agg, tags={"dataset": "v2", "details": str(OUT_DIR / f"{name}_details.json")},
        seed=SEEDS[0], split_path=str(SPLIT), dataset_path=str(DATASET),
        for_report=args.for_report)
    tracking.log_run(rec, json_path=str(OUT_DIR / f"{name}.json"), use_mlflow=not args.no_mlflow)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if name == "xgboost":
        saved.save_model(str(MODEL_DIR / "xgboost_binary_seed0.json"))
    else:
        import joblib
        joblib.dump(saved, MODEL_DIR / "random_forest_binary_seed0.joblib")

    print(f"\n[{name}] TEST binary F1 {agg['f1_mean']:.4f} +/- {agg['f1_std']:.4f} | "
          f"FPR {agg['fpr_mean']:.5f} | PR-AUC {agg['pr_auc_mean']:.4f} | "
          f"macro-F1 {agg['macro_f1_mean']:.4f} | latency p95 {lat['p95_ms']:.2f} ms")
    print("  per-class recall:", {k[7:-5]: round(v, 3) for k, v in agg.items()
                                  if k.startswith("recall_") and k.endswith("_mean")
                                  and k != "recall_mean"})
    print("  top features:", [(f, round(v, 3)) for f, v in imps[:10]])


if __name__ == "__main__":
    main()
