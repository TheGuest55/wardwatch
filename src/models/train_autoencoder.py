"""Experiment A (time-based split) for the autoencoder, the unsupervised anomaly detector.

    python -m src.models.train_autoencoder --experiment A [--for-report] [--no-mlflow]

Records: one for_report record per seed (A), per (class, seed) (B) and per (direction, variant, seed) (C);
the *_summary.json/.md files are derived from them. Predictions are saved per run (LSTM column layout,
val + test) and, for A seed 0, via predictions.save_predictions. B and C read the architecture from
runs/expA/<model>_selected.json. C scales both domains with the SOURCE train-only scaler (never refit
on the target) and also reports a threshold-refit variant (target benign train+val only).

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
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .. import common
from . import data as mdata
from . import metrics as mmetrics
from . import predictions
from . import sequence_c as sc
from . import tracking
from . import train_tree as tt
from . import autoencoder as ae

V2 = common.PROCESSED_DIR / "v2"
SPLIT = common.SESSIONS_DIR / "split_v2.json"
DATASET = V2 / "testbed_windows.parquet"
OUT_DIR = V2 / "runs" / "expA"
MODEL_DIR = V2 / "models" / "expA"
SEEDS = (0, 1, 2, 3, 4)

CIC_DIR = common.PROCESSED_DIR / "cic"
SHARED_CLASSES = ("arp_mitm", "recon_scan", "syn_flood", "mqtt_flood")   # testbed <-> CIC overlap

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


def _fit_one(params, seed, Xtr_normal, Xva_normal, extra=None):
    """Train one autoencoder (one architecture, one seed) on benign-only data. Returns the model."""
    ae.set_determinism(seed, 1)
    cfg = ae.make_cfg(seed=seed, n_feat=Xtr_normal.shape[1], **params, **(extra or {}))
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


def _fit_detector(model_name, params, seed, Xtr_normal, Xva_normal, extra=None):
    """Fit either anomaly detector on benign-only data. Returns (score_fn, info, fitted). score_fn(X) ->
    higher = more anomalous. `extra` overrides AE cfg keys (tests only; the report runs pass none)."""
    if model_name == "autoencoder":
        model, info = _fit_one(params, seed, Xtr_normal, Xva_normal, extra)
        return (lambda X: ae.anomaly_scores(model, X)), info, model
    clf = fit_isolation_forest(params, seed, Xtr_normal)
    return (lambda X: if_scores(clf, X)), {"epochs_run": None, "best_epoch": None}, clf


def _detector_scorer(model_name, params, seed, Xtr_normal, Xva_normal):
    """Compatibility wrapper: just the score function."""
    return _fit_detector(model_name, params, seed, Xtr_normal, Xva_normal)[0]


# --------------------------------------------------------------------------------------- environment
@dataclass
class Env:
    """Where a run reads and writes. The defaults are the real v2 paths; the tests point them at a
    temp dir and shrink the grids/epochs."""
    dataset: Path = DATASET
    fl: Path = V2 / "feature_list.json"
    split: Path = SPLIT
    runs: Path = V2 / "runs"
    model_dir: Path = MODEL_DIR
    pred_dir: Path = None                       # None -> predictions.PRED_DIR
    cic_dir: Path = CIC_DIR
    for_report: bool = False
    no_mlflow: bool = True
    seeds: tuple = SEEDS
    ae_cfg: dict = field(default_factory=dict)
    ae_grid: list = None
    if_grid: list = None

    def exp_dir(self, exp):
        d = Path(self.runs) / f"exp{exp}"
        d.mkdir(parents=True, exist_ok=True)
        return d


def preflight(env):
    """Fail fast, before any training, if a report-grade record would be refused for null provenance."""
    if not env.for_report:
        return None
    rec = tracking.build_run_record("A", "autoencoder", "anomaly", seed=0, split_path=str(env.split),
                                    dataset_path=str(env.dataset), for_report=True)
    tracking.assert_report_provenance(rec)
    return rec["provenance"]


def load_selected(model_name, env):
    """The architecture Experiment A selected (B and C read it; nothing is hard-coded)."""
    p = env.exp_dir("A") / f"{model_name}_selected.json"
    if not p.exists():
        raise SystemExit(f"{p} is missing: run Experiment A first (B and C use the architecture A selected)")
    params = json.loads(p.read_text())["params"]
    if "hidden" in params:
        params["hidden"] = tuple(params["hidden"])
    return params


def _tag(params):
    return "_".join(f"{k}{'-'.join(map(str, v)) if isinstance(v, (tuple, list)) else v}"
                    for k, v in params.items())


# --------------------------------------------------------------------------------------- metrics
def _view(y, score, thr):
    pred = ae.predict(score, thr)
    m = mmetrics.binary_metrics(y, y_score=score, y_pred=pred)
    m["fpr"] = float((pred.astype(bool) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return m


def _blocks(prefix, meta, y, score, thr, out):
    """Binary metrics on all rows and on rows with host_role != attacker (plus FPR and recall per
    attack class on each view), and recall on attacker rows and broker rows (attack rows only).
    Mirrors run_lstm._blocks. Keys: <prefix>_{all,noatk}_<metric>, <prefix>_recall_{attacker,broker}."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    role = meta["host_role"].to_numpy()
    cls = meta["attack_class"].astype(str).to_numpy()
    for tag, mask in (("all", np.ones(len(y), bool)), ("noatk", role != "attacker")):
        for k, v in _view(y[mask], score[mask], thr).items():
            out[f"{prefix}_{tag}_{k}"] = v
        for c in sorted(set(cls[mask & (y == 1)])):
            m = mask & (y == 1) & (cls == c)
            out[f"{prefix}_{tag}_recall_{c}"] = float((score[m] >= thr).mean())
    for r in ("attacker", "broker"):
        m = (y == 1) & (role == r)
        out[f"{prefix}_recall_{r}"] = float((score[m] >= thr).mean()) if m.any() else float("nan")
        out[f"{prefix}_n_attack_{r}"] = int(m.sum())


PRED_COLS = ["split", "session", "host_mac", "host_role", "window_start_epoch", "run_id", "attack_class",
             "label", "y_bin", "score", "pred"]


def _pred_frame(split, meta, y, score, thr, domain=None, dataset=None):
    f = meta[["session", "host_mac", "host_role", "window_start_epoch", "run_id", "attack_class",
              "label"]].reset_index(drop=True).copy()
    f.insert(0, "split", split)
    f["y_bin"] = np.asarray(y).astype(int)
    f["score"] = np.asarray(score, dtype=float)
    f["pred"] = ae.predict(score, thr)
    if domain is not None:
        f.insert(1, "domain", domain)
        f.insert(2, "dataset", dataset)
    return f


def _write_preds(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


def _log(env, rec, path):
    tracking.log_run(rec, json_path=str(path), use_mlflow=not env.no_mlflow)


def _record(env, exp, model_name, params, metrics, tags, seed):
    return tracking.build_run_record(
        experiment=exp, model=model_name, role="anomaly", params=params, metrics=metrics, tags=tags,
        seed=seed, split_path=str(env.split), dataset_path=str(env.dataset), for_report=env.for_report)


def _load(env, scale):
    df, fl = mdata.load_dataset(env.dataset, env.fl)
    return mdata.Dataset(df, fl, scale=scale)


# ------------------------------------------------------------------------------------ summaries
_METRICS = ("f1", "precision", "recall", "fpr", "pr_auc", "roc_auc")


def _ms(v):
    v = np.array(v, dtype=float)
    return float(np.nanmean(v)), float(np.nanstd(v))          # population std, as the old aggregate


def _records(env, exp, pattern):
    import re
    d = env.exp_dir(exp)
    out = []
    for p in sorted(d.glob("*.json")):
        if re.fullmatch(pattern, p.name):
            r = json.loads(p.read_text())
            if "params" in r and "metrics" in r:
                out.append(r)
    return out


def write_summary(env, exp, model_name):
    """<model>_summary.json/.md from the per-seed records (the summary is not a record)."""
    import re
    pat = {"A": rf"{model_name}_s\d+\.json", "B": rf"{model_name}_.+_s\d+\.json",
           "C": rf"{model_name}_(tb2cic|cic2tb)_(astrained|refit)_s\d+\.json"}[exp]
    recs = [r for r in _records(env, exp, pat) if r["experiment"] == exp and r["model"] == model_name]
    d = env.exp_dir(exp)
    S = {"experiment": exp, "model": model_name, "n_records": len(recs)}
    L = [f"# {model_name} Experiment {exp} summary", "", f"{len(recs)} per-seed records.", ""]
    if exp == "A" and recs:
        for view in ("all", "noatk"):
            S[view] = {m: _ms([r["metrics"][f"test_{view}_{m}"] for r in recs if f"test_{view}_{m}" in r["metrics"]])
                       for m in _METRICS}
        S["recall_attacker"] = _ms([r["metrics"]["test_recall_attacker"] for r in recs])
        S["recall_broker"] = _ms([r["metrics"]["test_recall_broker"] for r in recs])
        L += ["| view | " + " | ".join(_METRICS) + " |", "|---" * (len(_METRICS) + 1) + "|"]
        for view in ("all", "noatk"):
            L.append(f"| {view} | " + " | ".join(f"{S[view][m][0]:.4f} ± {S[view][m][1]:.4f}" for m in _METRICS) + " |")
        L += ["", f"recall on attacker rows {S['recall_attacker'][0]:.4f} ± {S['recall_attacker'][1]:.4f}; "
                  f"broker rows {S['recall_broker'][0]:.4f} ± {S['recall_broker'][1]:.4f}"]
    elif exp == "B" and recs:
        classes = sorted({r["params"]["held_out_class"] for r in recs})
        seeds = sorted({r["params"]["seed"] for r in recs})
        S["classes"], S["per_class"] = classes, {}
        L += ["| class | view | " + " | ".join(_METRICS) + " |", "|---" * (len(_METRICS) + 2) + "|"]
        for view in ("all", "noatk"):
            for c in classes:
                rs = [r for r in recs if r["params"]["held_out_class"] == c]
                S["per_class"].setdefault(view, {})[c] = {m: _ms([r["metrics"].get(f"test_{view}_{m}", np.nan) for r in rs])
                                                          for m in _METRICS}
                L.append(f"| {c} | {view} | " + " | ".join(
                    f"{S['per_class'][view][c][m][0]:.4f} ± {S['per_class'][view][c][m][1]:.4f}" for m in _METRICS) + " |")
        # B score (selection rule): per seed the mean over classes, then mean/std over seeds
        S["b_score"] = {}
        for view in ("all", "noatk"):
            S["b_score"][view] = {}
            for m in _METRICS:
                per_seed = []
                for sd in seeds:
                    v = [r["metrics"].get(f"test_{view}_{m}", np.nan) for r in recs
                         if r["params"]["seed"] == sd]
                    per_seed.append(np.nanmean(v))
                S["b_score"][view][m] = _ms(per_seed)
        L += ["", "## B score (per seed mean over withheld classes, then over seeds)", ""]
        for view in ("all", "noatk"):
            L.append(f"- {view}: " + ", ".join(f"{m} {S['b_score'][view][m][0]:.4f} ± {S['b_score'][view][m][1]:.4f}"
                                              for m in _METRICS))
    elif exp == "C" and recs:
        S["by"] = {}
        for direction in ("tb2cic", "cic2tb"):
            for variant in ("astrained", "refit"):
                rs = [r for r in recs if r["params"]["direction"] == direction and r["params"]["variant"] == variant]
                if not rs:
                    continue
                for dom in ("tgt_test", "src_test"):
                    for view in ("all", "noatk"):
                        S["by"][f"{direction}/{variant}/{dom}/{view}"] = {
                            m: _ms([r["metrics"].get(f"{dom}_{view}_{m}", np.nan) for r in rs]) for m in _METRICS}
        by_seed = {}
        for r in recs:
            if r["params"]["variant"] == "astrained":
                by_seed.setdefault(r["params"]["seed"], {})[r["params"]["direction"]] = r["metrics"]["tgt_test_all_f1"]
        both = [np.mean(list(v.values())) for v in by_seed.values() if len(v) == 2]
        S["selection_score_astrained_f1"] = _ms(both) if both else None
        L += ["| run | " + " | ".join(_METRICS) + " |", "|---" * (len(_METRICS) + 1) + "|"]
        for k, v in S["by"].items():
            L.append(f"| {k} | " + " | ".join(f"{v[m][0]:.4f} ± {v[m][1]:.4f}" for m in _METRICS) + " |")
        if both:
            L += ["", f"Selection score (as-trained F1, mean of both directions per seed, all rows): "
                      f"{S['selection_score_astrained_f1'][0]:.4f} ± {S['selection_score_astrained_f1'][1]:.4f}. "
                      "The refit variant is reported alongside and is NOT used for selection."]
    (d / f"{model_name}_summary.json").write_text(json.dumps(S, indent=2, default=str))
    (d / f"{model_name}_summary.md").write_text("\n".join(L) + "\n")
    return S


# ------------------------------------------------------------------------------------ Experiment A
def _save_ae(env, model, thr, feats, scaler):
    env.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "model_args": {"n_in": model.n_in, "hidden": list(model.hidden), "bottleneck": model.bottleneck},
                "threshold": float(thr), "feature_names": feats,
                "scaler_pickle": pickle.dumps(scaler)}, env.model_dir / "autoencoder_seed0.pt")


def run_a(model_name, env):
    """Experiment A. Validation grid at seed 0 (records with for_report=False + <model>_selected.json),
    then one for_report record per seed with the frozen architecture. The model, split, threshold rule
    and metrics are unchanged; the aggregate is derived from the per-seed records."""
    preflight(env)
    ds = _load(env, scale=True)
    feats = ds.feature_names
    Xtr_n, Xva_n = ds.normal_only("train"), ds.normal_only("val")
    Xva, yva, Xte, yte = ds.X("val"), ds.y("val"), ds.X("test"), ds.y("test")
    mva, mte = ds.meta("val"), ds.meta("test")
    print(f"[{model_name}] features={len(feats)} train_normal={len(Xtr_n)} val={len(yva)} "
          f"({int(yva.sum())} attack) test={len(yte)} ({int(yte.sum())} attack)")
    d = env.exp_dir("A")
    cand = env.ae_grid if env.ae_grid is not None else (list(grid()) if model_name == "autoencoder" else None)
    if model_name != "autoencoder":
        cand = env.if_grid if env.if_grid is not None else list(if_grid())
    search = []
    for params in cand:
        t0 = time.perf_counter()
        score_fn, info, _ = _fit_detector(model_name, params, 0, Xtr_n, Xva_n, env.ae_cfg)
        sva = score_fn(Xva)
        thr = ae.pick_threshold(yva, sva)
        m = mmetrics.binary_metrics(yva, y_score=sva, y_pred=ae.predict(sva, thr))
        row = {"params": params, "val_f1": m["f1"], "val_threshold": thr,
               "epochs": info["epochs_run"], "fit_s": time.perf_counter() - t0}
        search.append(row)
        grid_rec = tracking.build_run_record(
            "A", model_name, "anomaly", params={**params, "seed": 0, "threshold": thr, "stage": "grid"},
            metrics={"val_f1": m["f1"]}, tags={"stage": "grid"}, seed=0, split_path=str(env.split),
            dataset_path=str(env.dataset), for_report=False)
        _log(env, grid_rec, d / f"{model_name}_grid_{_tag(params)}.json")
        print(f"  grid {params} -> val F1 {m['f1']:.4f} @ thr {thr:.4f}")
    best = max(search, key=lambda r: r["val_f1"])
    params = best["params"]
    (d / f"{model_name}_selected.json").write_text(json.dumps(
        {"model": model_name, "params": params, "val_f1": best["val_f1"],
         "rule": "highest validation binary F1 at the val-max-F1 threshold (seed 0)"}, indent=2, default=str))
    print(f"[{model_name}] selected {params} (val F1 {best['val_f1']:.4f})")

    per_seed, pfe = [], None
    for seed in env.seeds:
        t0 = time.perf_counter()
        score_fn, info, fitted = _fit_detector(model_name, params, seed, Xtr_n, Xva_n, env.ae_cfg)
        fit_s = time.perf_counter() - t0
        s_va, s_te = score_fn(Xva), score_fn(Xte)
        thr = ae.pick_threshold(yva, s_va)
        metrics = {"train_s": fit_s}
        if info["epochs_run"] is not None:
            metrics["epochs_run"] = info["epochs_run"]
        _blocks("val", mva, yva, s_va, thr, metrics)
        _blocks("test", mte, yte, s_te, thr, metrics)
        rec = _record(env, "A", model_name,
                      {**params, "seed": seed, "threshold": thr, "best_epoch": info["best_epoch"],
                       "threshold_rule": "val-max-F1", "trained_on": "benign-only"},
                      metrics, {"dataset": "v2", "stage": "A"}, seed)
        _write_preds(d / "predictions" / f"{model_name}_s{seed}.parquet",
                     [_pred_frame("val", mva, yva, s_va, thr), _pred_frame("test", mte, yte, s_te, thr)])
        _log(env, rec, d / f"{model_name}_s{seed}.json")
        per_seed.append({**{k: v for k, v in metrics.items() if k.startswith("test_all_")}, "threshold": thr})
        print(f"  seed {seed}: test F1 {metrics['test_all_f1']:.4f} (attacker-excluded "
              f"{metrics['test_noatk_f1']:.4f}) FPR {metrics['test_all_fpr']:.5f}")
        if seed == env.seeds[0]:
            predictions.save_predictions(mte, yte, s_te, ae.predict(s_te, thr), "A", model_name,
                                         out_dir=env.pred_dir)
            if model_name == "autoencoder":
                pfe = per_feature_error(fitted, Xte, feats)
                _save_ae(env, fitted, thr, feats, ds.scaler)
            else:
                import joblib
                env.model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump({"model": fitted, "threshold": float(thr), "feature_names": feats,
                             "scaler": ds.scaler}, env.model_dir / "isolation_forest_seed0.joblib")
    (d / f"{model_name}_details.json").write_text(json.dumps(
        {"grid_search": search, "per_seed": per_seed, "per_feature_error_seed0": pfe, "features": feats},
        indent=2, default=str))
    S = write_summary(env, "A", model_name)
    print(f"\n[{model_name}] TEST binary F1 (all rows) {S['all']['f1'][0]:.4f} +/- {S['all']['f1'][1]:.4f} | "
          f"attacker-excluded {S['noatk']['f1'][0]:.4f} +/- {S['noatk']['f1'][1]:.4f}")
    return S


# ------------------------------------------------------------------------------------ Experiment B
def run_b(model_name, ds, feats, args=None, env=None):
    """Leave-one-attack-out. The detector trains on BENIGN-ONLY traffic, so withholding class X never
    changes training; the model is therefore fitted ONCE per seed and reused for every class. X is
    withheld from the validation rows that set the threshold, and TEST is benign + X only. One
    for_report record per (class, seed); the architecture is the one Experiment A selected."""
    env = env or Env(for_report=getattr(args, "for_report", False), no_mlflow=getattr(args, "no_mlflow", True))
    params = load_selected(model_name, env)
    preflight(env)
    Xtr_n, Xva_n = ds.normal_only("train"), ds.normal_only("val")
    mva, mte = ds.meta("val"), ds.meta("test")

    def cls_of(m):
        return np.where(m["label"].to_numpy() == "attack", m["attack_class"].astype(str), "benign")

    c_va, c_te = cls_of(mva), cls_of(mte)
    Xva_all, yva, Xte, yte = ds.X("val"), ds.y("val"), ds.X("test"), ds.y("test")
    classes = sorted(set(cls_of(ds.meta("train"))) - {"benign"})
    d = env.exp_dir("B")
    for X in classes:
        if int(yte[(c_te == "benign") | (c_te == X)].sum()) == 0:
            raise SystemExit(f"class {X} has no test rows: Experiment B undefined for it")
    for seed in env.seeds:
        t0 = time.perf_counter()
        score_fn, info, fitted = _fit_detector(model_name, params, seed, Xtr_n, Xva_n, env.ae_cfg)
        fit_s = time.perf_counter() - t0
        for X in classes:
            kva = c_va != X                              # withhold X from the threshold-fitting val rows
            kte = (c_te == "benign") | (c_te == X)       # test: benign + X only
            yva_x, mva_x, s_va = yva[kva], mva[kva], score_fn(Xva_all[kva])
            yte_x, mte_x, s_te = yte[kte], mte[kte], score_fn(Xte[kte])
            thr = ae.pick_threshold(yva_x, s_va)
            metrics = {"train_s": fit_s, "n_test_attack": int(yte_x.sum())}
            if info["epochs_run"] is not None:
                metrics["epochs_run"] = info["epochs_run"]
            _blocks("val", mva_x, yva_x, s_va, thr, metrics)
            _blocks("test", mte_x, yte_x, s_te, thr, metrics)
            name = f"{model_name}_{X}_s{seed}"
            rec = _record(env, "B", model_name,
                          {**params, "seed": seed, "threshold": thr, "held_out_class": X,
                           "best_epoch": info["best_epoch"], "threshold_rule": "val-max-F1",
                           "withheld_from": "val+test (training is benign-only)", "trained_on": "benign-only"},
                          metrics, {"dataset": "v2", "stage": "B"}, seed)
            _write_preds(d / "predictions" / f"{name}.parquet",
                         [_pred_frame("val", mva_x, yva_x, s_va, thr), _pred_frame("test", mte_x, yte_x, s_te, thr)])
            _log(env, rec, d / f"{name}.json")
            print(f"  seed {seed} withhold {X:13s} F1 {metrics['test_all_f1']:.4f} "
                  f"(attacker-excluded {metrics['test_noatk_f1']:.4f}) recall {metrics['test_all_recall']:.4f} "
                  f"FPR {metrics['test_all_fpr']:.5f} n_test={int(yte_x.sum())}")
            if metrics["test_noatk_f1"] >= 0.98:                       # working rule 3
                top = (per_feature_error(fitted, Xte[kte & (yte == 1)], feats)[:10]
                       if model_name == "autoencoder" else "n/a (Isolation Forest)")
                with open(d / "FLAG.md", "a") as f:
                    f.write(f"- {model_name} class={X} seed={seed} attacker-excluded F1="
                            f"{metrics['test_noatk_f1']:.4f}; top per-feature reconstruction errors: {top}\n")
    S = write_summary(env, "B", model_name)
    print(f"\n[{model_name}] EXPERIMENT B (all rows) F1 {S['b_score']['all']['f1'][0]:.4f} +/- "
          f"{S['b_score']['all']['f1'][1]:.4f} | attacker-excluded {S['b_score']['noatk']['f1'][0]:.4f}")
    return S


# ------------------------------------------------------------------------------------ Experiment C
def refit_threshold(score_fn, scaler, tgt, fpr_src, seed, n_max=1000):
    """Threshold-refit variant: the (1 - fpr_src) quantile of the scores of a seeded sample of TARGET
    BENIGN rows drawn from target train U val. Never the target test split, never a target attack
    row. Same model and scaler; only the threshold moves. Returns (thr_refit, n_refit)."""
    X = np.concatenate([tgt.X("train")[tgt.y("train") == 0], tgt.X("val")[tgt.y("val") == 0]])
    if len(X) == 0:
        raise SystemExit("no benign target train/val rows to refit the threshold on")
    n = min(int(n_max), len(X))
    idx = np.sort(np.random.RandomState(int(seed)).choice(len(X), size=n, replace=False))
    s = score_fn(scaler.transform(X[idx]))
    return float(np.quantile(s, 1.0 - float(fpr_src))), int(n)


def source_scalers(doms):
    """One scaler per SOURCE domain: StandardScaler on its train rows (after the tree filter). The
    target is always transformed with the source's scaler and never refit."""
    return {k: StandardScaler().fit(v.X("train")) for k, v in doms.items()}


def _sha(path):
    return tracking._sha256_file(path) if Path(path).exists() else None


def c_seed(env, model_name, params, doms, scalers, src, tgt, seed, tags, write=True):
    """One (direction, seed): fit on source benign (source scaler), threshold on source val, then both
    variants. Returns everything a test needs to check scaler/model identity between the variants."""
    ds_s, ds_t, scaler = doms[src], doms[tgt], scalers[src]
    score_fn, info, fitted = _fit_detector(model_name, params, seed,
                                           scaler.transform(ds_s.normal_only("train")),
                                           scaler.transform(ds_s.normal_only("val")), env.ae_cfg)
    yva = ds_s.y("val")
    s_val = score_fn(scaler.transform(ds_s.X("val")))
    thr = ae.pick_threshold(yva, s_val)
    s_src = score_fn(scaler.transform(ds_s.X("test")))
    s_tgt = score_fn(scaler.transform(ds_t.X("test")))          # SOURCE scaler on the target: never refit
    fpr_src = float((s_val[yva == 0] >= thr).mean())
    thr_refit, n_refit = refit_threshold(score_fn, scaler, ds_t, fpr_src, seed)
    out = {"score_fn": score_fn, "model": fitted, "scaler": scaler, "thr": thr, "thr_refit": thr_refit,
           "n_refit": n_refit, "fpr_src": fpr_src, "s_tgt": s_tgt, "s_src": s_src, "s_val": s_val}
    if not write:
        return out
    d = env.exp_dir("C")
    tag = {"testbed": "tb2cic", "cic": "cic2tb"}[src]
    mv, ms, mt = ds_s.meta("val"), ds_s.meta("test"), ds_t.meta("test")
    for variant, t_thr in (("astrained", thr), ("refit", thr_refit)):
        metrics = {}
        _blocks("val", mv, yva, s_val, thr, metrics)
        _blocks("src_test", ms, ds_s.y("test"), s_src, thr, metrics)       # in-domain reference
        _blocks("tgt_test", mt, ds_t.y("test"), s_tgt, t_thr, metrics)
        if info["epochs_run"] is not None:
            metrics["epochs_run"] = info["epochs_run"]
        p = {**params, "seed": seed, "direction": tag, "variant": variant, "source": src, "target": tgt,
             "threshold": thr, "target_threshold": t_thr, "best_epoch": info["best_epoch"],
             "threshold_rule": "source-val-max-F1 (as trained)" if variant == "astrained" else
             "refit: (1 - fpr_src) quantile of target benign train+val sample scores",
             "shared_classes": list(SHARED_CLASSES), "n_features": len(ds_s.feature_names),
             "trained_on": "benign-only", "scaler": "source train-only, never refit on the target"}
        if variant == "refit":
            p.update({"n_refit": n_refit, "fpr_src": fpr_src, "thr_refit": thr_refit})
        rec = _record(env, "C", model_name, p, metrics, {"dataset": "v2+cic", "stage": "C", **tags}, seed)
        name = f"{model_name}_{tag}_{variant}_s{seed}"
        _write_preds(d / "predictions" / f"{name}.parquet",
                     [_pred_frame("val", mv, yva, s_val, thr, "val", src),
                      _pred_frame("test", ms, ds_s.y("test"), s_src, thr, "src_test", src),
                      _pred_frame("test", mt, ds_t.y("test"), s_tgt, t_thr, "tgt_test", tgt)])
        _log(env, rec, d / f"{name}.json")
        print(f"  {tag} {variant:9s} seed {seed}: target F1 {metrics['tgt_test_all_f1']:.4f} "
              f"(attacker-excluded {metrics['tgt_test_noatk_f1']:.4f}) | source F1 {metrics['src_test_all_f1']:.4f}")
    return out


def load_c_domains(env):
    """Both domains under the trees' shared-feature contract (train_tree._domain: unscaled, shared
    features, benign + shared classes), the feature lists asserted equal, the DDoS stems rejected."""
    cic_parquet = Path(env.cic_dir) / "cic_windows.parquet"
    if not cic_parquet.exists():
        raise SystemExit(
            f"Experiment C needs {cic_parquet} (CICIoMT2024 extracted features), which is not present. "
            f"Run `python -m src.features.extract_cic` on the CIC PCAPs first. Experiment C deferred -- "
            f"NOT run (CLAUDE.md: never fake a result).")
    cic_fl = Path(env.cic_dir) / "feature_list.json"
    tb = tt._domain(env.dataset, env.fl)
    cic = tt._domain(cic_parquet, cic_fl)
    if tb.feature_names != cic.feature_names:
        raise SystemExit(f"feature mismatch: {set(tb.feature_names) ^ set(cic.feature_names)} "
                         f"(or a different order)")
    stems = sc.check_cic_files(pd.read_parquet(cic_parquet, columns=["run_id", "split"]))
    tags = {"cic_files": stems, "cic_windows_sha256": _sha(cic_parquet), "cic_feature_list_sha256": _sha(cic_fl),
            "cic_extract_report_sha256": _sha(Path(env.cic_dir) / "extract_report.json")}
    return {"testbed": tb, "cic": cic}, tags


def run_c(model_name, env):
    """Cross-dataset, as trained AND with the threshold refit. Each direction: fit on SOURCE benign
    traffic scaled with the SOURCE train-only scaler, threshold on SOURCE val, score the TARGET test
    with that same scaler (never refit). One record per (direction, variant, seed)."""
    preflight(env)
    params = load_selected(model_name, env)
    doms, tags = load_c_domains(env)
    scalers = source_scalers(doms)
    for src, tgt in (("testbed", "cic"), ("cic", "testbed")):
        for seed in env.seeds:
            c_seed(env, model_name, params, doms, scalers, src, tgt, seed, tags)
    S = write_summary(env, "C", model_name)
    sel = S.get("selection_score_astrained_f1")
    if sel:
        print(f"\n[{model_name}] EXPERIMENT C as-trained F1 (both directions) {sel[0]:.4f} +/- {sel[1]:.4f}")
    return S


def run_isolation_forest(*a, **k):                       # kept for callers of the old name
    raise NotImplementedError("use run_a(model_name='isolation_forest', env)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["autoencoder", "isolation_forest"], default="autoencoder",
                    help="which anomaly detector to train (the role has two: AE + IF baseline)")
    ap.add_argument("--experiment", choices=["A", "B", "C"], default="A",
                    help="A = time split; B = leave-one-attack-out; C = cross-dataset (needs cic/)")
    ap.add_argument("--for-report", action="store_true",
                    help="refuse to write records unless provenance is complete (checked before training)")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--summary", action="store_true", help="rebuild the summary from the per-seed records")
    args = ap.parse_args(argv)
    env = Env(for_report=args.for_report, no_mlflow=args.no_mlflow)
    if args.summary:
        write_summary(env, args.experiment, args.model)
        return 0
    if args.experiment == "A":
        run_a(args.model, env)
    elif args.experiment == "B":
        ds = _load(env, scale=True)
        run_b(args.model, ds, ds.feature_names, env=env)
    else:
        run_c(args.model, env)
    return 0


if __name__ == "__main__":
    main()
