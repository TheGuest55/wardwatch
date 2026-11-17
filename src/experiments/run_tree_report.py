"""Report-grade XGBoost / Random Forest runs for Experiments A, B and C.

    python -m src.experiments.run_tree_report --model xgboost --experiment A --for-report
    python -m src.experiments.run_tree_report --model random_forest --experiment B --for-report
    python -m src.experiments.run_tree_report --model xgboost --experiment C --for-report
    python -m src.experiments.run_tree_report --model xgboost --experiment A --summary   # rebuild the summary

Why this exists: src/models/train_tree.py writes ONE aggregated record per model and experiment, all rows
only, without saved predictions or a DDoS guard on the CIC frame. The autoencoder, Isolation Forest and
LSTM report runs write one for_report record per seed (per class in B, per direction in C), report every
metric on all rows AND with the attacker host excluded, and save their predictions. This wrapper gives the
trees the same contract WITHOUT changing their protocol: it calls train_tree's own functions (make_model,
grid, best_threshold, importances, multiclass_targets, _domain) and train_tree.py is not modified.

Protocol (identical to train_tree; tests/pipeline/test_run_tree_report.py checks the numbers match):
  * Inputs: feature_list.json features, unscaled (trees). victim/boundary rows excluded (harness).
  * Hyper-parameters: seed-0 grid, chosen by validation binary F1 at the validation-max-F1 threshold.
  * Per seed: fit, threshold = validation-max-F1, score test once. 5 seeds.
  * A also: multiclass model (benign + named class, balanced sample weights, train_tree step 3).
    test_macro_f1 uses the LSTM's definition (lstm.macro_f1: argmax over the fixed 9-class order,
    labels = range(9), zero_division = 0) so the classifiers are comparable; train_tree's own definition
    (sklearn's default label set, rows of classes unseen in training dropped) is kept as
    test_macro_f1_traintree.
  * B: class X is withheld from train AND validation; the grid is re-run on that validation (train_tree's
    B protocol; the LSTM used one frozen config instead -- both leak-free, protocols differ).
    Test = benign + X.
  * C: shared features and classes only (train_tree._domain); source-domain grid and threshold; the target
    is scored as trained (no refit for classifiers). Single-attacker CIC files only (DDoS stops the run).

Outputs (data/processed/v2/runs/exp{A,B,C}/):
  A: <model>_s{seed}.json, <model>_grid_<params>.json (for_report=false), <model>_selected.json,
     <model>_details.json, predictions/<model>_s{seed}.parquet, runs/predictions/A_<model>.parquet (seed 0),
     models/expA/<model>_binary_seed0.{json,joblib} (the files calibrate_explain reads).
  B: <model>_<class>_s{seed}.json, predictions/..., <model>_details.json.
  C: <model>_<tb2cic|cic2tb>_astrained_s{seed}.json, predictions/..., <model>_details.json.
  Each: <model>_summary.{md,json}; FLAG.md if a result reaches 0.98 F1 (working rule 3).
Threads: train_tree.N_JOBS (from ANALYSIS_CPUS) is recorded in every record's params.
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import lstm
from ..models import predictions
from ..models import sequence as seq
from ..models import tracking
from ..models import train_autoencoder as ta
from ..models import train_tree as tt

MODELS = ("xgboost", "random_forest")
FLAG_F1 = 0.98
MACRO_DEF = "lstm.macro_f1: argmax over CLASS_ORDER (benign + 8 attack classes), labels=range(9), zero_division=0"
MACRO_DEF_TT = "train_tree step 3: sklearn f1_score macro over the labels present, test rows of classes unseen in train dropped"


@dataclass
class Env(ta.Env):
    """ta.Env (paths, seeds, for_report, mlflow) plus an optional grid override for the tests."""
    tree_grid: dict = None


def _grid(env, name):
    if env.tree_grid and name in env.tree_grid:
        return [dict(p) for p in env.tree_grid[name]]
    return list(tt.grid(name))


def _base_params():
    return {"n_jobs": tt.N_JOBS, "threshold_rule": "val-max-F1",
            "selection": "seed-0 grid by validation binary F1 at the val-max-F1 threshold",
            "protocol": "src.models.train_tree (unchanged), wrapped by src.experiments.run_tree_report"}


def _record(env, exp, name, params, metrics, tags, seed, for_report=None):
    return tracking.build_run_record(
        experiment=exp, model=name, role="classifier", params=params, metrics=metrics, tags=tags, seed=seed,
        split_path=str(env.split), dataset_path=str(env.dataset),
        for_report=env.for_report if for_report is None else for_report)


def _proba(clf, X):
    return clf.predict_proba(X)[:, 1]


def _select(env, name, Xtr, ytr, Xva, yva):
    """train_tree's selection: seed-0 grid, best validation F1 at the val-max-F1 threshold (first max wins)."""
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    search = []
    for params in _grid(env, name):
        t0 = time.perf_counter()
        clf = tt.make_model(name, params, 0, n_pos, n_neg).fit(Xtr, ytr)
        thr, f1 = tt.best_threshold(yva, _proba(clf, Xva))
        search.append({"params": params, "val_f1": f1, "val_threshold": thr, "fit_s": time.perf_counter() - t0})
    return max(search, key=lambda r: r["val_f1"]), search, n_pos, n_neg


def _fit_seed(name, params, seed, Xtr, ytr, Xva, yva, n_pos, n_neg):
    t0 = time.perf_counter()
    clf = tt.make_model(name, params, seed, n_pos, n_neg).fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    s_va = _proba(clf, Xva)
    thr, val_f1 = tt.best_threshold(yva, s_va)
    return clf, thr, s_va, fit_s


def _flag(d, line):
    with open(d / "FLAG.md", "a") as f:
        f.write(line + "\n")
    print("FLAG:", line, flush=True)


def _save_model(env, name, clf):
    env.model_dir.mkdir(parents=True, exist_ok=True)
    if name == "xgboost":
        clf.save_model(str(env.model_dir / "xgboost_binary_seed0.json"))
    else:
        import joblib
        joblib.dump(clf, env.model_dir / "random_forest_binary_seed0.joblib")


def _macro_f1(name, params, seed, n_pos, n_neg, ds, Xtr, Xte):
    """Multiclass macro-F1 on test (Experiment A), both definitions. train_tree step 3 model."""
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    ytr_m, yte_m = tt.multiclass_targets(ds.meta("train")), tt.multiclass_targets(ds.meta("test"))
    unknown = sorted(set(yte_m) - set(seq.CLASS_ORDER))
    if unknown:
        raise SystemExit(f"test classes outside the fixed 9-class order: {unknown}")
    enc = LabelEncoder().fit(ytr_m)
    mc = tt.make_model(name, params, seed, n_pos, n_neg, multiclass=True)
    mc.fit(Xtr, enc.transform(ytr_m), sample_weight=compute_sample_weight("balanced", ytr_m))
    P_enc = mc.predict_proba(Xte)                                   # columns in enc.classes_ order
    P = np.zeros((len(Xte), len(seq.CLASS_ORDER)))
    for j, c in enumerate(enc.classes_):
        P[:, seq.CLASS_ORDER.index(c)] = P_enc[:, j]
    y_idx = np.array([seq.CLASS_ORDER.index(c) for c in yte_m])
    known = np.isin(yte_m, enc.classes_)
    pm = enc.inverse_transform(mc.predict(Xte[known]))
    return {"test_macro_f1": lstm.macro_f1(y_idx, P),
            "test_macro_f1_traintree": float(f1_score(yte_m[known], pm, average="macro", zero_division=0)),
            "n_test_unseen_class_rows": int((~known).sum())}


# ------------------------------------------------------------------------------------ Experiment A
def run_a(name, env):
    ta.preflight(env)
    d = env.exp_dir("A")
    ds = ta._load(env, scale=False)
    feats = ds.feature_names
    Xtr, ytr, Xva, yva, Xte, yte = (ds.X("train"), ds.y("train"), ds.X("val"), ds.y("val"),
                                    ds.X("test"), ds.y("test"))
    mva, mte = ds.meta("val"), ds.meta("test")
    best, search, n_pos, n_neg = _select(env, name, Xtr, ytr, Xva, yva)
    params = best["params"]
    for s in search:
        g = _record(env, "A", name, {**s["params"], "seed": 0, "threshold": s["val_threshold"], "stage": "grid"},
                    {"val_f1": s["val_f1"], "fit_s": s["fit_s"]}, {"dataset": "v2", "stage": "grid"}, 0,
                    for_report=False)
        ta._log(env, g, d / f"{name}_grid_{ta._tag(s['params'])}.json")
    (d / f"{name}_selected.json").write_text(json.dumps(
        {"model": name, "params": params, "val_f1": best["val_f1"],
         "rule": "highest validation binary F1 at the val-max-F1 threshold (seed 0)"}, indent=2, default=str))
    print(f"[{name}] A selected {params} (val F1 {best['val_f1']:.4f})", flush=True)

    imps = None
    for seed in env.seeds:
        clf, thr, s_va, fit_s = _fit_seed(name, params, seed, Xtr, ytr, Xva, yva, n_pos, n_neg)
        s_te = _proba(clf, Xte)
        metrics = {"train_s": fit_s}
        ta._blocks("val", mva, yva, s_va, thr, metrics)
        ta._blocks("test", mte, yte, s_te, thr, metrics)
        metrics.update(_macro_f1(name, params, seed, n_pos, n_neg, ds, Xtr, Xte))
        rec = _record(env, "A", name, {**params, **_base_params(), "seed": seed, "threshold": thr,
                                       "macro_f1_definition": MACRO_DEF, "macro_f1_traintree_definition": MACRO_DEF_TT},
                      metrics, {"dataset": "v2", "stage": "A"}, seed)
        ta._write_preds(d / "predictions" / f"{name}_s{seed}.parquet",
                        [ta._pred_frame("val", mva, yva, s_va, thr), ta._pred_frame("test", mte, yte, s_te, thr)])
        ta._log(env, rec, d / f"{name}_s{seed}.json")
        print(f"  seed {seed}: test F1 {metrics['test_all_f1']:.4f} (attacker-excluded {metrics['test_noatk_f1']:.4f}) "
              f"macro-F1 {metrics['test_macro_f1']:.4f} (train_tree def {metrics['test_macro_f1_traintree']:.4f})",
              flush=True)
        if seed == env.seeds[0]:
            predictions.save_predictions(mte, yte, s_te, (s_te >= thr).astype(int), "A", name, out_dir=env.pred_dir)
            _save_model(env, name, clf)
            imps = tt.importances(name, clf, feats)[:20]
    (d / f"{name}_details.json").write_text(json.dumps(
        {"grid_search": search, "selected": params, "feature_importance_seed0": imps, "features": feats},
        indent=2, default=str))
    return summary(env, "A", name)


# ------------------------------------------------------------------------------------ Experiment B
def run_b(name, env):
    ta.preflight(env)
    d = env.exp_dir("B")
    ds = ta._load(env, scale=False)
    feats = ds.feature_names
    c_tr, c_va, c_te = (tt.multiclass_targets(ds.meta(s)) for s in ("train", "val", "test"))
    classes = sorted(set(c_tr) - {"benign"})
    details = {}
    for X in classes:
        ktr, kva = c_tr != X, c_va != X                       # X never reaches training or tuning
        kte = (c_te == "benign") | (c_te == X)                # test: benign + X only
        Xtr, ytr = ds.X("train")[ktr], ds.y("train")[ktr]
        Xva, yva, mva = ds.X("val")[kva], ds.y("val")[kva], ds.meta("val")[kva].reset_index(drop=True)
        Xte, yte, mte = ds.X("test")[kte], ds.y("test")[kte], ds.meta("test")[kte].reset_index(drop=True)
        if yte.sum() == 0:
            raise SystemExit(f"class {X} has no test rows: Experiment B undefined for it")
        best, search, n_pos, n_neg = _select(env, name, Xtr, ytr, Xva, yva)
        params = best["params"]
        imp = None
        for seed in env.seeds:
            clf, thr, s_va, fit_s = _fit_seed(name, params, seed, Xtr, ytr, Xva, yva, n_pos, n_neg)
            s_te = _proba(clf, Xte)
            metrics = {"train_s": fit_s, "n_test_attack": int(yte.sum()), "n_train_attack": n_pos}
            ta._blocks("val", mva, yva, s_va, thr, metrics)
            ta._blocks("test", mte, yte, s_te, thr, metrics)
            rec = _record(env, "B", name, {**params, **_base_params(), "seed": seed, "threshold": thr,
                                           "held_out_class": X, "withheld_from": "train+val",
                                           "tuning": "grid re-run on validation without the withheld class"},
                          metrics, {"dataset": "v2", "stage": "B"}, seed)
            ta._write_preds(d / "predictions" / f"{name}_{X}_s{seed}.parquet",
                            [ta._pred_frame("val", mva, yva, s_va, thr), ta._pred_frame("test", mte, yte, s_te, thr)])
            ta._log(env, rec, d / f"{name}_{X}_s{seed}.json")
            if seed == env.seeds[0]:
                imp = tt.importances(name, clf, feats)[:10]
            if metrics["test_noatk_f1"] >= FLAG_F1:
                _flag(d, f"{name} withhold {X} seed {seed}: attacker-excluded F1 {metrics['test_noatk_f1']:.4f}; "
                         f"top importances {imp}")
            print(f"  seed {seed} withhold {X:13s} F1 {metrics['test_all_f1']:.4f} "
                  f"(attacker-excluded {metrics['test_noatk_f1']:.4f}) recall {metrics['test_all_recall']:.4f}",
                  flush=True)
        details[X] = {"params": params, "grid_search": search, "top_features_seed0": imp,
                      "n_test_attack": int(yte.sum()), "n_train_attack": n_pos}
    (d / f"{name}_details.json").write_text(json.dumps(details, indent=2, default=str))
    return summary(env, "B", name)


# ------------------------------------------------------------------------------------ Experiment C
def run_c(name, env):
    ta.preflight(env)
    d = env.exp_dir("C")
    doms, tags = ta.load_c_domains(env)                    # shared features/classes + DDoS guard + cic_files
    details = {}
    for src, tgt in (("testbed", "cic"), ("cic", "testbed")):
        tag = "tb2cic" if src == "testbed" else "cic2tb"
        ds_s, ds_t = doms[src], doms[tgt]
        Xtr, ytr, Xva, yva = ds_s.X("train"), ds_s.y("train"), ds_s.X("val"), ds_s.y("val")
        mva, ms, mt = ds_s.meta("val"), ds_s.meta("test"), ds_t.meta("test")
        best, search, n_pos, n_neg = _select(env, name, Xtr, ytr, Xva, yva)
        params = best["params"]
        imp = None
        for seed in env.seeds:
            clf, thr, s_va, fit_s = _fit_seed(name, params, seed, Xtr, ytr, Xva, yva, n_pos, n_neg)
            s_src, s_tgt = _proba(clf, ds_s.X("test")), _proba(clf, ds_t.X("test"))
            metrics = {"train_s": fit_s}
            ta._blocks("val", mva, yva, s_va, thr, metrics)
            ta._blocks("src_test", ms, ds_s.y("test"), s_src, thr, metrics)     # in-domain reference
            ta._blocks("tgt_test", mt, ds_t.y("test"), s_tgt, thr, metrics)     # as trained: source threshold
            rec = _record(env, "C", name, {**params, **_base_params(), "seed": seed, "threshold": thr,
                                           "direction": tag, "variant": "astrained", "source": src, "target": tgt,
                                           "n_features": len(ds_s.feature_names),
                                           "shared_classes": list(tt.SHARED_CLASSES)},
                          metrics, {"dataset": "v2+cic", "stage": "C", **tags}, seed)
            ta._write_preds(d / "predictions" / f"{name}_{tag}_astrained_s{seed}.parquet",
                            [ta._pred_frame("val", mva, yva, s_va, thr, "val", src),
                             ta._pred_frame("test", ms, ds_s.y("test"), s_src, thr, "src_test", src),
                             ta._pred_frame("test", mt, ds_t.y("test"), s_tgt, thr, "tgt_test", tgt)])
            ta._log(env, rec, d / f"{name}_{tag}_astrained_s{seed}.json")
            if seed == env.seeds[0]:
                imp = tt.importances(name, clf, ds_s.feature_names)[:10]
            top = max(metrics["tgt_test_all_f1"], metrics["tgt_test_noatk_f1"])
            if top >= FLAG_F1:
                _flag(d, f"{name} {tag} seed {seed}: cross-dataset F1 {top:.4f}; top importances {imp}")
            print(f"  {tag} seed {seed}: target F1 {metrics['tgt_test_all_f1']:.4f} (attacker-excluded "
                  f"{metrics['tgt_test_noatk_f1']:.4f}) | source F1 {metrics['src_test_all_f1']:.4f}", flush=True)
        details[tag] = {"params": params, "grid_search": search, "top_features_seed0": imp}
    (d / f"{name}_details.json").write_text(json.dumps({**details, "tags": tags}, indent=2, default=str))
    return summary(env, "C", name)


# ------------------------------------------------------------------------------------ summaries
def summary(env, exp, name):
    """train_autoencoder.write_summary (same layout as AE/IF); A adds both macro-F1 definitions."""
    S = ta.write_summary(env, exp, name)
    if exp == "A":
        recs = ta._records(env, "A", rf"{name}_s\d+\.json")
        recs = [r for r in recs if r["experiment"] == "A" and r["model"] == name]
        S["macro_f1"] = ta._ms([r["metrics"]["test_macro_f1"] for r in recs])
        S["macro_f1_traintree"] = ta._ms([r["metrics"]["test_macro_f1_traintree"] for r in recs])
        d = env.exp_dir("A")
        (d / f"{name}_summary.json").write_text(json.dumps(S, indent=2, default=str))
        with open(d / f"{name}_summary.md", "a") as f:
            f.write(f"\nmulticlass macro-F1 (LSTM definition) {S['macro_f1'][0]:.4f} ± {S['macro_f1'][1]:.4f}; "
                    f"train_tree definition {S['macro_f1_traintree'][0]:.4f} ± {S['macro_f1_traintree'][1]:.4f}\n")
    if exp == "A":
        print(f"\n[{name}] A test F1 {S['all']['f1'][0]:.4f} +/- {S['all']['f1'][1]:.4f} | attacker-excluded "
              f"{S['noatk']['f1'][0]:.4f} | macro-F1 {S['macro_f1'][0]:.4f}")
    elif exp == "B":
        print(f"\n[{name}] B score F1 {S['b_score']['all']['f1'][0]:.4f} +/- {S['b_score']['all']['f1'][1]:.4f} | "
              f"attacker-excluded {S['b_score']['noatk']['f1'][0]:.4f}")
    elif S.get("selection_score_astrained_f1"):
        sel = S["selection_score_astrained_f1"]
        print(f"\n[{name}] C as-trained F1 (both directions) {sel[0]:.4f} +/- {sel[1]:.4f}")
    return S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--experiment", choices=["A", "B", "C"], required=True)
    ap.add_argument("--for-report", action="store_true",
                    help="refuse to write records unless provenance is complete (checked before training)")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--summary", action="store_true", help="rebuild the summary from the per-seed records")
    args = ap.parse_args(argv)
    env = Env(for_report=args.for_report, no_mlflow=args.no_mlflow)
    if args.summary:
        summary(env, args.experiment, args.model)
        return 0
    print(f"[{args.model}] Experiment {args.experiment}, n_jobs={tt.N_JOBS}, seeds={env.seeds}", flush=True)
    {"A": run_a, "B": run_b, "C": run_c}[args.experiment](args.model, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
