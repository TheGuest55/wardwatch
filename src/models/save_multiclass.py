"""Refit and save the seed-0 multiclass XGBoost that names the attack type for "Known attack" windows.

    python -m src.models.save_multiclass            # -> models/expA/xgboost_multiclass_seed0.json (+ classes)

Experiment A trained this model (benign + the named classes, balanced sample weights, train_tree step 3) only to
report macro-F1 and never saved it. The triage layer needs it to NAME the type, so it is refit here exactly as
run_tree_report._macro_f1 does: same hyper-parameters (the A-selected ones), same data (A train), seed 0, using
train_tree.make_model / multiclass_targets (train_tree.py is not modified).

Identity (asserted, a stop if it fails): the refit model's test macro-F1 (the LSTM definition, lstm.macro_f1, and
train_tree's own) equals the value recorded in runs/expA/xgboost_s0.json. If the numbers differ the refit is not the
model Experiment A reported, and nothing is saved.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import common
from . import lstm
from . import sequence as seq
from . import train_autoencoder as ta
from . import train_tree as tt

V2 = tt.V2
OUT = tt.MODEL_DIR / "xgboost_multiclass_seed0.json"
CLASSES = tt.MODEL_DIR / "xgboost_multiclass_seed0_classes.json"
RECORD = tt.OUT_DIR / "xgboost_s0.json"
TOL = 1e-12


def fit(env=None, params=None, seed=0):
    """Returns (model, classes, ds, metrics). The same fit as run_tree_report._macro_f1."""
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    env = env or ta.Env()
    ds = ta._load(env, scale=False)
    Xtr, ytr, Xte = ds.X("train"), ds.y("train"), ds.X("test")
    params = params or json.loads((tt.OUT_DIR / "xgboost_selected.json").read_text())["params"]
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    ytr_m, yte_m = tt.multiclass_targets(ds.meta("train")), tt.multiclass_targets(ds.meta("test"))
    enc = LabelEncoder().fit(ytr_m)
    mc = tt.make_model("xgboost", params, seed, n_pos, n_neg, multiclass=True)
    mc.fit(Xtr, enc.transform(ytr_m), sample_weight=compute_sample_weight("balanced", ytr_m))
    P_enc = mc.predict_proba(Xte)
    P = np.zeros((len(Xte), len(seq.CLASS_ORDER)))
    for j, c in enumerate(enc.classes_):
        P[:, seq.CLASS_ORDER.index(c)] = P_enc[:, j]
    y_idx = np.array([seq.CLASS_ORDER.index(c) for c in yte_m])
    known = np.isin(yte_m, enc.classes_)
    pm = enc.inverse_transform(mc.predict(Xte[known]))
    metrics = {"test_macro_f1": lstm.macro_f1(y_idx, P),
               "test_macro_f1_traintree": float(f1_score(yte_m[known], pm, average="macro", zero_division=0))}
    return mc, [str(c) for c in enc.classes_], ds, metrics


def check_identity(metrics, record_path=RECORD):
    rec = json.loads(record_path.read_text())["metrics"]
    diff = {k: abs(metrics[k] - rec[k]) for k in ("test_macro_f1", "test_macro_f1_traintree")}
    bad = {k: (metrics[k], rec[k]) for k, d in diff.items() if not d <= TOL}
    if bad:
        raise SystemExit(f"[save_multiclass] identity FAILED, refit macro-F1 != recorded seed-0 value: {bad}")
    return diff


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    mc, classes, ds, metrics = fit()
    diff = check_identity(metrics)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mc.save_model(str(out))
    CLASSES.write_text(json.dumps({"classes": classes, "features": ds.feature_names,
                                   "test_macro_f1": metrics["test_macro_f1"],
                                   "test_macro_f1_traintree": metrics["test_macro_f1_traintree"],
                                   "recorded_in": str(RECORD)}, indent=2))
    print(f"[save_multiclass] identity OK (test macro-F1 {metrics['test_macro_f1']:.6f}, train_tree def "
          f"{metrics['test_macro_f1_traintree']:.6f}; |diff| {diff}); saved {out} and {CLASSES}")


if __name__ == "__main__":
    main()
