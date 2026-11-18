"""Seed-0 multiclass test predictions of the saved XGBoost, for the dashboard's multiclass confusion matrix.

    python -m src.models.multiclass_predictions   # -> runs/expA/predictions/xgboost_multiclass_s0.parquet (+ _meta.json)

Experiment A trained a multiclass XGBoost (benign + the named attack classes) only to report macro-F1. save_multiclass
refit and saved the seed-0 model (models/expA/xgboost_multiclass_seed0.json + _classes.json, identity-checked). This
scores the Experiment-A TEST rows with that saved model and writes one `p_<class>` column per class, so the dashboard
can draw a multiclass confusion matrix. It is SEED 0 ONLY (the other seeds' multiclass models were never saved) and
DERIVED: not a logged metric.

Identity (asserted; a mismatch stops and writes nothing): the macro-F1 computed from these probabilities (the LSTM
definition, lstm.macro_f1, and train_tree's own) equals the value recorded in runs/expA/xgboost_s0.json, exactly as
save_multiclass checks its refit.

Columns: split (always "test"), session, host_mac, host_role, window_start_epoch, run_id, attack_class, label, y_bin,
p_<class> for the 9 classes, score (= 1 - p_benign, a derived attack score) and pred (1 when the most likely class is an
attack class), both derived from the probabilities.
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import lstm
from . import save_multiclass as smc
from . import sequence as seq
from . import train_autoencoder as ta
from . import train_tree as tt

OUT = tt.OUT_DIR / "predictions" / "xgboost_multiclass_s0.parquet"
META = tt.OUT_DIR / "predictions" / "xgboost_multiclass_s0_meta.json"


def score_test(env=None, model_path=smc.OUT, classes_path=smc.CLASSES):
    """(frame, metrics) for the Experiment-A test rows with the saved multiclass model; metrics are the macro-F1s."""
    from sklearn.metrics import f1_score
    from xgboost import XGBClassifier
    env = env or ta.Env()
    meta = json.loads(classes_path.read_text())
    ds = ta._load(env, scale=False)
    if list(ds.feature_names) != meta["features"]:
        raise SystemExit("[multiclass_predictions] the saved model's features differ from the dataset's")
    classes = meta["classes"]
    clf = XGBClassifier()
    clf.load_model(str(model_path))
    Xte, mte = ds.X("test"), ds.meta("test")
    P_enc = clf.predict_proba(Xte)                                   # columns in `classes` (LabelEncoder) order
    yte_m = tt.multiclass_targets(mte)
    P = np.zeros((len(Xte), len(seq.CLASS_ORDER)))
    for j, c in enumerate(classes):
        P[:, seq.CLASS_ORDER.index(c)] = P_enc[:, j]
    y_idx = np.array([seq.CLASS_ORDER.index(c) for c in yte_m])
    known = np.isin(yte_m, classes)
    pred_known = np.array(classes)[P_enc[known].argmax(axis=1)]
    metrics = {"test_macro_f1": lstm.macro_f1(y_idx, P),
               "test_macro_f1_traintree": float(f1_score(yte_m[known], pred_known, average="macro", zero_division=0))}
    f = mte[["session", "host_mac", "host_role", "window_start_epoch", "run_id", "attack_class", "label"]].reset_index(drop=True).copy()
    f.insert(0, "split", "test")
    f["y_bin"] = np.asarray(ds.y("test")).astype(int)
    for j, c in enumerate(classes):
        f[f"p_{c}"] = P_enc[:, j]
    f["score"] = 1.0 - f["p_benign"]
    f["pred"] = (np.array(classes)[P_enc.argmax(axis=1)] != "benign").astype(int)
    return f, metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    f, metrics = score_test()
    diff = smc.check_identity(metrics)                                # SystemExit on a mismatch: nothing is written
    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    out.with_name(META.name).write_text(json.dumps({"seed": 0, "seed_note": "seed 0 only: no other seed's multiclass model was saved",
                                "derived": True, "n_rows": int(len(f)), "test_macro_f1": metrics["test_macro_f1"],
                                "test_macro_f1_traintree": metrics["test_macro_f1_traintree"],
                                "recorded_macro_f1_diff": diff, "model": str(smc.OUT)}, indent=2))
    print(f"[multiclass_predictions] identity OK (macro-F1 {metrics['test_macro_f1']:.6f}, train_tree def "
          f"{metrics['test_macro_f1_traintree']:.6f}, |diff| {diff}); {len(f)} test rows -> {out}")


if __name__ == "__main__":
    main()
