"""Per-window test predictions in one shared format, so any model's results can be bootstrapped and
compared (src/models/bootstrap_ci.py). One parquet per (experiment, model):

    data/processed/v2/runs/predictions/<experiment>_<model>.parquet

Columns: session, window_start_epoch, host_mac, label, attack_class, run_id (metadata kept for
block resampling only — never model inputs), y (1 = attack), score (higher = more attack-like; any
scale), pred (0/1 at the model's VALIDATION-chosen threshold).

Teammates (autoencoder, IF, LSTM, rule baseline): call save_predictions(meta, y, score, pred,
"A", "autoencoder") on your Experiment A test split, then run
    python -m src.models.bootstrap_ci --pred A_xgboost --pred A_autoencoder
"""
import json

import numpy as np
import pandas as pd

from .train_tree import DATASET, MODEL_DIR, V2

PRED_DIR = V2 / "runs" / "predictions"
META_KEEP = ("session", "window_start_epoch", "host_mac", "label", "attack_class", "run_id")


def save_predictions(meta, y, score, pred, experiment, model, out_dir=None):
    out_dir = out_dir or PRED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df = meta.loc[:, list(META_KEEP)].reset_index(drop=True).copy()
    df["y"] = np.asarray(y).astype(int)
    df["score"] = np.asarray(score, dtype=float)
    df["pred"] = np.asarray(pred).astype(int)
    if len(df) != len(meta):
        raise ValueError("predictions and metadata differ in length")
    path = out_dir / f"{experiment}_{model}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_predictions(name_or_path):
    from pathlib import Path
    p = Path(name_or_path)
    if not p.suffix:
        p = PRED_DIR / f"{name_or_path}.parquet"
    df = pd.read_parquet(p)
    missing = [c for c in META_KEEP + ("y", "score", "pred") if c not in df.columns]
    if missing:
        raise ValueError(f"{p}: missing columns {missing}")
    return p.stem, df


def export_experiment_a(name):
    """Experiment A seed-0 test predictions for a tree model, at its validation-chosen threshold
    (read from the Experiment A details file; nothing is re-fitted)."""
    from . import data as mdata
    from .calibrate_explain import load_model
    details = json.loads((V2 / "runs" / "expA" / f"{name}_details.json").read_text())
    thr = float(details["per_seed"][0]["threshold"])
    df, fl = mdata.load_dataset(DATASET, V2 / "feature_list.json")
    ds = mdata.Dataset(df, fl, scale=False)
    model = load_model(name)
    score = model.predict_proba(ds.X("test"))[:, 1]
    return save_predictions(ds.meta("test"), ds.y("test"), score, score >= thr, "A", name)
