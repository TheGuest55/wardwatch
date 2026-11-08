"""Split-respecting dataset loader for the model harness.

Discipline (./CLAUDE.md leakage control): the feature matrix uses only the columns in
feature_list.json (identifiers/time/raw-topic metadata are never inputs); splits come from the
`split` column (time-based, never a random shuffle); and any scaler is fit on **train only** and
applied to val/test — fitting on val/test would leak. Cross-dataset runs (Experiment C) drop the
testbed-only features via include_testbed_only=False.
"""
import json

import numpy as np
import pandas as pd

from .. import common

SPLITS = ("train", "val", "test")


def load_dataset(data_path=None, feature_list_path=None):
    data_path = data_path or (common.PROCESSED_DIR / "testbed_windows.parquet")
    fl_path = feature_list_path or (common.PROCESSED_DIR / "feature_list.json")
    df = pd.read_parquet(data_path)
    feature_list = json.loads(fl_path.read_text())
    return df, feature_list


def feature_columns(feature_list, df=None, include_testbed_only=True):
    feats = list(feature_list["features"])
    if not include_testbed_only:
        tb = set(feature_list.get("testbed_only", []))
        feats = [c for c in feats if c not in tb]
    if df is not None:
        feats = [c for c in feats if c in df.columns]
    return feats


def binary_target(df):
    """1 = attack, 0 = benign (the common metric for every experiment)."""
    return (df["label"] == "attack").astype(int).to_numpy()


def _matrix(df, feats):
    return (df[feats].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float))


class Dataset:
    """Holds per-split feature matrices, binary targets, and row metadata. A StandardScaler, when
    requested, is fit on train only."""

    def __init__(self, df, feature_list, include_testbed_only=True, scale=True):
        self.feature_names = feature_columns(feature_list, df, include_testbed_only)
        self._mask = {s: (df["split"] == s).to_numpy() for s in SPLITS}
        X = _matrix(df, self.feature_names)
        y = binary_target(df)
        self._meta = {s: df.loc[self._mask[s]].reset_index(drop=True) for s in SPLITS}
        self.scaler = None
        if scale:
            from sklearn.preprocessing import StandardScaler
            tr = self._mask["train"]
            if tr.sum() == 0:
                raise ValueError("no train rows: cannot fit a scaler (fit train-only discipline)")
            self.scaler = StandardScaler().fit(X[tr])   # TRAIN ONLY — never val/test
            X = self.scaler.transform(X)
        self._X = {s: X[self._mask[s]] for s in SPLITS}
        self._y = {s: y[self._mask[s]] for s in SPLITS}

    def X(self, split):
        return self._X[split]

    def y(self, split):
        return self._y[split]

    def meta(self, split):
        return self._meta[split]

    def normal_only(self, split="train"):
        """Train matrix restricted to benign rows (for the autoencoder / Isolation Forest)."""
        yb = self._y[split] == 0
        return self._X[split][yb]
