"""The deployed detector pair as one scoring object: XGBoost (calibrated, banded, type-naming) + Isolation Forest.

score(X_raw) takes RAW feature rows (51 columns in the triage config's feature order) and returns, per row:
  raw      XGBoost seed-0 raw score (what Experiment A saved)
  p        calibrated attack probability (XGBoost seed-0 + validation-fitted calibrator)
  band     0 confident benign / 1 unsure / 2 confident attack (validation cut-offs)
  if_score Isolation Forest anomaly score (scaler inside the call); anomalous = if_score >= the validation threshold
  cell     one of the six triage cells; named = the multiclass XGBoost's most likely ATTACK class, computed only
           for the rows in the "known" cell (the type is named only for Known attack)
The same object serves the offline triage (a whole session at once) and the live replay (one tick at a time).

Loading mirrors tree_latency / ae_latency, whose identity and equivalence checks already cover single-row vs
batch scoring of these exact models.
"""
import json
from pathlib import Path

import numpy as np

from .. import common
from ..experiments import ae_latency as al
from ..models import calibrate_explain as ce
from ..models import data as mdata
from ..models import save_multiclass as smc
from ..triage import verdict as V


class Pair:
    def __init__(self, n_jobs=1, model_dir=ce.MODEL_DIR):
        self.xgb = ce.CalibratedDetector.load("xgboost")
        if hasattr(self.xgb.model, "set_params"):
            self.xgb.model.set_params(n_jobs=n_jobs)
        self.feats = list(self.xgb.features)
        self.lo, self.hi = float(self.xgb.config["lo"]), float(self.xgb.config["hi"])
        self.iforest, if_feats = al.load_scorer("isolation_forest", model_dir)
        if list(if_feats) != self.feats:
            raise SystemExit("XGBoost and Isolation Forest feature lists differ")
        from xgboost import XGBClassifier
        self.mc = XGBClassifier()
        self.mc.load_model(str(Path(model_dir) / smc.OUT.name))
        self.mc.set_params(n_jobs=n_jobs)
        meta = json.loads((Path(model_dir) / smc.CLASSES.name).read_text())
        if meta["features"] != self.feats:
            raise SystemExit("multiclass model features differ from the triage config")
        self.classes = meta["classes"]
        self.attack_cols = [i for i, c in enumerate(self.classes) if c != "benign"]

    def matrix(self, rows):
        """Rows (dicts from the window builder) -> raw feature matrix in model order; NaN/inf -> 0 as in training."""
        X = np.array([[r[f] for f in self.feats] for r in rows], dtype=float)
        X[~np.isfinite(X)] = 0.0
        return X

    def score(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        raw = self.xgb.model.predict_proba(X)[:, 1]
        p = self.xgb.calibrator.predict(raw)
        band = ce.band(p, self.lo, self.hi)
        s_if = self.iforest.score_raw(X)
        anomalous = s_if >= self.iforest.threshold
        c = V.cells(anomalous, band)
        named = np.full(len(X), None, dtype=object)
        known = np.flatnonzero(c == "known")
        if len(known):
            P = self.mc.predict_proba(X[known])
            idx = np.array(self.attack_cols)[np.argmax(P[:, self.attack_cols], axis=1)]
            named[known] = [self.classes[i] for i in idx]
        return {"raw": raw, "p": p, "band": band, "if_score": s_if, "anomalous": anomalous, "cell": c, "named": named}


def feature_matrix(df, feats):
    return mdata._matrix(df, feats)
