"""Single-window latency of the deployed anomaly detectors, timed from RAW input.

Run in its OWN process on an idle machine (torch.set_num_threads(1)):
  python -m src.experiments.ae_latency

score_one(raw_row) is the whole per-window work: take ONE unscaled feature row (values in
feature_names order), scale it with the model's train-only scaler (inside the timed call), score it
and apply the validation-chosen threshold. Rows are real test rows. Equivalence (asserted): over an
evenly spaced subset of the test rows, score_one on raw rows matches the offline batch scores within
|d| <= 1e-6 * max(1, |score|), and no row changes its decision at the threshold. The tolerance scales
with the score because the AE computes in float32: one float32 step (ULP) at a reconstruction error of
128-256 is 2^-16 = 1.5e-5, and single-row vs batch arithmetic can differ by that much. Flood windows
score in the hundreds, so a fixed 1e-6 fails on rounding alone (17 Nov 2026: 1.53e-5 at score 180.27,
0 decision flips). One for_report record per model (experiment "A", tag latency) is written to
runs/expA/latency_<model>.json.
"""
import argparse
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from .. import common
from ..models import ae_explain as ex
from ..models import autoencoder as ae
from ..models import data as mdata
from ..models import metrics as mm
from ..models import tracking
from ..models import train_autoencoder as ta

V2 = common.PROCESSED_DIR / "v2"
EQUIV_TOL = 1e-6


class Scorer:
    """model + scaler + threshold. score_one takes a RAW row."""

    def __init__(self, name, model, scaler, threshold):
        self.name, self.model, self.scaler, self.threshold = name, model, scaler, float(threshold)

    def score_raw(self, X):
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        if self.name == "autoencoder":
            return ae.anomaly_scores(self.model, Xs)
        return ta.if_scores(self.model, Xs)

    def score_one(self, raw_row):
        s = float(self.score_raw(np.asarray(raw_row, dtype=float).reshape(1, -1))[0])
        return s, int(s >= self.threshold)


def load_scorer(name, model_dir):
    model_dir = Path(model_dir)
    if name == "autoencoder":
        a = ex.load_artefact(model_dir / "autoencoder_seed0.pt")
        return Scorer(name, a["model"], a["scaler"], a["threshold"]), a["feature_names"]
    import joblib
    b = joblib.load(model_dir / "isolation_forest_seed0.joblib")
    return Scorer(name, b["model"], b["scaler"], b["threshold"]), b["feature_names"]


def check_equivalence(scorer, X_raw, n_equiv=2000, tol=EQUIV_TOL):
    """score_one on raw rows vs the offline batch scores of the same rows.

    Passes when every |d| <= tol * max(1, |offline|) and no decision flips at the threshold.
    Returns {"max_abs_diff", "max_scaled_diff" (max of |d| / max(1, |offline|)), "n", "decision_flips"}."""
    idx = np.unique(np.linspace(0, len(X_raw) - 1, min(n_equiv, len(X_raw))).astype(int))
    offline = scorer.score_raw(X_raw[idx])
    got = np.array([scorer.score_one(X_raw[i])[0] for i in idx])
    d = np.abs(got - offline)
    scaled = d / np.maximum(1.0, np.abs(offline))
    flips = int(((got >= scorer.threshold) != (offline >= scorer.threshold)).sum())
    out = {"max_abs_diff": float(d.max()), "max_scaled_diff": float(scaled.max()), "n": int(len(idx)),
           "decision_flips": flips}
    if not out["max_scaled_diff"] <= tol:
        raise AssertionError(f"raw-path scores differ from the offline scores: max |d|/max(1,|s|)="
                             f"{out['max_scaled_diff']} > {tol} (max_abs_diff={out['max_abs_diff']})")
    if flips:
        raise AssertionError(f"raw-path decisions differ from the offline decisions on {flips} rows")
    return out


def run(name, model_dir, X_raw, out_json=None, n_calls=500, n_equiv=2000, for_report=False,
        split_path=ta.SPLIT, dataset_path=ta.DATASET):
    torch.set_num_threads(1)
    scorer, _ = load_scorer(name, model_dir)
    eq = check_equivalence(scorer, X_raw, n_equiv)
    print(f"[{name}] equivalence OK on {eq['n']} rows, max_abs_diff={eq['max_abs_diff']:.3g}, "
          f"max_scaled_diff={eq['max_scaled_diff']:.3g}, decision flips {eq['decision_flips']}", flush=True)
    rows = [X_raw[i] for i in range(min(len(X_raw), 500))]
    lat = mm.single_window_latency_ms(scorer.score_one, rows, n_calls=n_calls)
    print(f"[{name}] latency from raw input: median {lat['median_ms']:.3f} ms, p95 {lat['p95_ms']:.3f} ms "
          f"({lat['n']} calls, {torch.get_num_threads()} thread)", flush=True)
    metrics = {"median_ms": lat["median_ms"], "p95_ms": lat["p95_ms"], "n": lat["n"], "warmup": lat["warmup"],
               "equivalence_max_abs_diff": eq["max_abs_diff"], "equivalence_max_scaled_diff": eq["max_scaled_diff"],
               "equivalence_decision_flips": eq["decision_flips"], "equivalence_n_compared": eq["n"]}
    tags = {"latency": True, "input": "raw (unscaled) row; scaler inside the timed call",
            "equivalence_rule": f"|d| <= {EQUIV_TOL} * max(1, |score|) and 0 decision flips",
            "cpu": {"processor": platform.processor(), "machine": platform.machine(),
                    "platform": platform.platform(), "cpu_count": os.cpu_count(),
                    "torch": torch.__version__, "torch_threads": torch.get_num_threads()}}
    rec = tracking.build_run_record("A", name, "anomaly", params={"threads": 1, "n_calls": lat["n"],
                                    "artefact_dir": str(model_dir)}, metrics=metrics, tags=tags, seed=0,
                                    split_path=str(split_path), dataset_path=str(dataset_path),
                                    for_report=for_report)
    if out_json is not None:
        tracking.log_run(rec, json_path=str(out_json), use_mlflow=False)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calls", type=int, default=500)
    args = ap.parse_args(argv)
    os.chdir(common.ROOT)
    df, fl = mdata.load_dataset(ta.DATASET, V2 / "feature_list.json")
    ds = mdata.Dataset(df, fl, scale=False)            # RAW rows
    X = ds.X("test")
    total = 0.0
    for name in ("autoencoder", "isolation_forest"):
        rec = run(name, ta.MODEL_DIR, X, V2 / "runs" / "expA" / f"latency_{name}.json",
                  n_calls=args.n_calls, for_report=True)
        total += rec["metrics"]["p95_ms"]
    print(f"pair p95 sum (AE + IF, from raw input): {total:.3f} ms (the pair budget is 50 ms with the classifier)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
