"""Single-window latency of the deployed anomaly detectors, timed from RAW input.

Run in its OWN process on an idle machine (torch.set_num_threads(1)):
  python -m src.experiments.ae_latency

score_one(raw_row) is the whole per-window work: take ONE unscaled feature row (values in
feature_names order), scale it with the model's train-only scaler (inside the timed call), score it
and apply the validation-chosen threshold. Rows are real test rows. Equivalence (asserted, 1e-6):
score_one on raw rows equals the offline scores computed on the scaled matrix, over an evenly spaced
subset of the test rows. One for_report record per model (experiment "A", tag latency) is written to
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
    """score_one on raw rows vs the offline batch scores of the same rows. Returns (max_abs_diff, n)."""
    idx = np.unique(np.linspace(0, len(X_raw) - 1, min(n_equiv, len(X_raw))).astype(int))
    offline = scorer.score_raw(X_raw[idx])
    got = np.array([scorer.score_one(X_raw[i])[0] for i in idx])
    mx = float(np.max(np.abs(got - offline)))
    if not mx <= tol:
        raise AssertionError(f"raw-path scores differ from the offline scores: max_abs_diff={mx} > {tol}")
    return mx, int(len(idx))


def run(name, model_dir, X_raw, out_json=None, n_calls=500, n_equiv=2000, for_report=False,
        split_path=ta.SPLIT, dataset_path=ta.DATASET):
    torch.set_num_threads(1)
    scorer, _ = load_scorer(name, model_dir)
    mx, n_cmp = check_equivalence(scorer, X_raw, n_equiv)
    print(f"[{name}] equivalence OK on {n_cmp} rows, max_abs_diff={mx:.3g}", flush=True)
    rows = [X_raw[i] for i in range(min(len(X_raw), 500))]
    lat = mm.single_window_latency_ms(scorer.score_one, rows, n_calls=n_calls)
    print(f"[{name}] latency from raw input: median {lat['median_ms']:.3f} ms, p95 {lat['p95_ms']:.3f} ms "
          f"({lat['n']} calls, {torch.get_num_threads()} thread)", flush=True)
    metrics = {"median_ms": lat["median_ms"], "p95_ms": lat["p95_ms"], "n": lat["n"], "warmup": lat["warmup"],
               "equivalence_max_abs_diff": mx, "equivalence_n_compared": n_cmp}
    tags = {"latency": True, "input": "raw (unscaled) row; scaler inside the timed call",
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
