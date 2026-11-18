"""Single-window latency of the deployed classifier (XGBoost), timed from RAW input.

Run in its OWN process on an idle machine, one CPU (docker --cpus 1; the model is set to n_jobs=1):
  python -m src.experiments.tree_latency                # XGBoost record -> runs/expA/latency_xgboost.json
  python -m src.experiments.tree_latency --pair         # deployed-pair p95 from the two latency records only

What is timed (the 50 ms scoring path): score_one(raw_row) takes ONE feature row (values in the triage
config's feature order; trees use unscaled features, so there is no scaler), runs predict_proba, the
validation-fitted calibrator, and the validation cut-offs (confident benign / unsure / confident attack).
The deployed object is calibrate_explain.CalibratedDetector.load(model), i.e. the Experiment A seed-0
model plus the calibrator and triage config that `python -m src.models.calibrate_explain` wrote.
TreeSHAP (CalibratedDetector.explain) runs AFTER an alert, outside the scoring path, so it is timed
separately and stored as shap_* metrics; it is not part of p95_ms.

Asserted before timing:
  * artefact identity: batch predict_proba on the test rows equals the seed-0 Experiment A prediction
    file (runs/expA/predictions/<model>_s0.parquet) row for row, so the timed model is the recorded one;
  * equivalence (ae_latency.check_equivalence): single-row calibrated probabilities match the batch ones
    within |d| <= 1e-6 * max(1, |p|), no flip at the confident-attack cut-off, and no band changes.
"""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import common
from ..models import calibrate_explain as ce
from ..models import data as mdata
from ..models import metrics as mm
from ..models import tracking
from ..models import train_tree as tt
from . import ae_latency as al

V2 = common.PROCESSED_DIR / "v2"
EXPA = V2 / "runs" / "expA"
KEYS = ["session", "host_mac", "window_start_epoch"]


class TreeScorer:
    """CalibratedDetector wrapped for ae_latency.check_equivalence: score = calibrated probability,
    threshold = the confident-attack cut-off `hi`."""

    def __init__(self, det):
        self.det = det
        self.threshold = float(det.config["hi"])
        self.lo = float(det.config["lo"])

    def score_raw(self, X):
        return np.asarray(self.det.probability(np.asarray(X, dtype=float)), dtype=float)

    def score_one(self, raw_row):
        p = float(self.score_raw(np.asarray(raw_row, dtype=float).reshape(1, -1))[0])
        return p, int(ce.band(p, self.lo, self.threshold))


def load_detector(name, n_jobs=1):
    det = ce.CalibratedDetector.load(name)
    if hasattr(det.model, "set_params") and "n_jobs" in det.model.get_params():
        det.model.set_params(n_jobs=n_jobs)
    return det


def check_identity(det, X, meta, pred_file, tol=al.EQUIV_TOL):
    """Batch raw scores of the loaded model vs the recorded seed-0 test predictions, row for row."""
    p = pd.read_parquet(pred_file)
    p = p[p["split"] == "test"].reset_index(drop=True)
    if len(p) != len(X) or not p[KEYS].reset_index(drop=True).equals(meta[KEYS].reset_index(drop=True)):
        raise AssertionError(f"{pred_file}: test rows differ from the dataset's test rows")
    s = det.model.predict_proba(X)[:, 1]
    d = np.abs(s - p["score"].to_numpy())
    if not (d <= tol * np.maximum(1.0, np.abs(s))).all():
        raise AssertionError(f"loaded model is not the recorded seed-0 model: max |d| = {d.max()}")
    return {"identity_max_abs_diff": float(d.max()), "identity_n": int(len(d))}


def check_bands(scorer, X, idx):
    batch = ce.band(scorer.score_raw(X[idx]), scorer.lo, scorer.threshold)
    single = np.array([scorer.score_one(X[i])[1] for i in idx])
    flips = int((batch != single).sum())
    if flips:
        raise AssertionError(f"single-row bands differ from batch bands on {flips} rows")
    return flips


def run(name, X, meta, out_json, n_calls=500, n_equiv=2000, pred_file=None, for_report=False,
        split_path=tt.SPLIT, dataset_path=tt.DATASET):
    det = load_detector(name)
    ident = check_identity(det, X, meta, pred_file or (EXPA / "predictions" / f"{name}_s0.parquet"))
    scorer = TreeScorer(det)
    eq = al.check_equivalence(scorer, X, n_equiv)
    idx = np.unique(np.linspace(0, len(X) - 1, min(n_equiv, len(X))).astype(int))
    band_flips = check_bands(scorer, X, idx)
    print(f"[{name}] identity OK ({ident['identity_n']} rows, max |d| {ident['identity_max_abs_diff']:.3g}); "
          f"equivalence OK on {eq['n']} rows, max_scaled_diff={eq['max_scaled_diff']:.3g}, "
          f"cut-off flips {eq['decision_flips']}, band flips {band_flips}", flush=True)
    rows = [X[i] for i in range(min(len(X), 500))]
    lat = mm.single_window_latency_ms(scorer.score_one, rows, n_calls=n_calls)
    shap = mm.single_window_latency_ms(lambda x: det.explain(x, 3), rows, n_calls=n_calls)
    print(f"[{name}] scoring + calibration from raw input: median {lat['median_ms']:.3f} ms, "
          f"p95 {lat['p95_ms']:.3f} ms ({lat['n']} calls) | SHAP explanation alone (post-alert): "
          f"median {shap['median_ms']:.3f} ms, p95 {shap['p95_ms']:.3f} ms", flush=True)
    metrics = {"median_ms": lat["median_ms"], "p95_ms": lat["p95_ms"], "n": lat["n"], "warmup": lat["warmup"],
               "shap_median_ms": shap["median_ms"], "shap_p95_ms": shap["p95_ms"], "shap_n": shap["n"],
               "equivalence_max_abs_diff": eq["max_abs_diff"], "equivalence_max_scaled_diff": eq["max_scaled_diff"],
               "equivalence_decision_flips": eq["decision_flips"], "equivalence_n_compared": eq["n"],
               "equivalence_band_flips": band_flips, **ident}
    import xgboost
    tags = {"latency": True,
            "input": "raw (unscaled) feature row; predict_proba + calibrator + cut-offs inside the timed call",
            "timed_path": "scoring + calibration + band (50 ms path); SHAP timed separately (post-alert)",
            "equivalence_rule": f"|d| <= {al.EQUIV_TOL} * max(1, |p|), 0 flips at hi, 0 band changes",
            "artefacts": {"model": str(ce.MODEL_DIR / f"{name}_binary_seed0.json"),
                          "calibrator": str(ce.MODEL_DIR / f"{name}_calibrator.joblib"),
                          "triage_config": str(ce.MODEL_DIR / f"{name}_triage_config.json")},
            "calibration": det.config.get("calibration"), "cutoffs": {"lo": scorer.lo, "hi": scorer.threshold},
            "cpu": {"processor": platform.processor(), "machine": platform.machine(),
                    "platform": platform.platform(), "cpu_count": os.cpu_count(),
                    "xgboost": xgboost.__version__, "model_n_jobs": 1}}
    rec = tracking.build_run_record("A", name, "classifier", params={"threads": 1, "n_calls": lat["n"]},
                                    metrics=metrics, tags=tags, seed=0, split_path=str(split_path),
                                    dataset_path=str(dataset_path), for_report=for_report)
    if out_json is not None:
        tracking.log_run(rec, json_path=str(out_json), use_mlflow=False)
    return rec


def pair(classifier_json, anomaly_json, out_json):
    """Deployed-pair p95 from the two latency records only (sum of p95s: an upper bound on the p95 of the
    sum when both run back to back). Writes a derived summary, not a run record."""
    recs = {}
    for role, p in (("classifier", classifier_json), ("anomaly", anomaly_json)):
        r = json.loads(Path(p).read_text())
        if not (r.get("for_report") and (r.get("tags") or {}).get("latency")):
            raise SystemExit(f"{p} is not a report-grade latency record")
        recs[role] = {"path": str(p), "model": r["model"], "p95_ms": r["metrics"]["p95_ms"],
                      "median_ms": r["metrics"]["median_ms"],
                      "git_commit": r["provenance"]["git_commit"], "utc": r.get("utc")}
    total = recs["classifier"]["p95_ms"] + recs["anomaly"]["p95_ms"]
    out = {"pair": f"{recs['classifier']['model']} + {recs['anomaly']['model']}", "pair_p95_sum_ms": total,
           "budget_ms": 50.0, "within_budget": total < 50.0, "sources": recs,
           "note": "derived from the two latency records; sum of p95s; SHAP is post-alert and excluded"}
    Path(out_json).write_text(json.dumps(out, indent=2))
    print(f"pair {out['pair']}: p95 {recs['classifier']['p95_ms']:.3f} + {recs['anomaly']['p95_ms']:.3f} = "
          f"{total:.3f} ms (budget 50 ms) -> {out_json}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest"])
    ap.add_argument("--n-calls", type=int, default=500)
    ap.add_argument("--pair", action="store_true", help="only combine the two existing latency records")
    ap.add_argument("--anomaly", default="isolation_forest", choices=["isolation_forest", "autoencoder"])
    args = ap.parse_args(argv)
    os.chdir(common.ROOT)
    if args.pair:
        pair(EXPA / f"latency_{args.model}.json", EXPA / f"latency_{args.anomaly}.json",
             EXPA / f"latency_pair_{args.model}_{args.anomaly}_summary.json")
        return 0
    df, fl = mdata.load_dataset(tt.DATASET, V2 / "feature_list.json")
    ds = mdata.Dataset(df, fl, scale=False)                       # trees: unscaled = raw rows
    cfg = json.loads((ce.MODEL_DIR / f"{args.model}_triage_config.json").read_text())
    if cfg["features"] != ds.feature_names:
        raise SystemExit("triage config feature order differs from the dataset's feature order")
    run(args.model, ds.X("test"), ds.meta("test"), EXPA / f"latency_{args.model}.json",
        n_calls=args.n_calls, for_report=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
