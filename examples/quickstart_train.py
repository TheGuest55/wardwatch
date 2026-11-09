"""Teammate quickstart: load the shared v2 dataset through the harness, train a TOY XGBoost on the
train split, evaluate it on the test split with the fixed binary metrics, and write one run record to
JSON (and MLflow if a backend is configured).

This is a smoke test that the harness + dataset are wired up on your machine — NOT a reportable
result (50 shallow trees, no tuning/calibration, single seed). Run it inside the analysis image:

    docker run --rm -v ${PWD}:/work -w /work wardwatch-analysis:0.2.0 python examples/quickstart_train.py
"""
import json
import os
from pathlib import Path

# mlflow imports GitPython; the analysis image has no git binary, so quiet its probe.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from src import common
from src.models import data as mdata
from src.models import metrics as mmetrics
from src.models import tracking

V2 = common.PROCESSED_DIR / "v2"
SPLIT = common.SESSIONS_DIR / "split_v2.json"


def main():
    df, feature_list = mdata.load_dataset()               # defaults to data/processed/v2
    ds = mdata.Dataset(df, feature_list, scale=False)     # trees don't need scaling
    Xtr, ytr = ds.X("train"), ds.y("train")
    Xte, yte = ds.X("test"), ds.y("test")
    print(f"loaded v2: {len(df)} rows, {len(ds.feature_names)} features | "
          f"train={len(ytr)} rows, test={len(yte)} rows")

    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.3,
                        n_jobs=1, eval_metric="logloss", random_state=0)
    clf.fit(Xtr, ytr)
    y_score = clf.predict_proba(Xte)[:, 1]

    metrics = mmetrics.binary_metrics(yte, y_score=y_score, threshold=0.5)
    print("toy XGBoost — test binary metrics:")
    print(json.dumps(metrics, indent=2))

    rec = tracking.build_run_record(
        experiment="quickstart", model="xgboost_toy", role="classifier",
        params={"n_estimators": 50, "max_depth": 4, "learning_rate": 0.3},
        metrics=metrics, tags={"dataset": "v2"},
        seed=0, split_path=str(SPLIT),
        dataset_path=str(V2 / "testbed_windows.parquet"))
    out = V2 / "runs" / "quickstart.json"
    tracking.log_run(rec, json_path=str(out), use_mlflow=True)
    prov = rec["provenance"]
    print(f"logged run -> {out}")
    print(f"  provenance: seed={prov['seed']} git={prov['git_commit']} "
          f"dirty={prov['git_dirty']} dataset_sha256={str(prov['dataset_sha256'])[:12]}...")


if __name__ == "__main__":
    main()
