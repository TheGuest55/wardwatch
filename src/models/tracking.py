"""MLflow-compatible run record. Every number in the report must trace to a logged run
(./CLAUDE.md), so each evaluation emits one record with a fixed schema: experiment (A-E), model
name + role, params, metrics, and tags (dataset, split, seed, direction for cross-dataset). MLflow
is optional — when unavailable or disabled, build_run_record still returns the dict so callers can
persist it as JSON and the schema is identical either way."""
import json
import time


def build_run_record(experiment, model, role, params=None, metrics=None, tags=None):
    """One run's record. experiment: 'A'..'E'; role: 'classifier'|'anomaly'|'rule'; model: e.g.
    'xgboost','autoencoder','lstm','random_forest','isolation_forest','rule_baseline'."""
    return {
        "experiment": experiment,
        "model": model,
        "role": role,
        "params": dict(params or {}),
        "metrics": {k: v for k, v in (metrics or {}).items()},
        "tags": dict(tags or {}),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def log_run(record, use_mlflow=True, experiment_name="wardwatch"):
    """Log a record to MLflow when available/enabled; always return the record. Never raises if
    MLflow is missing — the harness must run in the analysis image with or without it."""
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=f"{record['experiment']}:{record['model']}"):
                mlflow.set_tags({**record["tags"], "role": record["role"],
                                 "experiment": record["experiment"], "model": record["model"]})
                mlflow.log_params(record["params"])
                mlflow.log_metrics({k: v for k, v in record["metrics"].items()
                                    if isinstance(v, (int, float))})
        except Exception as exc:  # mlflow missing or backend unset: fall back to the returned dict
            record = {**record, "_mlflow_skipped": str(exc)}
    return record


def dumps(record):
    return json.dumps(record, indent=2, default=str)
