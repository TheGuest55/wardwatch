"""MLflow-compatible run record. Every number in the report must trace to a logged run
(./CLAUDE.md), so each evaluation emits one record with a fixed schema: experiment (A-E), model
name + role, params, metrics, tags, and an auto-filled provenance block.

Review N5: every record must carry seed, split file + its sha256, the dataset sha256, the git commit
and a dirty-tree flag (m-new-6), and the analysis image digest — filled in automatically — and must
always be persisted to JSON even when MLflow is absent, so the schema is identical either way.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def _sha256_file(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_and_dirty():
    """(commit, dirty) from git if available (works on the host, or in an image that ships git);
    else fall back to env vars the host can inject (WARDWATCH_GIT_COMMIT / WARDWATCH_GIT_DIRTY), else
    (None, None). dirty = are there uncommitted tracked changes (review m-new-6)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode()
        return commit, bool(status.strip())
    except Exception:
        commit = os.environ.get("WARDWATCH_GIT_COMMIT")
        dirty_env = os.environ.get("WARDWATCH_GIT_DIRTY")
        if dirty_env is None:
            dirty = None
        else:
            v = dirty_env.strip()
            # Accept an explicit flag (1/true/yes/0/false/no) OR raw `git status --porcelain`
            # output: non-empty => dirty, empty => clean. This lets the host pass
            # WARDWATCH_GIT_DIRTY="$(git status --porcelain)" unchanged in PowerShell and bash.
            dirty = bool(v) and v.lower() not in ("0", "false", "no")
        return commit, dirty


def run_provenance(seed=None, split_path=None, dataset_path=None):
    """The auto-filled provenance block required on every record (review N5 + m-new-6). Image digest
    comes from WARDWATCH_IMAGE_DIGEST (injected by the host, since the no-network analysis container
    can't query docker), falling back to the ANALYSIS_IMAGE name tag."""
    commit, dirty = _git_commit_and_dirty()
    return {
        "seed": seed,
        "split_file": str(split_path) if split_path else None,
        "split_sha256": _sha256_file(split_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "git_commit": commit,
        "git_dirty": dirty,
        "image_digest": os.environ.get("WARDWATCH_IMAGE_DIGEST") or os.environ.get("ANALYSIS_IMAGE"),
    }


# Every provenance field a report-grade run record must carry (review: no null provenance in a
# record that feeds the report). git_dirty=False and seed=0 are valid (not null) — the check is
# strictly `is None`.
PROVENANCE_REQUIRED = ("seed", "split_file", "split_sha256", "dataset_sha256",
                       "git_commit", "git_dirty", "image_digest")


def build_run_record(experiment, model, role, params=None, metrics=None, tags=None,
                     seed=None, split_path=None, dataset_path=None, for_report=False):
    """One run's record with an auto-filled provenance block. experiment: 'A'..'E'; role:
    'classifier'|'anomaly'|'rule'; model: e.g. 'xgboost','autoencoder','lstm','random_forest',
    'isolation_forest','rule_baseline'. Set for_report=True for a run whose numbers go in the
    report; log_run then refuses to persist it if any provenance field is null."""
    return {
        "experiment": experiment,
        "model": model,
        "role": role,
        "params": dict(params or {}),
        "metrics": {k: v for k, v in (metrics or {}).items()},
        "tags": dict(tags or {}),
        "for_report": bool(for_report),
        "provenance": run_provenance(seed=seed, split_path=split_path, dataset_path=dataset_path),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def assert_report_provenance(record):
    """Raise if a report-grade record has any null provenance field. The provenance must be filled in
    (seed, split file + sha256, dataset sha256, git commit + dirty flag, image digest) so every
    reported number traces to an exact code + data + image state."""
    prov = record.get("provenance", {})
    missing = [k for k in PROVENANCE_REQUIRED if prov.get(k) is None]
    if missing:
        raise ValueError(
            f"report run record '{record.get('experiment')}:{record.get('model')}' has null "
            f"provenance fields {missing}; pass them in (e.g. WARDWATCH_GIT_COMMIT / "
            f"WARDWATCH_GIT_DIRTY / WARDWATCH_IMAGE_DIGEST env, and seed/split/dataset paths).")


def write_json(record, path):
    """Persist a record to JSON (always done, review N5 — independent of MLflow)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps(record))
    return str(p)


def log_run(record, json_path=None, use_mlflow=True, experiment_name="wardwatch"):
    """Always persist the record to JSON when json_path is given (review N5), then log to MLflow when
    available/enabled. Never raises if MLflow is missing — the harness must run with or without it.
    A record marked for_report is refused (ValueError) if any provenance field is null, BEFORE any
    write, so an unreproducible number can never reach the report."""
    if record.get("for_report"):
        assert_report_provenance(record)
    if json_path:
        write_json(record, json_path)
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=f"{record['experiment']}:{record['model']}"):
                mlflow.set_tags({**record["tags"], **record["provenance"],
                                 "role": record["role"], "experiment": record["experiment"],
                                 "model": record["model"]})
                mlflow.log_params(record["params"])
                mlflow.log_metrics({k: v for k, v in record["metrics"].items()
                                    if isinstance(v, (int, float))})
        except Exception as exc:  # mlflow missing or backend unset: the JSON is the source of truth
            record = {**record, "_mlflow_skipped": str(exc)}
    return record


def dumps(record):
    return json.dumps(record, indent=2, default=str)
