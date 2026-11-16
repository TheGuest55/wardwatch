"""Read-only loaders. Nothing here raises on missing or malformed input: every loader returns a
`Loaded(data, reason)`, and `reason` says why a thing is "not available".

Record discovery is layout-agnostic (the models write to different places): any *.json under a runs
directory that has `experiment`, `model` and a `metrics` dict is a run record. Headline membership is
decided by the record's `for_report` flag, plus three explicit exclusions that the flag alone does not
cover (all listed on the page's provenance expander):
  * smoke runs  - `for_report=True` is set on them by the smoke harness, so they sit in a `smoke/`
                  directory or carry a `smoke_` run name; they are never results.
  * latency runs - a record tagged `latency` is one timing measurement, not a seed of an experiment.
  * records without a seed - the headline is mean +- std over seeds.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from . import config

MODELS = ("xgboost", "random_forest", "autoencoder", "isolation_forest", "lstm", "rule_baseline")
ROLE_OF = {"xgboost": "classifier", "random_forest": "classifier", "lstm": "classifier",
           "autoencoder": "anomaly", "isolation_forest": "anomaly", "rule_baseline": "rule"}
# params that define a configuration (as opposed to a seed or a measured value)
CONFIG_KEYS = ("k", "hidden", "use_presence")
VIEW_PREFIX = {"all": "all", "noatk": "noatk"}   # "all rows" / "attacker host excluded"


@dataclass
class Loaded:
    data: Any = None
    reason: Optional[str] = None

    @property
    def ok(self):
        return self.reason is None and self.data is not None


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:                               # missing, unreadable, malformed
        return exc


def load_json(path, what="file"):
    p = Path(path)
    if not p.exists():
        return Loaded(None, f"{what} not found ({p.name})")
    d = _read_json(p)
    if isinstance(d, Exception):
        return Loaded(None, f"{what} unreadable ({p.name}): {type(d).__name__}")
    return Loaded(d)


# ---------------------------------------------------------------------------------------- run records
def _config_label(rec):
    p = rec.get("params") or {}
    parts = []
    if p.get("k") is not None:
        parts.append(f"k={p['k']}")
    if p.get("hidden") is not None:
        parts.append(f"h={p['hidden']}")
    if p.get("use_presence") is False:
        parts.append("no is_present")
    return " ".join(parts) or "default"


def _threshold_mode(rec):
    rule = str((rec.get("params") or {}).get("threshold_rule", "")).lower()
    if "refit" in rule:
        return "refit"
    return "as_trained" if rule else None


def _exclusion(rec, rel_parts, run_name):
    tags = rec.get("tags") or {}
    if not rec.get("for_report"):
        return "not for_report"
    if "smoke" in rel_parts[:-1] or str(run_name).startswith("smoke"):
        return "smoke run"
    if tags.get("latency"):
        return "latency measurement"
    if (rec.get("provenance") or {}).get("seed") is None:
        return "no seed"
    return None


def load_records(root=None):
    """All run records under the configured run roots as one flat DataFrame (metrics expanded into
    columns). Columns: model, role, experiment, kind, config, held_out_class, direction, threshold_mode,
    seed, run_name, for_report, eligible, exclude_reason, path, git_commit, git_dirty, image_digest, utc."""
    rows, problems = [], []
    for base in config.run_roots(root):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            rec = _read_json(path)
            if isinstance(rec, Exception):
                problems.append(f"{path.name}: unreadable ({type(rec).__name__})")
                continue
            if not (isinstance(rec, dict) and {"experiment", "model"} <= rec.keys()
                    and isinstance(rec.get("metrics"), dict)):
                continue                                         # not a run record (summary, curves, ...)
            rel = path.relative_to(base).parts
            tags, prov, params = rec.get("tags") or {}, rec.get("provenance") or {}, rec.get("params") or {}
            run_name = tags.get("run_name") or path.stem
            why = _exclusion(rec, rel, run_name)
            row = {
                "model": rec["model"], "role": rec.get("role") or ROLE_OF.get(rec["model"]),
                "experiment": str(rec["experiment"]),
                "kind": "latency" if tags.get("latency") else "metrics",
                "config": _config_label(rec), "held_out_class": params.get("held_out_class"),
                "direction": params.get("direction"), "threshold_mode": _threshold_mode(rec),
                "seed": prov.get("seed"), "run_name": run_name, "for_report": bool(rec.get("for_report")),
                "eligible": why is None, "exclude_reason": why, "path": str(path),
                "git_commit": prov.get("git_commit"), "git_dirty": prov.get("git_dirty"),
                "image_digest": prov.get("image_digest"), "utc": rec.get("utc"),
                "threshold": params.get("threshold"),
            }
            row.update({k: v for k, v in rec["metrics"].items() if k not in row})
            rows.append(row)
    if not rows:
        return Loaded(None, "no run records found under " + ", ".join(str(r) for r in config.run_roots(root))
                      + (f" ({'; '.join(problems)})" if problems else ""))
    df = pd.DataFrame(rows)
    df.attrs["problems"] = problems
    return Loaded(df)


def eligible(df, **where):
    """Headline-eligible metric records only (for_report, not smoke, not latency, seeded)."""
    m = df["eligible"] & (df["kind"] == "metrics")
    for k, v in where.items():
        m &= df[k] == v
    return df[m]


def metric_columns(df, split="test", view="all"):
    pre = f"{split}_{VIEW_PREFIX[view]}_"
    return {c[len(pre):]: c for c in df.columns if c.startswith(pre)}


def model_status(df):
    """model -> why it has no report-grade records (or None when it has some)."""
    out = {}
    for m in MODELS:
        if df is None:
            out[m] = "no run records"
            continue
        sub = df[df["model"] == m]
        if sub.empty:
            out[m] = "no run records on disk"
        elif not sub["eligible"].any():
            out[m] = "records exist but none is report-grade: " + ", ".join(sorted(set(sub["exclude_reason"].dropna())))
        else:
            out[m] = None
    return out


# -------------------------------------------------------------------------------------- predictions
def _pred_candidates(rec, root):
    base = Path(rec["path"])
    names = [rec["run_name"], base.stem]
    runs = Path(root) / "runs" if root is not None else config.data_root() / "runs"
    out = []
    for n in names:
        out += [base.parent / "predictions" / f"{n}.parquet", runs / rec["model"] / "predictions" / f"{n}.parquet"]
    out.append(runs / "predictions" / f"{rec['experiment']}_{rec['model']}.parquet")   # tree layout
    return out


def load_predictions(rec, root=None, split="test"):
    """Normalised prediction frame for one run record (a row/dict from `load_records`):
    columns split, y, score, pred, attack_class, host_role (+ p_<class> when present).
    `pred` is the run's own saved prediction or score >= the run's recorded threshold."""
    for p in dict.fromkeys(_pred_candidates(dict(rec), root)):
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            return Loaded(None, f"predictions unreadable ({p.name}): {type(exc).__name__}")
        y = df["y_bin"] if "y_bin" in df else df["y"] if "y" in df else (df["label"] == "attack").astype(int) \
            if "label" in df else None
        if y is None or "score" not in df:
            return Loaded(None, f"predictions {p.name} lack a label or score column")
        out = pd.DataFrame({"split": df["split"] if "split" in df else "test", "y": y.astype(int),
                            "score": df["score"].astype(float)})
        thr = rec.get("threshold")
        if "pred" in df:
            out["pred"] = df["pred"].astype(int)
        elif thr is not None and pd.notna(thr):
            out["pred"] = (out["score"] >= float(thr)).astype(int)
        else:
            return Loaded(None, f"{p.name}: no saved prediction and no recorded threshold")
        for c in ("attack_class", "host_role", "run_id", "session", "host_mac", "window_start_epoch"):
            out[c] = df[c] if c in df else None
        for c in df.columns:
            if c.startswith("p_"):
                out[c] = df[c]
        out = out[out["split"] == split].reset_index(drop=True)
        if out.empty:
            return Loaded(None, f"{p.name} has no '{split}' rows")
        out.attrs["source"] = str(p)
        return Loaded(out)
    return Loaded(None, "no predictions file for this run")


def drop_attacker(pred):
    """The 'attacker host excluded' view of a prediction frame. Needs host_role."""
    if pred["host_role"].isna().all():
        return Loaded(None, "predictions carry no host_role, so the attacker host cannot be excluded")
    return Loaded(pred[pred["host_role"] != "attacker"].reset_index(drop=True))


# ------------------------------------------------------------------------------- everything else
def load_dataset_summary(root=None):
    """Rows per (split, class) from the processed corpus parquet (never data/raw)."""
    root = Path(root) if root is not None else config.data_root()
    p = root / "testbed_windows.parquet"
    if not p.exists():
        return Loaded(None, f"{p.name} not found under {root}")
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        return Loaded(None, f"{p.name} unreadable: {type(exc).__name__}")
    need = {"split", "label"}
    if not need <= set(df.columns):
        return Loaded(None, f"{p.name} lacks columns {sorted(need - set(df.columns))}")
    cls = df["attack_class"].where(df["attack_class"].notna(), df["label"]) if "attack_class" in df else df["label"]
    tab = pd.crosstab(cls.fillna("benign"), df["split"])
    tab.index.name = "class"
    return Loaded(tab[[c for c in ("train", "val", "test") if c in tab.columns]])


def load_lstm_curves(rec):
    p = Path(rec["path"]).parent / "curves" / (str(rec["run_name"]) + ".json")
    r = load_json(p, "training curves")
    if r.ok and isinstance(r.data, dict) and isinstance(r.data.get("history"), list):
        return Loaded(pd.DataFrame(r.data["history"]))
    return Loaded(None, r.reason or "curves file has no history")


def load_latency(df):
    """model/config -> median/p95 ms, from latency-kind records (for_report only)."""
    if df is None:
        return Loaded(None, "no run records")
    sub = df[(df["kind"] == "latency") & df["for_report"]]
    if sub.empty or "p95_ms" not in sub:
        return Loaded(None, "no latency records")
    return Loaded(sub[["model", "config", "median_ms", "p95_ms", "path"]].reset_index(drop=True))


def load_lstm_explanations(root=None):
    root = Path(root) if root is not None else config.data_root()
    r = load_json(root / "runs" / "lstm" / "explanations.json", "LSTM explanations")
    if r.ok and not isinstance(r.data.get("classes"), dict):
        return Loaded(None, "explanations.json has no 'classes'")
    return r


def load_calibration_details(model, root=None):
    root = Path(root) if root is not None else config.data_root()
    return load_json(root / "runs" / "calibration" / f"{model}_details.json", f"{model} calibration/SHAP details")


def load_ae_details(root=None):
    root = Path(root) if root is not None else config.data_root()
    r = load_json(root / "runs" / "expA" / "autoencoder_details.json", "autoencoder details")
    if r.ok and "per_feature_error_seed0" not in r.data:
        return Loaded(None, "autoencoder details have no per_feature_error_seed0")
    return r


def load_bootstrap(root=None):
    root = Path(root) if root is not None else config.data_root()
    d = root / "runs" / "bootstrap"
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    if not files:
        return Loaded(None, "no bootstrap CI files (runs/bootstrap/)")
    out = {}
    for f in files:
        r = load_json(f, "bootstrap file")
        if r.ok:
            out[f.stem] = r.data
    return Loaded(out) if out else Loaded(None, "bootstrap files unreadable")


INCIDENT_FIELDS = ("incident_id", "target_host", "source", "first_seen", "last_seen", "verdict",
                   "top_features", "pcap_slice", "pcap_sha256")


def load_incidents(root=None):
    """live/incidents.jsonl, one JSON object per line (format in docs/dashboard-notes.md)."""
    from . import config as c
    p = c.incidents_path(root)
    if not p.exists():
        return Loaded(None, "not available yet: live/incidents.jsonl has not been written")
    rows, bad = [], 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            if isinstance(d, dict) and "incident_id" in d:
                rows.append(d)
            else:
                bad += 1
        except Exception:
            bad += 1
    if not rows:
        return Loaded(None, f"incidents.jsonl has no valid incidents ({bad} bad lines)")
    df = pd.DataFrame(rows)
    df.attrs["bad_lines"] = bad
    return Loaded(df)
