"""Tables for the Triage / E page, shaped from the logged records only (runs/triage/*.json, runs/expE/E_*.json).
Nothing here computes a metric: every number is a value in a record; this module only arranges it into tables."""
from pathlib import Path

import pandas as pd

from . import config
from .loaders import Loaded, load_json

CELLS = ("known", "novel", "review_anomaly_only", "review_classifier_only", "low_priority", "no_alert")
SESSIONS = ("r05", "r07")


def load_triage_record(name, root=None):
    root = Path(root) if root is not None else config.data_root()
    return load_json(root / "runs" / "triage" / f"{name}.json", f"{name} record")


def load_e_records(root=None):
    """{session: Loaded} for runs/expE/E_<session>.json (r05, r07, and any other E_*.json present)."""
    root = Path(root) if root is not None else config.data_root()
    d = root / "runs" / "expE"
    names = sorted({p.stem[2:] for p in d.glob("E_*.json")} | set(SESSIONS)) if d.is_dir() else list(SESSIONS)
    return {s: load_json(d / f"E_{s}.json", f"E_{s} record") for s in names}


def provenance_line(rec):
    p = rec.get("provenance") or {}
    return (f"commit `{str(p.get('git_commit'))[:7]}` · dirty {p.get('git_dirty')} · image `{str(p.get('image_digest'))[:19]}` · "
            f"for_report {rec.get('for_report')} · {rec.get('utc')}")


# ------------------------------------------------------------------------------------ triage_A
def sessions_in(rec):
    return list(((rec.get("tags") or {}).get("per_session") or {}))


def cell_table(rec, session):
    t = ((rec.get("tags") or {}).get("per_session") or {}).get(session, {}).get("cell_table") or {}
    df = pd.DataFrame(t).T
    return df.reindex(columns=[c for c in CELLS if c in df.columns]) if not df.empty else df


def detection_table(rec, session):
    det = ((rec.get("tags") or {}).get("per_session") or {}).get(session, {}).get("detection") or {}
    rows = []
    for model, views in det.items():
        for view, d in views.items():
            rows.append({"detector": model, "rows": "all rows" if view == "all" else "attacker host excluded",
                         "precision": d.get("precision"), "recall": d.get("recall"), "F1": d.get("f1"),
                         "TP": d.get("tp"), "FP": d.get("fp"), "FN": d.get("fn")})
    return pd.DataFrame(rows)


def grouping_summary(rec, session):
    s = ((rec.get("tags") or {}).get("per_session") or {}).get(session, {})
    g = s.get("grouping") or {}
    return {"accuracy": g.get("accuracy"), "runs detected": g.get("n_detected"), "runs correct": len(g.get("correct") or []),
            "purity": g.get("purity"), "incidents": g.get("n_incidents"), "false incidents": g.get("n_false_incidents"),
            "incidents per detected run": g.get("incidents_per_run"), "undetected runs": g.get("undetected")}


def grouping_grid(rec):
    grid = (rec.get("tags") or {}).get("grouping_grid") or []
    return pd.DataFrame([{"key": g["key"], "gap (s)": g["gap_s"], "accuracy": g["accuracy"],
                          "correct": g["correct"], "detected": g["detected"]} for g in grid])


def validation_tied(rec):
    """True when every validation (key, gap) setting has the same logged accuracy (the selection had no signal)."""
    g = grouping_grid(rec)
    return bool(len(g) > 1 and g["accuracy"].nunique() == 1)


def ttd_table(rec, session):
    t = ((rec.get("tags") or {}).get("per_session") or {}).get(session, {}).get("ttd") or {}
    return pd.DataFrame([{"run": r, "time to detect (s)": v if v is not None else "undetected"}
                         for r, v in (t.get("per_run") or {}).items()])


def false_novel_table(rec, session):
    f = ((rec.get("tags") or {}).get("per_session") or {}).get(session, {}).get("false_novel") or {}
    return pd.DataFrame([{"event": k, "events": v.get("n_events"), "benign windows": v.get("n_windows"),
                          "in Possible novel": v.get("novel"), "in review": v.get("review"),
                          "any alerting verdict": v.get("alert")} for k, v in f.items()])


# ------------------------------------------------------------------------------------ triage_B_withheld
def withheld_table(rec):
    w = (rec.get("tags") or {}).get("withheld") or {}
    rows = []
    for cls, v in w.items():
        a = v.get("agg") or {}
        f = lambda k: f"{a[k]['mean']:.3f} ± {a[k]['std']:.3f}" if k in a else "not available"
        rows.append({"withheld class": cls, "windows": v.get("n_windows"), "runs": v.get("n_runs"),
                     "novel or review": f("novel_or_review"), "novel": f("novel"), "review": f("review"),
                     "known attack": f("known"), "low priority": f("low_priority"), "no alert": f("no_alert"),
                     "runs reaching novel/review": f("runs_reaching_novel_or_review")})
    return pd.DataFrame(rows)


def loud_known(rec):
    """Withheld classes whose windows are 'Known attack' at mean share >= 0.99 (the loud ones)."""
    w = (rec.get("tags") or {}).get("withheld") or {}
    return sorted(c for c, v in w.items() if ((v.get("agg") or {}).get("known") or {}).get("mean", 0) >= 0.99)


# ------------------------------------------------------------------------------------ grouping_diagnosis
def diagnosis_runs(rec, session):
    runs = ((rec.get("tags") or {}).get("per_session") or {}).get(session, {}).get("runs") or {}
    rows = []
    for run, x in runs.items():
        causes = {}
        for e in x.get("extra") or []:
            causes[e["cause"]] = causes.get(e["cause"], 0) + 1
        rows.append({"run": run, "class": x.get("class"), "alerted windows": x.get("n_alerted_windows"),
                     "incidents": x.get("n_incidents"), "main incident alerts": x.get("main_alerts"),
                     "extra by cause": ", ".join(f"{k} {v}" for k, v in causes.items()) or "-",
                     "mixes other runs": ", ".join(x.get("mixes_other_runs") or []) or "-"})
    return pd.DataFrame(rows)


def diagnosis_totals(rec):
    t = (rec.get("tags") or {}).get("totals") or {}
    return t.get("by_cause") or {}, t.get("by_transition") or {}


def false_alarms(rec, session):
    return ((rec.get("tags") or {}).get("per_session") or {}).get(session, {}).get("false_alarm") or {}


# ------------------------------------------------------------------------------------ E
def e_latency_table(rec):
    m = rec.get("metrics") or {}
    rows = [{"scope": "per 5 s tick (all hosts)", "median ms": m.get("tick_median_ms"), "p95 ms": m.get("tick_p95_ms"),
             "max ms": m.get("tick_max_ms")},
            {"scope": "per row (tick / rows)", "median ms": m.get("row_median_ms"), "p95 ms": m.get("row_p95_ms"),
             "max ms": m.get("row_max_ms")}]
    return pd.DataFrame(rows)


def e_summary(rec):
    m, t = rec.get("metrics") or {}, rec.get("tags") or {}
    return {"budget ms": m.get("budget_ms"), "ticks over budget": m.get("ticks_over_budget"), "ticks": m.get("n_ticks"),
            "max replay lag s": m.get("lag_s_max"), "p95 replay lag s": m.get("lag_s_p95"),
            "ingest per packet mean us": m.get("ingest_us_mean"), "SHAP p95 s (post-alert)": m.get("shap_p95_s"),
            "slice p95 s (post-alert)": m.get("slice_p95_s"), "wall s": m.get("wall_s"), "incidents": m.get("n_incidents")}


def e_detection(rec):
    m = rec.get("metrics") or {}
    return pd.DataFrame([{"detector": "pair (alerting verdicts)", "precision": m.get("pair_precision"),
                          "recall": m.get("pair_recall"), "F1": m.get("pair_f1")},
                         {"detector": "XGBoost confident attack alone", "precision": None, "recall": None, "F1": m.get("xgboost_f1")},
                         {"detector": "Isolation Forest alone", "precision": None, "recall": None, "F1": m.get("isolation_forest_f1")}])


def e_ttd(rec):
    t = rec.get("tags") or {}
    off, live = (t.get("ttd_offline") or {}).get("per_run") or {}, t.get("ttd_live") or {}
    return pd.DataFrame([{"run": r, "offline (window end) s": off.get(r), "live (incl. processing) s": live.get(r)}
                         for r in sorted(set(off) | set(live))])


def e_equivalence(rec):
    """Flags exactly as logged: late packets, vs-offline row/band/verdict flips, saved-A-s0 identity, incident equality."""
    e = (rec.get("tags") or {}).get("equivalence") or {}
    ro, ident, inc = e.get("rows_vs_offline") or {}, e.get("vs_saved_A_s0") or {}, e.get("incidents_vs_offline") or {}
    return {"late packets": e.get("late_packets"), "rows compared with offline": ro.get("n_rows"),
            "max relative score difference": ro.get("max_rel_score_diff"), "band flips": ro.get("band_flips"),
            "verdict flips": ro.get("verdict_flips"), "rows compared with saved A s0": ident.get("xgboost_rows"),
            "max |score difference| vs saved A s0": ident.get("xgboost_max_abs_diff"),
            "incidents equal to offline": inc.get("equal"), "incidents": inc.get("n_incidents")}
