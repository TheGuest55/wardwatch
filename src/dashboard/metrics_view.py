"""Headline tables from run records. Every number here is a logged number (or a mean/std over logged
per-seed numbers): the dashboard never computes precision/recall/F1 itself.

Selection rule (CLAUDE.md): within each role, highest binary F1 averaged over Experiments B and C,
with C = the as-trained score averaged over BOTH directions. A winner is declared only when every
candidate has both inputs; otherwise `missing` lists what is absent and `winner` is None.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .loaders import MODELS, ROLE_OF, eligible, load_latency

CLASSES = ("arp_mitm", "bruteforce", "eavesdrop", "mqtt_flood", "pump_cmd", "recon_scan",
           "spoof_vitals", "syn_flood")
HEADLINE_COLS = {"f1": "F1", "pr_auc": "PR-AUC", "fpr_at_recall_90": "FPR @ 90% recall"}


def mean_std(values):
    v = pd.Series(values, dtype=float).dropna()
    if v.empty:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), int(len(v))


def fmt(mean, std=None, n=None, digits=3):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "not available"
    s = f"{mean:.{digits}f}"
    if std is not None and not np.isnan(std):
        s += f" ± {std:.{digits}f}"
    return s + (f" (n={n})" if n is not None else "")


def selected_a_config(df_model_a, root=None):
    """The configuration whose Experiment A numbers are the model's headline. LSTM: the logged
    selected_config.json (rule fixed in advance). Otherwise the only config, or the one with the best
    VALIDATION PR-AUC among those with is_present on. None when it cannot be decided."""
    configs = sorted(df_model_a["config"].unique())
    if len(configs) == 1:
        return configs[0]
    root = Path(root) if root is not None else config.data_root()
    sel = root / "runs" / "lstm" / "selected_config.json"
    if df_model_a["model"].iloc[0] == "lstm" and sel.exists():
        try:
            s = json.loads(sel.read_text())
            return f"k={s['k']} h={s['hidden']}"
        except Exception:
            pass
    if "best_val_pr_auc" in df_model_a:
        cand = df_model_a[~df_model_a["config"].str.contains("no is_present")]
        g = cand.groupby("config")["best_val_pr_auc"].mean().dropna()
        if len(g):
            return g.idxmax()
    return None


def experiment_a(df, model, view="all", root=None):
    sub = eligible(df, model=model, experiment="A")
    if sub.empty:
        return None
    cfg = selected_a_config(sub, root)
    if cfg is None:
        return {"config": None, "reason": "several configurations and no selection record"}
    sub = sub[sub["config"] == cfg]
    out = {"config": cfg, "n_seeds": int(sub["seed"].nunique()), "runs": sub}
    for key in HEADLINE_COLS:
        col = f"test_{view}_{key}"
        out[key] = mean_std(sub[col]) if col in sub else (np.nan, np.nan, 0)
    col = f"test_{view}_macro_f1"
    out["macro_f1"] = mean_std(sub[col]) if col in sub else (np.nan, np.nan, 0)
    return out


def experiment_b(df, model, view="all"):
    """Per held-out class mean F1 over seeds, then mean over classes."""
    sub = eligible(df, model=model, experiment="B")
    col = f"test_{view}_f1"
    if sub.empty or col not in sub or sub["held_out_class"].isna().all():
        return None
    per = sub.groupby("held_out_class")[col].agg(["mean", "std", "count"])
    return {"per_class": per, "mean_f1": float(per["mean"].mean()), "n_classes": int(len(per)),
            "missing_classes": [c for c in CLASSES if c not in per.index], "runs": sub}


def experiment_c(df, model, view="all", mode="as_trained"):
    """Cross-domain (target test) F1 per direction, then the mean of the two directions."""
    sub = eligible(df, model=model, experiment="C")
    if not sub.empty and "threshold_mode" in sub:
        sub = sub[sub["threshold_mode"].fillna("as_trained") == mode]
    col = f"tgt_test_{view}_f1"
    if sub.empty or col not in sub:
        return None
    per = sub.groupby("direction")[col].agg(["mean", "std", "count"])
    both = {"tb2cic", "cic2tb"} <= set(per.index)
    return {"per_direction": per, "score": float(per["mean"].mean()) if both else np.nan,
            "both_directions": both, "runs": sub}


def headline(df, view="all", root=None):
    """One row per model; each cell is a formatted string plus a numeric twin for sorting."""
    rows = []
    lat = load_latency(df)
    for m in MODELS:
        r = {"model": m, "role": ROLE_OF[m]}
        a = experiment_a(df, m, view, root) if df is not None else None
        if a is None or a.get("config") is None:
            for k in ("F1", "PR-AUC", "FPR @ 90% recall"):
                r[k] = "not available"
            r["config"] = a["reason"] if a else "no report-grade Experiment A records"
        else:
            r["config"] = f"{a['config']} · {a['n_seeds']} seeds"
            for key, name in HEADLINE_COLS.items():
                r[name] = fmt(*a[key][:2], digits=4 if key == "fpr_at_recall_90" else 3)
            for key, name in HEADLINE_COLS.items():
                r[f"_{name}"] = a[key][0]
        r["p95 latency (ms)"] = "not available"
        if lat.ok:
            ls = lat.data[lat.data["model"] == m]
            if not ls.empty:
                r["p95 latency (ms)"] = f"{ls['p95_ms'].iloc[0]:.2f}"
        b = experiment_b(df, m, view) if df is not None else None
        r["B mean F1"] = "not available" if b is None else f"{b['mean_f1']:.3f} ({b['n_classes']}/8 classes)"
        r["_B"] = np.nan if b is None else b["mean_f1"]
        c = experiment_c(df, m, view) if df is not None else None
        if c is None:
            r["C as-trained F1"] = "not available"
        elif not c["both_directions"]:
            r["C as-trained F1"] = "not available (one direction only)"
        else:
            r["C as-trained F1"] = f"{c['score']:.3f}"
        r["_C"] = np.nan if c is None else c["score"]
        rows.append(r)
    return pd.DataFrame(rows)


def selection(head):
    """Selection-rule inputs per role, and a winner only when every candidate has both inputs."""
    out = {}
    for role, cands in (("anomaly", ("autoencoder", "isolation_forest")),
                        ("classifier", ("xgboost", "random_forest", "lstm"))):
        sub = head[head["model"].isin(cands)]
        missing, scores = [], {}
        for _, r in sub.iterrows():
            for key, label in (("_B", "Experiment B"), ("_C", "Experiment C as-trained (both directions)")):
                if pd.isna(r[key]):
                    missing.append(f"{r['model']}: {label}")
            if not (pd.isna(r["_B"]) or pd.isna(r["_C"])):
                scores[r["model"]] = (r["_B"] + r["_C"]) / 2
        winner = max(scores, key=scores.get) if not missing and scores else None
        out[role] = {"scores": scores, "missing": missing, "winner": winner}
    return out
