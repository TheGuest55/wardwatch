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
from .loaders import MODELS, ROLE_OF, eligible, load_latency, not_applicable

CLASSES = ("arp_mitm", "bruteforce", "eavesdrop", "mqtt_flood", "pump_cmd", "recon_scan",
           "spoof_vitals", "syn_flood")
HEADLINE_COLS = {"f1": "F1", "pr_auc": "PR-AUC", "fpr_at_recall_90": "FPR @ 90% recall"}


NA_NOATK_MACRO = "not available (no attacker-excluded macro-F1 logged)"


def macro_series(frame, view):
    """The logged macro-F1 per record for a view, or None when no record logged one. All rows: `test_all_macro_f1`
    where a record has it, else the plain `test_macro_f1` (tree records and the LSTM's all-rows value), per record.
    Attacker excluded: only `test_noatk_macro_f1` (LSTM); trees log none, and the plain key is never reused for it."""
    names = ("test_all_macro_f1", "test_macro_f1") if view == "all" else ("test_noatk_macro_f1",)
    have = [frame[c] for c in names if c in frame.columns]
    if not have:
        return None
    out = have[0]
    for s in have[1:]:
        out = out.fillna(s)
    return out if out.notna().any() else None


def macro_cell(g, model, view):
    """(formatted cell, numeric twin, not-applicable reason) for one group of records of one model."""
    why = not_applicable(model, "macro_f1")
    if why:
        return "not applicable", np.nan, why
    ser = macro_series(g, view)
    if ser is None:
        return (NA_NOATK_MACRO if view == "noatk" else "not available"), np.nan, None
    m = mean_std(ser)
    return fmt(*m[:2]), m[0], None


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
    ser = macro_series(sub, view)
    out["macro_f1"] = mean_std(ser) if ser is not None else (np.nan, np.nan, 0)
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


# ------------------------------------------------------------------ tables for the comparison page
def a_by_config(df, view="all"):
    """Experiment A, every configuration of every model (ablations included): mean ± std over seeds."""
    sub = eligible(df, experiment="A")
    rows = []
    for (model, cfg), g in sub.groupby(["model", "config"]):
        r = {"model": model, "config": cfg, "seeds": int(g["seed"].nunique())}
        for key, name in {"f1": "F1", "pr_auc": "PR-AUC", "fpr_at_recall_90": "FPR @ 90% recall"}.items():
            col = f"test_{view}_{key}"
            r[name] = fmt(*mean_std(g[col])[:2]) if col in g else "not available"
            r["_" + name] = mean_std(g[col])[0] if col in g else np.nan
        r["macro-F1"], r["_macro-F1"], _ = macro_cell(g, model, view)
        rows.append(r)
    return pd.DataFrame(rows)


def b_per_class(df, view="all"):
    """Rows = models, columns = held-out class; cells mean ± std of test F1 over seeds."""
    sub = eligible(df, experiment="B")
    col = f"test_{view}_f1"
    if sub.empty or col not in sub:
        return pd.DataFrame()
    g = sub.groupby(["model", "held_out_class"])[col].agg(["mean", "std", "count"]).reset_index()
    g["cell"] = [fmt(r["mean"], r["std"]) for _, r in g.iterrows()]
    out = g.pivot(index="model", columns="held_out_class", values="cell")
    return out.reindex(columns=[c for c in CLASSES if c in out.columns]).fillna("not available")


def c_by_direction(df, view="all"):
    sub = eligible(df, experiment="C")
    col = f"tgt_test_{view}_f1"
    if sub.empty or col not in sub:
        return pd.DataFrame()
    g = sub.groupby(["model", "threshold_mode", "direction"], dropna=False)[col].agg(["mean", "std", "count"])
    g = g.reset_index()
    g["F1 (target test)"] = [fmt(r["mean"], r["std"]) for _, r in g.iterrows()]
    return g[["model", "threshold_mode", "direction", "count", "F1 (target test)"]]


def c_sensitivity(df, view="all"):
    """Frame-size sensitivity: Experiment C as-trained F1 (mean of both directions, over seeds) with and without the
    seven frame-size features, per model. The 'without' records are tagged sensitivity and are NOT headline-eligible;
    this is the only place they are shown. Each cell is a logged mean over records (nothing recomputed)."""
    if df is None or "sensitivity" not in df:
        return pd.DataFrame()
    col = f"tgt_test_{view}_f1"
    sens = df[(df["experiment"] == "C") & (df["sensitivity"] == "nosize") & df["for_report"]
              & df["seed"].notna() & (df["kind"] == "metrics")]
    if "threshold_mode" in sens:
        sens = sens[sens["threshold_mode"].fillna("as_trained") == "as_trained"]
    if sens.empty or col not in sens:
        return pd.DataFrame()
    rows = []
    for model, g in sens.groupby("model"):
        per = g.groupby("direction")[col].mean()
        if not {"tb2cic", "cic2tb"} <= set(per.index):
            continue
        no = float(per.mean())
        with_ = experiment_c(df, model, view)
        w = with_["score"] if with_ and with_["both_directions"] else np.nan
        rows.append({"model": model, "seeds": int(g["seed"].nunique()),
                     "C F1 with size features (headline)": "not available" if np.isnan(w) else f"{w:.3f}",
                     "C F1 without size features": f"{no:.3f}",
                     "difference": "not available" if np.isnan(w) else f"{no - w:+.3f}"})
    return pd.DataFrame(rows)


D_TASKS = (("2", "2-class (benign vs attack)"), ("6", "6 categories"), ("19", "19 classes"))


def d_table(df):
    """Experiment D (indicative): CIC's own feature CSVs and official split, XGBoost / RF, with and without the dropped
    features. NOT comparable to A-C and never in the headline or the selection. Read straight from the D records
    (experiment "D", or "D-drop-<feature>" for the leakage re-run); the published RF value is the logged mean minus the logged
    difference. All-rows only: CIC's CSVs carry no host column, so the attacker-excluded view is not applicable."""
    if df is None or df.empty:
        return pd.DataFrame()
    sub = df[df["experiment"].astype(str).str.startswith("D") & df["for_report"] & (df["kind"] == "metrics")]
    rows = []
    for _, r in sub.sort_values(["experiment", "model"]).iterrows():
        variant = "all 45 features" if r["experiment"] == "D" else "without " + r["experiment"][len("D-drop-"):].upper()
        for t, label in D_TASKS:
            k = f"{t}c_f1_macro_mean"
            if k not in r or pd.isna(r[k]):
                continue
            d = r.get(f"{t}c_f1_macro_minus_published_rf")
            row = {"model": r["model"], "variant": variant, "task": label,
                   "accuracy": fmt(r.get(f"{t}c_accuracy_mean"), r.get(f"{t}c_accuracy_std")),
                   "precision (macro)": fmt(r.get(f"{t}c_precision_macro_mean"), r.get(f"{t}c_precision_macro_std")),
                   "recall (macro)": fmt(r.get(f"{t}c_recall_macro_mean"), r.get(f"{t}c_recall_macro_std")),
                   "F1 (macro)": fmt(r[k], r.get(f"{t}c_f1_macro_std")),
                   "published RF F1": "not available" if d is None or pd.isna(d) else f"{r[k] - d:.3f}",
                   "F1 minus published RF": "not available" if d is None or pd.isna(d) else f"{d:+.3f}"}
            if t == "2" and f"2c_f1_attack_mean" in r and pd.notna(r["2c_f1_attack_mean"]):
                row["attack-class F1 (2-class)"] = fmt(r["2c_f1_attack_mean"], r.get("2c_f1_attack_std"))
            rows.append(row)
    return pd.DataFrame(rows)


def provenance_table(df):
    """Records behind a table: id, seed, commit, dirty flag, image digest, and whether it counted."""
    cols = ["model", "experiment", "config", "held_out_class", "direction", "seed", "run_name", "git_commit",
            "git_dirty", "image_digest", "eligible", "exclude_reason"]
    return df[[c for c in cols if c in df.columns]].sort_values(["model", "experiment", "run_name"])
