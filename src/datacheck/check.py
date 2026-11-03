"""Data-quality checks + leakage probe over data/processed/testbed_windows.parquet.

    python -m src.datacheck.check [--out docs/data-report.md]

Every number here is computed from the dataset and the captured sessions, and copied into the
report. Checks (./CLAUDE.md Task 5): rows per class x session, normal/attack ratio, boundary
windows dropped; every ground-truth run -> >=1 attack window; no attack window outside a logged
interval; every class in train/val/test; per-session benign-event counts (reboots + reconnects)
and legit nurse pump-command count; hosts/window and missing/inf per feature; and a leakage probe
(metadata-only vs features-only classifier). "Suspiciously good results are bugs until proven
otherwise": if identifiers/time dominate, the report flags it instead of celebrating.
"""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from .. import common
from ..labels import label as labelling


# ----------------------------- gather -----------------------------

def benign_event_counts(session):
    """Return {reboot, reconnect, nurse_command} counts from a session's benign_events.jsonl."""
    path = common.SESSIONS_DIR / session / "benign_events.jsonl"
    counts = Counter()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                counts[json.loads(line).get("event", "?")] += 1
            except ValueError:
                counts["?"] += 1
    return {"reboot": counts.get("reboot", 0), "reconnect": counts.get("reconnect", 0),
            "nurse_command": counts.get("nurse_command", 0)}


def per_session_parquet(session):
    p = common.PROCESSED_DIR / "sessions" / f"{session}.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ----------------------------- leakage probe -----------------------------

def _binary_target(df):
    return (df["label"] == "attack").astype(int)


def _fit_score(X_tr, y_tr, X_te, y_te):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score
    if y_tr.nunique() < 2 or len(X_te) == 0:
        return None
    clf = RandomForestClassifier(n_estimators=120, random_state=0, n_jobs=1, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return {
        "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "clf": clf,
    }


def _split_xy(df, cols):
    """Return (X_train, y_train, X_test, y_test) using the time-based split column when both
    train and test are present; otherwise a deterministic 70/30 fallback (flagged by caller)."""
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = _binary_target(df)
    splits = set(df["split"].unique())
    if {"train", "test"} <= splits:
        tr, te = df["split"] == "train", df["split"] == "test"
        return X[tr], y[tr], X[te], y[te], False
    rng = np.random.RandomState(0)
    mask = rng.rand(len(df)) < 0.7
    return X[mask], y[mask], X[~mask], y[~mask], True


def leakage_probe(df, feature_list):
    """Compare a metadata/identifier-only classifier against a features-only one."""
    out = {}

    # Metadata-only: pure identifiers + time, one-hot encoded. A high score here is the leakage we
    # deliberately keep OUT of the model by treating these as metadata.
    meta_src = df[["host_ip", "host_role", "window_start_epoch"]].copy()
    meta_src["host_ip"] = meta_src["host_ip"].astype(str)
    meta_enc = pd.get_dummies(meta_src, columns=["host_ip", "host_role"], dummy_na=True)
    Xm = meta_enc
    ym = _binary_target(df)
    splits = set(df["split"].unique())
    if {"train", "test"} <= splits:
        tr, te = df["split"] == "train", df["split"] == "test"
        m = _fit_score(Xm[tr], ym[tr], Xm[te], ym[te])
        fallback = False
    else:
        rng = np.random.RandomState(0)
        mask = rng.rand(len(df)) < 0.7
        m = _fit_score(Xm[mask], ym[mask], Xm[~mask], ym[~mask])
        fallback = True
    out["metadata_only"] = {k: m[k] for k in ("f1", "accuracy")} if m else None
    out["split_fallback_random"] = fallback

    # Features-only: the actual model inputs.
    feats = [c for c in feature_list["features"] if c in df.columns]
    Xtr, ytr, Xte, yte, _ = _split_xy(df, feats)
    f = _fit_score(Xtr, ytr, Xte, yte)
    if f:
        imp = sorted(zip(feats, f["clf"].feature_importances_), key=lambda t: -t[1])[:15]
        out["features_only"] = {"f1": f["f1"], "accuracy": f["accuracy"],
                                "top15": [[n, round(float(v), 4)] for n, v in imp]}
    else:
        out["features_only"] = None

    # Features-only with the attacker host excluded (detection must not rely on the attacker's own
    # row being trivially separable).
    no_atk = df[df["host_role"] != "attacker"]
    if len(no_atk):
        Xtr, ytr, Xte, yte, _ = _split_xy(no_atk, feats)
        f2 = _fit_score(Xtr, ytr, Xte, yte)
        out["features_only_no_attacker"] = ({"f1": f2["f1"], "accuracy": f2["accuracy"]}
                                            if f2 else None)
    return out


# ----------------------------- report -----------------------------

def run(out_path):
    data_path = common.PROCESSED_DIR / "testbed_windows.parquet"
    fl_path = common.PROCESSED_DIR / "feature_list.json"
    if not data_path.exists() or not fl_path.exists():
        raise SystemExit("run `make dataset` first (missing testbed_windows.parquet / feature_list.json)")
    df = pd.read_parquet(data_path)
    feature_list = json.loads(fl_path.read_text())
    sessions = sorted(df["session"].unique())

    lines = ["# WardWatch testbed dataset — data report", "",
             f"- rows: **{len(df)}**  |  features: **{feature_list['n_features']}**  "
             f"|  sessions: {', '.join(sessions)}", ""]

    # class x session
    lines += ["## Rows per class x session", "",
              "Label counts per session (attack windows broken out by class below).", ""]
    piv = (df.assign(cls=df["attack_class"].fillna(df["label"]))
             .pivot_table(index="cls", columns="session", values="pkts_total",
                          aggfunc="count", fill_value=0))
    lines.append("```")
    lines.append(piv.to_string())
    lines.append("```")
    tot = len(df)
    n_atk = int((df["label"] == "attack").sum())
    lines += ["", f"- attack rows: {n_atk} ({n_atk / tot:.1%}); "
                  f"benign rows: {tot - n_atk} ({(tot - n_atk) / tot:.1%})", ""]

    # class coverage across splits
    lines += ["## Every class present in train / val / test", ""]
    classes = sorted(c for c in df["attack_class"].dropna().unique())
    cov_ok = True
    for c in classes:
        present = {s: int(((df["attack_class"] == c) & (df["split"] == s)).sum())
                   for s in ("train", "val", "test")}
        missing = [s for s, n in present.items() if n == 0]
        if missing:
            cov_ok = False
        flag = "OK" if not missing else f"MISSING in {missing}"
        lines.append(f"- `{c}`: {present}  -> {flag}")
    lines.append("")

    # ground-truth run coverage + no-window-outside-interval
    lines += ["## Ground-truth run coverage", "",
              "Every logged run should produce >=1 attack window; no attack window should carry a "
              "run_id that isn't in ground truth.", ""]
    all_runs, run_issue = set(), False
    for s in sessions:
        sdir = common.SESSIONS_DIR / s
        intervals, _, _ = labelling.load_ground_truth(sdir)
        gt_runs = {iv["run_id"] for iv in intervals}
        all_runs |= gt_runs
        got = set(df[(df["session"] == s) & (df["label"] == "attack")]["run_id"])
        no_window = gt_runs - got
        orphan = got - gt_runs
        if no_window or orphan:
            run_issue = True
        if gt_runs:
            lines.append(f"- {s}: {len(gt_runs)} runs, "
                         f"{len(got & gt_runs)} with >=1 window"
                         + (f"; NO WINDOW: {sorted(no_window)}" if no_window else "")
                         + (f"; ORPHAN run_id: {sorted(orphan)}" if orphan else ""))
    lines.append("")

    # benign-event + legit-command counts per session
    lines += ["## Benign events per session", "",
              "Reboots + nurse reconnects keep benign wildcard-SUBSCRIBE recurring; legit nurse "
              "commands keep pump-command traffic in benign data. A session with zero reconnects or "
              "zero commands is flagged.", ""]
    be_issue = False
    for s in sessions:
        be = benign_event_counts(s)
        flags = []
        if be["reconnect"] == 0:
            flags.append("ZERO reconnects")
        if be["nurse_command"] == 0:
            flags.append("ZERO nurse commands")
        if flags:
            be_issue = True
        lines.append(f"- {s}: {be}" + (f"  <- {', '.join(flags)}" if flags else ""))
    lines.append("")

    # hosts per window + data hygiene
    hpw = df.groupby(["session", "window_start_epoch"])["host_mac"].nunique()
    feats = [c for c in feature_list["features"] if c in df.columns]
    fx = df[feats].apply(pd.to_numeric, errors="coerce")
    n_inf = int(np.isinf(fx.to_numpy(dtype=float, na_value=np.nan)).sum())
    na_by_col = fx.isna().sum()
    na_cols = {c: int(n) for c, n in na_by_col.items() if n}
    lines += ["## Data hygiene", "",
              f"- hosts per window: min {int(hpw.min())}, median {int(hpw.median())}, "
              f"max {int(hpw.max())}",
              f"- infinite feature cells: {n_inf}",
              f"- columns with NaN: {na_cols if na_cols else 'none'}", ""]

    # boundary windows (from per-session parquets, which still carry is_boundary)
    boundary = {}
    for s in sessions:
        pssn = per_session_parquet(s)
        if pssn is not None:
            boundary[s] = int(pssn["is_boundary"].sum())
    lines += ["## Boundary windows dropped (per session)", "", f"- {boundary}", ""]

    # leakage probe
    lines += ["## Leakage probe", "",
              "Binary attack-vs-normal F1. `metadata_only` uses ONLY identifiers + time (host_ip, "
              "host_role, window_start) — these are excluded from the model; a high score here is "
              "the leakage we avoid by keeping them as metadata. `features_only` uses the real model "
              "inputs.", ""]
    probe = leakage_probe(df, feature_list)
    lines.append("```")
    lines.append(json.dumps(probe, indent=2))
    lines.append("```")
    leak_flag = False
    fo = probe.get("features_only") or {}
    for name, _imp in fo.get("top15", []):
        if name in feature_list.get("metadata", []):
            leak_flag = True
    lines += ["", f"- split used: {'RANDOM 70/30 fallback (no real train/test yet)' if probe['split_fallback_random'] else 'time-based train/test'}",
              f"- metadata-only F1: {fo and probe['metadata_only']}",
              f"- any metadata column among top-15 features: {'YES — INVESTIGATE' if leak_flag else 'no'}", ""]

    # verdict
    problems = []
    if not cov_ok:
        problems.append("a class is missing from a split")
    if run_issue:
        problems.append("a run has no window or an orphan run_id exists")
    if be_issue:
        problems.append("a session has zero reconnects or zero nurse commands")
    if leak_flag:
        problems.append("a metadata column leaked into the top features")
    lines = (["> **Verdict:** " + ("PASS" if not problems else "REVIEW — " + "; ".join(problems)),
              ""] + lines)

    out = common.ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} | verdict: {'PASS' if not problems else 'REVIEW: ' + '; '.join(problems)}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data-report.md")
    args = ap.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
