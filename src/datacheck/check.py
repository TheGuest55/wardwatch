"""Data-quality checks + leakage probe over data/processed/testbed_windows.parquet.

    python -m src.datacheck.check [--in-dir data/processed] [--out docs/data-report.md]

Every number here is computed from the dataset and the captured sessions, and copied into the
report. Checks (./CLAUDE.md Task 5): rows per class x session, normal/attack ratio, boundary
windows dropped; every ground-truth run -> >=1 attack window; no attack window outside a logged
interval; every class in train/val/test; per-session benign-event counts (reboots + reconnects)
and legit nurse pump-command count; hosts/window and missing/inf per feature; and a leakage probe
(metadata-only vs features-only classifier). "Suspiciously good results are bugs until proven
otherwise": if identifiers/time dominate, the report flags it instead of celebrating.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .. import common
from ..labels import label as labelling


# ----------------------------- integrity -----------------------------

def manifest_verify(session):
    """Re-verify a session's manifest.json SHA-256 against the files on disk (review M10). Returns
    {ok, n_files, mismatches, missing, unlisted} — empty lists mean the session verifies."""
    sdir = common.SESSIONS_DIR / session
    man_path = sdir / "manifest.json"
    if not man_path.exists():
        return {"ok": False, "n_files": 0, "mismatches": [], "missing": ["manifest.json"],
                "unlisted": []}
    manifest = json.loads(man_path.read_text())
    listed = manifest.get("files", manifest) if isinstance(manifest, dict) else {}
    # Accept either {"files": {path: sha}} or {path: sha}.
    if "files" in manifest and isinstance(manifest["files"], dict):
        listed = manifest["files"]
    elif all(isinstance(v, str) for v in manifest.values()):
        listed = manifest
    mismatches, missing = [], []
    for rel, expected in sorted(listed.items()):
        # merge_session.py writes each entry as {"sha256": ..., "bytes": ...}; older/synthetic
        # manifests used a bare hex string. Accept both.
        exp_sha = expected.get("sha256") if isinstance(expected, dict) else expected
        fp = sdir / rel
        if not fp.exists():
            missing.append(rel)
            continue
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != exp_sha:
            mismatches.append(rel)
    return {"ok": not (mismatches or missing), "n_files": len(listed),
            "mismatches": mismatches, "missing": missing, "unlisted": []}


def attack_windows_within_intervals(df, session):
    """Review M10 (real check): every attack-labelled window must lie fully inside some logged
    attack interval (by time, not by run_id — the run_id check was circular). Returns the count of
    attack rows whose [w, w+WINDOW) is not contained in any ground-truth interval for the session."""
    sdir = common.SESSIONS_DIR / session
    intervals, _, _ = labelling.load_ground_truth(sdir)
    # load_ground_truth already converts each interval's start/end to epoch seconds.
    spans = [(iv["start"], iv["end"]) for iv in intervals if "start" in iv and "end" in iv]
    W = common.WINDOW_SECONDS
    atk = df[(df["session"] == session) & (df["label"] == "attack")]
    outside = 0
    for w in atk["window_start_epoch"]:
        if not any(s <= w and (w + W) <= e for s, e in spans):
            outside += 1
    return outside


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


def per_session_parquet(session, in_dir=None):
    base = Path(in_dir) if in_dir is not None else common.PROCESSED_DIR
    p = base / "sessions" / f"{session}.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ----------------------------- leakage probe -----------------------------

PROXY_AUC = 0.95       # single-feature ROC-AUC vs attacker-host at/above which a feature is a proxy
ETA2_FLAG = 0.99       # per-session eta^2 at/above which a feature is a session fingerprint (leakage)


def _eta_squared(x, g):
    """Fraction of x's variance explained by grouping g (SS_between / SS_total), in [0, 1]. 1 means
    x is constant within each group (an identity fingerprint); 0 means group tells you nothing."""
    x = np.asarray(x, dtype=float)
    ss_tot = float(((x - x.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return 0.0
    ss_between = 0.0
    for lvl in np.unique(g):
        xi = x[g == lvl]
        ss_between += len(xi) * (xi.mean() - x.mean()) ** 2
    return float(ss_between / ss_tot)


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


def _matrix(df, cols):
    return (df[cols].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0))


def _split_masks(df):
    """Boolean train/test masks (index-aligned to df). Uses the time-based split column when both
    train and test are present; otherwise a deterministic 70/30 fallback (flagged by caller)."""
    splits = set(df["split"].unique())
    if {"train", "test"} <= splits:
        return (df["split"] == "train"), (df["split"] == "test"), False
    r = pd.Series(np.random.RandomState(0).rand(len(df)), index=df.index)
    return (r < 0.7), (r >= 0.7), True


def _split_xy(df, cols):
    """Return (X_train, y_train, X_test, y_test, fallback) for a quick classifier."""
    X, y = _matrix(df, cols), _binary_target(df)
    tr, te, fallback = _split_masks(df)
    return X[tr], y[tr], X[te], y[te], fallback


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

    # Features-only: the actual model inputs. Also derive per-class recall from the SAME fitted
    # classifier's predictions on the test split, so a high binary F1 can't hide a weak class.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score

    feats = [c for c in feature_list["features"] if c in df.columns]
    tr, te, _ = _split_masks(df)
    X, y = _matrix(df, feats), _binary_target(df)
    out["features_only"] = None
    out["per_class_recall"] = None
    if y[tr].nunique() == 2 and int(te.sum()) > 0:
        clf = RandomForestClassifier(n_estimators=120, random_state=0, n_jobs=1,
                                     class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred = pd.Series(clf.predict(X[te]), index=df.index[te])
        imp = sorted(zip(feats, clf.feature_importances_), key=lambda t: -t[1])[:15]
        out["features_only"] = {
            "f1": round(float(f1_score(y[te], pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y[te], pred)), 4),
            "top15": [[n, round(float(v), 4)] for n, v in imp],
        }
        # Identifier-proxy check (review N6): ANY feature (not just the top-15) whose single-feature
        # ROC-AUC against the attacker-host target on train is >= PROXY_AUC is acting as an
        # identity/lifetime proxy for the attacker rather than generalisable attack signal. The 0.95
        # threshold is deliberately sensitive; the features-only-no-attacker F1 is the companion
        # guard that detection doesn't merely exploit the attacker row.
        from sklearn.metrics import roc_auc_score
        atk_host = (df["host_role"] == "attacker").astype(int)
        proxies = []
        if atk_host[tr].nunique() == 2:
            yv = atk_host[tr].to_numpy()
            for name in feats:
                col = X.loc[tr, name].to_numpy(dtype=float)
                try:
                    auc = roc_auc_score(yv, col)
                except ValueError:
                    continue
                auc = max(auc, 1.0 - auc)   # direction-agnostic
                if auc >= PROXY_AUC:
                    proxies.append([name, round(float(auc), 4)])
            proxies.sort(key=lambda t: -t[1])
        out["attacker_host_proxies"] = proxies

        # Per-session and per-host eta^2 (review N6): the fraction of a feature's variance explained
        # by session identity or host identity. A value near 1 means the feature is ~constant within
        # a session (a session fingerprint, which a session/time split would leak) or within a host
        # (a host identity). Reported for all; session-eta^2 >= ETA2_FLAG is treated as leakage.
        if "session" in df.columns and "host_mac" in df.columns:
            g_sess = df.loc[tr, "session"].to_numpy()
            g_host = df.loc[tr, "host_mac"].to_numpy()
            eta_sess, eta_host = {}, {}
            for name in feats:
                col = X.loc[tr, name].to_numpy(dtype=float)
                eta_sess[name] = round(_eta_squared(col, g_sess), 4)
                eta_host[name] = round(_eta_squared(col, g_host), 4)
            out["eta2_session_top"] = sorted(eta_sess.items(), key=lambda t: -t[1])[:8]
            out["eta2_host_top"] = sorted(eta_host.items(), key=lambda t: -t[1])[:8]
            out["eta2_session_flags"] = sorted(
                ([n, v] for n, v in eta_sess.items() if v >= ETA2_FLAG), key=lambda t: -t[1])

        te_df = df[te]
        atk_classes = sorted(te_df.loc[te_df["label"] == "attack", "attack_class"].dropna().unique())
        recall, support = {}, {}
        for c in atk_classes:
            idx = te_df.index[(te_df["label"] == "attack") & (te_df["attack_class"] == c)]
            support[c] = int(len(idx))
            if len(idx):
                recall[c] = round(float((pred.loc[idx] == 1).mean()), 4)
        benign_idx = te_df.index[te_df["label"] == "benign"]
        out["per_class_recall"] = {
            "recall": recall,
            "benign_specificity": (round(float((pred.loc[benign_idx] == 0).mean()), 4)
                                   if len(benign_idx) else None),
            "test_support": support,
        }

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

def run(out_path, in_dir=None):
    in_dir = Path(in_dir) if in_dir is not None else common.PROCESSED_DIR
    data_path = in_dir / "testbed_windows.parquet"
    fl_path = in_dir / "feature_list.json"
    if not data_path.exists() or not fl_path.exists():
        raise SystemExit(f"run `make dataset` first (missing {data_path} / {fl_path})")
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

    # Clean attack windows per class x session (benign excluded, boundary already dropped).
    atk = df[df["label"] == "attack"]
    if len(atk):
        cw = atk.pivot_table(index="attack_class", columns="session",
                             values="pkts_total", aggfunc="count", fill_value=0)
        cw["TOTAL"] = cw.sum(axis=1)
        sessions_atk = sorted(atk["session"].unique())
        zeros = [(c, s) for c in cw.index for s in sessions_atk
                 if int(((atk["attack_class"] == c) & (atk["session"] == s)).sum()) == 0]
        lines += ["## Clean attack windows per class x session", "",
                  "Attack windows only (benign excluded; boundary windows already dropped). A 0 "
                  "means that class produced no clean labelled window in that session.", "",
                  "```", cw.to_string(), "```", "",
                  f"- (class, session) with zero clean windows: {zeros if zeros else 'none'}", ""]

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
    all_runs, run_issue, outside_issue = set(), False, False
    for s in sessions:
        sdir = common.SESSIONS_DIR / s
        intervals, _, _ = labelling.load_ground_truth(sdir)
        gt_runs = {iv["run_id"] for iv in intervals}
        all_runs |= gt_runs
        got = set(df[(df["session"] == s) & (df["label"] == "attack")]["run_id"])
        no_window = gt_runs - got
        orphan = got - gt_runs
        # Review M10: a direct timestamp check (not the circular run_id test) — attack windows must
        # lie fully inside a logged interval.
        outside = attack_windows_within_intervals(df, s)
        if no_window or orphan:
            run_issue = True
        if outside:
            outside_issue = True
        if gt_runs:
            lines.append(f"- {s}: {len(gt_runs)} runs, "
                         f"{len(got & gt_runs)} with >=1 window"
                         + (f"; NO WINDOW: {sorted(no_window)}" if no_window else "")
                         + (f"; ORPHAN run_id: {sorted(orphan)}" if orphan else "")
                         + (f"; {outside} ATTACK WINDOWS OUTSIDE ANY INTERVAL" if outside
                            else "; all attack windows within intervals"))
    lines.append("")

    # manifest re-verification (review M10): the recorded SHA-256 manifest must still match disk.
    lines += ["## Manifest re-verification", "",
              "Each session's manifest.json SHA-256 is re-checked against the files on disk.", ""]
    manifest_issue = False
    for s in sessions:
        mv = manifest_verify(s)
        if not mv["ok"]:
            manifest_issue = True
        lines.append(f"- {s}: {mv['n_files']} files -> "
                     + ("OK" if mv["ok"]
                        else f"MISMATCH {mv['mismatches']} MISSING {mv['missing']}"))
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
        pssn = per_session_parquet(s, in_dir)
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

    fo = probe.get("features_only") or {}
    pcr = probe.get("per_class_recall") or {}
    if pcr:
        lines += ["", "### Features-only per-class recall (test split)", "",
                  f"Binary classifier recall on each attack class's test windows (benign shown as "
                  f"specificity). A low value is a hard class the single probe misses even at "
                  f"overall F1 {fo.get('f1')} — expected for stolen-credential classes.", ""]
        for c, r in sorted(pcr.get("recall", {}).items()):
            lines.append(f"- `{c}`: recall {r} (n={pcr.get('test_support', {}).get(c, 0)})")
        lines += [f"- benign specificity: {pcr.get('benign_specificity')}", ""]

    # Review N6: the feature/metadata separation is now an invariant asserted directly (features are
    # an explicit allow-list disjoint from metadata), not inferred from whether a metadata name
    # happened to land in the top-15. Raises loudly if the contract is ever violated.
    overlap_ok = True
    try:
        common.assert_feature_metadata_disjoint()
    except ValueError as exc:
        overlap_ok = False
        lines += [f"- FEATURE/METADATA OVERLAP: {exc}"]
    proxies = probe.get("attacker_host_proxies") or []
    eta_flags = probe.get("eta2_session_flags") or []
    lines += ["", f"- split used: {'RANDOM 70/30 fallback (no real train/test yet)' if probe['split_fallback_random'] else 'time-based train/test'}",
              f"- metadata-only F1: {fo and probe['metadata_only']}",
              f"- features & metadata disjoint (allow-list): {'yes' if overlap_ok else 'NO — INVESTIGATE'}",
              f"- features that alone identify the attacker host (single-feature AUC>={PROXY_AUC} "
              f"vs attacker-host): {proxies if proxies else 'none'}",
              f"- top per-session eta^2 (feature ~constant within a session = fingerprint): "
              f"{probe.get('eta2_session_top')}",
              f"- top per-host eta^2: {probe.get('eta2_host_top')}",
              f"- session-fingerprint features (eta^2>={ETA2_FLAG}): {eta_flags if eta_flags else 'none'}",
              ""]

    # verdict
    problems = []
    if not cov_ok:
        problems.append("a class is missing from a split")
    if run_issue:
        problems.append("a run has no window or an orphan run_id exists")
    if outside_issue:
        problems.append("an attack window falls outside every logged interval")
    if manifest_issue:
        problems.append("a session manifest no longer verifies")
    if be_issue:
        problems.append("a session has zero reconnects or zero nurse commands")
    if not overlap_ok:
        problems.append("features and metadata overlap")
    if proxies:
        problems.append(f"{len(proxies)} feature(s) identify the attacker host at AUC>={PROXY_AUC}")
    if eta_flags:
        problems.append("a feature is a per-session fingerprint (eta^2>=%.2f)" % ETA2_FLAG)
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
    ap.add_argument("--in-dir", default=str(common.PROCESSED_DIR),
                    help="directory holding testbed_windows.parquet, feature_list.json and the "
                         "per-session parquet cache to check (default data/processed)")
    args = ap.parse_args()
    run(args.out, args.in_dir)


if __name__ == "__main__":
    main()
