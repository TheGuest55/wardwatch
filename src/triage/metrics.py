"""Triage metrics. Every definition here was fixed before any test result was seen.

Grouping accuracy: a DETECTED run (>= 1 alerted attack window) is correct iff all its alerted attack windows sit in
ONE incident and that incident holds no alerted attack window of any other run. Victim and boundary rows are ignored.
Also reported: incident purity (share of an incident's alerted attack windows that belong to its plurality run) and
incidents per detected run; "false incidents" hold no attack window at all.

Time to detect (per run): end of the first alerting attack window (window start + 5 s) minus the run's start_utc.
Undetected runs are listed, not averaged in.

B withheld share: for a withheld class X, the share of X's attack windows (and of runs with >= 1 X window) whose
cell is "novel" or either review cell; also reported: known, low priority, no alert. IF uses the fold's `pred`;
XGBoost bands use the A-selected calibration method (isotonic) fitted on THAT FOLD's validation rows (where X is
absent) plus confidence_cutoffs on those rows. Nothing is fitted on test (fold_bands refuses test rows).

False-novel rate: for each benign disruption event (reboot: windows overlapping [start, end]; reconnect: the window
holding the timestamp), the benign windows of the event's host (matched by host_ip) are classed by cell. Reported:
share in "novel" (the false-novel rate), share in either review cell, share in any alerting cell. Nurse commands
are reported on the same terms but separately, as non-disruptions.
"""
import numpy as np

from .. import common
from ..models import calibrate_explain as ce
from . import verdict as V

W = common.WINDOW_SECONDS
REVIEW_CELLS = ("review_anomaly_only", "review_classifier_only")
NOVEL_OR_REVIEW = ("novel",) + REVIEW_CELLS


def fold_bands(val, test, method="isotonic", target_precision=0.95, target_recall=0.95):
    """Calibrate raw XGBoost scores for a B fold. `val` and `test` are frames with `split` and raw `score` (and
    `y_bin` for val). Fit only on split == 'val'; refuses anything else as the fitting set. Returns
    (p_test, band_test, (lo, hi), calibrator)."""
    if not (val["split"] == "val").all():
        raise ValueError("fold calibration may only be fitted on validation rows")
    if (test["split"] != "test").any():
        raise ValueError("fold_bands scores test rows only")
    cal = ce.Calibrator(method).fit(val["score"].to_numpy(float), val["y_bin"].to_numpy(int))
    p_val = cal.predict(val["score"].to_numpy(float))
    lo, hi = ce.confidence_cutoffs(val["y_bin"].to_numpy(int), p_val, target_precision, target_recall)
    p = cal.predict(test["score"].to_numpy(float))
    return p, ce.band(p, lo, hi), (lo, hi), cal


def cell_table(labels, classes, cells):
    """{class: {cell: count}} for the six cells. labels: 'benign'/'attack'/'victim'/'boundary'; classes: attack_class
    (None for benign). Rows are grouped as benign, <attack class>, victim:<class>, boundary."""
    out = {}
    for lab, cls, c in zip(labels, classes, cells):
        key = "benign" if lab == "benign" else (str(cls) if lab == "attack" else
                                                f"victim:{cls}" if lab == "victim" else "boundary")
        row = out.setdefault(key, {k: 0 for k in V.CELLS})
        row[c] += 1
    return out


def prf(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp, fp, fn = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def grouping(alerts, members, runs):
    """alerts: {(host_mac, window_start): run_id or None} for alerted rows labelled attack/benign (None = benign
    alert). members: {incident_id: [(host_mac, window_start)]}. runs: iterable of all run ids (undetected included).
    Returns {"accuracy", "n_runs", "n_detected", "correct": [...], "incidents_per_run", "purity", "n_incidents",
    "n_false_incidents"}."""
    inc_of = {}
    for iid, mem in members.items():
        for m in mem:
            inc_of[m] = iid
    by_run, by_inc = {}, {}
    for k, run in alerts.items():
        if run is None or k not in inc_of:
            continue
        by_run.setdefault(run, set()).add(inc_of[k])
        by_inc.setdefault(inc_of[k], {}).setdefault(run, 0)
        by_inc[inc_of[k]][run] += 1
    correct, wrong = [], []
    for run, incs in sorted(by_run.items()):
        ok = len(incs) == 1 and len(by_inc[next(iter(incs))]) == 1
        (correct if ok else wrong).append(run)
    n_det = len(by_run)
    purities = [max(c.values()) / sum(c.values()) for c in by_inc.values()]
    return {"n_runs": len(set(runs)), "n_detected": n_det, "correct": correct, "incorrect": wrong,
            "accuracy": len(correct) / n_det if n_det else float("nan"),
            "incidents_per_run": float(np.mean([len(v) for v in by_run.values()])) if by_run else float("nan"),
            "purity": float(np.mean(purities)) if purities else float("nan"),
            "n_incidents": len(members), "n_false_incidents": len(members) - len(by_inc),
            "undetected": sorted(set(runs) - set(by_run))}


def time_to_detect(alert_windows, run_starts):
    """alert_windows: {run_id: [window_start of every alerted attack window]}; run_starts: {run_id: start epoch}.
    Returns {"per_run": {run: seconds or None}, "detected", "median_s", "mean_s", "undetected": [...]}."""
    per = {}
    for run, t0 in run_starts.items():
        ws = alert_windows.get(run)
        per[run] = float(min(ws) + W - t0) if ws else None
    got = [v for v in per.values() if v is not None]
    return {"per_run": per, "detected": len(got), "undetected": sorted(r for r, v in per.items() if v is None),
            "median_s": float(np.median(got)) if got else None, "mean_s": float(np.mean(got)) if got else None}


def event_windows(events, wins_by_ip, kind):
    """Benign windows of the event host matched by ip. events: dicts with ip and (start,end) epochs or ts epoch.
    wins_by_ip: {ip: [(window_start, row_index)]} of BENIGN windows. Returns the set of row indices."""
    hit = set()
    for e in events:
        for ws, i in wins_by_ip.get(e["ip"], ()):
            if kind == "reboot":
                ok = ws < e["end"] and ws + W > e["start"]
            else:
                ok = ws <= e["ts"] < ws + W
            if ok:
                hit.add(i)
    return hit


def event_rates(idx, cells):
    """Share of the given benign rows in each cell group."""
    n = len(idx)
    if not n:
        return {"n_windows": 0, "novel": None, "review": None, "alert": None, "low_priority": None}
    c = [cells[i] for i in sorted(idx)]
    return {"n_windows": n, "novel": sum(x == "novel" for x in c) / n,
            "review": sum(x in REVIEW_CELLS for x in c) / n, "alert": sum(V.is_alerting(x) for x in c) / n,
            "low_priority": sum(x == "low_priority" for x in c) / n}


def withheld_share(cells, run_ids):
    """cells/run_ids over the withheld class's attack windows. Returns window shares and run share reaching
    novel-or-review."""
    n = len(cells)
    if not n:
        return None
    c = np.array(cells, dtype=object)
    reach = np.isin(c, NOVEL_OR_REVIEW)
    runs = sorted(set(run_ids))
    r_reach = sum(bool(reach[np.array(run_ids, dtype=object) == r].any()) for r in runs)
    return {"n_windows": n, "n_runs": len(runs), "novel_or_review": float(reach.mean()),
            "novel": float((c == "novel").mean()), "review": float(np.isin(c, REVIEW_CELLS).mean()),
            "known": float((c == "known").mean()), "low_priority": float((c == "low_priority").mean()),
            "no_alert": float((c == "no_alert").mean()), "runs_reaching_novel_or_review": r_reach / len(runs)}
