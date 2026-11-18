"""Offline triage of the held-out sessions and the Experiment-B "withheld attack" analysis.

    python -m src.triage.offline [--for-report]

1. Choose the incident grouping (key x gap) on the VALIDATION sessions only (r04, r06): criterion = pooled
   grouping accuracy, ties to the simpler key and the smaller gap. Test is never used to choose anything.
2. Triage the test sessions (r05, r07): every window of the stored session parquet is scored with the deployed pair
   (XGBoost + calibrator + cut-offs, Isolation Forest) and turned into a verdict; before anything is reported the
   raw XGBoost score and IF decision are checked equal to the saved Experiment-A seed-0 predictions on the rows
   those files hold (victim/boundary rows are not in them; they are scored here too, as a live system would, and
   counted separately in the tables).
3. Incidents (with TreeSHAP reasons and a hashed pcap slice) go to data/processed/v2/triage/incidents_offline.jsonl.
4. Records: runs/triage/triage_A.json, triage_B_withheld.json (+ .md), for_report with provenance.

PCAP slices hold plaintext lab MQTT credentials: they are only written under data/processed/v2/triage/slices/
(gitignored); never commit or publish them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import common
from ..labels import label as labelling
from ..models import data as mdata
from ..models import metrics as mm
from ..models import tracking
from ..models import train_tree as tt
from ..realtime import stream
from ..realtime.pipeline import Pair
from . import engine, incidents as inc_mod, metrics as tm, pcap_slice
from . import verdict as V

V2 = common.PROCESSED_DIR / "v2"
TRI = V2 / "triage"
RUNS = V2 / "runs" / "triage"
EXPA, EXPB = V2 / "runs" / "expA", V2 / "runs" / "expB"
VAL, TEST = ("r04", "r06"), ("r05", "r07")
KEYS = ("source", "source+target")
GAPS = (10, 30, 60, 120)
EQ_TOL = 1e-6


# ------------------------------------------------------------------------------------ data
def load_peers(session, peers_dir=TRI):
    p = pd.read_parquet(Path(peers_dir) / f"peers_{session}.parquet")
    return {(m, int(w)): json.loads(js) for m, w, js in zip(p["host_mac"], p["window_start_epoch"], p["peer_pkts"])}


class Session:
    """One stored session scored with the deployed pair."""

    def __init__(self, name, pair, parquet_dir=V2 / "sessions", peers_dir=TRI, df=None, peers=None):
        self.name = name
        df = pd.read_parquet(Path(parquet_dir) / f"{name}.parquet") if df is None else df
        self.df = df.sort_values(["window_start_epoch", "host_mac"]).reset_index(drop=True)
        self.X = mdata._matrix(self.df, pair.feats)
        self.res = pair.score(self.X)
        self.peers = load_peers(name, peers_dir) if peers is None else peers
        self.intervals, _, _ = labelling.load_ground_truth(common.SESSIONS_DIR / name)

    @classmethod
    def from_parts(cls, name, df, X, res, peers):
        """A scored session assembled from parts (the live replay's rows and scores), for evaluate_session."""
        o = cls.__new__(cls)
        o.name, o.df, o.X, o.res, o.peers = name, df, X, res, peers
        o.intervals, _, _ = labelling.load_ground_truth(common.SESSIONS_DIR / name)
        return o

    def identity_check(self, split_for, pred_dir=EXPA / "predictions"):
        """Raw XGBoost score and IF decision vs the saved A seed-0 predictions (rows they hold). Raises on any
        difference; returns the max relative score difference and the number of rows compared."""
        out = {}
        for model, col in (("xgboost", "raw"), ("isolation_forest", "anomalous")):
            p = pd.read_parquet(Path(pred_dir) / f"{model}_s0.parquet")
            p = p[(p["session"] == self.name) & (p["split"] == split_for)]
            m = p.merge(self.df[["host_mac", "window_start_epoch"]].assign(i=np.arange(len(self.df))),
                        on=["host_mac", "window_start_epoch"], how="left")
            if m["i"].isna().any():
                raise AssertionError(f"{self.name}: {int(m['i'].isna().sum())} saved {model} rows missing from the session")
            i = m["i"].to_numpy(int)
            if model == "xgboost":
                d = np.abs(self.res["raw"][i] - m["score"].to_numpy(float))
                bad = ~(d <= EQ_TOL * np.maximum(1.0, np.abs(m["score"].to_numpy(float))))
                out["xgboost_max_abs_diff"] = float(d.max())
            else:
                bad = self.res["anomalous"][i].astype(int) != m["pred"].to_numpy(int)
            if bad.any():
                raise AssertionError(f"{self.name}: {int(bad.sum())} {model} rows differ from the saved A s0 predictions")
            out[f"{model}_rows"] = int(len(m))
        return out


def explain_fn_for(pair):
    def fn(feat):
        return [{"feature": r["feature"], "text": r["text"], "delta": r["shap"]} for r in pair.xgb.explain(feat, 3)]
    return fn


def make_post(pair, session, slice_dir, rel_root):
    """Inline post-alert step: TreeSHAP reasons and the hashed slice (path relative to the data root)."""
    explain = explain_fn_for(pair)
    pcaps = stream.pcap_paths(session)
    for p in pcaps:
        pcap_slice.get_index(p)

    def slice_fn(first, last_end, macs, iid):
        out = Path(slice_dir) / f"{iid}.pcap"
        s = pcap_slice.write_slice(pcaps, macs, first, last_end, out)
        s["path"] = str(Path(rel_root) / f"{iid}.pcap")
        return s

    def post(rec, inc):
        inc_mod.complete_record(rec, inc, explain, slice_fn)
    return post


def run_triager(sess, key, gap, ip_roles, post=None):
    """Feed one scored session through the Triager window by window; returns the finished Triager."""
    tr = engine.Triager(sess.name, ip_roles, key, gap, post=post)
    df, res = sess.df, sess.res
    wins = df["window_start_epoch"].to_numpy()
    cuts = np.flatnonzero(np.diff(wins)) + 1
    for idx in np.split(np.arange(len(df)), cuts):
        w = int(wins[idx[0]])
        sub = {k: v[idx] for k, v in res.items()}
        tr.window(w, df["host_mac"].to_numpy()[idx].tolist(), df["host_ip"].to_numpy()[idx].tolist(),
                  sess.X[idx], sub, sess.peers)
    tr.finish()
    return tr


# ------------------------------------------------------------------------------------ evaluation
def _attack_alerts(sess, tr):
    """{(mac, win): run_id | None} for alerted rows labelled attack (run id) or benign (None)."""
    out = {}
    lab = sess.df.set_index(["host_mac", "window_start_epoch"])
    for mac, win, c, *_ in tr.cell_log:
        if not V.is_alerting(c):
            continue
        row = lab.loc[(mac, win)]
        if row["label"] == "attack":
            out[(mac, win)] = row["run_id"]
        elif row["label"] == "benign":
            out[(mac, win)] = None
    return out


def evaluate_session(sess, tr):
    df, res = sess.df, sess.res
    lab = df["label"].to_numpy()
    ev = np.isin(lab, ("benign", "attack"))
    y = (lab == "attack").astype(int)
    alerting = np.isin(res["cell"], V.ALERTING_CELLS)
    noatk = ev & (df["host_role"].to_numpy() != "attacker")
    det = {}
    for name, pred in (("pair_alerting", alerting), ("xgboost_confident_attack", res["band"] == 2),
                       ("isolation_forest", res["anomalous"])):
        det[name] = {"all": tm.prf(y[ev], pred[ev]), "noatk": tm.prf(y[noatk], pred[noatk])}
    table = tm.cell_table(lab, df["attack_class"].to_numpy(), res["cell"])
    alerts = _attack_alerts(sess, tr)
    runs = [iv["run_id"] for iv in sess.intervals]
    grp = tm.grouping(alerts, tr.members, runs)
    by_run = {}
    for (mac, win), run in alerts.items():
        if run is not None:
            by_run.setdefault(run, []).append(win)
    ttd = tm.time_to_detect(by_run, {iv["run_id"]: iv["start"] for iv in sess.intervals})
    return {"detection": det, "cell_table": table, "grouping": grp, "ttd": ttd,
            "false_novel": false_novel(sess),
            "n_rows": int(len(df)), "n_alert_rows": int(alerting.sum()), "n_incidents": len(tr.records)}


def load_events(session):
    ev = {"reboot": [], "reconnect": [], "nurse_command": []}
    for line in (common.SESSIONS_DIR / session / "benign_events.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        e = {"ip": r["ip"]}
        if r["event"] == "reboot":
            e.update(start=common.iso_to_epoch(r["start_utc"]), end=common.iso_to_epoch(r["end_utc"]))
        else:
            e["ts"] = common.iso_to_epoch(r["ts"])
        ev[r["event"]].append(e)
    return ev


def false_novel(sess):
    """Per event type: benign windows of the event host (by host_ip), classed by cell. Nurse commands: the
    nurse's benign windows holding the command ts, reported separately as non-disruptions."""
    df = sess.df
    cells = sess.res["cell"]
    benign = df.index[df["label"] == "benign"]
    wins_by_ip = {}
    for i in benign:
        wins_by_ip.setdefault(df.at[i, "host_ip"], []).append((int(df.at[i, "window_start_epoch"]), int(i)))
    ev = load_events(sess.name)
    out = {}
    for kind in ("reboot", "reconnect", "nurse_command"):
        idx = tm.event_windows(ev[kind], wins_by_ip, "reboot" if kind == "reboot" else "ts")
        out[kind] = {"n_events": len(ev[kind]), **tm.event_rates(idx, cells)}
    return out


def pooled(per_session, key_path):
    """Sum prf counts over sessions and recompute P/R/F1 (pooled, not averaged)."""
    out = {}
    for name in per_session[next(iter(per_session))]["detection"]:
        out[name] = {}
        for view in ("all", "noatk"):
            tp = sum(s["detection"][name][view]["tp"] for s in per_session.values())
            fp = sum(s["detection"][name][view]["fp"] for s in per_session.values())
            fn = sum(s["detection"][name][view]["fn"] for s in per_session.values())
            p, r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
            out[name][view] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                               "tp": tp, "fp": fp, "fn": fn}
    return out


# ------------------------------------------------------------------------------------ grouping on validation
def choose_grouping(sessions, ip_roles, keys=KEYS, gaps=GAPS):
    grid = []
    for key in keys:
        for gap in gaps:
            correct = detected = 0
            per = {}
            for s in sessions:
                tr = run_triager(s, key, gap, ip_roles)
                runs = [iv["run_id"] for iv in s.intervals]
                g = tm.grouping(_attack_alerts(s, tr), tr.members, runs)
                correct += len(g["correct"])
                detected += g["n_detected"]
                per[s.name] = {"accuracy": g["accuracy"], "n_incidents": g["n_incidents"],
                               "incidents_per_run": g["incidents_per_run"], "purity": g["purity"]}
            grid.append({"key": key, "gap_s": gap, "accuracy": correct / detected if detected else float("nan"),
                         "correct": correct, "detected": detected, "per_session": per})
    rank = lambda g: (-g["accuracy"], keys.index(g["key"]), g["gap_s"])
    best = sorted(grid, key=rank)[0]
    return best, grid


# ------------------------------------------------------------------------------------ B withheld
def b_withheld(classes, seeds=(0, 1, 2, 3, 4), pred_dir=EXPB / "predictions"):
    out = {}
    for cls in classes:
        per_seed = []
        for s in seeds:
            x = pd.read_parquet(Path(pred_dir) / f"xgboost_{cls}_s{s}.parquet")
            f = pd.read_parquet(Path(pred_dir) / f"isolation_forest_{cls}_s{s}.parquet")
            val, test = x[x["split"] == "val"], x[x["split"] == "test"].reset_index(drop=True)
            keys = ["session", "host_mac", "window_start_epoch"]
            ft = f[f["split"] == "test"].reset_index(drop=True)
            if not test[keys].equals(ft[keys]):
                raise AssertionError(f"B {cls} s{s}: xgboost and isolation_forest test rows differ")
            if (val["attack_class"] == cls).any():
                raise AssertionError(f"B {cls} s{s}: withheld class present in the fold's validation rows")
            p, band, cut, _ = tm.fold_bands(val, test)
            cells = V.cells(ft["pred"].to_numpy(int), band)
            w = (test["label"] == "attack") & (test["attack_class"] == cls)
            r = tm.withheld_share(cells[w.to_numpy()].tolist(), test.loc[w, "run_id"].tolist())
            ben = (test["label"] == "benign").to_numpy()
            r["benign_alerting_share"] = float(np.isin(cells[ben], V.ALERTING_CELLS).mean())
            r["cutoffs"] = {"lo": cut[0], "hi": cut[1]}
            per_seed.append(r)
        agg = {}
        for k in ("novel_or_review", "novel", "review", "known", "low_priority", "no_alert",
                  "runs_reaching_novel_or_review", "benign_alerting_share"):
            v = np.array([r[k] for r in per_seed], dtype=float)
            agg[k] = {"mean": float(v.mean()), "std": float(v.std())}
        out[cls] = {"n_windows": per_seed[0]["n_windows"], "n_runs": per_seed[0]["n_runs"], "agg": agg,
                    "per_seed": per_seed}
    return out


# ------------------------------------------------------------------------------------ main
def _flatten(prefix, d, out):
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten(f"{prefix}{k}_", v, out)
        elif isinstance(v, (int, float)) and v is not None:
            out[f"{prefix}{k}"] = v
    return out


def run(for_report=False, out_dir=RUNS, tri_dir=TRI, sessions=TEST, val=VAL):
    cfg = common.load_config()
    roles = common.ip_role_map(cfg)
    pair = Pair()
    t0 = time.time()
    val_s = [Session(n, pair) for n in val]
    best, grid = choose_grouping(val_s, roles)
    print(f"[triage] grouping chosen on validation {list(val)}: key={best['key']} gap={best['gap_s']} s "
          f"(accuracy {best['accuracy']:.3f}, {best['correct']}/{best['detected']} runs)", flush=True)
    for g in grid:
        print(f"    {g['key']:14s} gap {g['gap_s']:>3d}: {g['accuracy']:.3f} ({g['correct']}/{g['detected']})")
    slice_dir = Path(tri_dir) / "slices"
    per, all_records, ident = {}, [], {}
    for n in sessions:
        s = Session(n, pair)
        ident[n] = s.identity_check("test")
        print(f"[triage] {n}: identical to the saved A s0 predictions {ident[n]}", flush=True)
        tr = run_triager(s, best["key"], best["gap_s"], roles,
                         post=make_post(pair, n, slice_dir, "triage/slices"))
        per[n] = evaluate_session(s, tr)
        all_records += tr.records
        print(f"[triage] {n}: {len(tr.records)} incidents; pair F1 {per[n]['detection']['pair_alerting']['all']['f1']:.3f}; "
              f"grouping accuracy {per[n]['grouping']['accuracy']:.3f}", flush=True)
    inc_mod.write_jsonl(all_records, Path(tri_dir) / "incidents_offline.jsonl")
    from ..models import sequence as seq
    b = b_withheld(list(seq.ATTACK_CLASSES))
    prov = dict(seed=0, split_path=str(tt.SPLIT), dataset_path=str(tt.DATASET), for_report=for_report)
    params = {"models": "xgboost (calibrated, isotonic) + isolation_forest, Experiment A seed 0",
              "grouping_key": best["key"], "grouping_gap_s": best["gap_s"], "grouping_chosen_on": list(val),
              "cutoffs": {"lo": pair.lo, "hi": pair.hi}}
    metrics = {}
    for n, r in per.items():
        _flatten(f"{n}_", {"pair_f1": r["detection"]["pair_alerting"]["all"]["f1"],
                           "grouping_accuracy": r["grouping"]["accuracy"], "n_incidents": r["n_incidents"],
                           "ttd_median_s": r["ttd"]["median_s"] or float("nan")}, metrics)
    pool = pooled(per, None)
    _flatten("pooled_", {m: {v: {k: x for k, x in d.items() if k in ("precision", "recall", "f1")}
                             for v, d in views.items()} for m, views in pool.items()}, metrics)
    rec_a = tracking.build_run_record("triage", "xgboost+isolation_forest", "pair", params=params, metrics=metrics,
                                      tags={"stage": "triage_A", "sessions": list(sessions), "identity": ident,
                                            "grouping_grid": grid, "per_session": per, "pooled": pool,
                                            "slices_note": "slices hold plaintext lab credentials; gitignored data only"},
                                      **prov)
    rec_b = tracking.build_run_record("triage", "xgboost+isolation_forest", "pair", params=params,
                                      metrics={f"{c}_novel_or_review_mean": v["agg"]["novel_or_review"]["mean"]
                                               for c, v in b.items()},
                                      tags={"stage": "triage_B_withheld", "withheld": b,
                                            "calibration": "isotonic fitted on each fold's B validation + confidence_cutoffs; "
                                                           "IF pred as saved"}, **prov)
    out_dir = Path(out_dir)
    tracking.log_run(rec_a, json_path=str(out_dir / "triage_A.json"), use_mlflow=False)
    tracking.log_run(rec_b, json_path=str(out_dir / "triage_B_withheld.json"), use_mlflow=False)
    (out_dir / "triage_summary.md").write_text(summary_md(rec_a, rec_b))
    print(f"[triage] done in {time.time() - t0:.0f} s; records in {out_dir}", flush=True)
    return rec_a, rec_b


def summary_md(a, b):
    t = a["tags"]
    L = ["# Offline triage (deployed pair: XGBoost + Isolation Forest)", "",
         f"Grouping chosen on validation {a['params']['grouping_chosen_on']}: key `{a['params']['grouping_key']}`, "
         f"gap {a['params']['grouping_gap_s']} s.", ""]
    for n, r in t["per_session"].items():
        L += [f"## {n}", "", "| cell | " + " | ".join(V.CELLS) + " |", "|---|" + "---|" * len(V.CELLS)]
        for cls, row in sorted(r["cell_table"].items()):
            L.append(f"| {cls} | " + " | ".join(str(row[c]) for c in V.CELLS) + " |")
        g, tt_ = r["grouping"], r["ttd"]
        L += ["", f"Pair F1 {r['detection']['pair_alerting']['all']['f1']:.3f}; incidents {r['n_incidents']}; grouping accuracy "
              f"{g['accuracy']:.3f} ({len(g['correct'])}/{g['n_detected']} detected runs; undetected {g['undetected']}); "
              f"purity {g['purity']:.3f}; false incidents {g['n_false_incidents']}; "
              f"TTD median {tt_['median_s']} s.", ""]
        L += ["False-novel (benign windows of the event host):", "", "| event | events | windows | novel | review | any alert |",
              "|---|---|---|---|---|---|"]
        for k, v in r["false_novel"].items():
            f = lambda x: "-" if x is None else f"{x:.3f}"
            L.append(f"| {k} | {v['n_events']} | {v['n_windows']} | {f(v['novel'])} | {f(v['review'])} | {f(v['alert'])} |")
        L.append("")
    L += ["## B: withheld attack classes reaching novel / review (mean ± std over 5 seeds)", "",
          "| class | windows | runs | novel or review | novel | review | known | low | no alert | runs reaching |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in b["tags"]["withheld"].items():
        g = lambda k: f"{v['agg'][k]['mean']:.3f} ± {v['agg'][k]['std']:.3f}"
        L.append(f"| {c} | {v['n_windows']} | {v['n_runs']} | {g('novel_or_review')} | {g('novel')} | {g('review')} | "
                 f"{g('known')} | {g('low_priority')} | {g('no_alert')} | {g('runs_reaching_novel_or_review')} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--for-report", action="store_true")
    args = ap.parse_args(argv)
    run(for_report=args.for_report)


if __name__ == "__main__":
    main()
