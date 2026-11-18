"""Experiment E: replay a recorded held-out session through the deployed pair as a live system.

    python -m src.experiments.run_e --session r05 [--speed 1] [--for-report]

The pcaps of the session are replayed against the wall clock (speed 1 = real time) into the incremental window
builder; a timer closes each 5 s window 1 s after it ends; the TIMED path of every tick (all hosts of one window) is
    finalize rows into feature order -> XGBoost (calibrated p, band) + Isolation Forest + the type name for Known
    attack rows -> verdicts -> attribution and incident tracker update
measured with perf_counter per tick and per row against the 50 ms budget. Measured separately: per-packet ingest,
replay lag, and the post-alert work (TreeSHAP reasons and the hashed pcap slice, done in a worker thread).
This is a REPLAY of a recorded session at real speed, not a live capture.

The live system does not know the attacker: the builder gets sentinels, so host_role (display only) comes from the
config's IP -> role map; no feature depends on it.

Equivalence with the offline run (a stop if broken; checks are exact, not "suspiciously good" results):
  * the raw XGBoost score of every row equals the saved Experiment-A seed-0 test prediction (|d| <= 1e-6 * max(1,|s|)),
    the IF decision equals the saved one;
  * against the offline scoring of the stored session parquet: 0 band flips, 0 cell (verdict) flips, same rows;
  * the incidents equal the offline incidents of this session (ids, every field, sha256), apart from the slice
    directory (live/ vs triage/).
Output: data/processed/v2/live/incidents_<session>.jsonl and runs/expE/E_<session>.json. `python -m src.realtime.merge_live`
merges the per-session files into live/incidents.jsonl (what the dashboard shows) and verifies every slice hash.
"""
import argparse
import json
import os
import platform
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import common
from ..features import windows
from ..models import tracking
from ..models import train_tree as tt
from ..realtime import stream
from ..realtime.pipeline import Pair
from ..triage import engine, incidents as inc_mod, metrics as tm, offline, pcap_slice
from ..triage import verdict as V

V2 = common.PROCESSED_DIR / "v2"
LIVE = V2 / "live"
EXPE = V2 / "runs" / "expE"
BUDGET_MS = 50.0
SENTINEL = "-"
EQ_TOL = 1e-6


class LivePost:
    """Post-alert worker: TreeSHAP + slice for each closed incident, off the scoring path."""

    def __init__(self, pair, session, live_dir, write_path):
        self.explain = offline.explain_fn_for(pair)
        self.pcaps = stream.pcap_paths(session)
        for p in self.pcaps:                      # index the pcaps now, before the replay starts
            pcap_slice.get_index(p)
        self.live_dir, self.write_path = Path(live_dir), write_path
        self.q, self.done, self.shap_s, self.slice_s = queue.Queue(), [], [], []
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()

    def __call__(self, rec, inc):
        self.q.put((rec, inc))

    def _slice(self, first, last_end, macs, iid):
        out = self.live_dir / "slices" / f"{iid}.pcap"
        s = pcap_slice.write_slice(self.pcaps, macs, first, last_end, out)
        s["path"] = str(Path("live/slices") / f"{iid}.pcap")
        return s

    def _work(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            rec, inc = item
            _, _, top = inc_mod._summary(inc)
            t0 = time.perf_counter()
            rec["top_features"] = self.explain(top.feat)
            t1 = time.perf_counter()
            inc_mod.complete_record(rec, inc, None, self._slice)
            self.shap_s.append(t1 - t0)
            self.slice_s.append(time.perf_counter() - t1)
            self.done.append(rec)
            inc_mod.write_jsonl(self.done, self.write_path)
            self.q.task_done()

    def close(self):
        self.q.put(None)
        self.q.join()


def _pcts(a):
    a = np.asarray(a, dtype=float)
    if not len(a):
        return {"median": None, "p95": None, "max": None, "n": 0}
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "n": int(len(a))}


def replay_session(session, pair, key, gap, speed=1.0, max_seconds=None, live_dir=LIVE, packet_iter=None):
    """Run the replay. Returns a dict with the rows, per-row results, tick timings, stats and the triager."""
    cfg = common.load_config()
    roles = common.ip_role_map(cfg)
    wb = windows.WindowBuilder(session, SENTINEL, SENTINEL, roles, with_peers=True)
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    post = LivePost(pair, session, live_dir, live_dir / f"incidents_{session}.jsonl")
    tr = engine.Triager(session, roles, key, gap, post=post)
    rows_all, tick_s, tick_n, results, tick_done = [], [], [], {}, {}
    peers = {}

    def on_tick(rows, ctx):
        t0 = time.perf_counter()
        rows = sorted(rows, key=lambda r: (r["window_start_epoch"], r["host_mac"]))
        X = pair.matrix(rows)
        res = pair.score(X)
        wins = np.array([r["window_start_epoch"] for r in rows])
        for r in rows:
            peers[(r["host_mac"], r["window_start_epoch"])] = r["peer_pkts"]
        cuts = np.flatnonzero(np.diff(wins)) + 1
        for idx in np.split(np.arange(len(rows)), cuts):
            tr.window(int(wins[idx[0]]), [rows[i]["host_mac"] for i in idx], [rows[i]["host_ip"] for i in idx],
                      X[idx], {k: v[idx] for k, v in res.items()}, peers)
        dt = time.perf_counter() - t0
        done_at = ctx["replay_now"] + dt * (speed or 1.0)
        tick_s.append(dt)
        tick_n.append(len(rows))
        for i, r in enumerate(rows):
            k = (r["host_mac"], r["window_start_epoch"])
            results[k] = {c: res[c][i] for c in ("raw", "p", "band", "if_score", "anomalous", "cell", "named")}
            tick_done[k] = done_at
            rows_all.append({k2: v for k2, v in r.items() if k2 != "peer_pkts"})

    stats = stream.ReplayStats()
    it = packet_iter if packet_iter is not None else stream.session_packets(session, max_seconds=max_seconds)
    stream.replay(it, wb, on_tick, speed=speed, stats=stats)
    tr.finish()
    post.close()
    return {"rows": rows_all, "results": results, "tick_done": tick_done, "tick_s": tick_s, "tick_n": tick_n,
            "stats": stats, "triager": tr, "builder": wb, "post": post, "peers": peers}


def to_session(session, out, pair, df_ref):
    """The live rows + results as an offline.Session (labels joined from the stored parquet by key)."""
    live = pd.DataFrame(out["rows"])
    keys = ["host_mac", "window_start_epoch"]
    # evaluation labels, and the ground-truth role for the attacker-excluded VIEW (not used by any decision)
    lab_cols = ["label", "attack_class", "run_id", "is_boundary", "host_role"]
    df = live.drop(columns=[c for c in lab_cols if c in live.columns]).merge(
        df_ref[keys + lab_cols], on=keys, how="left")
    df = df.sort_values(["window_start_epoch", "host_mac"]).reset_index(drop=True)
    idx = {k: i for i, k in enumerate(zip(df["host_mac"], df["window_start_epoch"]))}
    order = [out["results"][k] for k in zip(df["host_mac"], df["window_start_epoch"])]
    res = {c: np.array([o[c] for o in order], dtype=object if c in ("cell", "named") else None)
           for c in ("raw", "p", "band", "if_score", "anomalous", "cell", "named")}
    res["anomalous"] = res["anomalous"].astype(bool)
    X = pair.matrix(df.to_dict("records"))
    return offline.Session.from_parts(session, df, X, res, out["peers"])


def compare_rows(live_sess, off_sess):
    """Live vs offline scoring of the same rows: raw score rel diff, band flips, cell flips. Raises on a break."""
    a = live_sess.df[["host_mac", "window_start_epoch"]].assign(i=np.arange(len(live_sess.df)))
    b = off_sess.df[["host_mac", "window_start_epoch"]].assign(j=np.arange(len(off_sess.df)))
    m = a.merge(b, on=["host_mac", "window_start_epoch"], how="outer")
    if m["i"].isna().any() or m["j"].isna().any():
        raise AssertionError(f"live and offline rows differ: {int(m['i'].isna().sum())} only offline, "
                             f"{int(m['j'].isna().sum())} only live")
    i, j = m["i"].to_numpy(int), m["j"].to_numpy(int)
    d = np.abs(live_sess.res["raw"][i] - off_sess.res["raw"][j]) / np.maximum(1.0, np.abs(off_sess.res["raw"][j]))
    band_flips = int((live_sess.res["band"][i] != off_sess.res["band"][j]).sum())
    cell_flips = int((live_sess.res["cell"][i] != off_sess.res["cell"][j]).sum())
    out = {"n_rows": int(len(m)), "max_rel_score_diff": float(d.max()), "band_flips": band_flips,
           "verdict_flips": cell_flips}
    if not d.max() <= EQ_TOL or band_flips or cell_flips:
        raise AssertionError(f"live scoring differs from offline: {out}")
    return out


def compare_incidents(live_recs, off_recs):
    """Every field equal (ids, sha256 included), modulo the slice directory."""
    norm = lambda r: {**r, "pcap_slice": Path(r["pcap_slice"]).name if r.get("pcap_slice") else None}
    L = {r["incident_id"]: norm(r) for r in live_recs}
    O = {r["incident_id"]: norm(r) for r in off_recs}
    probs = []
    if set(L) != set(O):
        probs.append(f"incident ids differ: only live {sorted(set(L) - set(O))[:5]}, only offline {sorted(set(O) - set(L))[:5]}")
    for k in sorted(set(L) & set(O)):
        for f in sorted(set(L[k]) | set(O[k])):
            if L[k].get(f) != O[k].get(f):
                probs.append(f"{k}.{f}: live {str(L[k].get(f))[:100]} != offline {str(O[k].get(f))[:100]}")
    if probs:
        raise AssertionError("; ".join(probs[:8]))
    return {"n_incidents": len(L), "equal": True}


def run(session, speed=1.0, max_seconds=None, for_report=False, live_dir=LIVE, exp_dir=EXPE, tri_dir=offline.TRI,
        a_record=offline.RUNS / "triage_A.json"):
    if for_report and (max_seconds or session not in offline.TEST):
        raise SystemExit("a report-grade E run is a full replay of a held-out test session")
    a = json.loads(Path(a_record).read_text())
    key, gap = a["params"]["grouping_key"], a["params"]["grouping_gap_s"]
    pair = Pair()
    print(f"[E] replay {session} at speed {speed} (grouping {key}, gap {gap} s), pair loaded", flush=True)
    out = replay_session(session, pair, key, gap, speed, max_seconds, live_dir)
    tr, st = out["triager"], out["stats"]
    ref = pd.read_parquet(offline.V2 / "sessions" / f"{session}.parquet")
    live = to_session(session, out, pair, ref)
    summ = st.summary()
    late = out["builder"].late_packets
    print(f"[E] {summ['n_packets']} packets, {summ['n_ticks']} ticks, wall {summ['wall_s']:.0f} s, late packets {late}, "
          f"max lag {summ['lag_s_max']:.3f} s", flush=True)
    if late:
        raise SystemExit(f"[E] STOP: {late} late packets")
    eq = {"late_packets": late}
    if not max_seconds:
        off = offline.Session(session, pair, df=ref, peers=live.peers)
        eq["rows_vs_offline"] = compare_rows(live, off)
        eq["vs_saved_A_s0"] = live.identity_check("test")
        tr_off = [r for r in (json.loads(l) for l in (Path(tri_dir) / "incidents_offline.jsonl").read_text().splitlines())
                  if r["session"] == session]
        eq["incidents_vs_offline"] = compare_incidents(
            [r for r in tr.records], tr_off)
        print(f"[E] equivalence with offline OK: {json.dumps(eq)}", flush=True)
    ev = offline.evaluate_session(live, tr)
    # live time to detect: replay clock when the tick holding the first alerting attack window finished
    run_starts = {iv["run_id"]: iv["start"] for iv in live.intervals}
    first_alert_done = {}
    for (mac, win), run_id in offline._attack_alerts(live, tr).items():
        if run_id is not None:
            t = out["tick_done"][(mac, win)]
            first_alert_done[run_id] = min(first_alert_done.get(run_id, np.inf), t)
    live_ttd = {r: (float(first_alert_done[r] - t0) if r in first_alert_done else None) for r, t0 in run_starts.items()}
    got = [v for v in live_ttd.values() if v is not None]
    tick_ms = np.array(out["tick_s"]) * 1000
    row_ms = tick_ms / np.maximum(np.array(out["tick_n"]), 1)
    lat = {"tick_ms": _pcts(tick_ms), "row_ms": _pcts(row_ms), "rows_per_tick": _pcts(out["tick_n"])}
    post = out["post"]
    metrics = {"tick_median_ms": lat["tick_ms"]["median"], "tick_p95_ms": lat["tick_ms"]["p95"],
               "tick_max_ms": lat["tick_ms"]["max"], "row_median_ms": lat["row_ms"]["median"],
               "row_p95_ms": lat["row_ms"]["p95"], "row_max_ms": lat["row_ms"]["max"],
               "ticks_over_budget": int((tick_ms > BUDGET_MS).sum()), "budget_ms": BUDGET_MS,
               "shap_p95_s": _pcts(post.shap_s)["p95"], "slice_p95_s": _pcts(post.slice_s)["p95"],
               "ingest_us_mean": summ["ingest_us_mean"], "ingest_us_p95": summ["ingest_us_p95"],
               "lag_s_max": summ["lag_s_max"], "lag_s_p95": summ["lag_s_p95"], "wall_s": summ["wall_s"],
               "n_packets": summ["n_packets"], "n_ticks": summ["n_ticks"], "n_incidents": len(tr.records),
               "pair_precision": ev["detection"]["pair_alerting"]["all"]["precision"],
               "pair_recall": ev["detection"]["pair_alerting"]["all"]["recall"],
               "pair_f1": ev["detection"]["pair_alerting"]["all"]["f1"],
               "xgboost_f1": ev["detection"]["xgboost_confident_attack"]["all"]["f1"],
               "isolation_forest_f1": ev["detection"]["isolation_forest"]["all"]["f1"],
               "grouping_accuracy": ev["grouping"]["accuracy"],
               "ttd_offline_median_s": ev["ttd"]["median_s"], "ttd_live_median_s": float(np.median(got)) if got else None}
    rec = tracking.build_run_record(
        "E", "xgboost+isolation_forest", "pair",
        params={"session": session, "speed": speed, "grace_s": stream.GRACE_S, "grouping_key": key, "grouping_gap_s": gap,
                "max_seconds": max_seconds, "threads": 1},
        metrics=metrics,
        tags={"stage": "E", "replay": "recorded held-out session replayed against the wall clock; not a live capture",
              "equivalence": eq, "detection": ev["detection"], "cell_table": ev["cell_table"], "grouping": ev["grouping"],
              "ttd_offline": ev["ttd"], "ttd_live": live_ttd, "false_novel": ev["false_novel"], "latency": lat,
              "replay_stats": summ, "uptime": open("/proc/uptime").read().split()[0] if os.path.exists("/proc/uptime") else None,
              "loadavg": list(os.getloadavg()),
              "cpu": {"cpu_count": os.cpu_count(), "platform": platform.platform()},
              "slices_note": "slices hold plaintext lab credentials; gitignored data only"},
        seed=0, split_path=str(tt.SPLIT), dataset_path=str(tt.DATASET), for_report=for_report)
    path = Path(exp_dir) / f"E_{session}.json"
    tracking.log_run(rec, json_path=str(path), use_mlflow=False)
    print(f"[E] tick median {lat['tick_ms']['median']:.2f} ms, p95 {lat['tick_ms']['p95']:.2f} ms, max {lat['tick_ms']['max']:.2f} ms "
          f"(budget {BUDGET_MS} ms; {metrics['ticks_over_budget']} ticks over); {len(tr.records)} incidents; "
          f"pair F1 {metrics['pair_f1']:.3f}; record {path}", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--for-report", action="store_true")
    args = ap.parse_args(argv)
    try:
        run(args.session, args.speed, args.max_seconds, args.for_report)
    except (AssertionError, SystemExit) as e:
        EXPE.mkdir(parents=True, exist_ok=True)
        (EXPE / "STOP.md").write_text(f"# STOP: E {args.session}\n\n{e}\n")
        raise


if __name__ == "__main__":
    main()
