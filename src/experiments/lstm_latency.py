"""Single-window latency of the deployed LSTM path, plus an equivalence check against the offline run.

Run in its OWN process on an idle machine (torch.set_num_threads(1), deterministic settings):
  python -m src.experiments.lstm_latency

score_one(item) is the full per-window work of the live path: take ONE raw window row (the 51 feature
values in feature_names order, unscaled, plus window_start_epoch), push it into that host's
RollingBuffer (which scales internally and fills grid gaps), run the forward pass and the softmax,
return (1 - P(benign), class probabilities). Inputs are REAL test rows replayed per host in time order
from the host's first window in the session, so the buffer state is realistic.

Equivalence (asserted, 1e-6): the replayed scores equal the saved offline scores in
predictions/<run>.parquet (test split) for every (session, host_mac, window_start_epoch).
The record logs median/p95 ms (metrics.single_window_latency_ms, >= 200 calls), the equivalence
max_abs_diff and CPU info. p95 over 20 ms stops the run (the 50 ms budget is for the whole pair).
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import common
from ..models import lstm
from ..models import metrics as mm
from ..models import sequence as seq
from ..models import tracking

V2 = common.PROCESSED_DIR / "v2"
RUNS = V2 / "runs" / "lstm"
RUN_NAME = "A_k12_h32_s0"
SPLIT_PATH = "data/raw/sessions/split_v2.json"
DATASET_PATH = "data/processed/v2/testbed_windows.parquet"
P95_STOP_MS = 20.0
EQUIV_TOL = 1e-6


class Scorer:
    """Per-(session, host) RollingBuffers + one model. The unit the live pipeline would run."""

    def __init__(self, art):
        self.model = art["model"].eval()
        self.scaler = art["scaler"]
        self.k = art["k"]
        self.names = art["feature_names"]
        self.use_presence = art["cfg"]["use_presence"]
        self.buffers = {}

    def score_one(self, item):
        """item = {"host": hashable host key, "epoch": window_start_epoch, "values": {feature: raw value},
        "first": True for the host's first window (starts a fresh buffer)}."""
        buf = self.buffers.get(item["host"])
        if buf is None or item["first"]:
            buf = self.buffers[item["host"]] = seq.RollingBuffer(
                self.k, self.scaler, self.names, self.use_presence)
        x = buf.push(item["values"], item["epoch"])                # scales + fills gaps -> [1, k, d]
        with torch.no_grad():
            p = torch.softmax(self.model(torch.as_tensor(x)), dim=1).numpy()
        return float((1.0 - p[:, 0])[0]), p[0]


def host_streams(df, names):
    """{(session, host_mac): [items in time order]} from raw rows (ALL labels: victim and boundary
    windows advance the buffer exactly as in the offline tensors)."""
    out = {}
    for (sess, mac), g in df.sort_values("window_start_epoch").groupby(["session", "host_mac"], sort=True):
        vals = g[names].to_dict("records")
        out[(sess, mac)] = [{"host": (sess, mac), "epoch": int(e), "values": v, "first": i == 0}
                            for i, (e, v) in enumerate(zip(g["window_start_epoch"], vals))]
    return out


def has_gaps(items):
    e = np.array([i["epoch"] for i in items])
    return len(e) > 1 and bool((np.diff(e) != seq.STEP).any())


def replay(scorer, streams):
    """Untimed full replay. Returns {(session, host_mac, epoch): score}."""
    out = {}
    for (sess, mac), items in streams.items():
        for it in items:
            out[(sess, mac, it["epoch"])] = scorer.score_one(it)[0]
    return out


def check_equivalence(replayed, preds, tol=EQUIV_TOL):
    """Compare replayed scores with the saved offline scores for the same keys. Returns
    (max_abs_diff, n_compared); raises AssertionError on a missing key or a diff over tol."""
    missing = 0
    diffs = []
    for r in preds.itertuples():
        v = replayed.get((r.session, r.host_mac, int(r.window_start_epoch)))
        if v is None:
            missing += 1
            continue
        diffs.append(abs(v - float(r.score)))
    if missing:
        raise AssertionError(f"{missing} saved predictions have no replayed score")
    mx = float(max(diffs)) if diffs else float("nan")
    if not diffs or not mx <= tol:
        raise AssertionError(f"replayed scores differ from the offline scores: max_abs_diff={mx} > {tol}")
    return mx, len(diffs)


def timing_inputs(streams, n_hosts=6):
    """Contiguous per-host streams: first the hosts with gaps (>= 1), then a spread of the others, so
    cycling through the list wraps from the end of one host to the start of another (first=True resets)."""
    keys = sorted(streams)
    gapped = [k for k in keys if has_gaps(streams[k])]
    rest = [k for k in keys if k not in set(gapped)]
    chosen = gapped[:2] + rest[::max(1, len(rest) // max(1, n_hosts - 2))][:n_hosts - 2]
    items = [it for k in chosen for it in streams[k]]
    return items, chosen


def run(artefact_path, df_test, preds_test, out_json=None, n_calls=None, threads=1, seed=0,
        for_report=False, split_path=SPLIT_PATH, dataset_path=DATASET_PATH, run_name=RUN_NAME):
    lstm.set_determinism(seed, threads)
    art = lstm.load_artefact(artefact_path)
    scorer = Scorer(art)
    streams = host_streams(df_test, art["feature_names"])
    t0 = time.perf_counter()
    replayed = replay(scorer, streams)
    max_diff, n_cmp = check_equivalence(replayed, preds_test)
    print(f"equivalence OK: {n_cmp} test rows, max_abs_diff={max_diff:.3g} "
          f"(replay {time.perf_counter() - t0:.1f}s)", flush=True)

    items, chosen = timing_inputs(streams)
    n_calls = max(200, int(n_calls) if n_calls else len(items))
    fresh = Scorer(art)                                            # empty buffers, like a fresh process
    lat = mm.single_window_latency_ms(fresh.score_one, items, n_calls=n_calls)
    print(f"latency: median {lat['median_ms']:.3f} ms, p95 {lat['p95_ms']:.3f} ms "
          f"({lat['n']} calls, {lat['warmup']} warm-up, {len(chosen)} hosts, "
          f"{sum(has_gaps(streams[c]) for c in chosen)} with gaps, {threads} thread)", flush=True)

    params = {"k": art["k"], "hidden": art["cfg"]["hidden"], "threads": threads, "seed": seed,
              "n_calls": lat["n"], "artefact": str(artefact_path), "n_hosts_timed": len(chosen)}
    metrics = {"median_ms": lat["median_ms"], "p95_ms": lat["p95_ms"], "n": lat["n"],
               "warmup": lat["warmup"], "equivalence_max_abs_diff": max_diff,
               "equivalence_n_compared": n_cmp}
    tags = {"latency": True, "cpu": {"processor": platform.processor(), "machine": platform.machine(),
                                     "platform": platform.platform(), "cpu_count": os.cpu_count(),
                                     "torch": torch.__version__, "torch_threads": torch.get_num_threads()}}
    rec = tracking.build_run_record("A", "lstm", "classifier", params=params, metrics=metrics, tags=tags,
                                    seed=seed, split_path=split_path, dataset_path=dataset_path,
                                    for_report=for_report)
    if lat["p95_ms"] > P95_STOP_MS:
        print(f"STOP: p95 {lat['p95_ms']:.2f} ms > {P95_STOP_MS} ms; no record written", flush=True)
        return rec, 2
    tracking.log_run(rec, json_path=str(out_json) if out_json else None)
    return rec, 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN_NAME)
    ap.add_argument("--n-calls", type=int, default=None)
    args = ap.parse_args(argv)
    os.chdir(common.ROOT)
    df = seq.load_sessions(V2 / "sessions", SPLIT_PATH)
    test = df[df["split"] == "test"]
    preds = pd.read_parquet(RUNS / "predictions" / f"{args.run}.parquet")
    preds = preds[preds["split"] == "test"]
    rec, code = run(V2 / "models" / "lstm" / f"{args.run}.pt", test, preds,
                    out_json=RUNS / f"latency_{args.run}.json", n_calls=args.n_calls, for_report=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
