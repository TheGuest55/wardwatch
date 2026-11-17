"""Equivalence check: the streaming window builder must reproduce a stored session parquet exactly.

    python -m src.realtime.check_session --session r05 [--sessions n02 r01 ...]

Runs WindowBuilder as fast as possible (replay speed 0, timer closes) over every pcap of the session and asserts
against data/processed/v2/sessions/<S>.parquet:
  * the set of (window_start_epoch, host_mac) keys is identical (no extra, no missing rows);
  * every column the extractor emits at build time (the 51 features and the metadata host_ip, host_role,
    touches_attacker, window_start_iso) is exactly equal, NaN equal to NaN, int/float values bit for bit,
    including the causal flow features nf_active_flows / nf_new_flows / nf_mean_flow_age;
  * late_packets == 0.
Any difference raises SystemExit with the first offending rows. It then writes the peer sidecar
data/processed/v2/triage/peers_<S>.parquet (session, host_mac, window_start_epoch, peer_pkts as JSON: the host's
unicast packets [sent, recv] per peer MAC), METADATA used for source attribution only.

The ground truth's attacker MAC/IP are passed to the builder here so the label-side metadata columns can be
compared too; the live path (run_e) does not know the attacker and passes sentinels, which changes only
host_role / touches_attacker, never a feature.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import common
from ..features import windows
from ..labels import label as labelling
from . import stream

V2 = common.PROCESSED_DIR / "v2"
PEER_DIR = V2 / "triage"
KEYS = ["window_start_epoch", "host_mac"]


def ground_truth_attacker(session, cfg):
    _, ip, mac = labelling.load_ground_truth(common.SESSIONS_DIR / session)
    return mac, ip or cfg["attacker"]["ip"]


def build_session(session, speed=0, max_seconds=None, with_peers=True):
    """Stream the whole session through the builder; returns (rows, builder, stats)."""
    cfg = common.load_config()
    mac, ip = ground_truth_attacker(session, cfg)
    wb = windows.WindowBuilder(session, mac, ip, common.ip_role_map(cfg), with_peers=with_peers)
    out = []
    st = stream.replay(stream.session_packets(session, max_seconds=max_seconds), wb,
                       lambda rows, ctx: out.extend(rows), speed=speed)
    return out, wb, st


def compare(rows, ref, max_report=5):
    """Exact comparison of streamed rows with the stored parquet. Returns a list of problems (empty = equal)."""
    got = pd.DataFrame([{k: v for k, v in r.items() if k != "peer_pkts"} for r in rows])
    probs = []
    gk = set(zip(got["window_start_epoch"], got["host_mac"]))
    rk = set(zip(ref["window_start_epoch"], ref["host_mac"]))
    if len(got) != len(gk):
        probs.append(f"{len(got) - len(gk)} duplicate (window, host) rows in the stream")
    if gk - rk:
        probs.append(f"{len(gk - rk)} extra rows, e.g. {sorted(gk - rk)[:max_report]}")
    if rk - gk:
        probs.append(f"{len(rk - gk)} missing rows, e.g. {sorted(rk - gk)[:max_report]}")
    if probs:
        return probs
    got = got.sort_values(KEYS).reset_index(drop=True)
    ref = ref.sort_values(KEYS).reset_index(drop=True)
    cols = [c for c in got.columns if c in ref.columns]
    missing = [c for c in got.columns if c not in ref.columns]
    if missing:
        probs.append(f"streamed columns absent from the parquet: {missing}")
    for c in cols:
        a, b = got[c], ref[c]
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            av, bv = a.to_numpy(float), b.to_numpy(float)
            same = (av == bv) | (np.isnan(av) & np.isnan(bv))
        else:
            same = (a.astype(object) == b.astype(object)) | (a.isna() & b.isna())
        if not bool(np.all(same)):
            bad = np.flatnonzero(~np.asarray(same))
            probs.append(f"column {c}: {len(bad)} differ, e.g. row {int(bad[0])} "
                         f"{got.loc[bad[0], KEYS].tolist()} stream={a.iloc[bad[0]]!r} parquet={b.iloc[bad[0]]!r}")
    return probs


def write_peers(session, rows, out_dir=PEER_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"session": session,
                       "host_mac": [r["host_mac"] for r in rows],
                       "window_start_epoch": [r["window_start_epoch"] for r in rows],
                       "peer_pkts": [json.dumps(r["peer_pkts"], sort_keys=True) for r in rows]})
    df = df.sort_values(KEYS).reset_index(drop=True)
    p = out_dir / f"peers_{session}.parquet"
    df.to_parquet(p, index=False)
    return p


def check(session, write=True, parquet_dir=V2 / "sessions", peers_dir=PEER_DIR):
    t0 = time.perf_counter()
    rows, wb, st = build_session(session)
    ref = pd.read_parquet(Path(parquet_dir) / f"{session}.parquet")
    probs = compare(rows, ref)
    if wb.late_packets:
        probs.append(f"{wb.late_packets} late packets (windows {sorted(wb.late_windows)[:5]})")
    res = {"session": session, "n_rows": len(rows), "n_ref_rows": len(ref), "n_packets": wb.n_packets,
           "late_packets": wb.late_packets, "exact": not probs, "problems": probs,
           "seconds": round(time.perf_counter() - t0, 1)}
    if not probs and write:
        res["peers"] = str(write_peers(session, rows, peers_dir))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--out", default=str(PEER_DIR / "check_session.json"))
    args = ap.parse_args(argv)
    results = []
    for s in args.session:
        r = check(s)
        results.append(r)
        print(json.dumps(r, indent=2, default=str), flush=True)
        if not r["exact"]:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2, default=str))
            raise SystemExit(f"[check_session] {s}: streamed rows differ from the stored parquet: {r['problems']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"[check_session] {len(results)} session(s) exactly equal to the stored parquet, 0 late packets")


if __name__ == "__main__":
    main()
