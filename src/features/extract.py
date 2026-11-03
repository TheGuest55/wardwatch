"""Extract one session into a labelled host-window feature table.

Usage (inside the analysis container):
    python -m src.features.extract --session t02 [--out data/processed/sessions/t02.parquet]

Streams every pcap in the session (so session-wide state like client-id ownership persists),
builds per-host-window features, enriches with NFStream flow counts when available, labels each
window from ground_truth.jsonl, and writes a per-session parquet. dataset.py concatenates these.
"""
import argparse
import itertools
import json
import math
import sys

import numpy as np
import pandas as pd

from .. import common
from . import packets, windows, nfstream_flows
from ..labels import label as labelling


def _session_dir(session):
    return common.SESSIONS_DIR / session


def _pcap_paths(session_dir):
    pdir = session_dir / "pcap"
    return sorted(pdir.glob("*.pcap")) if pdir.exists() else []


def extract_session(session):
    """Return (df, info). info carries counts used by datacheck and the CLI summary."""
    sdir = _session_dir(session)
    if not sdir.exists():
        raise SystemExit(f"session not found: {sdir}")

    cfg = common.load_config()
    ip_roles = common.ip_role_map(cfg)
    config_json = json.loads((sdir / "config.json").read_text()) if (sdir / "config.json").exists() else {}
    dry_run = bool(config_json.get("dry_run", False))

    intervals, attacker_ip, attacker_mac = labelling.load_ground_truth(sdir)
    attacker_ip = attacker_ip or cfg["attacker"]["ip"]

    pcaps = _pcap_paths(sdir)
    if not pcaps:
        raise SystemExit(f"no pcaps under {sdir}/pcap")

    packet_iter = itertools.chain.from_iterable(packets.iter_packets(p) for p in pcaps)
    rows, conflict_windows = windows.build_rows(
        packet_iter, session, attacker_mac, attacker_ip, ip_roles)

    # NFStream enrichment (optional): sum per-pcap flow features onto (mac, window).
    nf = {}
    nf_available = nfstream_flows.AVAILABLE
    if nf_available:
        for p in pcaps:
            part = nfstream_flows.window_flow_features(p)
            if part is None:
                nf_available = False
                break
            for key, cell in part.items():
                tgt = nf.setdefault(key, {"nf_active_flows": 0, "nf_new_flows": 0, "nf_dur_sum": 0.0})
                for k, v in cell.items():
                    tgt[k] += v

    for r in rows:
        key = (r["host_mac"], r["window_start_epoch"])
        if nf_available and key in nf:
            cell = nf[key]
            r["nf_active_flows"] = cell["nf_active_flows"]
            r["nf_new_flows"] = cell["nf_new_flows"]
            r["nf_mean_flow_dur"] = (cell["nf_dur_sum"] / cell["nf_active_flows"]
                                     if cell["nf_active_flows"] else 0.0)
        elif nf_available:
            r["nf_active_flows"] = 0
            r["nf_new_flows"] = 0
            r["nf_mean_flow_dur"] = 0.0
        else:
            r["nf_active_flows"] = math.nan
            r["nf_new_flows"] = math.nan
            r["nf_mean_flow_dur"] = math.nan

    df = pd.DataFrame(rows)
    df = labelling.apply_labels(df, intervals)
    df["dry_run"] = dry_run
    df = df.sort_values(["window_start_epoch", "host_mac"]).reset_index(drop=True)

    info = {
        "session": session,
        "dry_run": dry_run,
        "n_rows": len(df),
        "n_pcaps": len(pcaps),
        "nfstream": nf_available,
        "n_runs": len(intervals),
        "attacker_mac": attacker_mac,
        "attacker_ip": attacker_ip,
        "conflict_windows": len(conflict_windows),
        "label_counts": df["label"].value_counts().to_dict(),
        "attack_by_class": (df[df["label"] == "attack"]["attack_class"]
                            .value_counts().to_dict()),
        "runs_with_window": sorted(set(df[df["label"] == "attack"]["run_id"])),
        "all_run_ids": [iv["run_id"] for iv in intervals],
    }
    return df, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df, info = extract_session(args.session)
    out = args.out or str(common.PROCESSED_DIR / "sessions" / f"{args.session}.parquet")
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    if not info["nfstream"]:
        print("[DATA] nfstream unavailable in image; nf_* flow features left NaN "
              "(per-window stats from packets are unaffected)", file=sys.stderr)
    print(json.dumps({**info, "out": out}, indent=2, default=str))
    # Sanity surfaced immediately: every non-dry run should land >=1 attack window.
    if not info["dry_run"]:
        missing = set(info["all_run_ids"]) - set(info["runs_with_window"])
        if missing:
            print(f"[WARN] runs with no attack window: {sorted(missing)}", file=sys.stderr)


if __name__ == "__main__":
    main()
