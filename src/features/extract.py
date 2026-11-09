"""Extract one session into a labelled host-window feature table.

Usage (inside the analysis container):
    python -m src.features.extract --session t02 [--out data/processed/sessions/t02.parquet]

Streams every pcap in the session (so session-wide state like client-id ownership persists),
builds per-host-window features (incl. causal flow stats), labels each
window from ground_truth.jsonl, and writes a per-session parquet. dataset.py concatenates these.
"""
import argparse
import itertools
import json
import re

import sys

import numpy as np
import pandas as pd

from .. import common
from . import mqtt, packets, windows
from ..labels import label as labelling


def _session_dir(session):
    return common.SESSIONS_DIR / session


def _pcap_paths(session_dir):
    pdir = session_dir / "pcap"
    return sorted(pdir.glob("*.pcap")) if pdir.exists() else []


def _bed_from_text(text):
    m = re.search(r"bed(\d+)", text or "")
    return int(m.group(1)) if m else None


def _spoof_tap(iterable, spoof_ivs, attacker_ip, attacker_mac, obs):
    """Single-pass observer (review N2): while packets stream to build_rows, record which bed the
    attacker actually PUBLISHed to (and the bed in its CONNECT client id) inside each spoof interval,
    to confirm the notes-derived victim against the PCAP. Cheap: only attacker MQTT packets touched."""
    for rec in iterable:
        if rec.get("mqtt") and (rec.get("ip_src") == attacker_ip or rec.get("eth_src") == attacker_mac):
            ts = rec["ts"]
            for iv in spoof_ivs:
                if iv["start"] <= ts <= iv["end"]:
                    o = obs.setdefault(iv["run_id"], {"beds": set(), "client_beds": set(),
                                                      "publishes": 0})
                    for m in rec["mqtt"]:
                        if m.get("type") == mqtt.PUBLISH and m.get("topic"):
                            b = _bed_from_text(m["topic"])
                            if b is not None:
                                o["beds"].add(b)
                            o["publishes"] += 1
                        elif m.get("type") == mqtt.CONNECT and m.get("client_id"):
                            b = _bed_from_text(m["client_id"])
                            if b is not None:
                                o["client_beds"].add(b)
        yield rec


def _confirm_spoof_victims(spoof_ivs, obs):
    """Compare the notes-parsed spoof victim bed with the bed(s) the attacker actually published to /
    connected as in the PCAP. Raises SystemExit on a genuine disagreement (review N2: stop and tell);
    records confirmed/unconfirmed status otherwise. Returns {run_id: {...}}."""
    result = {}
    for iv in spoof_ivs:
        notes_bed = labelling._spoof_bed_from_notes(iv["notes"])
        o = obs.get(iv["run_id"], {"beds": set(), "client_beds": set(), "publishes": 0})
        seen = o["beds"] | o["client_beds"]
        if notes_bed is not None and seen and seen != {notes_bed}:
            raise SystemExit(
                f"[N2] spoof victim disagreement in {iv['run_id']}: notes say bed{notes_bed}, "
                f"PCAP shows publish-topic beds {sorted(o['beds'])} / client-id beds "
                f"{sorted(o['client_beds'])}. Stopping — do not label or build until resolved.")
        result[iv["run_id"]] = {
            "notes_bed": notes_bed,
            "pcap_publish_beds": sorted(o["beds"]),
            "pcap_client_beds": sorted(o["client_beds"]),
            "pcap_publishes": o["publishes"],
            "confirmed": bool(seen) and notes_bed is not None and seen == {notes_bed},
        }
    return result


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

    spoof_ivs = [iv for iv in intervals if iv["class"] == "spoof_vitals"]
    spoof_obs = {}
    packet_iter = itertools.chain.from_iterable(packets.iter_packets(p) for p in pcaps)
    if spoof_ivs:
        packet_iter = _spoof_tap(packet_iter, spoof_ivs, attacker_ip, attacker_mac, spoof_obs)
    rows, conflict_windows = windows.build_rows(
        packet_iter, session, attacker_mac, attacker_ip, ip_roles)
    # Confirm the spoof victim against the PCAP (raises on disagreement, review N2).
    spoof_confirm = _confirm_spoof_victims(spoof_ivs, spoof_obs)

    # Flow features (nf_active_flows/nf_new_flows/nf_mean_flow_age) are now computed causally inside
    # windows.build_rows (review B1): per window, from packets seen up to the window end only — so
    # they are live-computable and never encode a flow's whole future lifetime. nfstream_flows.py is
    # retained as an optional offline diagnostic (not on the model path) and is not used here.

    df = pd.DataFrame(rows)
    df = labelling.apply_labels(df, intervals, cfg)
    df["dry_run"] = dry_run
    df = df.sort_values(["window_start_epoch", "host_mac"]).reset_index(drop=True)

    info = {
        "session": session,
        "dry_run": dry_run,
        "n_rows": len(df),
        "n_pcaps": len(pcaps),
        "n_runs": len(intervals),
        "attacker_mac": attacker_mac,
        "attacker_ip": attacker_ip,
        "conflict_windows": len(conflict_windows),
        "label_counts": df["label"].value_counts().to_dict(),
        "victim_by_class": (df[df["label"] == "victim"]["attack_class"]
                            .value_counts().to_dict()),
        "spoof_victim_confirm": spoof_confirm,
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

    print(json.dumps({**info, "out": out}, indent=2, default=str))
    # Sanity surfaced immediately: every non-dry run should land >=1 attack window.
    if not info["dry_run"]:
        missing = set(info["all_run_ids"]) - set(info["runs_with_window"])
        if missing:
            print(f"[WARN] runs with no attack window: {sorted(missing)}", file=sys.stderr)


if __name__ == "__main__":
    main()
