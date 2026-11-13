"""Run the SAME host-window extractor on CICIoMT2024 Wi-Fi/MQTT PCAPs (Experiment C).

    python -m src.features.extract_cic [--max-packets 3000000] [--force]

Input : data/raw/ciciomt2024/{train,test}/*.pcap   (CIC's official split, one class per file)
Output: data/processed/cic/sessions/<file>.parquet  (per-file cache)
        data/processed/cic/cic_windows.parquet      (all files, same columns as the testbed table)
        data/processed/cic/feature_list.json        (same features; testbed-only ones flagged)
        data/processed/cic/extract_report.json      (per-file counts — goes in REPORT_NOTES)

Discipline (./CLAUDE.md):
  * One extractor for both datasets: rows come from windows.build_rows, unchanged. Nothing here
    computes a feature. Payload-plausibility features need our JSON vitals, so they come out as
    zeros on CIC and are flagged testbed_only (Experiment C drops them).
  * Labels follow the TESTBED rule as closely as CIC allows: in an attack file, a host-window is
    `attack` iff it holds traffic from/to the attacker's MAC or IP (build_rows' touches_attacker,
    the same test the testbed labeller uses); every other row in that file is ordinary bystander
    traffic and is `benign`. Every row in a Benign file is `benign` (CIC's per-file label). The whole
    attack file stands in for the "logged interval" — CIC gives no per-run timing — so any
    attacker-touching window in an attack file counts, which is coarser than the testbed. Nothing is
    marked boundary/victim.
  * The attacker is not documented per file by CIC; it was identified from the traffic and is
    VERIFIED per attack file before labelling (see ATTACKER and _verify_attacker). Disagreement stops
    the build rather than guessing.
  * LABEL_MAP is the explicit CIC-class -> testbed-class map. Classes with no counterpart map to
    None and are excluded from Experiment C.
  * Split: CIC train files -> train/val by TIME within each file (first 75% of its windows train,
    last 25% val — never shuffled); CIC test files -> test.
  * --max-packets caps the packets read per file (floods are tens of millions of packets). A capped
    file is a prefix in capture order, recorded in the report; it is never a random sample.
"""
import argparse
import itertools
import json
import re
import time
from pathlib import Path

import pandas as pd

from .. import common
from . import mqtt, packets, windows

RAW_DIR = common.ROOT / "data" / "raw" / "ciciomt2024"
LABEL_RULE = "touches-attacker-v3"   # bump to invalidate cached per-file extractions
OUT_DIR = common.PROCESSED_DIR / "cic"
VAL_FRACTION = 0.25

# (filename regex, CIC class, testbed class or None). First match wins. File names follow
# CIC's layout: <Class>[<part>]_{train,test}.pcap.
LABEL_MAP = [
    (r"^Benign",                    "Benign",                  "benign"),
    (r"^ARP_Spoofing",              "ARP_Spoofing",            "arp_mitm"),
    (r"^Recon-Port_Scan",           "Recon-Port_Scan",         "recon_scan"),
    (r"^Recon-OS_Scan",             "Recon-OS_Scan",           "recon_scan"),
    (r"^Recon-VulScan",             "Recon-VulScan",           "recon_scan"),
    (r"^Recon-Ping_Sweep",          "Recon-Ping_Sweep",        None),   # ICMP sweep: no TCP scan analogue
    (r"^TCP_IP-DoS-SYN",            "TCP_IP-DoS-SYN",          "syn_flood"),
    (r"^TCP_IP-DDoS-SYN",           "TCP_IP-DDoS-SYN",         "syn_flood"),
    (r"^MQTT-DoS-Connect_Flood",    "MQTT-DoS-Connect_Flood",  "mqtt_flood"),
    (r"^MQTT-DoS-Publish_Flood",    "MQTT-DoS-Publish_Flood",  "mqtt_flood"),
    (r"^MQTT-DDoS-Connect_Flood",   "MQTT-DDoS-Connect_Flood", "mqtt_flood"),
    (r"^MQTT-DDoS-Publish_Flood",   "MQTT-DDoS-Publish_Flood", "mqtt_flood"),
    (r"^MQTT-Malformed_Data",       "MQTT-Malformed_Data",     None),
    (r"^TCP_IP-D?DoS-(ICMP|UDP|TCP)", "TCP_IP-flood-other",    None),
]
# The CIC attacker, identified from the PCAPs (2026-11-10, logged in REPORT_NOTES):
#   ARP_Spoofing  : the only MAC claiming the gateway IP 192.168.137.1 (owned by 8c:ae:4c:dd:1e:39);
#                   that conflict never occurs in the Benign captures.
#   Recon-Port_Scan: top sender, ~34k distinct destination ports per window.
#   TCP_IP-DoS-SYN : sends ~1.1M SYNs in the 3M-packet prefix.
# Same host in every attack file checked. It is also present (normally) in the Benign captures.
ATTACKER = {"mac": "dc:a6:32:dc:27:d5", "ip": "192.168.137.48"}
# Benign files: pass a sentinel so build_rows marks nothing as attacker-touching. A non-None
# value keeps its test (`rec.get("ip_src") == attacker_ip`) from matching non-IP packets.
_NO_ATTACKER = "-"


# Legitimate IP->MAC bindings, taken from the Benign captures (where nothing is spoofed): the
# gateway 192.168.137.1 is answered for by the router. Order-independent ground truth for the ARP
# check — "whoever answered first" fails when the spoofer happens to answer before the router.
# Re-verified on every Benign file at extraction (_check_bindings); a contradiction stops the build.
LEGIT_BINDINGS = {"192.168.137.1": "8c:ae:4c:dd:1e:39"}


def _sender_tap(iterable, obs):
    """Single pass, alongside build_rows: per SOURCE MAC, count the behaviour each attack class is
    made of. Window features can't do this — they count a packet for both its sender and its
    receiver (review M4), so a flood's victim scores as high as its attacker."""
    for rec in iterable:
        src = rec["eth_src"]
        o = obs.setdefault(src, {"pure_syn": 0, "dports": set(), "mqtt_connect": 0,
                                 "mqtt_publish": 0, "arp_claim_conflicts": 0})
        if rec["l4"] == "tcp" and rec.get("syn") and not rec.get("ack"):
            o["pure_syn"] += 1
        if rec["l4"] in ("tcp", "udp") and rec.get("dport") and len(o["dports"]) < 70000:
            o["dports"].add(rec["dport"])
        for m in rec.get("mqtt") or ():
            if m.get("type") == mqtt.CONNECT:
                o["mqtt_connect"] += 1
            elif m.get("type") == mqtt.PUBLISH:
                o["mqtt_publish"] += 1
        if rec["l3"] == "arp" and rec.get("arp_op") == 2:
            owner = LEGIT_BINDINGS.get(rec.get("arp_psrc"))
            if owner and rec.get("arp_hwsrc") != owner:   # answers for an IP it does not own
                o["arp_claim_conflicts"] += 1
        yield rec


SIGNATURE = {"ARP_Spoofing": "arp_claim_conflicts",
             "Recon-Port_Scan": "dports", "Recon-OS_Scan": "dports", "Recon-VulScan": "dports",
             "TCP_IP-DoS-SYN": "pure_syn", "TCP_IP-DDoS-SYN": "pure_syn",
             "MQTT-DoS-Connect_Flood": "mqtt_connect", "MQTT-DDoS-Connect_Flood": "mqtt_connect",
             "MQTT-DoS-Publish_Flood": "mqtt_publish", "MQTT-DDoS-Publish_Flood": "mqtt_publish"}


def _verify_attacker(obs, stem, cic_cls):
    """Stop unless ATTACKER is the top SENDER of the class's signature behaviour in this file (see
    _sender_tap). Returns a dict for the report."""
    mac, sig = ATTACKER["mac"], SIGNATURE.get(cic_cls)
    if sig is None:
        raise SystemExit(f"[cic] no attacker signature for class {cic_cls}: add one deliberately")
    val = {m: (len(o[sig]) if sig == "dports" else o[sig]) for m, o in obs.items()}
    if mac not in val:
        raise SystemExit(f"[cic] attacker {mac} sent nothing in {stem}: identify it before labelling")
    ranked = sorted(val.items(), key=lambda kv: -kv[1])
    if ranked[0][0] != mac or ranked[0][1] == 0:
        raise SystemExit(f"[cic] {stem}: top sender by {sig} is {ranked[0]}, not attacker {mac}. "
                         f"Stopping.")
    return {"attacker_mac": mac, "check": sig, "attacker_value": int(ranked[0][1]),
            "runner_up": [ranked[1][0], int(ranked[1][1])] if len(ranked) > 1 else None}


def classify(stem):
    for pat, cic_cls, tb_cls in LABEL_MAP:
        if re.search(pat, stem):
            return cic_cls, tb_cls
    raise SystemExit(f"no LABEL_MAP entry for CIC file {stem!r}: add one deliberately")


def extract_file(path, cic_split, max_packets):
    stem = path.stem
    cic_cls, tb_cls = classify(stem)
    it = packets.iter_packets(path)
    counter = {"n": 0}

    def counted(src):
        for rec in src:
            counter["n"] += 1
            yield rec

    capped = itertools.islice(it, max_packets) if max_packets else it
    is_benign = tb_cls == "benign"
    a_mac, a_ip = (_NO_ATTACKER, _NO_ATTACKER) if is_benign else (ATTACKER["mac"], ATTACKER["ip"])
    t0 = time.perf_counter()
    obs = {}
    stream = _sender_tap(counted(capped), obs)   # benign too: re-checks LEGIT_BINDINGS
    rows, _ = windows.build_rows(stream, stem, a_mac, a_ip, {})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"{path}: no rows extracted")
    df = df.sort_values(["window_start_epoch", "host_mac"]).reset_index(drop=True)

    if is_benign:
        bad = {m: o["arp_claim_conflicts"] for m, o in obs.items() if o["arp_claim_conflicts"]}
        if bad:
            raise SystemExit(f"[cic] {stem}: LEGIT_BINDINGS contradicted in a Benign capture by "
                             f"{bad}. Stopping — the gateway binding is not what we assumed.")
        verify = {"legit_bindings_hold": True, "bindings": LEGIT_BINDINGS}
    else:
        verify = _verify_attacker(obs, stem, cic_cls)
    is_attack = (~pd.Series(is_benign, index=df.index)) & df["touches_attacker"].astype(bool)
    df["label"] = is_attack.map({True: "attack", False: "benign"})
    df["attack_class"] = [(tb_cls or "unmapped") if a else None for a in is_attack]
    df["cic_class"] = cic_cls
    df["run_id"] = stem
    df["is_boundary"] = False
    df["dry_run"] = False
    df["host_role"] = df["host_role"].where(df["host_mac"] != ATTACKER["mac"], "attacker")

    if cic_split == "test":
        df["split"] = "test"
    else:                                   # time-based train/val inside the file
        wins = sorted(df["window_start_epoch"].unique())
        cut = wins[int(len(wins) * (1 - VAL_FRACTION))] if len(wins) > 3 else float("inf")
        df["split"] = (df["window_start_epoch"] >= cut).map({True: "val", False: "train"})

    info = {
        "file": path.name, "cic_split": cic_split, "cic_class": cic_cls, "testbed_class": tb_cls,
        "packets_read": counter["n"], "capped": bool(max_packets and counter["n"] >= max_packets),
        "rows": int(len(df)), "hosts": int(df["host_mac"].nunique()),
        "attack_rows": int(is_attack.sum()), "attacker_verify": verify,
        "windows": int(df["window_start_epoch"].nunique()),
        "seconds_covered": float(df["window_start_epoch"].max() - df["window_start_epoch"].min()
                                 + common.WINDOW_SECONDS),
        "split_rows": df["split"].value_counts().to_dict(),
        "rows_with_mqtt": int((df["mqtt_pkts"] > 0).sum()),
        "extract_s": round(time.perf_counter() - t0, 1),
    }
    return df, info


def _extract_job(path, cic_split, max_packets, sess_dir):
    """Worker: extract one file, write its parquet, return its report entry."""
    df, info = extract_file(path, cic_split, max_packets)
    df.to_parquet(Path(sess_dir) / f"{path.stem}.parquet", index=False)
    info.update({"max_packets": max_packets, "pcap_bytes": path.stat().st_size,
                 "label_rule": LABEL_RULE})
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-packets", type=int, default=3_000_000,
                    help="read at most this many packets per file, in capture order (0 = all)")
    ap.add_argument("--workers", type=int, default=4, help="files extracted in parallel")
    ap.add_argument("--force", action="store_true", help="re-extract files even if cached")
    args = ap.parse_args()

    sess_dir = OUT_DIR / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "extract_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    jobs, todo = [], []
    for cic_split in ("train", "test"):
        for path in sorted((RAW_DIR / cic_split).glob("*.pcap")):
            jobs.append(path)
            key = path.name
            fresh = ((sess_dir / f"{path.stem}.parquet").exists() and not args.force
                     and key in report
                     and report[key].get("max_packets") == args.max_packets
                     and report[key].get("label_rule") == LABEL_RULE
                     and report[key].get("pcap_bytes") == path.stat().st_size)
            if not fresh:
                todo.append((path, cic_split, args.max_packets, str(sess_dir)))
    if not jobs:
        raise SystemExit(f"no pcaps under {RAW_DIR}/{{train,test}}")

    # Files are independent, so extract them in parallel (largest first). Each worker writes its
    # own parquet; only this parent process writes the report, so there is no write race.
    todo.sort(key=lambda j: -j[0].stat().st_size)
    if todo:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(todo)))) as pool:
            futs = {pool.submit(_extract_job, *j): j[0] for j in todo}
            for fut in as_completed(futs):
                info = fut.result()          # a SystemExit in a worker re-raises here and stops
                report[info["file"]] = info
                report_path.write_text(json.dumps(report, indent=2))
                print(f"[cic] extracted {info['file']}: {info['packets_read']} pkts"
                      f"{' (CAPPED)' if info['capped'] else ''} -> {info['rows']} rows "
                      f"({info['attack_rows']} attack), {info['hosts']} hosts, "
                      f"{info['seconds_covered']:.0f}s, {info['extract_s']}s  "
                      f"[{info['cic_class']} -> {info['testbed_class']}]", flush=True)
    frames = [pd.read_parquet(sess_dir / f"{p.stem}.parquet") for p in jobs]

    df = pd.concat(frames, ignore_index=True)
    extra = set(df.columns) - set(common.FEATURES) - set(common.META_COLUMNS) - {
        "touches_attacker", "dry_run", "cic_class"}
    missing = set(common.FEATURES) - set(df.columns)
    if extra or missing:
        raise SystemExit(f"column contract violated: unexpected {sorted(extra)}, missing {sorted(missing)}")
    df.to_parquet(OUT_DIR / "cic_windows.parquet", index=False)
    feature_list = {
        "n_features": len(common.FEATURES), "features": list(common.FEATURES),
        "metadata": [c for c in df.columns if c not in common.FEATURES],
        "testbed_only": list(common.TESTBED_ONLY_FEATURES),
        "label_map": [{"pattern": p, "cic_class": c, "testbed_class": t} for p, c, t in LABEL_MAP],
    }
    (OUT_DIR / "feature_list.json").write_text(json.dumps(feature_list, indent=2))

    summary = (df.assign(cls=df["attack_class"].fillna("benign"))
               .groupby(["cls", "split"]).size().unstack(fill_value=0))
    print(f"\n[cic] {len(df)} rows from {len(frames)} files -> {OUT_DIR / 'cic_windows.parquet'}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
