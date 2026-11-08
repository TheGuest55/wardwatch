"""Audit that attacker payloads overlap benign traffic on BOTH total size and per-field value
representation (review B2 dry-run proof). Runs in the analysis image:

    python scripts/audit_payloads.py <session>

For monitor/vitals (benign real monitors vs the spoof_vitals attacker) and pump/cmd (benign nurse vs
the pump_cmd attacker) it reports the serialised-length range for each side (must overlap) and the
per-field numeric representation (hr/spo2 must be float on both sides; bp/rate int) — so no single
field or threshold separates attack from benign.
"""
import collections
import json
import sys

from src import common
from src.features import packets


def _repr_tag(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, float):
        return "float"
    if isinstance(v, int):
        return "int"
    return type(v).__name__


def audit(session):
    cfg = common.load_config()
    attacker_ip = cfg["attacker"]["ip"]
    pdir = common.SESSIONS_DIR / session / "pcap"
    pcaps = sorted(pdir.glob("*.pcap"))

    vit_len = {"attacker": [], "benign": []}
    cmd_len = {"attacker": [], "benign": []}
    tokens = collections.defaultdict(collections.Counter)   # (side, field) -> repr counts
    connect_ids = collections.Counter()

    for p in pcaps:
        for rec in packets.iter_packets(p):
            for m in rec.get("mqtt", []):
                tn = m.get("type_name")
                if tn == "connect":
                    cid = m.get("client_id")
                    if cid:
                        connect_ids[cid] += 1
                    continue
                if tn != "publish":
                    continue
                topic = m.get("topic") or ""
                ap = m.get("app_payload") or b""
                side = "attacker" if rec.get("ip_src") == attacker_ip else "benign"
                if topic.endswith("monitor/vitals"):
                    vit_len[side].append(len(ap))
                    try:
                        obj = json.loads(ap.decode())
                        for f in ("hr", "spo2", "bp_sys", "bp_dia"):
                            if f in obj:
                                tokens[(side, f)][_repr_tag(obj[f])] += 1
                    except Exception:
                        pass
                elif topic.endswith("pump/cmd"):
                    cmd_len[side].append(len(ap))

    def rng(xs):
        return f"n={len(xs)} min={min(xs)} max={max(xs)} median={sorted(xs)[len(xs)//2]}" if xs else "n=0"

    def overlaps(a, b):
        return bool(a) and bool(b) and max(min(a), min(b)) <= min(max(a), max(b))

    print(f"== payload audit: {session} (attacker_ip={attacker_ip}) ==")
    print("monitor/vitals  attacker:", rng(vit_len["attacker"]))
    print("monitor/vitals  benign  :", rng(vit_len["benign"]))
    print("  length ranges overlap :", overlaps(vit_len["attacker"], vit_len["benign"]))
    print("pump/cmd        attacker:", rng(cmd_len["attacker"]))
    print("pump/cmd        benign  :", rng(cmd_len["benign"]))
    print("  length ranges overlap :", overlaps(cmd_len["attacker"], cmd_len["benign"]))
    print("per-field representation (monitor/vitals):")
    for f in ("hr", "spo2", "bp_sys", "bp_dia"):
        print(f"  {f:7} attacker={dict(tokens[('attacker', f)])}  benign={dict(tokens[('benign', f)])}")
    print("CONNECT client-ids seen (parser sanity):", dict(connect_ids))


if __name__ == "__main__":
    audit(sys.argv[1])
