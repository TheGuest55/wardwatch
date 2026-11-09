"""Turn ground-truth attack intervals into per-host-window labels.

Rule (./CLAUDE.md):
- A host-window is `attack` iff it lies fully inside a logged attack interval AND it holds traffic
  involving the attacker (by IP, or by MAC for ARP spoofing — forwarded packets keep the victims'
  IPs, so MAC is authoritative). The `touches_attacker` flag from windows.py folds in both tests.
- Review N2 (victim rows): inside a STOLEN-CREDENTIAL interval whose class impersonates/addresses a
  specific device — `spoof_vitals` (the impersonated monitor) and `pump_cmd` (the commanded pumps) —
  a fully-inside window belonging to that victim device that does NOT touch the attacker is labelled
  `victim` (its attack_class and run_id are kept as metadata). Victim rows are ambiguous training
  signal (the victim's own, still-normal traffic during its impersonation), so they are excluded from
  training and from binary metrics exactly like boundary rows, and counted per class in datacheck.
  `eavesdrop` has NO victim device: it is a passive wildcard SUBSCRIBE, so the nurse's (and every
  device's) own traffic is unchanged — there is no impersonated/addressed device whose rows turn
  ambiguous, so no victim label is applied for it.
- Windows straddling an interval boundary are `boundary` (dropped and counted), never labelled.

Victim identity is derived from ground truth + topology, never guessed: pump_cmd from the structured
`params.commands[].bed`; spoof_vitals from the bed named in `notes` (confirmed against the PCAP topic
in extract.py — see confirm_spoof_victims). Future runs also carry a structured `victims` field.
"""
import json
import re

from .. import common

WINDOW = int(common.WINDOW_SECONDS)

VICTIM_CLASSES = ("spoof_vitals", "pump_cmd")   # eavesdrop intentionally excluded (passive read)
_KIND_OFFSET = {"monitor": 1, "pump": 2, "wearable": 3}


def load_ground_truth(session_dir):
    """Return (intervals, attacker_ip, attacker_mac). Each interval: class, run_id, start/end (epoch),
    and the raw params/notes (used to derive victim identity)."""
    path = session_dir / "ground_truth.jsonl"
    intervals, attacker_ip, attacker_mac = [], None, None
    if not path.exists():
        return intervals, attacker_ip, attacker_mac
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        attacker_ip = attacker_ip or r.get("src_ip")
        attacker_mac = attacker_mac or r.get("src_mac")
        intervals.append({
            "class": r["class"],
            "run_id": r.get("run_id", ""),
            "start": common.iso_to_epoch(r["start_utc"]),
            "end": common.iso_to_epoch(r["end_utc"]),
            "params": r.get("params", {}) or {},
            "notes": r.get("notes", "") or "",
            "victims": r.get("victims"),   # structured {ip,mac} list on future runs; else None
        })
    return intervals, attacker_ip, attacker_mac


def _device_ip(cfg, bed, kind):
    prefix = cfg["network"]["subnet"].rsplit(".", 1)[0] + "."
    return f"{prefix}{10 * int(bed) + _KIND_OFFSET[kind]}"


def _spoof_bed_from_notes(notes):
    """Bed the spoof targeted, parsed from the note 'to bedN as bedN-monitor'. None if absent."""
    m = re.search(r"to bed(\d+)", notes or "")
    return int(m.group(1)) if m else None


def victim_ips_for_interval(iv, cfg):
    """Set of victim device IPs for one interval (empty unless a victim class). Prefers a structured
    `victims` field (future runs); else derives from params/notes (review N2)."""
    if iv["class"] not in VICTIM_CLASSES:
        return set()
    if iv.get("victims"):
        return {v["ip"] for v in iv["victims"] if v.get("ip")}
    if iv["class"] == "pump_cmd":
        beds = {c["bed"] for c in iv["params"].get("commands", []) if "bed" in c}
        return {_device_ip(cfg, b, "pump") for b in beds}
    if iv["class"] == "spoof_vitals":
        bed = _spoof_bed_from_notes(iv["notes"])
        return {_device_ip(cfg, bed, "monitor")} if bed is not None else set()
    return set()


def label_window(win_start, touches_attacker, host_ip, intervals, victim_map):
    """Return (label, attack_class, run_id, is_boundary) for one host-window.

    victim_map: {id(interval): set(victim_ips)} precomputed once by apply_labels."""
    w0, w1 = win_start, win_start + WINDOW
    boundary = False
    inside = None
    victim_iv = None
    for iv in intervals:
        s, e = iv["start"], iv["end"]
        if s < w1 and e > w0:                      # overlaps at all
            if (w0 < s < w1) or (w0 < e < w1):     # straddles a boundary
                boundary = True
            if (s <= w0) and (e >= w1):            # fully inside this interval
                if inside is None:
                    inside = iv
                if victim_iv is None and host_ip in victim_map.get(id(iv), ()):
                    victim_iv = iv
    if boundary:
        return "boundary", None, None, True
    if inside is not None and touches_attacker:
        return "attack", inside["class"], inside["run_id"], False
    if victim_iv is not None:                       # victim device, inside, not touching attacker
        return "victim", victim_iv["class"], victim_iv["run_id"], False
    return "benign", None, None, False


def apply_labels(df, intervals, cfg):
    """Fill label/attack_class/run_id/is_boundary on a feature DataFrame in place (review N2 adds the
    `victim` label, which needs host_ip and the topology cfg)."""
    victim_map = {id(iv): victim_ips_for_interval(iv, cfg) for iv in intervals}
    labels, classes, run_ids, boundaries = [], [], [], []
    host_ips = df["host_ip"] if "host_ip" in df.columns else [None] * len(df)
    for win, touches, hip in zip(df["window_start_epoch"], df["touches_attacker"], host_ips):
        lab, cls, rid, bnd = label_window(int(win), bool(touches), hip, intervals, victim_map)
        labels.append(lab)
        classes.append(cls)
        run_ids.append(rid)
        boundaries.append(bnd)
    df["label"] = labels
    df["attack_class"] = classes
    df["run_id"] = run_ids
    df["is_boundary"] = boundaries
    return df
