"""Turn ground-truth attack intervals into per-host-window labels.

Rule (./CLAUDE.md): a host-window is `attack` iff it lies fully inside a logged attack interval AND
it holds traffic involving the attacker (by IP, or by MAC for ARP spoofing — forwarded packets keep
the victims' IPs, so MAC is authoritative). The `touches_attacker` flag computed in windows.py
already folds in both the IP and MAC tests. Windows that straddle an interval boundary are ambiguous
in time and are marked `is_boundary` (dropped and counted downstream), never labelled attack/benign.
"""
import json

from .. import common

WINDOW = int(common.WINDOW_SECONDS)


def load_ground_truth(session_dir):
    """Return (intervals, attacker_ip, attacker_mac). Each interval: dict with class, run_id,
    start (epoch), end (epoch)."""
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
        })
    return intervals, attacker_ip, attacker_mac


def label_window(win_start, touches_attacker, intervals):
    """Return (label, attack_class, run_id, is_boundary) for one window start (epoch)."""
    w0, w1 = win_start, win_start + WINDOW
    boundary = False
    inside = None
    for iv in intervals:
        s, e = iv["start"], iv["end"]
        if s < w1 and e > w0:  # overlaps at all
            straddles = (w0 < s < w1) or (w0 < e < w1)
            fully_inside = (s <= w0) and (e >= w1)
            if straddles:
                boundary = True
            if fully_inside and inside is None:
                inside = iv
    if boundary:
        return "boundary", None, None, True
    if inside is not None and touches_attacker:
        return "attack", inside["class"], inside["run_id"], False
    return "benign", None, None, False


def apply_labels(df, intervals):
    """Fill label/attack_class/run_id/is_boundary columns on a feature DataFrame in place."""
    labels, classes, run_ids, boundaries = [], [], [], []
    for win, touches in zip(df["window_start_epoch"], df["touches_attacker"]):
        lab, cls, rid, bnd = label_window(int(win), bool(touches), intervals)
        labels.append(lab)
        classes.append(cls)
        run_ids.append(rid)
        boundaries.append(bnd)
    df["label"] = labels
    df["attack_class"] = classes
    df["run_id"] = run_ids
    df["is_boundary"] = boundaries
    return df
