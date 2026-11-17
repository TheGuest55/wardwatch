"""Alerts -> incidents. Shared by the offline triage and the live replay, so both give identical incidents.

Attribution (per alerting host window H; uses packet COUNTS only, never roles or labels):
  1. P = H's unicast peer with the most packets (sent + received) in that window (ties: lowest MAC).
  2. If H sent more packets to P than it received from P: source = H, target = P; otherwise source = P, target = H.
  3. No unicast peer (broadcast only): source = H, target = broadcast.
A MAC is shown as "<ip> (<role>)", the IP being the one the stream first saw that MAC use and the role the
config's role for that IP. host_role / IP / MAC text is DISPLAY-ONLY: no decision below reads it.

Grouping: an alert joins the open incident with the same KEY whose last alert window ended no more than `gap`
seconds before the alert's window starts. KEY is "source" (the source MAC) or "source+target". Both use the
target device, the source and a time gap; the source key keeps a scan together and lists its targets in `scope`.
The key and the gap are chosen on the validation sessions (src/triage/offline.py), never on test.

Incident record (data/processed/v2/<live|triage>/incidents*.jsonl, one JSON per line): the dashboard's format
(docs/dashboard-notes.md on the dashboard branch) plus `cell`, `scope` and `session`:
  incident_id, session, target_host, source, first_seen, last_seen (ISO UTC; window starts of the first and last
  alert), verdict (one of the six), cell, n_alerts, calibrated_p (max), attack_class (Known attack windows only),
  scope [every host shown], timeline [{t, event}], top_features [{feature, text, delta}] (TreeSHAP of the
  max-p window), pcap_slice (path relative to the data root), pcap_sha256.
incident_id = <session>-<source role>-<first_seen epoch>[-n for a repeat]: deterministic, so offline and live agree.
"""
import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .. import common
from . import verdict as V

BROADCAST = "ff:ff:ff:ff:ff:ff"
W = int(common.WINDOW_SECONDS)
KEYS = ("source", "source+target")


def iso(epoch):
    return common.epoch_to_iso(int(epoch))


def attribute(host_mac, peer_pkts):
    """(source_mac, target_mac) for an alerting host window. peer_pkts: {peer_mac: [sent, recv]} (unicast only)."""
    if not peer_pkts:
        return host_mac, BROADCAST
    peer = sorted(peer_pkts.items(), key=lambda kv: (-(kv[1][0] + kv[1][1]), kv[0]))[0][0]
    sent, recv = peer_pkts[peer]
    return (host_mac, peer) if sent > recv else (peer, host_mac)


class HostNames:
    """MAC -> '<ip> (<role>)' for display. The IP is the first host_ip the stream showed for that MAC; the role
    comes from the config's IP -> role map. Display only."""

    def __init__(self, ip_roles):
        self.ip_roles, self.ip = ip_roles, {}

    def see(self, mac, ip):
        if ip and mac not in self.ip:
            self.ip[mac] = ip

    def name(self, mac):
        if mac == BROADCAST:
            return "broadcast"
        ip = self.ip.get(mac)
        return f"{ip} ({self.ip_roles.get(ip, 'unknown')})" if ip else mac

    def role(self, mac):
        return self.ip_roles.get(self.ip.get(mac), "unknown") if mac != BROADCAST else "broadcast"


@dataclass
class Alert:
    session: str
    host_mac: str
    window_start: int
    p: float                 # calibrated attack probability
    band: int
    score_if: float
    cell: str
    named: str = None        # multiclass name (Known attack only)
    source: str = None       # MAC
    target: str = None       # MAC
    feat: np.ndarray = None  # raw feature row (for the post-alert explanation)


@dataclass
class _Open:
    key: tuple
    session: str
    incident_id: str
    first: int
    last: int
    alerts: list = field(default_factory=list)
    cell: str = None
    timeline: list = field(default_factory=list)


class IncidentTracker:
    """Feed alerts in (window_start, host_mac) order; call close_idle(window_end) after each window and flush()
    at the end. finalize(open_incident) -> record is injected so the offline and live paths share one tracker."""

    def __init__(self, session, key="source", gap=30, names=None, finalize=None):
        if key not in KEYS:
            raise ValueError(key)
        self.session, self.key, self.gap, self.names = session, key, float(gap), names
        self.finalize = finalize or (lambda inc: inc)
        self.open, self.done, self._ids = {}, [], Counter()

    def _key(self, a):
        return (a.source,) if self.key == "source" else (a.source, a.target)

    def _role(self, mac):
        return self.names.role(mac) if self.names else "host"

    def update(self, a):
        k = self._key(a)
        inc = self.open.get(k)
        if inc is not None and a.window_start - (inc.last + W) > self.gap:
            self._close(k)
            inc = None
        if inc is None:
            base = f"{a.session}-{self._role(a.source)}-{a.window_start}"
            self._ids[base] += 1
            iid = base if self._ids[base] == 1 else f"{base}-{self._ids[base]}"
            inc = self.open[k] = _Open(k, a.session, iid, a.window_start, a.window_start)
            inc.cell = a.cell
            inc.timeline.append((a.window_start, f"first alert: {V.verdict_of(a.cell)} ({a.cell})"))
        else:
            if V.SEVERITY[a.cell] < V.SEVERITY[inc.cell]:
                inc.cell = a.cell
                inc.timeline.append((a.window_start, f"verdict escalated to {V.verdict_of(a.cell)} ({a.cell})"))
        inc.last = a.window_start
        inc.alerts.append(a)

    def _close(self, k):
        inc = self.open.pop(k)
        self.done.append(self.finalize(inc))

    def close_idle(self, window_end):
        """Close incidents that no alert at or after window_end can join any more."""
        for k in [k for k, inc in self.open.items() if window_end - (inc.last + W) > self.gap]:
            self._close(k)

    def flush(self):
        for k in sorted(self.open, key=lambda k: self.open[k].first):
            self._close(k)
        return self.done


def _summary(inc):
    A = inc.alerts
    src = Counter(a.source for a in A).most_common()
    source = sorted(m for m, c in src if c == src[0][1])[0]
    tg = Counter(a.target for a in A if a.source == source).most_common()
    target = sorted(m for m, c in tg if c == tg[0][1])[0] if tg else source
    top = max(A, key=lambda a: (a.p, -a.window_start, a.host_mac))
    return source, target, top


def build_record(inc, names):
    """The incident record without the post-alert fields (top_features, pcap_slice, pcap_sha256), which
    complete_record fills in. Everything here is cheap and needs no model or pcap."""
    A = inc.alerts
    source, target, _ = _summary(inc)
    cell = V.ALERTING_CELLS[min(V.SEVERITY[a.cell] for a in A)]
    named = Counter(a.named for a in A if a.cell == "known" and a.named).most_common()
    attack_class = sorted(c for c, n in named if n == named[0][1])[0] if named else None
    scope = sorted({names.name(m) for a in A for m in (a.host_mac, a.source, a.target)})
    return {"incident_id": inc.incident_id, "session": inc.session,
            "target_host": names.name(target), "source": names.name(source),
            "first_seen": iso(inc.first), "last_seen": iso(inc.last), "verdict": V.verdict_of(cell), "cell": cell,
            "n_alerts": len(A), "calibrated_p": float(max(a.p for a in A)), "attack_class": attack_class,
            "scope": scope,
            "timeline": [{"t": iso(t), "event": ev} for t, ev in inc.timeline]
            + ([{"t": iso(inc.last), "event": f"last alert ({len(A)} alerts)"}] if len(A) > 1 else []),
            "top_features": [], "pcap_slice": None, "pcap_sha256": None}


def complete_record(rec, inc, explain_fn=None, slice_fn=None):
    """Post-alert work (TreeSHAP of the max-p window, evidence slice). explain_fn(feat) -> top_features;
    slice_fn(first, last_end, [source_mac, target_mac], incident_id) -> {"path", "sha256"}."""
    source, target, top = _summary(inc)
    if explain_fn:
        rec["top_features"] = explain_fn(top.feat)
    if slice_fn:
        s = slice_fn(inc.first, inc.last + W, [source, target], inc.incident_id)
        rec["pcap_slice"], rec["pcap_sha256"] = s["path"], s["sha256"]
    return rec


def write_jsonl(records, path):
    from pathlib import Path
    import os
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n"
                           for r in sorted(records, key=lambda r: (r["first_seen"], r["incident_id"]))))
    os.replace(tmp, p)
