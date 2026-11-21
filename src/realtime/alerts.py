"""The alert stream: one JSON line per alerting window and per incident open / update / close, appended while a replay runs.

    {"header": {...}}                                   first line: mode, session, speed, git commit, image digest, started_utc
    {"ts", "event": "alert", "session", "host_mac", "host_ip", "verdict", "cell", "class", "p", "incident_id"}
    {"ts", "event": "incident_open" | "incident_update" | "incident_close", "session", "incident_id", "verdict", "cell",
     "n_alerts", "first_seen", "last_seen", ...}

`ts` is the END of the alerting 5 s window (UTC, ISO). The sink is called AFTER the timed scoring path of a tick, so Experiment E's
latency does not include it, and it only runs when a sink is passed (default None: the replay writes nothing extra).
It holds credentials of nothing: no payloads, no PCAP slices, no topics, only what the dashboard shows.
"""
import json
import os
import time
from pathlib import Path

from .. import common
from ..triage import incidents as inc_mod
from ..triage import verdict as V

W = int(common.WINDOW_SECONDS)


def header(mode, session, speed, **extra):
    """The first line of a stream. git commit and image digest come from the provenance environment the run scripts set."""
    h = {"mode": mode, "session": session, "speed": speed,
         "git_commit": os.environ.get("WARDWATCH_GIT_COMMIT") or None,
         "image_digest": os.environ.get("WARDWATCH_IMAGE_DIGEST") or None,
         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    h.update(extra)
    return h


class AlertSink:
    def __init__(self, path, head):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8")          # truncates: one replay per file
        self.seen, self.n_closed, self.last = set(), 0, {}
        self._write({"header": head})

    def _write(self, obj):
        self.f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        self.f.flush()

    def _incident_of(self, tr, mac, win, src, tgt):
        key = (src,) if tr.tracker.key == "source" else (src, tgt)
        inc = tr.tracker.open.get(key)
        if inc is not None and inc.alerts and inc.alerts[-1].host_mac == mac and inc.alerts[-1].window_start == win:
            return inc.incident_id
        for inc in tr.tracker.open.values():                  # fallbacks: another open incident, then an already closed one
            if any(a.host_mac == mac and a.window_start == win for a in inc.alerts):
                return inc.incident_id
        for iid, mem in tr.members.items():
            if (mac, win) in mem:
                return iid
        return None

    def tick(self, tr, win, macs, ips, res, peers):
        """After tr.window(...) for one window: the alerts, then incident open / update / close events."""
        ts = inc_mod.iso(win + W)
        touched = []
        order = sorted(range(len(macs)), key=lambda i: macs[i])
        for i in order:
            c = res["cell"][i]
            if not V.is_alerting(c):
                continue
            src, tgt = inc_mod.attribute(macs[i], peers.get((macs[i], win), {}))
            iid = self._incident_of(tr, macs[i], win, src, tgt)
            self._write({"ts": ts, "event": "alert", "session": tr.session, "host_mac": macs[i], "host_ip": ips[i],
                         "verdict": V.verdict_of(c), "cell": c, "class": res["named"][i] if c == "known" else None,
                         "p": float(res["p"][i]), "incident_id": iid})
            if iid is not None and iid not in touched:
                touched.append(iid)
        for iid in touched:
            inc = next((x for x in tr.tracker.open.values() if x.incident_id == iid), None)
            if inc is None:
                continue                                        # closed in the same tick: its close event carries the final state
            cell = inc.cell
            ev = "incident_update" if iid in self.seen else "incident_open"
            self.seen.add(iid)
            self._write({"ts": ts, "event": ev, "session": tr.session, "incident_id": iid, "verdict": V.verdict_of(cell),
                         "cell": cell, "n_alerts": len(inc.alerts), "first_seen": inc_mod.iso(inc.first),
                         "last_seen": inc_mod.iso(inc.last)})
        self._closed(tr, ts)

    def _closed(self, tr, ts):
        for rec in tr.records[self.n_closed:]:
            self._write({"ts": ts, "event": "incident_close", "session": rec["session"], "incident_id": rec["incident_id"],
                         "verdict": rec["verdict"], "cell": rec["cell"], "n_alerts": rec["n_alerts"],
                         "first_seen": rec["first_seen"], "last_seen": rec["last_seen"], "attack_class": rec["attack_class"],
                         "target_host": rec["target_host"], "source": rec["source"]})
        self.n_closed = len(tr.records)

    def finish(self, tr):
        """After tr.finish(): the incidents flushed at the end of the replay, then close the file."""
        last = max((r["last_seen"] for r in tr.records), default=None)
        self._closed(tr, last or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.f.close()
