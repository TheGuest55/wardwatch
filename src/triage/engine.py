"""Per-window triage driver shared by the offline run and the live replay.

Triager.window(...) takes the scored rows of ONE 5 s window and (1) records each row's cell, (2) turns the
alerting rows into attributed Alerts, (3) updates the incident tracker, (4) closes incidents no later alert can
join. finish() flushes the rest. Identical calls give identical incidents, which is how E is checked against the
offline run.

Post-alert work (TreeSHAP, the evidence slice) is injected as `post(record, incident)`; the offline run does it
inline, the live replay hands it to a worker thread so the scoring path never waits for pcap IO.
"""
import numpy as np

from . import incidents as inc_mod
from . import verdict as V


class Triager:
    def __init__(self, session, ip_roles, key="source", gap=30, post=None):
        self.session = session
        self.names = inc_mod.HostNames(ip_roles)
        self.records = []          # incident records, in closing order
        self.members = {}          # incident_id -> [(host_mac, window_start)] of its alerts
        self.cell_log = []         # (host_mac, window_start, cell, p, band, if_score) for every row seen
        self.post = post
        self.tracker = inc_mod.IncidentTracker(session, key, gap, self.names, finalize=self._finalize)

    def _finalize(self, inc):
        rec = inc_mod.build_record(inc, self.names)
        self.members[rec["incident_id"]] = [(a.host_mac, a.window_start) for a in inc.alerts]
        self.records.append(rec)
        if self.post:
            self.post(rec, inc)
        return rec

    def window(self, win, macs, ips, X, res, peers, window_end=None):
        """macs/ips: per-row host MAC / IP; X: raw feature rows; res: Pair.score(X); peers: {(mac, win):
        {peer: [sent, recv]}} or a callable (mac) -> dict. Rows are processed in MAC order."""
        order = np.argsort(np.array(macs, dtype=object), kind="stable")
        for i in order:
            self.names.see(macs[i], ips[i])
        get = peers if callable(peers) else (lambda m: peers.get((m, win), {}))
        for i in order:
            c = res["cell"][i]
            self.cell_log.append((macs[i], win, c, float(res["p"][i]), int(res["band"][i]), float(res["if_score"][i])))
            if not V.is_alerting(c):
                continue
            src, tgt = inc_mod.attribute(macs[i], get(macs[i]))
            self.tracker.update(inc_mod.Alert(self.session, macs[i], int(win), float(res["p"][i]), int(res["band"][i]),
                                              float(res["if_score"][i]), c, res["named"][i], src, tgt, X[i]))
        self.tracker.close_idle(win + inc_mod.W if window_end is None else window_end)

    def finish(self):
        self.tracker.flush()
        return self.records
