"""Verdict cells, attribution, grouping, incident record format, and the display-only rule for host_role."""
import json
import unittest

import numpy as np

from src.triage import engine, incidents as inc
from src.triage import verdict as V

# Verbatim copy of canonical() and VERDICTS from the dashboard branch (src/dashboard/incidents.py at fb5a94e), so
# the strings are checked against what the dashboard will do with them.
DASH_VERDICTS = ("Known attack", "Possible novel attack", "Detectors disagree: review",
                 "Low priority, logged", "No alert")
DASH_ALIASES = {"detectors disagree → review": "Detectors disagree: review",
                "detectors disagree -> review": "Detectors disagree: review",
                "detectors disagree": "Detectors disagree: review"}


def dash_canonical(v):
    s = str(v or "").strip()
    low = s.lower()
    for k, c in DASH_ALIASES.items():
        if low.startswith(k):
            return c
    for c in DASH_VERDICTS:
        if low.startswith(c.lower()):
            return c
    return s or "unknown"


A, B, C, BR = "02:00:00:00:00:0a", "02:00:00:00:00:0b", "02:00:00:00:00:0c", "02:00:00:00:00:10"
IPS = {A: "10.0.0.11", B: "10.0.0.12", C: "10.0.0.66", BR: "10.0.0.10"}
ROLES = {"10.0.0.11": "monitor", "10.0.0.12": "pump", "10.0.0.66": "attacker", "10.0.0.10": "broker"}


class TestVerdict(unittest.TestCase):
    def test_six_cells_exact_strings(self):
        want = {(True, 2): ("Known attack (type named)", "known"),
                (True, 1): ("Possible novel attack", "novel"),
                (True, 0): ("Detectors disagree → review", "review_anomaly_only"),
                (False, 2): ("Detectors disagree → review", "review_classifier_only"),
                (False, 1): ("Low priority, logged", "low_priority"),
                (False, 0): ("No alert", "no_alert")}
        for (anom, band), (verdict, cell) in want.items():
            self.assertEqual(V.cell(anom, band), cell)
            self.assertEqual(V.verdict_of(cell), verdict)
        self.assertEqual(len(set(V.CELLS)), 6)
        self.assertEqual(len({V.verdict_of(c) for c in V.CELLS}), 5)       # the two review cells share a string

    def test_vectorised_matches_scalar_and_alerting_set(self):
        a = np.array([1, 1, 1, 0, 0, 0], bool)
        b = np.array([2, 1, 0, 2, 1, 0])
        self.assertEqual(list(V.cells(a, b)), [V.cell(x, y) for x, y in zip(a, b)])
        self.assertEqual([V.is_alerting(c) for c in V.CELLS], [True, True, True, True, False, False])

    def test_every_verdict_maps_through_the_dashboard_canonical(self):
        for c in V.CELLS:
            self.assertIn(dash_canonical(V.verdict_of(c)), DASH_VERDICTS, V.verdict_of(c))
        self.assertEqual(dash_canonical(V.KNOWN), "Known attack")
        self.assertEqual(dash_canonical(V.REVIEW), "Detectors disagree: review")


class TestAttribution(unittest.TestCase):
    def test_sender_is_source_when_it_sent_more(self):
        self.assertEqual(inc.attribute(A, {BR: [90, 3], B: [1, 1]}), (A, BR))

    def test_receiver_side_flood_victim_row_points_back_at_the_sender(self):
        self.assertEqual(inc.attribute(BR, {C: [3, 900]}), (C, BR))            # broker received 900 from C

    def test_tie_and_peer_choice_are_deterministic(self):
        self.assertEqual(inc.attribute(A, {B: [5, 5], BR: [5, 5]}), (B, A))    # lowest MAC peer wins; sent == recv -> the peer is the source
        self.assertEqual(inc.attribute(A, {}), (A, inc.BROADCAST))


def _alert(win, mac, src, tgt, cell="known", p=0.9, named="syn_flood", feat=None):
    return inc.Alert("t", mac, win, p, 2 if cell in ("known", "review_classifier_only") else 1, 1.0, cell, named, src, tgt,
                     feat if feat is not None else np.zeros(3))


def _run_tracker(alerts, key, gap, roles=ROLES):
    names = inc.HostNames(roles)
    for m, ip in IPS.items():
        names.see(m, ip)
    tr = inc.IncidentTracker("t", key, gap, names, finalize=lambda i: i)
    for a in sorted(alerts, key=lambda a: (a.window_start, a.host_mac)):
        tr.update(a)
        tr.close_idle(a.window_start + 5)
    tr.flush()
    return tr.done, names


class TestGrouping(unittest.TestCase):
    def test_scan_stays_together_under_source_key_and_splits_under_source_target(self):
        scan = [_alert(1000 + 5 * i, C, C, t, "novel") for i, t in enumerate([A, B, BR, A, B, BR])]
        d_src, _ = _run_tracker(scan, "source", 30)
        d_st, _ = _run_tracker(scan, "source+target", 30)
        self.assertEqual(len(d_src), 1)
        self.assertEqual(len(d_st), 3)

    def test_gap_separates_two_runs_of_the_same_source(self):
        al = [_alert(1000, C, C, BR), _alert(1005, C, C, BR), _alert(1400, C, C, BR)]
        self.assertEqual([len(i.alerts) for i in _run_tracker(al, "source", 60)[0]], [2, 1])
        self.assertEqual([len(i.alerts) for i in _run_tracker(al, "source", 600)[0]], [3])

    def test_two_overlapping_runs_from_two_sources_stay_apart(self):
        a1 = [_alert(1000 + 5 * i, C, C, BR) for i in range(6)]
        a2 = [_alert(1010 + 5 * i, A, A, BR, "review_anomaly_only", 0.2, None) for i in range(6)]
        done, _ = _run_tracker(a1 + a2, "source", 30)
        self.assertEqual(sorted(len(i.alerts) for i in done), [6, 6])
        self.assertEqual({tuple(sorted({x.host_mac for x in i.alerts})) for i in done}, {(C,), (A,)})

    def test_most_severe_verdict_wins_and_timeline_records_changes(self):
        al = [_alert(1000, C, C, BR, "review_classifier_only", 0.5, None), _alert(1005, C, C, BR, "novel", 0.4, None),
              _alert(1010, C, C, BR, "known", 0.95, "mqtt_flood"), _alert(1015, C, C, BR, "known", 0.8, "syn_flood")]
        (i,), names = _run_tracker(al, "source", 30)
        rec = inc.build_record(i, names)
        self.assertEqual(rec["cell"], "known")
        self.assertEqual(rec["verdict"], V.KNOWN)
        self.assertEqual(rec["attack_class"], "mqtt_flood")                  # tie 1-1: alphabetical
        self.assertEqual(rec["n_alerts"], 4)
        self.assertAlmostEqual(rec["calibrated_p"], 0.95)
        ev = [t["event"] for t in rec["timeline"]]
        self.assertTrue(ev[0].startswith("first alert") and "escalated to Possible novel" in ev[1]
                        and "escalated to Known attack" in ev[2] and ev[-1].startswith("last alert"))

    def test_ids_are_deterministic_and_unique(self):
        al = [_alert(1000, C, C, BR), _alert(1000, A, A, BR)]
        d1, n1 = _run_tracker(al, "source", 30)
        d2, n2 = _run_tracker(list(reversed(al)), "source", 30)
        self.assertEqual([i.incident_id for i in d1], [i.incident_id for i in d2])
        self.assertEqual(len({i.incident_id for i in d1}), 2)
        self.assertTrue(d1[0].incident_id.startswith("t-"))


class TestRecordFormat(unittest.TestCase):
    def test_dashboard_fields_plus_cell_scope_session(self):
        (i,), names = _run_tracker([_alert(1000, C, C, BR, feat=np.array([1.0, 2.0]))], "source", 30)
        rec = inc.build_record(i, names)
        for f in ("incident_id", "target_host", "source", "first_seen", "last_seen", "verdict", "top_features",
                  "pcap_slice", "pcap_sha256", "n_alerts", "attack_class", "calibrated_p", "timeline",
                  "cell", "scope", "session"):
            self.assertIn(f, rec)
        json.dumps(rec)
        self.assertEqual(rec["source"], "10.0.0.66 (attacker)")
        self.assertEqual(rec["target_host"], "10.0.0.10 (broker)")
        self.assertRegex(rec["first_seen"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")
        self.assertIn(dash_canonical(rec["verdict"]), DASH_VERDICTS)

    def test_post_alert_fields_are_filled_by_complete_record(self):
        (i,), names = _run_tracker([_alert(1000, C, C, BR, p=0.3, feat=np.array([7.0])),
                                    _alert(1005, C, C, BR, p=0.9, feat=np.array([9.0]))], "source", 30)
        rec = inc.build_record(i, names)
        seen = {}
        inc.complete_record(rec, i, lambda f: (seen.__setitem__("f", f), [{"feature": "x", "text": "t", "delta": 1.0}])[1],
                            lambda first, last_end, macs, iid: {"path": f"live/slices/{iid}.pcap", "sha256": "ab" * 32})
        self.assertEqual(float(seen["f"][0]), 9.0)                           # explained on the max-p window
        self.assertEqual(rec["pcap_sha256"], "ab" * 32)
        self.assertEqual(rec["top_features"][0]["feature"], "x")


class TestRolesAreDisplayOnly(unittest.TestCase):
    """host_role / role text may only change names, never verdicts, attribution, grouping or membership."""

    def _run(self, roles):
        tr = engine.Triager("t", roles, "source", 30)
        rows = {(C, 1000): {BR: [3, 50]}, (BR, 1000): {C: [50, 3]}, (A, 1005): {BR: [4, 6]}, (C, 1005): {BR: [2, 60]}}
        res_by = {(C, 1000): ("known", .9), (BR, 1000): ("review_classifier_only", .8), (A, 1005): ("novel", .4),
                  (C, 1005): ("known", .95)}
        for win in (1000, 1005):
            keys = sorted(k for k in res_by if k[1] == win)
            macs = [k[0] for k in keys]
            res = {"cell": np.array([res_by[k][0] for k in keys], dtype=object), "p": np.array([res_by[k][1] for k in keys]),
                   "band": np.array([2, 2, 1, 2][:len(keys)]), "if_score": np.ones(len(keys)),
                   "named": np.array(["syn_flood"] * len(keys), dtype=object)}
            tr.window(win, macs, [IPS[m] for m in macs], np.zeros((len(keys), 2)), res, rows)
        tr.finish()
        return tr

    def test_permuting_roles_changes_names_only(self):
        base = self._run(ROLES)
        perm = self._run({"10.0.0.11": "broker", "10.0.0.12": "attacker", "10.0.0.66": "pump", "10.0.0.10": "monitor"})
        strip = lambda t: sorted((r["first_seen"], r["last_seen"], r["verdict"], r["cell"], r["n_alerts"],
                                  r["calibrated_p"], r["attack_class"]) for r in t.records)
        self.assertEqual(strip(base), strip(perm))
        mem = lambda t: sorted(sorted(v) for v in t.members.values())
        self.assertEqual(mem(base), mem(perm))
        self.assertEqual(base.cell_log, perm.cell_log)
        self.assertNotEqual({r["source"] for r in base.records}, {r["source"] for r in perm.records})   # display differs
        self.assertNotEqual(sorted(r["incident_id"] for r in base.records), sorted(r["incident_id"] for r in perm.records))


if __name__ == "__main__":
    unittest.main()
