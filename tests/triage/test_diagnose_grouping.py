"""diagnose_grouping on synthetic sessions: one test per cause, mixing, false alarms, and the equality assertion."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.triage import diagnose_grouping as dg
from src.triage import offline

ATT, BR, D1, N = "02:00:00:00:00:66", "02:00:00:00:00:10", "02:00:00:00:00:11", "02:00:00:00:00:20"
IP = {ATT: "10.0.0.66", BR: "10.0.0.10", D1: "10.0.0.11", N: "10.0.0.20"}
ROLE = {ATT: "attacker", BR: "broker", D1: "monitor", N: "nurse"}
IP_ROLES = {IP[m]: ROLE[m] for m in IP}


def session(rows, runs):
    """rows: dicts mac, win, label, run, cell, peers. runs: {run_id: class}."""
    df = pd.DataFrame({"host_mac": [r["mac"] for r in rows], "host_ip": [IP[r["mac"]] for r in rows],
                       "window_start_epoch": [r["win"] for r in rows], "label": [r["label"] for r in rows],
                       "run_id": [r.get("run") for r in rows], "host_role": [ROLE[r["mac"]] for r in rows]})
    cells = np.array([r["cell"] for r in rows], dtype=object)
    order = np.lexsort((df["host_mac"].to_numpy(), df["window_start_epoch"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)
    cells = cells[order]
    n = len(df)
    res = {"cell": cells, "p": np.full(n, 0.9), "band": np.full(n, 2), "if_score": np.ones(n),
           "named": np.array(["syn_flood"] * n, dtype=object), "raw": np.full(n, 0.9)}
    peers = {(r["mac"], r["win"]): r["peers"] for r in rows}
    return SimpleNamespace(name="s", df=df, X=np.zeros((n, 2)), res=res, peers=peers,
                           intervals=[{"run_id": k, "class": v, "start": 0.0, "end": 1.0} for k, v in runs.items()])


def R(mac, win, run, peers, cell="known", label="attack"):
    return {"mac": mac, "win": win, "label": label, "run": run, "cell": cell, "peers": peers}


def diag(rows, runs, key="source", gap=10):
    s = session(rows, runs)
    tr = offline.run_triager(s, key, gap, IP_ROLES)
    return dg.diagnose(s, tr, key, gap), tr


class TestCauses(unittest.TestCase):
    def test_single_incident_run_has_no_extra(self):
        rows = [R(ATT, 1000 + 5 * i, "r1", {BR: [50, 1]}) for i in range(4)] + \
               [R(BR, 1000 + 5 * i, "r1", {ATT: [1, 50]}) for i in range(4)]
        d, _ = diag(rows, {"r1": "syn_flood"})
        r = d["runs"]["r1"]
        self.assertEqual((r["n_incidents"], r["extra"]), (1, []))
        self.assertEqual(d["runs_single_incident"], 1)

    def test_attribution_cause_records_the_role_transition(self):
        rows = [R(ATT, 2000, "r2", {BR: [50, 1]}), R(BR, 2000, "r2", {ATT: [60, 2]})]      # the broker sent more: source = broker
        d, _ = diag(rows, {"r2": "recon_scan"})
        r = d["runs"]["r2"]
        self.assertEqual(r["n_incidents"], 2)
        self.assertEqual([e["cause"] for e in r["extra"]], ["attribution"])
        self.assertEqual(r["extra"][0]["transition"], "attacker->broker")
        self.assertEqual(d["extra_by_cause"], {"attribution": 1, "gap": 0, "other": 0})
        self.assertEqual(d["extra_by_transition"], {"attacker->broker": 1})

    def test_gap_cause_same_source_separated_by_more_than_the_gap(self):
        rows = [R(ATT, 3000, "r3", {BR: [50, 1]}), R(ATT, 3200, "r3", {BR: [50, 1]})]
        d, _ = diag(rows, {"r3": "bruteforce"}, gap=10)
        self.assertEqual([e["cause"] for e in d["runs"]["r3"]["extra"]], ["gap"])
        d2, _ = diag(rows, {"r3": "bruteforce"}, gap=300)
        self.assertEqual(d2["runs"]["r3"]["n_incidents"], 1)

    def test_other_is_flagged_when_neither_cause_holds(self):
        rows = [R(ATT, 4000, "r4", {BR: [50, 1]}), R(ATT, 4005, "r4", {D1: [50, 1]})]
        d, _ = diag(rows, {"r4": "scan"}, key="source+target", gap=10)      # same source, 0 s apart, different target
        self.assertEqual([e["cause"] for e in d["runs"]["r4"]["extra"]], ["other"])
        self.assertEqual(d["extra_by_cause"]["other"], 1)

    def test_incident_holding_two_runs_is_reported(self):
        rows = [R(ATT, 5000, "r5", {BR: [50, 1]}), R(ATT, 5005, "r6", {BR: [50, 1]})]
        d, _ = diag(rows, {"r5": "a", "r6": "b"})
        self.assertEqual(d["runs"]["r5"]["mixes_other_runs"], ["r6"])
        self.assertEqual(d["runs"]["r6"]["mixes_other_runs"], ["r5"])

    def test_benign_and_victim_alerts_inside_a_run_incident_are_flagged(self):
        rows = [R(ATT, 6000, "r7", {BR: [50, 1]}), R(ATT, 6005, None, {BR: [50, 1]}, "novel", "benign"),
                R(ATT, 6010, "r7", {BR: [50, 1]}, "novel", "victim")]
        d, _ = diag(rows, {"r7": "x"})
        r = d["runs"]["r7"]
        self.assertEqual(r["incidents_with_benign_alerts"], 1)
        self.assertEqual(r["incidents_with_victim_boundary"], 1)

    def test_undetected_run_is_listed_not_explained(self):
        d, _ = diag([R(ATT, 7000, "r8", {BR: [1, 1]}, "no_alert")], {"r8": "pump_cmd"})
        self.assertFalse(d["runs"]["r8"]["detected"])
        self.assertEqual(d["runs_detected"], 0)


class TestFalseAlarms(unittest.TestCase):
    def test_benign_only_vs_victim_boundary_by_role_and_verdict(self):
        rows = [R(N, 8000, None, {BR: [5, 1]}, "novel", "benign"),
                R(D1, 8100, None, {BR: [5, 1]}, "review_anomaly_only", "benign"),
                R(D1, 8200, None, {BR: [5, 1]}, "review_classifier_only", "victim"),
                R(ATT, 8300, "r9", {BR: [50, 1]})]
        d, _ = diag(rows, {"r9": "y"})
        fa = d["false_alarm"]
        self.assertEqual((fa["n"], fa["benign_only"], fa["with_victim_boundary"]), (3, 2, 1))
        self.assertEqual(fa["by_source_role"], {"monitor": 2, "nurse": 1})
        self.assertEqual(fa["by_cell"], {"novel": 1, "review_anomaly_only": 1, "review_classifier_only": 1})
        self.assertEqual(fa["by_verdict"]["Detectors disagree → review"], 2)


class TestEqualityAssertion(unittest.TestCase):
    def _recs(self):
        _, tr = diag([R(ATT, 1000, "r1", {BR: [50, 1]}), R(ATT, 1005, "r1", {BR: [50, 1]})], {"r1": "x"})
        return tr.records

    def test_equal_passes_and_any_field_difference_raises(self):
        recs = self._recs()
        self.assertEqual(dg.assert_equal_to_offline(recs, [dict(r) for r in recs]), len(recs))
        for f, v in (("n_alerts", 99), ("verdict", "No alert"), ("source", "x"), ("first_seen", "y"), ("cell", "z"),
                     ("target_host", "t"), ("last_seen", "l")):
            bad = [dict(r) for r in recs]
            bad[0][f] = v
            with self.assertRaises(AssertionError, msg=f):
                dg.assert_equal_to_offline(recs, bad)
        with self.assertRaises(AssertionError):
            dg.assert_equal_to_offline(recs, [])                     # missing / extra incidents


class TestSummary(unittest.TestCase):
    def test_summary_md_mentions_read_only_and_roles(self):
        d, _ = diag([R(ATT, 2000, "r2", {BR: [50, 1]}), R(BR, 2000, "r2", {ATT: [60, 2]})], {"r2": "scan"})
        rec = {"params": {"grouping_key": "source", "grouping_gap_s": 10},
               "tags": {"equal_to_offline": {"s": 2}, "per_session": {"s": d},
                        "totals": {"by_cause": d["extra_by_cause"], "by_transition": d["extra_by_transition"]}}}
        md = dg.summary_md(rec)
        self.assertIn("rule is unchanged", md)
        self.assertIn("evaluation only", md)
        self.assertIn("attacker->broker", md)


if __name__ == "__main__":
    unittest.main()
