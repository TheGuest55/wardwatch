"""Triage on the real v2 data (mounted read-only): multiclass identity, offline identity with the saved predictions,
B withheld analysis, and the live replay equal to the offline run on the same rows."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import common
from src.experiments import run_e
from src.features import windows
from src.models import save_multiclass as smc
from src.realtime import stream
from src.triage import offline, verdict as V
from tests.needs import needs
from tests.triage import real

FULL = ("DATASET", "RESULTS", "RAW", "MODELS")


def needs_missing():
    from tests import needs as _n
    return _n.missing(*FULL)


@needs("DATASET", "RESULTS", "MODELS")
class TestSaveMulticlass(unittest.TestCase):
    def test_refit_macro_f1_equals_the_recorded_seed0_value(self):
        mc, classes, ds, metrics = real.fitted()
        diff = smc.check_identity(metrics)
        self.assertTrue(all(d <= smc.TOL for d in diff.values()), diff)
        self.assertIn("benign", classes)
        self.assertEqual(len(classes), 9)

    def test_identity_failure_is_a_systemexit(self):
        with self.assertRaises(SystemExit):
            smc.check_identity({"test_macro_f1": 0.5, "test_macro_f1_traintree": 0.5})


@needs(*FULL)
class TestOfflineAgainstSavedPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pair = real.pair()

    def test_scores_equal_the_saved_a_seed0_predictions_on_the_test_rows(self):
        s = offline.Session("r05", self.pair, peers={})
        ident = s.identity_check("test")
        self.assertGreater(ident["xgboost_rows"], 10000)
        self.assertEqual(ident["xgboost_rows"], ident["isolation_forest_rows"])
        self.assertLessEqual(ident["xgboost_max_abs_diff"], 1e-6)
        # victim / boundary rows are scored here but are not in the saved files
        self.assertGreater(len(s.df), ident["xgboost_rows"])
        self.assertEqual(set(np.unique(s.res["cell"])) - set(V.CELLS), set())

    def test_identity_check_detects_a_perturbed_score(self):
        s = offline.Session("r05", self.pair, peers={})
        s.res["raw"] = s.res["raw"] + 1e-3
        with self.assertRaises(AssertionError):
            s.identity_check("test")

    def test_b_withheld_uses_validation_without_the_withheld_class(self):
        b = offline.b_withheld(["arp_mitm"], seeds=(0, 1))
        v = b["arp_mitm"]
        self.assertGreater(v["n_windows"], 0)
        for k, d in v["agg"].items():
            self.assertTrue(0.0 <= d["mean"] <= 1.0, k)
        for sd in v["per_seed"]:
            shares = sd["novel"] + sd["review"] + sd["known"] + sd["low_priority"] + sd["no_alert"]
            self.assertAlmostEqual(shares, 1.0)


class TestLiveEqualsOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pair = real.pair() if not needs_missing() else None

    @needs(*FULL)
    def test_replay_at_speed_zero_equals_offline_on_the_same_rows(self):
        cfg = common.load_config()
        roles = common.ip_role_map(cfg)
        pk = list(stream.session_packets("r05", max_seconds=240, start_offset=560))     # contains the first recon_scan
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = run_e.replay_session("r05", self.pair, "source", 30, speed=0, live_dir=tmp / "live",
                                       packet_iter=iter(pk))
            self.assertEqual(out["builder"].late_packets, 0)
            ref = pd.read_parquet(offline.V2 / "sessions" / "r05.parquet")
            live = run_e.to_session("r05", out, self.pair, ref)
            # offline on the same packets: batch rows (peers kept), the same scoring and triage code
            wb = windows.WindowBuilder("r05", run_e.SENTINEL, run_e.SENTINEL, roles, with_peers=True)
            for r in pk:
                wb.add(r)
            rows = wb.flush()
            peers = {(r["host_mac"], r["window_start_epoch"]): r["peer_pkts"] for r in rows}
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "peer_pkts"} for r in rows])
            off = offline.Session("r05", self.pair, df=df, peers=peers)
            eq = run_e.compare_rows(live, off)
            self.assertEqual(eq["band_flips"], 0)
            self.assertEqual(eq["verdict_flips"], 0)
            self.assertEqual(eq["max_rel_score_diff"], 0.0)
            tr = offline.run_triager(off, "source", 30, roles, post=offline.make_post(self.pair, "r05", tmp / "off", "triage/slices"))
            self.assertGreater(len(tr.records), 0)
            self.assertTrue(any(r["cell"] == "known" for r in tr.records) or any(r["n_alerts"] > 3 for r in tr.records))
            res = run_e.compare_incidents(out["triager"].records, tr.records)
            self.assertTrue(res["equal"])
            # evidence exists and re-hashes
            import hashlib
            for r in out["triager"].records:
                p = tmp / "live" / "slices" / f"{r['incident_id']}.pcap"
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), r["pcap_sha256"])
                self.assertTrue(r["top_features"])
            lines = (tmp / "live" / "incidents_r05.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), len(out["triager"].records))
            self.assertEqual(json.loads(lines[0])["session"], "r05")

    def test_compare_incidents_flags_any_field_difference(self):
        a = [{"incident_id": "x", "pcap_slice": "live/slices/x.pcap", "n_alerts": 3}]
        b = [{"incident_id": "x", "pcap_slice": "triage/slices/x.pcap", "n_alerts": 3}]
        self.assertTrue(run_e.compare_incidents(a, b)["equal"])
        with self.assertRaises(AssertionError):
            run_e.compare_incidents(a, [{**b[0], "n_alerts": 4}])

    @needs(*FULL)
    def test_run_writes_a_record_for_a_short_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = tmp / "triage_A.json"
            a.write_text(json.dumps({"params": {"grouping_key": "source", "grouping_gap_s": 30}}))
            from src.experiments import run_e as R
            real.pair()
            orig = R.Pair
            R.Pair = lambda: real.pair()
            try:
                rec = R.run("r05", speed=0, max_seconds=420, live_dir=tmp / "live", exp_dir=tmp / "expE", a_record=a)
            finally:
                R.Pair = orig
            self.assertFalse(rec["for_report"])
            self.assertEqual(rec["experiment"], "E")
            self.assertEqual(rec["tags"]["equivalence"]["late_packets"], 0)
            self.assertGreater(rec["metrics"]["n_ticks"], 50)
            self.assertLess(rec["metrics"]["tick_p95_ms"], R.BUDGET_MS * 10)
            self.assertTrue((tmp / "expE" / "E_r05.json").exists())
        with self.assertRaises(SystemExit):                       # a report-grade run must be a full test replay
            run_e.run("r05", max_seconds=60, for_report=True)


if __name__ == "__main__":
    unittest.main()
