"""Triage metric definitions, B calibration discipline, false-novel window matching and slice determinism."""
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.triage import metrics as tm
from src.triage import pcap_slice as ps
from src.triage import verdict as V
from tests.needs import needs


class TestGroupingAndTTD(unittest.TestCase):
    def test_grouping_accuracy_definition(self):
        alerts = {("m", 0): "r1", ("m", 5): "r1", ("m", 10): "r2", ("m", 15): "r2", ("m", 20): "r3", ("x", 0): None}
        members = {"i1": [("m", 0), ("m", 5)], "i2": [("m", 10), ("m", 20)], "i3": [("m", 15)], "i4": [("x", 0)]}
        g = tm.grouping(alerts, members, ["r1", "r2", "r3", "r4"])
        self.assertEqual(g["correct"], ["r1"])                 # r2 split over two incidents; r3 shares i2 with r2
        self.assertEqual(sorted(g["incorrect"]), ["r2", "r3"])
        self.assertEqual(g["undetected"], ["r4"])
        self.assertAlmostEqual(g["accuracy"], 1 / 3)
        self.assertEqual(g["n_false_incidents"], 1)            # i4 holds only a benign alert
        self.assertEqual(g["n_incidents"], 4)

    def test_time_to_detect_is_end_of_first_alert_window_minus_start(self):
        t = tm.time_to_detect({"r1": [1010, 1005, 1100]}, {"r1": 1001.5, "r2": 2000.0})
        self.assertAlmostEqual(t["per_run"]["r1"], 1005 + 5 - 1001.5)
        self.assertIsNone(t["per_run"]["r2"])
        self.assertEqual(t["undetected"], ["r2"])
        self.assertEqual(t["detected"], 1)


class TestFoldBands(unittest.TestCase):
    def _frames(self, n=400):
        rng = np.random.default_rng(0)
        y = (rng.random(n) < 0.3).astype(int)
        score = np.clip(y * 0.6 + rng.random(n) * 0.4, 0, 1)
        val = pd.DataFrame({"split": "val", "y_bin": y, "score": score})
        test = pd.DataFrame({"split": "test", "y_bin": y, "score": score})
        return val, test

    def test_fits_on_validation_only_and_refuses_test_rows(self):
        val, test = self._frames()
        p, band, (lo, hi), cal = tm.fold_bands(val, test)
        self.assertEqual(len(p), len(test))
        self.assertTrue(set(np.unique(band)) <= {0, 1, 2})
        self.assertLessEqual(lo, hi)
        with self.assertRaises(ValueError):
            tm.fold_bands(test, test)                          # test rows offered as the fitting set
        with self.assertRaises(ValueError):
            tm.fold_bands(val, val)                            # validation rows offered as the scored set

    def test_test_labels_cannot_change_the_fit(self):
        val, test = self._frames()
        a = tm.fold_bands(val, test)
        test2 = test.assign(y_bin=1 - test["y_bin"])           # flip every test label
        b = tm.fold_bands(val, test2)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[2], b[2])


class TestFalseNovelMatching(unittest.TestCase):
    def test_reboot_overlap_and_reconnect_timestamp_match_the_hosts_benign_windows(self):
        wins = {"10.0.0.11": [(1000, 0), (1005, 1), (1010, 2), (1015, 3)], "10.0.0.12": [(1005, 4)]}
        reboot = [{"ip": "10.0.0.11", "start": 1007.0, "end": 1011.0}]
        self.assertEqual(tm.event_windows(reboot, wins, "reboot"), {1, 2})      # [1005,1010) and [1010,1015) overlap
        rec = [{"ip": "10.0.0.11", "ts": 1015.0}, {"ip": "10.0.0.99", "ts": 1005.0}]
        self.assertEqual(tm.event_windows(rec, wins, "reconnect"), {3})         # window holding ts; unknown host: none

    def test_rates_split_novel_review_alert(self):
        cells = ["novel", "review_anomaly_only", "no_alert", "low_priority", "known"]
        r = tm.event_rates({0, 1, 2, 3}, cells)
        self.assertEqual((r["novel"], r["review"], r["alert"], r["low_priority"]), (0.25, 0.25, 0.5, 0.25))
        self.assertIsNone(tm.event_rates(set(), cells)["novel"])

    def test_withheld_share(self):
        r = tm.withheld_share(["novel", "known", "no_alert", "review_classifier_only"], ["a", "a", "b", "b"])
        self.assertEqual((r["novel_or_review"], r["known"], r["no_alert"]), (0.5, 0.25, 0.25))
        self.assertEqual(r["runs_reaching_novel_or_review"], 1.0)
        self.assertEqual(r["n_runs"], 2)

    def test_cell_table_keeps_victim_and_boundary_apart(self):
        t = tm.cell_table(["benign", "attack", "victim", "boundary"], [None, "syn_flood", "spoof_vitals", None],
                          ["no_alert", "known", "novel", "low_priority"])
        self.assertEqual(set(t), {"benign", "syn_flood", "victim:spoof_vitals", "boundary"})
        self.assertEqual(t["syn_flood"]["known"], 1)


def _pcap(path, pkts, linktype=1):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 262144, linktype))
        for ts, dst, src in pkts:
            data = bytes.fromhex(dst.replace(":", "")) + bytes.fromhex(src.replace(":", "")) + b"\x08\x00" + b"payload-secret"
            f.write(struct.pack("<IIII", int(ts), int((ts % 1) * 1e6), len(data), len(data)) + data)


M1, M2, M3 = "02:00:00:00:00:01", "02:00:00:00:00:02", "02:00:00:00:00:03"


class TestSlice(unittest.TestCase):
    def test_slice_filters_by_mac_and_time_is_deterministic_and_leaves_sources_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _pcap(d / "a.pcap", [(100.0, M2, M1), (101.5, M3, M2), (102.0, M3, M3), (106.0, M2, M1)])
            _pcap(d / "b.pcap", [(103.25, M1, M3), (104.0, M3, M3)])
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.glob("*.pcap")}
            s1 = ps.write_slice([d / "a.pcap", d / "b.pcap"], [M1, M2], 100.0, 105.0, d / "out" / "x.pcap")
            s2 = ps.write_slice([d / "a.pcap", d / "b.pcap"], [M2, M1], 100.0, 105.0, d / "out" / "y.pcap")
            self.assertEqual(s1["n_packets"], 3)               # 100.0, 101.5, 103.25; not 102.0 (no MAC), not 106.0
            self.assertEqual(s1["sha256"], s2["sha256"])       # same inputs -> same bytes and hash
            self.assertEqual(s1["sha256"], hashlib.sha256((d / "out" / "x.pcap").read_bytes()).hexdigest())
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.glob("*.pcap")})
            # window end is exclusive, start inclusive
            s3 = ps.write_slice([d / "a.pcap"], [M1], 100.0, 106.0, d / "out" / "z.pcap")
            self.assertEqual(s3["n_packets"], 1)
            s4 = ps.write_slice([d / "a.pcap"], [M1], 100.0, 106.000001, d / "out" / "w.pcap")
            self.assertEqual(s4["n_packets"], 2)

    def test_slice_is_a_readable_pcap_with_original_packet_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _pcap(d / "a.pcap", [(100.0, M2, M1)])
            out = ps.write_slice([d / "a.pcap"], [M1], 0, 1e9, d / "o.pcap")
            pk = [x for x in ps._read_pcap(d / "o.pcap") if x[0] != "hdr"]
            self.assertEqual(len(pk), 1)
            self.assertTrue(pk[0][4].endswith(b"payload-secret"))
            self.assertEqual(out["bytes"], (d / "o.pcap").stat().st_size)


class TestIndexedSliceEqualsFullScan(unittest.TestCase):
    def test_synthetic_with_small_timestamp_disorder(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(1)
            ts = np.sort(rng.uniform(1000, 3000, 9000))
            ts[::50] -= rng.uniform(0, 0.5, len(ts[::50]))          # slight disorder, far inside the 5 s slack
            macs = [M1, M2, M3]
            _pcap(d / "a.pcap", [(t, macs[i % 3], macs[(i + 1) % 3]) for i, t in enumerate(ts)])
            for a, b in ((1000, 1010), (1500.5, 1530), (2990, 3010), (50, 60)):
                full = ps.write_slice([d / "a.pcap"], [M1], a, b, d / "f.pcap", indexed=False)
                idx = ps.write_slice([d / "a.pcap"], [M1], a, b, d / "i.pcap", indexed=True)
                self.assertEqual(full["sha256"], idx["sha256"], (a, b))
                self.assertEqual(full["n_packets"], idx["n_packets"])

    @needs("RAW")
    def test_real_r05_ranges(self):
        pcaps = sorted(Path("data/raw/sessions/r05/pcap").glob("*.pcap"))
        first = ps.get_index(pcaps[0]).ts[0]
        for off, span, mac in ((606, 40, "c6:e6:a8:5f:29:af"), (1500, 120, "c6:e6:a8:5f:29:af"), (4200, 30, "c6:e6:a8:5f:29:af")):
            with tempfile.TemporaryDirectory() as d:
                a = ps.write_slice(pcaps, [mac], first + off, first + off + span, Path(d) / "a.pcap", indexed=False)
                b = ps.write_slice(pcaps, [mac], first + off, first + off + span, Path(d) / "b.pcap", indexed=True)
                self.assertEqual(a["sha256"], b["sha256"], off)
                self.assertEqual(a["n_packets"], b["n_packets"])
                if off == 606:
                    self.assertGreater(a["n_packets"], 100)               # the recon scan is in this range


if __name__ == "__main__":
    unittest.main()
