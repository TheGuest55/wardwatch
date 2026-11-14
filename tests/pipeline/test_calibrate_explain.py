"""Calibration / confidence bands / reasons (src/models/calibrate_explain.py)."""
import json
import unittest
from pathlib import Path

import numpy as np

from src.models.calibrate_explain import (FEATURE_TEXT, Calibrator, band, brier, confidence_cutoffs,
                                          ece, reliability_bins, select_method, top_reasons)


def _overconfident(n=4000, seed=0):
    """True attack rate p_true, but the raw score is pushed towards 0/1 (overconfident)."""
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    score = 1 / (1 + np.exp(-4 * np.log(np.clip(p_true, 1e-6, 1 - 1e-6) /
                                         np.clip(1 - p_true, 1e-6, 1))))
    return score, y


class TestCalibration(unittest.TestCase):
    def test_ece_zero_when_perfect_and_large_when_wrong(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(ece(y, np.array([0.0, 0.0, 1.0, 1.0])), 0.0)
        self.assertAlmostEqual(ece(y, np.array([1.0, 1.0, 0.0, 0.0])), 1.0)

    def test_reliability_bins_count_every_row(self):
        y = np.array([0, 1, 0, 1, 1])
        r = reliability_bins(y, np.array([0.0, 0.2, 0.5, 0.99, 1.0]), n_bins=10)
        self.assertEqual(sum(b["n"] for b in r["bins"]), 5)

    def test_both_methods_reduce_ece_on_overconfident_scores(self):
        s, y = _overconfident()
        s2, y2 = _overconfident(seed=1)
        raw = ece(y2, s2)
        for m in ("platt", "isotonic"):
            p = Calibrator(m).fit(s, y).predict(s2)
            self.assertLess(ece(y2, p), raw, m)
            self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertLess(brier(y2, Calibrator("platt").fit(s, y).predict(s2)), brier(y2, s2))

    def test_select_method_cross_fits_over_groups(self):
        s, y = _overconfident()
        groups = np.where(np.arange(len(y)) % 2 == 0, "r04", "r06")
        best, res = select_method(s, y, groups)
        self.assertIn(best, ("platt", "isotonic"))
        self.assertEqual(set(res), {"platt", "isotonic"})
        with self.assertRaises(ValueError):
            select_method(s, y, np.array(["r04"] * len(y)))

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            Calibrator("temperature")


class TestCutoffs(unittest.TestCase):
    def test_cutoffs_meet_targets_on_fit_data(self):
        s, y = _overconfident(n=6000)
        p = Calibrator("isotonic").fit(s, y).predict(s)
        lo, hi = confidence_cutoffs(y, p, 0.95, 0.95)
        self.assertLessEqual(lo, hi)
        self.assertGreaterEqual(y[p >= hi].mean(), 0.95)          # precision in confident attack
        self.assertGreaterEqual(y[p >= lo].sum() / y.sum(), 0.95)  # recall kept above lo

    def test_separable_data(self):
        y = np.array([0] * 90 + [1] * 10)
        p = np.concatenate([np.linspace(0, 0.1, 90), np.linspace(0.9, 1, 10)])
        lo, hi = confidence_cutoffs(y, p, 0.95, 0.95)
        self.assertGreaterEqual(lo, 0.1)
        self.assertLessEqual(hi, 0.9)
        b = band(p, lo, hi)
        self.assertTrue((b[y == 1] == 2).all())

    def test_band_edges(self):
        self.assertEqual(band([0.1, 0.5, 0.9, 0.2], 0.2, 0.9).tolist(), [0, 1, 2, 1])

    def test_one_class_rejected(self):
        with self.assertRaises(ValueError):
            confidence_cutoffs(np.zeros(5), np.linspace(0, 1, 5))


class TestReasons(unittest.TestCase):
    def test_positive_contributions_first(self):
        feats = ["tcp_syn", "mqtt_connect", "frac_arp", "pkts_total"]
        r = top_reasons(np.array([0.5, -2.0, 1.5, 0.1]), np.array([300, 0, 0.1, 400]), feats, k=2)
        self.assertEqual([x["feature"] for x in r], ["frac_arp", "tcp_syn"])
        self.assertIn("raises", r[0]["text"])

    def test_falls_back_to_largest_abs_when_few_positive(self):
        r = top_reasons(np.array([0.5, -2.0, -0.1]), np.zeros(3), ["tcp_syn", "tcp_rst", "tcp_fin"], 2)
        self.assertEqual([x["feature"] for x in r], ["tcp_syn", "tcp_rst"])
        self.assertIn("lowers", r[1]["text"])

    def test_every_model_input_has_plain_text(self):
        fl = Path("data/processed/v2/feature_list.json")
        if not fl.exists():
            self.skipTest("v2 dataset not present")
        feats = json.loads(fl.read_text())["features"]
        self.assertEqual(sorted(set(feats) - set(FEATURE_TEXT)), [])


if __name__ == "__main__":
    unittest.main()
