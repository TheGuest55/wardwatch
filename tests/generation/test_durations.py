"""Regression test for randomised attack durations within spec ranges: the runner draws a seeded
concrete duration from each run's duration_range, reproducible per session and always >= one 5 s
window. Also checks the default schedule uses ranges for the duration-based classes.
"""
import json
import pathlib
import unittest

from orchestrator import timing  # services/attacker on PYTHONPATH

REPO = pathlib.Path(__file__).resolve().parents[2]


class TestDurations(unittest.TestCase):
    def test_resolve_draws_within_range_and_sets_duration(self):
        rng = timing.session_rng(1337, "r01")
        p = timing.resolve_duration(rng, {"duration_range": [120, 480], "foo": 1})
        self.assertIn("duration", p)
        self.assertNotIn("duration_range", p)
        self.assertTrue(120 <= p["duration"] <= 480)
        self.assertEqual(p["foo"], 1)                 # other params preserved

    def test_reproducible_per_session(self):
        a = timing.resolve_duration(timing.session_rng(42, "r03"), {"duration_range": [45, 90]})
        b = timing.resolve_duration(timing.session_rng(42, "r03"), {"duration_range": [45, 90]})
        self.assertEqual(a["duration"], b["duration"])

    def test_no_range_is_passthrough(self):
        p = timing.resolve_duration(timing.session_rng(1, "r01"), {"duration": 60})
        self.assertEqual(p["duration"], 60)

    def test_all_durations_exceed_one_window(self):
        # every range lower bound must exceed the 5 s window so a run yields at least one clean window
        sched = json.loads((REPO / "schedules" / "default.json").read_text())
        for run in sched["runs"]:
            rng = run.get("params", {}).get("duration_range")
            if rng:
                self.assertGreater(rng[0], 5, f"{run['class']} min duration too short")

    def test_duration_based_classes_use_ranges(self):
        sched = json.loads((REPO / "schedules" / "default.json").read_text())
        by = {r["class"]: r.get("params", {}) for r in sched["runs"]}
        for cls in ("syn_flood", "mqtt_flood", "arp_mitm", "eavesdrop", "spoof_vitals", "pump_cmd"):
            self.assertIn("duration_range", by[cls], f"{cls} should randomise its duration")


if __name__ == "__main__":
    unittest.main()
