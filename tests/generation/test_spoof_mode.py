"""Regression test for review M6: spoof_vitals must alternate both sub-modes across sessions,
seeded and logged (it previously always ran client-id reuse; the harder second-client-id variant
had 0 runs). Tests the paho-free chooser and that the schedule no longer pins the mode.
"""
import json
import pathlib
import unittest

from orchestrator import timing  # services/attacker on PYTHONPATH

REPO = pathlib.Path(__file__).resolve().parents[2]


class TestSpoofMode(unittest.TestCase):
    def test_deterministic_per_session_rng(self):
        a = timing.choose_spoof_mode(timing.session_rng(1337, "r01"))
        b = timing.choose_spoof_mode(timing.session_rng(1337, "r01"))
        self.assertEqual(a, b)

    def test_both_modes_appear_across_sessions(self):
        modes = {timing.choose_spoof_mode(timing.session_rng(42, f"r{i:02d}")) for i in range(30)}
        self.assertEqual(modes, {True, False}, "both reuse and second-client-id must occur")

    def test_schedule_does_not_pin_mode(self):
        sched = json.loads((REPO / "schedules" / "default.json").read_text())
        spoof = next(r for r in sched["runs"] if r["class"] == "spoof_vitals")
        self.assertNotIn("client_id_reuse", spoof.get("params", {}),
                         "default schedule must leave the spoof mode to the seeded chooser")


if __name__ == "__main__":
    unittest.main()
