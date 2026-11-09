"""Regression test for review M6 + N1: spoof_vitals must run BOTH sub-modes across a batch, chosen
deterministically (it previously always ran client-id reuse; the harder second-client-id variant had
0 runs). The mode now alternates by session index (not a random coin flip that can strand a whole
batch on one mode), and a schedule can still pin it. Paho-free, host-importable.
"""
import json
import pathlib
import unittest

from orchestrator import timing  # services/attacker on PYTHONPATH

REPO = pathlib.Path(__file__).resolve().parents[2]


class TestSpoofMode(unittest.TestCase):
    def test_deterministic_from_session_id(self):
        self.assertEqual(timing.choose_spoof_mode("r06"), timing.choose_spoof_mode("r06"))

    def test_alternates_by_session_index(self):
        # Consecutive session indices flip the mode; same parity keeps it.
        self.assertNotEqual(timing.choose_spoof_mode("r06"), timing.choose_spoof_mode("r07"))
        self.assertEqual(timing.choose_spoof_mode("r06"), timing.choose_spoof_mode("r08"))

    def test_both_modes_appear_across_a_batch(self):
        modes = {timing.choose_spoof_mode(f"r{i:02d}") for i in range(1, 8)}
        self.assertEqual(modes, {True, False}, "both reuse and second-client-id must occur")

    def test_session_index_parses_trailing_number(self):
        self.assertEqual(timing.session_index("r06"), 6)
        self.assertEqual(timing.session_index("r07"), 7)
        # No trailing digits -> still deterministic (stable hash), just not a small int.
        self.assertEqual(timing.session_index("alpha"), timing.session_index("alpha"))

    def test_default_schedule_does_not_pin_mode(self):
        sched = json.loads((REPO / "schedules" / "default.json").read_text())
        spoof = next(r for r in sched["runs"] if r["class"] == "spoof_vitals")
        self.assertNotIn("client_id_reuse", spoof.get("params", {}),
                         "default schedule must leave the spoof mode to the chooser")

    def test_override_schedule_pins_second_client_id(self):
        sched = json.loads((REPO / "schedules" / "spoof_second_client.json").read_text())
        spoof = next(r for r in sched["runs"] if r["class"] == "spoof_vitals")
        self.assertEqual(spoof["params"].get("client_id_reuse"), False,
                         "override must pin spoof_vitals to second_client_id (reuse=False)")
        # Nothing else changed: same run classes in the same order as default.
        default = json.loads((REPO / "schedules" / "default.json").read_text())
        self.assertEqual([r["class"] for r in sched["runs"]],
                         [r["class"] for r in default["runs"]])


if __name__ == "__main__":
    unittest.main()
