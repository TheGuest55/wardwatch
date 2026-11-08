"""Regression test for review B2: attacker payloads must not be separable from benign by a
serialisation artefact. Replicates the real call sites (benign monitor / nurse vs attacker
spoof_vitals / pump_cmd) through the shared builder and asserts BOTH:

  * total serialised length ranges overlap (no single-threshold split), and
  * per-field value representation parity — hr/spo2 are floats on both sides (not int-where-the-
    device-sends-float), bp are ints, and the timestamp has the same format/length. A single field
    that is always an int where the device sends a float would still leak even if totals overlap.

Standard-library only; imports the shared builder, never the paho-dependent attack modules.
"""
import json
import random
import re
import unittest

import payloads  # services/shared/payloads.py (on PYTHONPATH)

FLOAT_RE = re.compile(r"^-?\d+\.\d+$")   # json.dumps(1.0) -> "1.0"
INT_RE = re.compile(r"^-?\d+$")          # json.dumps(1)   -> "1"


def _benign_monitor(rng):
    """Mirror monitor.py: bidmc-like whole-number floats and synthetic 1-dp floats, gauss bp."""
    hr = rng.choice([float(rng.randint(55, 105)), round(rng.uniform(55, 105), 1)])
    spo2 = rng.choice([float(rng.randint(94, 100)), round(rng.uniform(94, 100), 1)])
    return payloads.monitor_vitals(rng.randint(1, 4), hr, spo2,
                                   rng.gauss(120, 4), rng.gauss(78, 3), vitals_source="bidmc")


def _attack_spoof(rng):
    """Mirror spoof_vitals.py: int hr/spo2 (incl. implausible), randint bp."""
    implausible = rng.random() < 0.3
    hr = rng.choice([220, 20, 0]) if implausible else rng.randint(60, 100)
    spo2 = rng.choice([60, 40]) if implausible else rng.randint(94, 100)
    return payloads.monitor_vitals(rng.randint(1, 4), hr, spo2,
                                   rng.randint(90, 140), rng.randint(60, 90), vitals_source="bidmc")


def _token(value):
    return json.dumps(value)


def _lengths(payload_list):
    return [len(json.dumps(p)) for p in payload_list]


class TestPayloadParity(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(20261108)
        self.benign = [_benign_monitor(self.rng) for _ in range(400)]
        self.attack = [_attack_spoof(self.rng) for _ in range(400)]

    def test_field_order_identical(self):
        self.assertEqual(list(self.benign[0].keys()), list(self.attack[0].keys()))

    def test_hr_spo2_are_floats_on_both_sides(self):
        # The exact B2 leak class: an int where the device sends a float is a perfect tell.
        for name, rows in (("benign", self.benign), ("attack", self.attack)):
            for p in rows:
                self.assertRegex(_token(p["hr"]), FLOAT_RE, f"{name} hr not a float: {p['hr']!r}")
                self.assertRegex(_token(p["spo2"]), FLOAT_RE, f"{name} spo2 not a float")

    def test_bp_are_ints_on_both_sides(self):
        for rows in (self.benign, self.attack):
            for p in rows:
                self.assertRegex(_token(p["bp_sys"]), INT_RE)
                self.assertRegex(_token(p["bp_dia"]), INT_RE)

    def test_timestamp_format_and_length_match(self):
        # Device format (isoformat, "+00:00"); NOT the old 24-char "...mmmZ" attacker format.
        b_ts, a_ts = self.benign[0]["ts"], self.attack[0]["ts"]
        self.assertIn("+00:00", b_ts)
        self.assertIn("+00:00", a_ts)
        self.assertFalse(a_ts.endswith("Z"))
        self.assertEqual(len(b_ts), len(a_ts))

    def test_total_length_ranges_overlap(self):
        b, a = _lengths(self.benign), _lengths(self.attack)
        # Overlap = the two [min,max] intervals intersect (so no single threshold separates them).
        self.assertLessEqual(max(min(b), min(a)), min(max(b), max(a)),
                             f"spoof lengths {min(a)}-{max(a)} disjoint from benign {min(b)}-{max(b)}")

    def test_pump_command_parity(self):
        rng = random.Random(7)
        nurse = [payloads.pump_command("set_rate", rate=rng.choice([75, 100, 125, 150]))
                 for _ in range(50)]
        attacker = [payloads.pump_command("set_rate", rate=rng.choice([0, 75, 100, 125, 150, 999, 1200]))
                    for _ in range(50)]
        self.assertEqual(list(nurse[0].keys()), list(attacker[0].keys()))        # action, rate, ts
        nb, ab = _lengths(nurse), _lengths(attacker)
        self.assertLessEqual(max(min(nb), min(ab)), min(max(nb), max(ab)))
        ack = payloads.pump_command("ack_alarm")
        self.assertNotIn("rate", ack)
        self.assertEqual(list(ack.keys()), ["action", "ts"])


if __name__ == "__main__":
    unittest.main()
