"""Regression test for review M7: run onsets align to the 5 s window grid so the first window is
fully-inside the attack interval (and survives labelling) rather than straddling it.

Tests the paho-free aligner AND the labelling consequence via src.labels.label.label_window —
start_utc AT a boundary => onset window is attack; start_utc mid-window => it is dropped (boundary).
"""
import unittest

from orchestrator import timing          # services/attacker on PYTHONPATH
from src.labels.label import label_window  # repo root on sys.path (cwd)

W = 5


class TestAlignToGrid(unittest.TestCase):
    def test_returns_next_boundary_and_sleeps_to_boundary_plus_lead(self):
        slept = []
        b = timing.align_to_grid(window=5.0, lead=0.05,
                                 clock=lambda: 1000.3, sleep=slept.append)
        self.assertEqual(b, 1005.0)
        self.assertAlmostEqual(slept[0], 1005.0 + 0.05 - 1000.3, places=9)

    def test_advances_even_when_exactly_on_a_boundary(self):
        slept = []
        b = timing.align_to_grid(window=5.0, lead=0.05, clock=lambda: 1000.0, sleep=slept.append)
        self.assertEqual(b, 1005.0)
        self.assertAlmostEqual(slept[0], 5.05, places=9)

    def test_aligned_onset_window_is_attack(self):
        # start_utc AT the boundary 1000 => window [1000,1005) is fully inside [1000,1060].
        intervals = [{"class": "eavesdrop", "run_id": "r", "start": 1000.0, "end": 1060.0}]
        label, cls, rid, boundary = label_window(1000, True, None, intervals, {})
        self.assertEqual(label, "attack")
        self.assertFalse(boundary)

    def test_misaligned_onset_window_is_dropped(self):
        # The old behaviour: start_utc mid-window (1000.2) makes [1000,1005) straddle -> dropped.
        intervals = [{"class": "eavesdrop", "run_id": "r", "start": 1000.2, "end": 1060.2}]
        label, cls, rid, boundary = label_window(1000, True, None, intervals, {})
        self.assertTrue(boundary)
        self.assertEqual(label, "boundary")


if __name__ == "__main__":
    unittest.main()
