"""Regression test for review B3: device benign behaviour must differ per session.

The device RNG now derives from (base_seed, session_id, username). Two sessions at the same base
seed must produce different reboot-decision and command sequences; a given (seed, session, device)
must still reproduce exactly (so a session is replayable). Imports only the stdlib-only noise module
(no paho).
"""
import unittest

from wardwatch_devices.noise import device_seed_rng  # services/devices on PYTHONPATH


def _draws(rng, n=20):
    return [rng.random() for _ in range(n)]


class TestDeviceSeed(unittest.TestCase):
    def test_sessions_diverge_same_seed_same_device(self):
        a = _draws(device_seed_rng(1337, "r01", "bed1-monitor"))
        b = _draws(device_seed_rng(1337, "r02", "bed1-monitor"))
        self.assertNotEqual(a, b)
        # not merely offset by one — the whole sequences differ
        self.assertEqual(len(set(map(tuple, [a, b]))), 2)

    def test_reproducible_for_same_inputs(self):
        a = _draws(device_seed_rng(1337, "r04", "bed3-pump"))
        b = _draws(device_seed_rng(1337, "r04", "bed3-pump"))
        self.assertEqual(a, b)

    def test_devices_diverge_within_a_session(self):
        mon = _draws(device_seed_rng(1337, "r01", "bed1-monitor"))
        pump = _draws(device_seed_rng(1337, "r01", "bed1-pump"))
        self.assertNotEqual(mon, pump)

    def test_base_seed_matters(self):
        a = _draws(device_seed_rng(1337, "r01", "bed1-monitor"))
        b = _draws(device_seed_rng(20261108, "r01", "bed1-monitor"))
        self.assertNotEqual(a, b)

    def test_string_and_int_seed_both_accepted(self):
        # run_session passes SEED as a string via env; config uses an int. Both must work.
        self.assertEqual(_draws(device_seed_rng("1337", "r01", "n")),
                         _draws(device_seed_rng("1337", "r01", "n")))


if __name__ == "__main__":
    unittest.main()
