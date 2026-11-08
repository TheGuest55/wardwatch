"""Regression test for review M8 (device side): devices perform seeded quick reconnects so the
victim re-CONNECTs during an arp_mitm window. Tests the paho-free Noise schedule and asserts the
device base loop wires reconnects in (source check — base imports paho, so it isn't imported here).
"""
import os
import pathlib
import unittest

from wardwatch_devices import noise  # services/devices on PYTHONPATH

REPO = pathlib.Path(__file__).resolve().parents[2]


class TestReconnectSchedule(unittest.TestCase):
    def _noise(self, **env):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update({k: str(v) for k, v in env.items()})
        try:
            return noise.Noise("bed1-monitor", "172.28.10.11")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_interval_within_bounds_when_enabled(self):
        n = self._noise(SEED="1337", SESSION_ID="r01",
                        DEVICE_RECONNECT_MIN="60", DEVICE_RECONNECT_MAX="150")
        for _ in range(50):
            v = n.next_reconnect_in()
            self.assertIsNotNone(v)
            self.assertGreaterEqual(v, 60)
            self.assertLessEqual(v, 150)

    def test_disabled_returns_none(self):
        n = self._noise(SEED="1337", SESSION_ID="r01",
                        DEVICE_RECONNECT_MIN="0", DEVICE_RECONNECT_MAX="0")
        self.assertIsNone(n.next_reconnect_in())

    def test_base_loop_wires_reconnect(self):
        src = (REPO / "services/devices/wardwatch_devices/base.py").read_text()
        self.assertIn("def reconnect(self):", src)
        self.assertIn("next_reconnect", src)
        wear = (REPO / "services/devices/wardwatch_devices/wearable.py").read_text()
        self.assertIn("self.reconnect()", wear)


if __name__ == "__main__":
    unittest.main()
