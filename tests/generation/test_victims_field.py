"""Generation test for review N2: spoof_vitals and pump_cmd must emit a structured `victims` field
({ip, mac, role, bed}) that the runner records in ground truth for future runs. paho is stubbed so
the handlers import without the broker library (host generation tier is paho-free)."""
import sys
import types
import random
import unittest


def _install_paho_stub():
    if "paho.mqtt.client" in sys.modules:
        return
    paho = types.ModuleType("paho")
    mqtt_pkg = types.ModuleType("paho.mqtt")
    client = types.ModuleType("paho.mqtt.client")

    class _Client:
        def __init__(self, *a, **k): pass
        def username_pw_set(self, *a, **k): pass
        def connect(self, *a, **k): pass
        def loop_start(self): pass
        def loop_stop(self): pass
        def publish(self, *a, **k): pass
        def disconnect(self): pass

    class _CB:
        VERSION2 = 2

    client.Client = _Client
    client.CallbackAPIVersion = _CB
    paho.mqtt = mqtt_pkg
    mqtt_pkg.client = client
    sys.modules["paho"] = paho
    sys.modules["paho.mqtt"] = mqtt_pkg
    sys.modules["paho.mqtt.client"] = client


_install_paho_stub()
from orchestrator.attacks import spoof_vitals as sv, pump_cmd as pc  # noqa: E402
from orchestrator import util  # noqa: E402

PREFIX = "172.28.10."


class _Ctx:
    def __init__(self, session_id="r09"):
        self.session_id = session_id
        self.rng = random.Random(1)
        self.beds = 4
        self.broker_ip = f"{PREFIX}10"
        self.nurse_user = "nurse-station"
        self.monitor_users = [f"bed{b}-monitor" for b in range(1, 5)]
        self.user_ip = {f"bed{b}-monitor": f"{PREFIX}{10 * b + 1}" for b in range(1, 5)}
        self.cfg = {"network": {"subnet": "172.28.10.0/24"}}

    def load_stolen(self): return {}
    def provisioned(self, user): return "pw"   # fallback cred so the abuse class can run


class TestVictimsField(unittest.TestCase):
    def test_spoof_vitals_emits_victim(self):
        res = sv.spoof_vitals(_Ctx(), {"duration": 0})     # duration 0 -> no publish loop
        self.assertIn("victims", res)
        v = res["victims"][0]
        self.assertEqual(v["ip"], _Ctx().user_ip[f"bed{v['bed']}-monitor"])
        self.assertIn("mac", v)
        self.assertEqual(v["role"], "monitor")

    def test_pump_cmd_emits_victims(self):
        orig = util.monotonic
        calls = {"n": 0}

        def fake_monotonic():
            calls["n"] += 1
            return 0.0 if calls["n"] <= 2 else 1e9   # one loop iteration, then exit
        util.monotonic = fake_monotonic
        orig_sleep = pc.time.sleep
        pc.time.sleep = lambda *a, **k: None
        try:
            res = pc.pump_cmd(_Ctx(), {"duration": 5, "cmds_per_min": 60})
        finally:
            util.monotonic = orig
            pc.time.sleep = orig_sleep
        self.assertIn("victims", res)
        self.assertGreaterEqual(len(res["victims"]), 1)
        for v in res["victims"]:
            self.assertEqual(v["ip"], f"{PREFIX}{10 * v['bed'] + 2}")  # pump = 10*bed+2
            self.assertEqual(v["role"], "pump")


if __name__ == "__main__":
    unittest.main()
