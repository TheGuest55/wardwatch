"""Regression test: the device-reconnect mechanism must be wired into every device's compose env,
and the batch runner must accept per-session seeds.

Why: the arp_mitm credential-capture mechanism (review M8) depends on devices performing seeded
quick reconnects, enabled only when DEVICE_RECONNECT_MIN/MAX reach the container. In the t10/t11 dry
runs they were silently absent (a stale compose generated before the env existed), so no device ever
reconnected and arp_mitm could never capture. run_session.sh now regenerates compose before `up` and
asserts the live env; this test guards the generator and the batch-runner parse so the mechanism
can't go dormant again.
"""
import json
import pathlib
import subprocess
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import gen_lab  # noqa: E402  (scripts/ on path)


class TestComposeReconnect(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((REPO / "config" / "ward.json").read_text())
        self.hosts = gen_lab.build_hosts(self.cfg)

    def test_every_device_has_reconnect_env(self):
        devices = [h for h in self.hosts if h["role"] != "nurse-station"]
        self.assertTrue(devices)
        for h in devices:
            # device_service renders a compose YAML block (string) for the host.
            svc = gen_lab.device_service(h, self.cfg)
            self.assertIn("DEVICE_RECONNECT_MIN", svc, h["name"])
            self.assertIn("DEVICE_RECONNECT_MAX", svc, h["name"])
            # Compose default must be a positive cadence (not the disabled 0), via ${VAR:-NN}.
            for key in ("DEVICE_RECONNECT_MIN", "DEVICE_RECONNECT_MAX"):
                line = next(l for l in svc.splitlines() if key in l)
                default = line.split(":-")[-1].split("}")[0]
                self.assertGreater(float(default), 0, f"{h['name']} {line.strip()}")

    def test_run_session_regenerates_and_asserts_live_env(self):
        src = (REPO / "scripts" / "run_session.sh").read_text()
        self.assertIn("python3 scripts/gen_lab.py", src)
        self.assertIn("DEVICE_RECONNECT_MIN", src)

    def test_run_sessions_parses_id_and_seed(self):
        """`id:seed` tokens pass the seed through; a bare id does not. Checked with a stub
        run_session.sh that just echoes its id and SEED, so no lab is started."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "scripts").mkdir()
            (d / "data" / "raw" / "sessions").mkdir(parents=True)
            (d / "scripts" / "run_sessions.sh").write_text(
                (REPO / "scripts" / "run_sessions.sh").read_text())
            (d / "scripts" / "run_session.sh").write_text(
                '#!/usr/bin/env bash\necho "RAN id=$1 SEED=${SEED:-none}"\n')
            os.chmod(d / "scripts" / "run_session.sh", 0o755)
            out = subprocess.run(
                ["bash", "scripts/run_sessions.sh", "r01:10007", "r02"],
                cwd=d, capture_output=True, text=True, check=True).stdout
            self.assertIn("RAN id=r01 SEED=10007", out)
            self.assertIn("RAN id=r02 SEED=none", out)


if __name__ == "__main__":
    unittest.main()
