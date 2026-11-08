"""Pipeline tests for review M10: manifest re-verification, the direct attack-window-vs-interval
check, and the attacker-host identifier-proxy flag in the leakage probe."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.datacheck import check
from src import common


def _sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


class TestManifestVerify(unittest.TestCase):
    def _session(self, tmp, name="sX"):
        sdir = Path(tmp) / name
        sdir.mkdir(parents=True)
        (sdir / "a.txt").write_text("hello")
        (sdir / "b.bin").write_bytes(b"\x00\x01\x02")
        manifest = {"files": {"a.txt": _sha(sdir / "a.txt"), "b.bin": _sha(sdir / "b.bin")}}
        (sdir / "manifest.json").write_text(json.dumps(manifest))
        return sdir

    def test_ok_then_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdir = self._session(tmp)
            orig = common.SESSIONS_DIR
            common.SESSIONS_DIR = Path(tmp)
            try:
                mv = check.manifest_verify("sX")
                self.assertTrue(mv["ok"], mv)
                (sdir / "a.txt").write_text("tampered")  # change a file after manifest
                mv2 = check.manifest_verify("sX")
                self.assertFalse(mv2["ok"])
                self.assertIn("a.txt", mv2["mismatches"])
            finally:
                common.SESSIONS_DIR = orig


class TestAttackWindowsWithinIntervals(unittest.TestCase):
    def test_flags_window_outside_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdir = Path(tmp) / "sY"
            sdir.mkdir(parents=True)
            gt = {"run_id": "sY-x-01", "class": "syn_flood",
                  "start_utc": "2026-01-01T00:00:10.000Z", "end_utc": "2026-01-01T00:00:40.000Z",
                  "src_ip": "1.2.3.4", "src_mac": "aa:bb"}
            (sdir / "ground_truth.jsonl").write_text(json.dumps(gt) + "\n")
            s0 = common.iso_to_epoch(gt["start_utc"])
            df = pd.DataFrame([
                {"session": "sY", "label": "attack", "window_start_epoch": int(s0 + 5)},   # inside
                {"session": "sY", "label": "attack", "window_start_epoch": int(s0 + 100)},  # outside
                {"session": "sY", "label": "benign", "window_start_epoch": int(s0 + 100)},  # ignored
            ])
            orig = common.SESSIONS_DIR
            common.SESSIONS_DIR = Path(tmp)
            try:
                self.assertEqual(check.attack_windows_within_intervals(df, "sY"), 1)
            finally:
                common.SESSIONS_DIR = orig


class TestIdentifierProxy(unittest.TestCase):
    def test_attacker_host_proxy_flagged(self):
        # A feature equal to the attacker-host indicator must be flagged as a proxy; a noise feature
        # must not. Build a train/test split with an attacker host present in both.
        rng = np.random.RandomState(0)
        n = 400
        is_attacker = np.tile([0, 0, 0, 1], n // 4)
        df = pd.DataFrame({
            "label": np.where(is_attacker == 1, "attack", "benign"),
            "attack_class": np.where(is_attacker == 1, "syn_flood", None),
            "host_role": np.where(is_attacker == 1, "attacker", "monitor"),
            "host_ip": "10.0.0.1",
            "window_start_epoch": np.arange(n),
            "split": ["train"] * (n // 2) + ["test"] * (n // 2),
            "ident": is_attacker.astype(float),              # perfect attacker-host proxy
            "noise": rng.rand(n),                            # harmless
        })
        fl = {"features": ["ident", "noise"], "metadata": []}
        probe = check.leakage_probe(df, fl)
        proxies = [p[0] for p in (probe.get("attacker_host_proxies") or [])]
        self.assertIn("ident", proxies)
        self.assertNotIn("noise", proxies)


if __name__ == "__main__":
    unittest.main()
