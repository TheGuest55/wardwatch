"""Pipeline test for review N6: the leakage probe must flag a v1-like leak — a feature that alone
identifies the attacker host (single-feature AUC >= PROXY_AUC), and a feature that is constant within
a session (per-session eta^2 >= ETA2_FLAG). Also checks the eta^2 helper and the disjointness assert.
"""
import unittest

import numpy as np
import pandas as pd

from src.datacheck import check
from src import common


def _leaky_df():
    rng = np.random.RandomState(0)
    rows = []
    # 3 sessions: sA,sB train, sC test. Each has an attacker host and a monitor host, both labels.
    plan = [("sA", "train"), ("sB", "train"), ("sC", "test")]
    for si, (sess, split) in enumerate(plan, start=1):
        for host, role, hip in [("at", "attacker", "10.0.0.66"), ("mo", "monitor", "10.0.0.11")]:
            for k in range(12):
                label = "attack" if (role == "attacker" and k % 2 == 0) else "benign"
                rows.append({
                    "session": sess, "split": split,
                    "window_start_epoch": 1_000_000 + si * 1000 + k * 5,
                    "host_mac": f"00:00:00:00:0{si}:{1 if role=='attacker' else 2}",
                    "host_ip": hip, "host_role": role,
                    "label": label, "attack_class": "recon_scan" if label == "attack" else "",
                    "run_id": "r-x-01" if label == "attack" else "",
                    # LEAK 1: perfectly separates the attacker host (AUC 1.0)
                    "leaky_attacker": 1.0 if role == "attacker" else 0.0,
                    # LEAK 2: constant within each session (per-session eta^2 = 1.0)
                    "sess_fingerprint": float(si),
                    # honest noise
                    "noise": float(rng.rand()),
                })
    return pd.DataFrame(rows)


class TestN6(unittest.TestCase):
    FL = {"features": ["leaky_attacker", "sess_fingerprint", "noise"], "metadata": []}

    def test_attacker_host_proxy_flagged(self):
        probe = check.leakage_probe(_leaky_df(), self.FL)
        names = [n for n, _ in probe["attacker_host_proxies"]]
        self.assertIn("leaky_attacker", names)
        self.assertNotIn("noise", names)

    def test_session_fingerprint_flagged(self):
        probe = check.leakage_probe(_leaky_df(), self.FL)
        flagged = [n for n, _ in probe["eta2_session_flags"]]
        self.assertIn("sess_fingerprint", flagged)
        self.assertNotIn("noise", flagged)

    def test_eta_squared_bounds(self):
        x = np.array([1.0, 1.0, 5.0, 5.0])
        g = np.array(["a", "a", "b", "b"])
        self.assertAlmostEqual(check._eta_squared(x, g), 1.0)         # constant within group
        self.assertAlmostEqual(check._eta_squared(np.array([1.0, 2, 1, 2]), g), 0.0)  # no group signal

    def test_disjoint_assertion_catches_overlap(self):
        orig = common.FEATURES
        common.FEATURES = list(common.FEATURES) + ["host_ip"]  # a metadata column
        try:
            with self.assertRaises(ValueError):
                common.assert_feature_metadata_disjoint()
        finally:
            common.FEATURES = orig


if __name__ == "__main__":
    unittest.main()
