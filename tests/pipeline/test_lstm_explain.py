"""Tests for the occlusion explainer (src/models/lstm_explain.py): all-zero feature gives delta 0,
is_present and filled windows are never touched, the group mapping covers every feature exactly once."""
import unittest

import numpy as np
import torch
from torch import nn

from src import common
from src.models import lstm, lstm_explain as ex

NAMES = ["a", "b", "c", "d"]
K = 6


def tiny_model(n_in=len(NAMES) + 1, seed=0):
    lstm.set_determinism(seed, 1)
    return lstm.LSTMClassifier(n_in, 8, 1, 0.0, 3).eval()


def make_seq(present=(0, 1, 2, 3, 4, 5), seed=0, zero_col=None):
    rng = np.random.RandomState(seed)
    x = np.zeros((K, len(NAMES) + 1), dtype=np.float32)
    for p in present:
        x[p, :len(NAMES)] = rng.randn(len(NAMES)) + 0.5
        x[p, len(NAMES)] = 1.0
    if zero_col is not None:
        x[:, zero_col] = 0.0
    return x


class Spy(nn.Module):
    """Wraps a model and records every batch it is given."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone().numpy())
        return self.inner(x)


class TestGroups(unittest.TestCase):
    def test_every_feature_in_exactly_one_group(self):
        g = ex.build_groups()
        flat = [f for fs in g.values() for f in fs]
        self.assertEqual(sorted(flat), sorted(common.FEATURES))
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(g), {"MQTT", "TCP flags", "volume", "ARP", "flow"})
        self.assertTrue(all(g.values()))

    def test_restricted_features_and_stray_feature(self):
        g = ex.build_groups(["mqtt_publish", "arp_replies"])
        self.assertEqual(g["MQTT"], ["mqtt_publish"])
        self.assertEqual(g["ARP"], ["arp_replies"])
        with self.assertRaises(AssertionError):
            ex.build_groups(["not_a_feature"])

    def test_duplicate_or_missing_member_is_caught(self):
        old = {k: list(v) for k, v in ex.GROUP_MEMBERS.items()}
        try:
            ex.GROUP_MEMBERS["flow"].append("mqtt_publish")
            with self.assertRaises(AssertionError):
                ex.build_groups()
            ex.GROUP_MEMBERS["flow"].remove("mqtt_publish")
            ex.GROUP_MEMBERS["flow"].remove("nf_new_flows")
            with self.assertRaises(AssertionError):
                ex.build_groups()
        finally:
            ex.GROUP_MEMBERS.clear()
            ex.GROUP_MEMBERS.update(old)


class TestOcclusion(unittest.TestCase):
    def test_all_zero_feature_has_delta_zero(self):
        m = tiny_model()
        seq = make_seq(zero_col=1)                       # feature "b" is scaled zero everywhere
        e = ex.explain(m, seq, last_n=3, feature_names=NAMES)
        self.assertEqual(e["feature"]["delta"]["b"], 0.0)
        self.assertTrue(any(v != 0.0 for k, v in e["feature"]["delta"].items() if k != "b"))

    def test_only_feature_channels_of_the_last_n_present_windows_change(self):
        spy = Spy(tiny_model())
        seq = make_seq(present=(0, 1, 3, 4, 5))          # window 2 is a gap (zeros, is_present 0)
        e = ex.explain(spy, seq, last_n=3, groups={"g": ["a", "b"]}, feature_names=NAMES)
        self.assertEqual(e["present_positions"], [3, 4, 5])
        base = spy.seen[0][0]
        np.testing.assert_array_equal(base, seq)
        variants = np.concatenate(spy.seen[1:])
        self.assertEqual(len(variants), len(NAMES) + 1)
        for v in variants:
            diff = v != seq
            self.assertFalse(diff[:, len(NAMES)].any())            # is_present untouched
            self.assertFalse(diff[[0, 1, 2]].any())                 # windows before the last n untouched
            self.assertFalse(diff[2].any())                         # the filled gap untouched
            self.assertTrue((v[[3, 4, 5]][diff[[3, 4, 5]]] == 0).all())   # changed entries are scaled zero

    def test_fill_windows_inside_the_last_n_are_skipped_not_occluded(self):
        spy = Spy(tiny_model())
        seq = make_seq(present=(0, 1, 2, 3, 5))          # gap at 4
        e = ex.explain(spy, seq, last_n=3, feature_names=NAMES)
        self.assertEqual(e["present_positions"], [2, 3, 5])
        for v in np.concatenate(spy.seen[1:]):
            self.assertTrue((v[4] == seq[4]).all())

    def test_fewer_present_windows_than_last_n(self):
        e = ex.explain(tiny_model(), make_seq(present=(4, 5)), last_n=3, feature_names=NAMES)
        self.assertEqual(e["n_windows_occluded"], 2)
        self.assertIn("in the last 2 windows", e["feature"]["top"][0])

    def test_group_delta_equals_manual_occlusion_of_its_members(self):
        m = tiny_model()
        seq = make_seq()
        e = ex.explain(m, seq, last_n=2, groups={"ab": ["a", "b"], "cd": ["c", "d"]}, feature_names=NAMES)
        occ = seq.copy()
        occ[4:6, [0, 1]] = 0.0
        with torch.no_grad():
            s = 1 - torch.softmax(m(torch.as_tensor(occ)[None]), 1)[0, 0].item()
            s0 = 1 - torch.softmax(m(torch.as_tensor(seq)[None]), 1)[0, 0].item()
        logit = lambda p: np.log(p / (1 - p))
        self.assertAlmostEqual(e["group"]["delta_prob"]["ab"], s0 - s, places=6)
        self.assertAlmostEqual(e["group"]["delta"]["ab"], logit(s0) - logit(s), places=4)
        self.assertAlmostEqual(e["base_score"], s0, places=6)
        self.assertAlmostEqual(e["base_logodds"], logit(s0), places=4)

    def test_top5_sorted_plain_language_and_deterministic(self):
        m = tiny_model()
        names = [f"mqtt_publish_rate", "tcp_syn", "arp_replies", "pkts_total", "nf_new_flows", "mean_iat"]
        seq = np.zeros((K, len(names) + 1), dtype=np.float32)
        rng = np.random.RandomState(3)
        seq[:, :len(names)] = rng.randn(K, len(names))
        seq[:, len(names)] = 1.0
        m = tiny_model(len(names) + 1)
        e1 = ex.explain(m, seq, last_n=3, feature_names=names)
        e2 = ex.explain(m, seq, last_n=3, feature_names=names)
        self.assertEqual(e1["feature"]["top"], e2["feature"]["top"])
        self.assertEqual(len(e1["feature"]["top"]), 5)
        d = [e1["feature"]["delta"][n] for n in e1["feature"]["top_names"]]
        self.assertEqual(d, sorted(d, reverse=True))
        self.assertRegex(e1["feature"]["top"][0], r"^.+ in the last 3 windows (raised|lowered) the attack log-odds by \d+\.\d\d$")
        self.assertEqual(ex.humanise("mqtt_publish_rate"), "MQTT publish rate")
        self.assertEqual(ex.humanise("MQTT"), "MQTT group")

    def test_without_presence_channel_filled_windows_are_exact_zero_rows(self):
        spy = Spy(tiny_model(len(NAMES)))
        seq = make_seq(present=(1, 2, 3, 4, 5))[:, :len(NAMES)]      # window 0 = padding (all zeros)
        e = ex.explain(spy, seq, last_n=6, feature_names=NAMES, use_presence=False)
        self.assertEqual(e["present_positions"], [1, 2, 3, 4, 5])
        for v in np.concatenate(spy.seen[1:]):
            self.assertTrue((v[0] == 0).all())

    def test_logodds_does_not_saturate(self):
        # A model whose attack score is 1.0 in float32 both before and after occluding "a": the
        # probability delta is 0, the log-odds delta is the exact logit difference (20 * 1.0 * 3 = 60).
        class Lin(nn.Module):
            def forward(self, x):
                s = x[:, -3:, 0].sum(1, keepdim=True)
                z0 = torch.zeros_like(s)
                return torch.cat([z0, 30.0 + 20.0 * s, torch.full_like(s, -50.0)], 1)
        seq = make_seq()
        seq[:, 0] = 1.0
        e = ex.explain(Lin(), seq, last_n=3, feature_names=NAMES)
        self.assertEqual(e["base_score"], 1.0)
        self.assertEqual(e["feature"]["delta_prob"]["a"], 0.0)
        self.assertAlmostEqual(e["feature"]["delta"]["a"], 60.0, places=5)
        self.assertEqual(e["feature"]["delta"]["b"], 0.0)
        self.assertEqual(e["feature"]["top_names"][0], "a")

    def test_last_n_k_covers_every_present_window(self):
        e = ex.explain(tiny_model(), make_seq(present=(0, 2, 3, 5)), last_n=K, feature_names=NAMES)
        self.assertEqual(e["present_positions"], [0, 2, 3, 5])

    def test_bad_shapes_rejected(self):
        with self.assertRaises(ValueError):
            ex.explain(tiny_model(), make_seq()[:, :3], feature_names=NAMES)
        with self.assertRaises(ValueError):
            ex.explain(tiny_model(), make_seq(), feature_names=None)
        with self.assertRaises(ValueError):
            ex.explain(tiny_model(), make_seq(present=()), feature_names=NAMES)


if __name__ == "__main__":
    unittest.main()


class TestVerdictRule(unittest.TestCase):
    """The fixed rule in lstm_explain_run: top group must be an expected one AND >= 1 nat."""
    def setUp(self):
        from src.experiments import lstm_explain_run as r
        self.r = r

    def _e(self, groups, feats):
        order = lambda d: sorted(d, key=lambda n: (-d[n], n))
        return {"group": {"delta": groups, "top_names": order(groups)},
                "feature": {"delta": feats, "top_names": order(feats)}}

    def test_expected_names_exist_and_cover_all_classes(self):
        from src.models import sequence
        self.assertEqual(set(self.r.EXPECTED_GROUPS), set(sequence.ATTACK_CLASSES))
        self.assertEqual(set(self.r.EXPECTED_FEATURES), set(sequence.ATTACK_CLASSES))
        for c in sequence.ATTACK_CLASSES:
            self.assertTrue(self.r.EXPECTED_GROUPS[c] <= set(ex.GROUP_MEMBERS), c)
            self.assertTrue(self.r.EXPECTED_FEATURES[c] <= set(common.FEATURES), c)

    def test_rule(self):
        v = self.r.verdict("arp_mitm", self._e({"ARP": 2.0, "MQTT": 0.1}, {"ip_mac_conflict": 1.5}))
        self.assertTrue(v["useful"]) ; self.assertTrue(v["top_feature_expected"])
        v = self.r.verdict("arp_mitm", self._e({"ARP": 0.9, "MQTT": 0.1}, {"ip_mac_conflict": 0.5}))
        self.assertFalse(v["useful"])                       # below 1 nat
        v = self.r.verdict("arp_mitm", self._e({"MQTT": 3.0, "ARP": 2.0}, {"mqtt_pkts": 2.0}))
        self.assertFalse(v["useful"]) ; self.assertFalse(v["top_group_expected"])
