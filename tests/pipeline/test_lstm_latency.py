"""Latency script on a tiny model: score_one equals the offline score on a host with gaps, the
equivalence check bites, and run() works end to end and writes a record."""
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.experiments import lstm_latency as ll
from src.models import lstm
from src.models import sequence as seq

NAMES = ["f1", "f2", "f3"]
K = 4
T0 = 1_000_000


def synth_test_df():
    rng = np.random.RandomState(0)
    rows = []
    for host, steps in (("H1", [0, 1, 2, 3, 4, 7, 8, 9, 12, 13]),    # gaps
                        ("H2", [0, 1]),                               # fewer than k windows
                        ("H3", list(range(10)))):
        for i in steps:
            rows.append({"session": "s", "split": "test", "host_mac": host, "host_ip": "ip",
                         "host_role": "monitor", "run_id": None, "window_start_epoch": T0 + 5 * i,
                         **{n: float(v) for n, v in zip(NAMES, rng.randn(3) * 3 + 1)},
                         "label": "attack" if i == 2 else ("boundary" if i == 8 else "benign"),
                         "attack_class": "syn_flood" if i == 2 else None})
    return pd.DataFrame(rows)


def make_artefact(path, df):
    sc = StandardScaler().fit(df[NAMES].to_numpy())
    lstm.set_determinism(0, 1)
    model = lstm.LSTMClassifier(len(NAMES) + 1, 6, 1, 0.25, 9).eval()
    cfg = lstm.make_cfg(hidden=6, n_feat=len(NAMES))
    lstm.save_artefact(path, model, sc, K, NAMES, list(seq.CLASS_ORDER), 0.5, cfg)
    return sc, model


def offline_preds(df, sc, model):
    t = seq.build_sequences(df, sc, NAMES, K, splits=("test",))["test"]
    p = lstm.predict_proba(model, t.X)
    m = t.meta[["session", "host_mac", "window_start_epoch"]].copy()
    m["score"] = lstm.binary_score(p).astype(np.float64)
    return m


class TestLatency(unittest.TestCase):
    def setUp(self):
        self.df = synth_test_df()
        self.tmp = tempfile.TemporaryDirectory()
        self.art = os.path.join(self.tmp.name, "m.pt")
        self.sc, self.model = make_artefact(self.art, self.df)
        self.preds = offline_preds(self.df, self.sc, self.model)

    def tearDown(self):
        self.tmp.cleanup()

    def test_score_one_equals_offline_score_on_hosts_with_gaps(self):
        scorer = ll.Scorer(lstm.load_artefact(self.art))
        streams = ll.host_streams(self.df, NAMES)
        self.assertTrue(ll.has_gaps(streams[("s", "H1")]))
        self.assertFalse(ll.has_gaps(streams[("s", "H3")]))
        replayed = ll.replay(scorer, streams)
        mx, n = ll.check_equivalence(replayed, self.preds)
        self.assertEqual(n, len(self.preds))
        self.assertLess(mx, 1e-6)

    def test_score_one_returns_probabilities(self):
        scorer = ll.Scorer(lstm.load_artefact(self.art))
        it = ll.host_streams(self.df, NAMES)[("s", "H2")][0]
        score, probs = scorer.score_one(it)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertAlmostEqual(score, 1.0 - float(probs[0]), places=6)

    def test_equivalence_check_catches_a_difference_and_a_missing_row(self):
        scorer = ll.Scorer(lstm.load_artefact(self.art))
        replayed = ll.replay(scorer, ll.host_streams(self.df, NAMES))
        bad = self.preds.copy()
        bad.loc[bad.index[0], "score"] += 1e-3
        with self.assertRaises(AssertionError):
            ll.check_equivalence(replayed, bad)
        extra = pd.concat([self.preds, self.preds.iloc[:1].assign(window_start_epoch=1)])
        with self.assertRaises(AssertionError):
            ll.check_equivalence(replayed, extra)

    def test_timing_inputs_wrap_safely(self):
        streams = ll.host_streams(self.df, NAMES)
        items, chosen = ll.timing_inputs(streams)
        self.assertTrue(any(ll.has_gaps(streams[c]) for c in chosen))
        scorer = ll.Scorer(lstm.load_artefact(self.art))
        for i in range(3 * len(items)):                # cycling past the end must not raise
            scorer.score_one(items[i % len(items)])

    def test_run_end_to_end_on_a_tiny_model(self):
        out = os.path.join(self.tmp.name, "latency.json")
        rec, code = ll.run(self.art, self.df, self.preds, out_json=out, n_calls=250, for_report=False)
        self.assertEqual(code, 0)
        with open(out) as fh:
            stored = json.load(fh)
        m = stored["metrics"]
        self.assertGreater(m["median_ms"], 0)
        self.assertGreaterEqual(m["p95_ms"], m["median_ms"])
        self.assertGreaterEqual(m["n"], 200)
        self.assertLess(m["equivalence_max_abs_diff"], 1e-6)
        self.assertEqual(stored["params"]["threads"], 1)
        self.assertIs(stored["tags"]["latency"], True)
        self.assertIn("processor", stored["tags"]["cpu"])
        self.assertEqual(stored["experiment"], "A")


if __name__ == "__main__":
    unittest.main()
