"""Tests for the raw-input AE/IF latency path (src/experiments/ae_latency.py)."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments import ae_latency as lat
from src.models import train_autoencoder as ta
from tests.pipeline.synth_c import make_synthetic

ENV = {"WARDWATCH_GIT_COMMIT": "0" * 40, "WARDWATCH_GIT_DIRTY": "false",
       "WARDWATCH_IMAGE_DIGEST": "sha256:" + "a" * 64}


class TestLatency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old = {k: os.environ.get(k) for k in ENV}
        os.environ.update(ENV)
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.paths = make_synthetic(root / "d")
        cls.env = ta.Env(dataset=cls.paths["tb_windows"], fl=cls.paths["tb_fl"], split=cls.paths["split_path"],
                         runs=root / "r", model_dir=root / "m", pred_dir=root / "p", seeds=(0,),
                         ae_cfg={"epochs": 2}, ae_grid=[{"hidden": (8,), "bottleneck": 3}],
                         if_grid=[{"n_estimators": 10, "max_samples": 1.0}], for_report=True)
        for m in ("autoencoder", "isolation_forest"):
            ta.run_a(m, cls.env)
        cls.X = ta._load(cls.env, scale=False).X("test")           # RAW rows

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        cls._tmp.cleanup()

    def test_scaler_is_called_inside_the_timed_function_and_scores_match_offline(self):
        for name in ("autoencoder", "isolation_forest"):
            scorer, _ = lat.load_scorer(name, self.env.model_dir)
            calls = []
            real = scorer.scaler.transform
            scorer.scaler.transform = lambda x, _r=real: (calls.append(np.asarray(x).shape), _r(x))[1]
            s, pred = scorer.score_one(self.X[0])                   # a RAW row
            self.assertEqual(calls, [(1, self.X.shape[1])])         # scaled exactly once, inside the call
            self.assertEqual(pred, int(s >= scorer.threshold))
            scorer.scaler.transform = real
            mx, n = lat.check_equivalence(scorer, self.X, n_equiv=100)
            self.assertLess(mx, 1e-6)
            self.assertEqual(n, 100)

    def test_record_is_complete_and_from_raw_input(self):
        out = Path(self._tmp.name) / "lat.json"
        rec = lat.run("autoencoder", self.env.model_dir, self.X, out, n_calls=200, n_equiv=50, for_report=True,
                      split_path=self.paths["split_path"], dataset_path=self.paths["tb_windows"])
        self.assertTrue(out.exists() and rec["for_report"] and rec["tags"]["latency"])
        self.assertGreaterEqual(rec["metrics"]["n"], 200)
        self.assertIn("raw", rec["tags"]["input"])
        self.assertLess(rec["metrics"]["equivalence_max_abs_diff"], 1e-6)


if __name__ == "__main__":
    unittest.main()
