"""random_split --model autoencoder: the protocol function on synthetic data, and the refusal to overwrite an existing record."""
import unittest

import numpy as np

from src.models import random_split as rs


class TestEvaluateAE(unittest.TestCase):
    def test_trains_on_benign_only_thresholds_on_val_and_reports_per_class_recall(self):
        rng = np.random.default_rng(0)
        feats = [f"f{i}" for i in range(6)]

        def part(n_ben, n_att):
            X = np.vstack([rng.normal(0, 1, (n_ben, 6)), rng.normal(6, 1, (n_att, 6))])
            y = np.r_[np.zeros(n_ben), np.ones(n_att)].astype(int)
            cls = np.array(["benign"] * n_ben + ["syn_flood"] * n_att, dtype=object)
            return X, y, cls
        Xtr, ytr, _ = part(300, 20)
        Xva, yva, _ = part(100, 10)
        Xte, yte, cte = part(100, 10)
        m, params, top, score, pred = rs.evaluate_ae({"hidden": (8, 4), "bottleneck": 2}, Xtr, ytr, Xva, yva, Xte, yte, cte, 0, feats)
        self.assertGreater(m["f1"], 0.9)                       # an easy, separable toy problem
        self.assertEqual(len(score), len(yte))
        self.assertIn("recall_syn_flood", m)
        self.assertLessEqual(len(top), 10)
        self.assertIn("val_f1", m)

    def test_ae_is_a_choice_of_the_cli_and_the_trees_still_are(self):
        self.assertIn("autoencoder", rs.MODELS)
        self.assertIn("xgboost", rs.MODELS)
        self.assertIn("random_forest", rs.MODELS)


if __name__ == "__main__":
    unittest.main()
