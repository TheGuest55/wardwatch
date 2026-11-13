"""Tests for the LSTM classifier (src/models/lstm.py): shapes, bit-for-bit determinism, class weights,
noise mask, validation threshold, macro-F1, artefact round trip."""
import os
import tempfile
import unittest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.models import lstm

NF, K, C = 3, 4, 3


def toy_data(n, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, K, NF + 1).astype(np.float32)
    X[:, :, NF] = 1.0                                   # is_present
    X[:, :2, :] = 0.0                                   # two filled windows at the start of each seq
    cls = np.zeros(n, dtype=np.int64)
    sig = X[:, -1, 0]
    cls[sig > 0.5] = 1
    cls[sig > 1.3] = 2
    X[:, -1, 1] += cls                                  # class-dependent signal
    return X, cls


def run_once(seed, threads=1, epochs=3):
    lstm.set_determinism(seed, threads)
    Xtr, ytr = toy_data(400, 1)
    Xva, yva = toy_data(200, 2)
    cfg = lstm.make_cfg(hidden=8, epochs=epochs, patience=5, seed=seed, threads=threads,
                        batch_size=64, n_feat=NF)
    model = lstm.LSTMClassifier(NF + 1, 8, 1, cfg["dropout"], C)
    res = lstm.train(model, {"X_train": Xtr, "y_train": ytr, "X_val": Xva, "y_val": yva}, cfg)
    return model, res, lstm.predict_proba(model, Xva), cfg, (Xva, yva)


class TestModel(unittest.TestCase):
    def test_forward_shape_and_softmax(self):
        lstm.set_determinism(0, 1)
        m = lstm.LSTMClassifier(NF + 1, 8, 1, 0.25, C)
        x = torch.randn(5, K, NF + 1)
        self.assertEqual(tuple(m(x).shape), (5, C))
        p = lstm.predict_proba(m, x.numpy())
        self.assertEqual(p.shape, (5, C))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(lstm.binary_score(p), 1 - p[:, 0], atol=0)

    def test_two_layers_forward(self):
        m = lstm.LSTMClassifier(NF + 1, 8, 2, 0.25, C)
        self.assertEqual(tuple(m(torch.randn(2, K, NF + 1)).shape), (2, C))


class TestWeights(unittest.TestCase):
    def test_sqrt_ratio_formula(self):
        y = np.array([0] * 100 + [1] * 25 + [2] * 4)
        np.testing.assert_allclose(lstm.class_weights(y, 3), [1.0, 2.0, 5.0])

    def test_absent_class_gets_weight_one(self):
        y = np.array([0] * 50 + [1] * 2)
        w = lstm.class_weights(y, 4)
        self.assertEqual(w[2], 1.0)
        self.assertAlmostEqual(w[1], 5.0)

    def test_ignores_negative_labels(self):
        y = np.array([0] * 16 + [1] * 4 + [-1] * 3)
        np.testing.assert_allclose(lstm.class_weights(y, 2), [1.0, 2.0])


class TestNoise(unittest.TestCase):
    def test_noise_only_on_present_feature_channels(self):
        for use_presence in (True, False):
            x = torch.randn(64, K, NF + (1 if use_presence else 0))
            if use_presence:
                x[:, :, NF] = 1.0
                x[:, :2, :] = 0.0                       # filled windows: zeros and is_present = 0
            else:
                x[:, :2, :] = 0.0
            g = torch.Generator().manual_seed(0)
            y = lstm.add_noise(x, 0.1, NF, use_presence, g)
            self.assertTrue(torch.equal(y[:, :2], x[:, :2]))                 # filled windows untouched
            if use_presence:
                self.assertTrue(torch.equal(y[:, :, NF], x[:, :, NF]))       # is_present untouched
            self.assertFalse(torch.equal(y[:, 2:, :NF], x[:, 2:, :NF]))      # present features noised
            self.assertGreater(float((y - x).abs().max()), 0.0)
            self.assertLess(float((y - x).abs().max()), 1.0)

    def test_zero_sigma_is_identity(self):
        x = torch.randn(4, K, NF + 1)
        self.assertTrue(torch.equal(lstm.add_noise(x, 0.0, NF, True, torch.Generator()), x))

    def test_noise_is_seeded(self):
        x = torch.randn(8, K, NF + 1)
        a = lstm.add_noise(x, 0.1, NF, True, torch.Generator().manual_seed(3))
        b = lstm.add_noise(x, 0.1, NF, True, torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(a, b))


class TestTrainingDeterminism(unittest.TestCase):
    def test_same_seed_same_threads_is_bit_for_bit(self):
        m1, r1, p1, _, _ = run_once(7)
        m2, r2, p2, _, _ = run_once(7)
        self.assertTrue(np.array_equal(p1, p2))
        self.assertEqual(r1["history"], r2["history"])
        self.assertEqual(r1["best_epoch"], r2["best_epoch"])
        for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
            self.assertTrue(torch.equal(v1, v2), k1)

    def test_different_seed_differs(self):
        _, _, p1, _, _ = run_once(7)
        _, _, p2, _, _ = run_once(8)
        self.assertFalse(np.array_equal(p1, p2))

    def test_curves_and_early_stopping_bookkeeping(self):
        model, res, probs, cfg, (Xva, yva) = run_once(0, epochs=4)
        h = res["history"]
        self.assertEqual(len(h), res["epochs_run"])
        for key in ("epoch", "train_loss", "val_loss", "train_pr_auc", "val_pr_auc"):
            self.assertIn(key, h[0])
        best = max(h, key=lambda r: r["val_pr_auc"])
        self.assertEqual(res["best_epoch"], best["epoch"])
        self.assertAlmostEqual(res["best_val_pr_auc"], best["val_pr_auc"])
        # restored weights are the best epoch's: validation PR-AUC now equals the recorded best
        ap = lstm._pr_auc((yva > 0).astype(int), lstm.binary_score(probs))
        self.assertAlmostEqual(ap, best["val_pr_auc"], places=6)
        self.assertLessEqual(res["epochs_run"], 4)

    def test_patience_stops_training(self):
        lstm.set_determinism(0, 1)
        Xtr, ytr = toy_data(100, 1)
        rng = np.random.RandomState(5)
        Xva, _ = toy_data(100, 2)
        yva = rng.randint(0, C, size=100)               # unlearnable validation labels
        cfg = lstm.make_cfg(hidden=4, epochs=30, patience=2, batch_size=50, n_feat=NF)
        m = lstm.LSTMClassifier(NF + 1, 4, 1, 0.25, C)
        res = lstm.train(m, {"X_train": Xtr, "y_train": ytr, "X_val": Xva, "y_val": yva}, cfg)
        self.assertLess(res["epochs_run"], 30)


class TestThresholdAndMacro(unittest.TestCase):
    def test_threshold_maximises_validation_f1(self):
        y = [0, 0, 1, 1, 1]
        s = [0.1, 0.4, 0.35, 0.8, 0.9]
        # F1 by threshold: .1 -> .75, .35 -> .857, .4 -> .667, .8 -> .8, .9 -> .5
        self.assertEqual(lstm.pick_threshold(y, s), 0.35)

    def test_threshold_tie_goes_to_the_highest(self):
        y = [0, 1, 0, 1]
        s = [0.1, 0.9, 0.1, 0.9]
        self.assertEqual(lstm.pick_threshold(y, s), 0.9)

    def test_threshold_requires_positives(self):
        with self.assertRaises(ValueError):
            lstm.pick_threshold([0, 0], [0.1, 0.2])

    def test_macro_f1_toy(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        probs = np.eye(3)[[0, 1, 1, 1, 2, 0]]
        self.assertAlmostEqual(lstm.macro_f1(y, probs), (0.5 + 0.8 + 2 / 3) / 3, places=9)

    def test_macro_f1_zero_division_is_zero(self):
        y = np.array([0, 0, 1])
        probs = np.eye(3)[[0, 0, 2]]                    # class 2 predicted but absent; class 1 missed
        self.assertAlmostEqual(lstm.macro_f1(y, probs), (1.0 + 0.0 + 0.0) / 3, places=9)


class TestMacroFixedLabels(unittest.TestCase):
    def test_missing_class_counts_as_zero_unlike_sklearn_default(self):
        from sklearn.metrics import f1_score
        y = np.array([0, 0, 1, 1])
        probs = np.eye(3)[[0, 0, 1, 1]]                 # perfect on classes 0 and 1; class 2 absent everywhere
        default = f1_score(y, probs.argmax(1), average="macro", zero_division=0)
        self.assertAlmostEqual(default, 1.0)
        self.assertAlmostEqual(lstm.macro_f1(y, probs), 2 / 3, places=9)   # fixed set: class 2 -> F1 0
        self.assertNotEqual(lstm.macro_f1(y, probs), default)

    def test_n_classes_comes_from_probs_width(self):
        y = np.array([0, 1])
        probs = np.eye(9)[[0, 1]]
        self.assertAlmostEqual(lstm.macro_f1(y, probs), 2 / 9, places=9)


class TestArtefact(unittest.TestCase):
    def test_round_trip_gives_identical_scores(self):
        model, res, probs, cfg, (Xva, yva) = run_once(3)
        sc = StandardScaler().fit(np.random.RandomState(0).randn(20, NF))
        names = ["a", "b", "c"]
        order = ["benign", "x", "y"]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "m.pt")
            lstm.save_artefact(path, model, sc, K, names, order, 0.37, cfg)
            self.assertTrue(os.path.isfile(path))
            a = lstm.load_artefact(path)
        np.testing.assert_array_equal(lstm.predict_proba(a["model"], Xva), probs)
        self.assertEqual((a["k"], a["feature_names"], a["class_order"], a["threshold"]),
                         (K, names, order, 0.37))
        probe = np.random.RandomState(1).randn(5, NF)
        np.testing.assert_array_equal(a["scaler"].transform(probe), sc.transform(probe))
        self.assertEqual(a["cfg"]["hidden"], 8)


if __name__ == "__main__":
    unittest.main()
