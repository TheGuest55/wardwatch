"""Tests for the autoencoder anomaly detector (src/models/autoencoder.py): forward shape,
reconstruction-error properties, bit-for-bit training determinism, loss actually falls, the
validation threshold rule, cfg typo-guard, and the detection premise (attacks reconstruct worse than
benign). Mirrors tests/pipeline/test_lstm.py in style (unittest, tiny synthetic fixtures)."""
import unittest

import numpy as np
import torch

from src.models import autoencoder as ae

NF = 6


def normal_data(n, seed):
    """Benign-like rows: standardised-ish Gaussian features."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, NF).astype(np.float32)


def attack_data(n, seed):
    """Anomalous rows: shifted + inflated variance, so a model fit on normal_data rebuilds them badly."""
    rng = np.random.RandomState(seed)
    return (rng.randn(n, NF).astype(np.float32) * 4.0 + 6.0)


def train_small(seed, epochs=12):
    ae.set_determinism(seed, 1)
    Xtr = normal_data(400, 1)
    Xva = normal_data(150, 2)
    cfg = ae.make_cfg(hidden=(8, 4), bottleneck=2, epochs=epochs, patience=20,
                      batch_size=64, seed=seed, n_feat=NF)
    model = ae.AutoEncoder(NF, hidden=(8, 4), bottleneck=2)
    res = ae.train(model, Xtr, Xva, cfg)
    return model, res


class TestModel(unittest.TestCase):
    def test_forward_shape_matches_input(self):
        ae.set_determinism(0, 1)
        m = ae.AutoEncoder(NF, hidden=(8, 4), bottleneck=2)
        x = torch.randn(5, NF)
        self.assertEqual(tuple(m(x).shape), (5, NF))

    def test_bottleneck_is_narrowest(self):
        m = ae.AutoEncoder(NF, hidden=(8, 4), bottleneck=2)
        # encoder output width == bottleneck
        z = m.encoder(torch.randn(3, NF))
        self.assertEqual(tuple(z.shape), (3, 2))


class TestReconstructionError(unittest.TestCase):
    def test_error_is_nonneg_finite_and_one_per_row(self):
        m = ae.AutoEncoder(NF, hidden=(8, 4), bottleneck=2)
        X = normal_data(20, 5)
        err = ae.reconstruction_error(m, X)
        self.assertEqual(err.shape, (20,))
        self.assertTrue(np.isfinite(err).all())
        self.assertTrue((err >= 0).all())

    def test_anomaly_scores_alias_equals_reconstruction_error(self):
        m = ae.AutoEncoder(NF, hidden=(8, 4), bottleneck=2)
        X = normal_data(10, 6)
        np.testing.assert_array_equal(ae.anomaly_scores(m, X), ae.reconstruction_error(m, X))

    def test_batching_matches_single_pass(self):
        m = ae.AutoEncoder(NF, hidden=(8, 4), bottleneck=2)
        X = normal_data(100, 7)
        np.testing.assert_allclose(ae.reconstruction_error(m, X, batch_size=8),
                                   ae.reconstruction_error(m, X, batch_size=100), atol=1e-6)


class TestTraining(unittest.TestCase):
    def test_loss_falls(self):
        _, res = train_small(0)
        h = res["history"]
        self.assertLess(h[-1]["train_mse"], h[0]["train_mse"])

    def test_same_seed_is_bit_for_bit(self):
        m1, r1 = train_small(3)
        m2, r2 = train_small(3)
        self.assertEqual(r1["history"], r2["history"])
        for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
            self.assertTrue(torch.equal(v1, v2), k1)

    def test_different_seed_differs(self):
        m1, _ = train_small(3)
        m2, _ = train_small(4)
        same = all(torch.equal(v1, v2) for v1, v2 in
                   zip(m1.state_dict().values(), m2.state_dict().values()))
        self.assertFalse(same)

    def test_detects_attacks_after_training(self):
        """The whole premise: trained on normal, attack rows reconstruct worse than benign rows."""
        model, _ = train_small(0)
        benign_err = ae.reconstruction_error(model, normal_data(200, 9))
        attack_err = ae.reconstruction_error(model, attack_data(200, 10))
        self.assertGreater(attack_err.mean(), benign_err.mean())


class TestThreshold(unittest.TestCase):
    def test_threshold_maximises_validation_f1(self):
        y = [0, 0, 1, 1, 1]
        s = [0.1, 0.4, 0.35, 0.8, 0.9]
        self.assertEqual(ae.pick_threshold(y, s), 0.35)

    def test_threshold_tie_goes_to_the_highest(self):
        y = [0, 1, 0, 1]
        s = [0.1, 0.9, 0.1, 0.9]
        self.assertEqual(ae.pick_threshold(y, s), 0.9)

    def test_threshold_requires_positives(self):
        with self.assertRaises(ValueError):
            ae.pick_threshold([0, 0], [0.1, 0.2])

    def test_predict_applies_threshold(self):
        s = np.array([0.1, 0.5, 0.9])
        np.testing.assert_array_equal(ae.predict(s, 0.5), [0, 1, 1])   # >= threshold is attack


class TestConfig(unittest.TestCase):
    def test_make_cfg_rejects_unknown_key(self):
        with self.assertRaises(ValueError):
            ae.make_cfg(bottlneck=4)                    # typo of "bottleneck"

    def test_make_cfg_allows_n_feat_passthrough(self):
        cfg = ae.make_cfg(n_feat=NF, seed=2)
        self.assertEqual(cfg["seed"], 2)


if __name__ == "__main__":
    unittest.main()
