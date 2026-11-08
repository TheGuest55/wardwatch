"""Pipeline tests for the model harness: split-respecting loader with train-only fit discipline,
and metrics that match sklearn on a toy set."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from src.models import data as mdata
from src.models import metrics as mmetrics
from src.models import tracking


def _toy_df():
    n = 90
    rng = np.random.RandomState(0)
    split = np.array((["train"] * 50) + (["val"] * 20) + (["test"] * 20))
    label = np.where(rng.rand(n) < 0.3, "attack", "benign")
    return pd.DataFrame({
        "f1": rng.rand(n) * 10 + 5,
        "f2": rng.rand(n) * 2,
        "host_ip": "10.0.0.1", "host_mac": "aa", "host_role": "monitor",
        "label": label, "attack_class": np.where(label == "attack", "syn_flood", None),
        "split": split,
    })


FL = {"features": ["f1", "f2"], "metadata": ["host_ip", "host_mac", "host_role"],
      "testbed_only": ["f2"]}


class TestLoader(unittest.TestCase):
    def test_scaler_fit_train_only(self):
        df = _toy_df()
        ds = mdata.Dataset(df, FL, scale=True)
        raw = df.loc[df["split"] == "train", ["f1", "f2"]].to_numpy(dtype=float)
        np.testing.assert_allclose(ds.scaler.mean_, raw.mean(axis=0), rtol=1e-9)
        # transformed train is ~zero-mean; val/test are NOT used to fit, so their mean is generally !=0
        self.assertTrue(np.allclose(ds.X("train").mean(axis=0), 0.0, atol=1e-9))

    def test_splits_disjoint_and_complete(self):
        df = _toy_df()
        ds = mdata.Dataset(df, FL, scale=False)
        total = sum(len(ds.X(s)) for s in ("train", "val", "test"))
        self.assertEqual(total, len(df))
        self.assertEqual(len(ds.X("train")), 50)

    def test_include_testbed_only_toggle(self):
        df = _toy_df()
        full = mdata.Dataset(df, FL, scale=False)
        cross = mdata.Dataset(df, FL, include_testbed_only=False, scale=False)
        self.assertIn("f2", full.feature_names)
        self.assertNotIn("f2", cross.feature_names)   # dropped for cross-dataset (Experiment C)

    def test_normal_only_is_benign(self):
        df = _toy_df()
        ds = mdata.Dataset(df, FL, scale=False)
        n_benign_train = int((df[(df["split"] == "train")]["label"] == "benign").sum())
        self.assertEqual(len(ds.normal_only("train")), n_benign_train)


class TestMetrics(unittest.TestCase):
    def test_matches_sklearn(self):
        rng = np.random.RandomState(1)
        y = (rng.rand(200) < 0.4).astype(int)
        score = np.clip(y * 0.6 + rng.rand(200) * 0.5, 0, 1)
        m = mmetrics.binary_metrics(y, y_score=score, threshold=0.5)
        pred = (score >= 0.5).astype(int)
        self.assertAlmostEqual(m["f1"], f1_score(y, pred), places=10)
        self.assertAlmostEqual(m["roc_auc"], roc_auc_score(y, score), places=10)
        self.assertAlmostEqual(m["pr_auc"], average_precision_score(y, score), places=10)

    def test_fpr_at_recall_separable(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # perfectly separable
        fpr, thr, rec = mmetrics.fpr_at_recall(y, score, target_recall=1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(fpr, 0.0)

    def test_fpr_unreachable_when_no_positives(self):
        y = np.zeros(5, dtype=int)
        fpr, thr, rec = mmetrics.fpr_at_recall(y, np.arange(5) / 5.0)
        self.assertTrue(np.isnan(fpr))


class TestTracking(unittest.TestCase):
    def test_record_schema_and_log_never_raises(self):
        rec = tracking.build_run_record("A", "xgboost", "classifier",
                                        params={"max_depth": 6}, metrics={"f1": 0.9},
                                        tags={"dataset": "testbed", "seed": 0})
        for k in ("experiment", "model", "role", "params", "metrics", "tags", "utc"):
            self.assertIn(k, rec)
        out = tracking.log_run(rec, use_mlflow=True)   # mlflow may be absent; must not raise
        self.assertEqual(out["model"], "xgboost")


if __name__ == "__main__":
    unittest.main()
