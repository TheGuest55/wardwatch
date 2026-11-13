"""Pipeline tests for the model harness: split-respecting loader with train-only fit discipline,
and metrics that match sklearn on a toy set."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

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
        self.assertAlmostEqual(m["accuracy"], float((pred == y).mean()), places=10)

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

    def test_fpr_at_recall_handles_tied_scores(self):
        # Review N3: two of the three negatives tie with the positive at 0.5. Reaching recall 1.0
        # needs threshold 0.5, which predicts BOTH tied negatives positive -> fpr = 2/3, not 0.
        y = np.array([1, 1, 0, 0, 0])
        score = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
        fpr, thr, rec = mmetrics.fpr_at_recall(y, score, target_recall=1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(thr, 0.5)
        self.assertAlmostEqual(fpr, 2.0 / 3.0, places=10)

    def test_single_window_latency_shape(self):
        # Review N4: >=200 timed single-window calls after warm-up, median and p95 reported.
        calls = {"n": 0}

        def score_one(x):
            calls["n"] += 1
            return float(np.asarray(x, dtype=float).sum())

        inputs = [np.arange(51, dtype=float) for _ in range(8)]  # ~ one feature row each
        res = mmetrics.single_window_latency_ms(score_one, inputs, n_calls=10, warmup=20)
        self.assertEqual(res["n"], 200)                 # floored at 200 even if asked fewer
        self.assertEqual(calls["n"], 200 + 20)          # warm-up calls happened too
        self.assertGreaterEqual(res["p95_ms"], res["median_ms"])
        self.assertTrue(np.isfinite(res["median_ms"]))


class TestLoaderPaths(unittest.TestCase):
    def _write(self, d, features):
        d = Path(d)
        df = _toy_df()
        df.to_parquet(d / "testbed_windows.parquet", index=False)
        (d / "feature_list.json").write_text(json.dumps({"features": features}))
        return str(d / "testbed_windows.parquet"), str(d / "feature_list.json")

    def test_accepts_string_paths(self):
        with tempfile.TemporaryDirectory() as d:
            dp, fp = self._write(d, ["f1", "f2"])
            df, fl = mdata.load_dataset(dp, fp)          # strings, not Path (m-new-3)
            self.assertEqual(fl["features"], ["f1", "f2"])

    def test_rejects_identifier_columns(self):
        with tempfile.TemporaryDirectory() as d:
            dp, fp = self._write(d, ["f1", "host_ip"])   # host_ip is metadata
            with self.assertRaises(ValueError):
                mdata.load_dataset(dp, fp)

    def test_default_is_v2(self):
        self.assertEqual(mdata.V2_DIR.name, "v2")
        self.assertEqual(mdata.V2_DIR.parent.name, "processed")


class TestProvenance(unittest.TestCase):
    def test_record_has_autofilled_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            split = Path(d) / "split_v2.json"
            split.write_text('{"train":["r01"],"val":["r04"],"test":["r05"]}')
            dataset = Path(d) / "testbed_windows.parquet"
            dataset.write_bytes(b"not-a-real-parquet-but-hashable")
            rec = tracking.build_run_record("A", "xgboost", "classifier",
                                            seed=60077, split_path=str(split),
                                            dataset_path=str(dataset))
            prov = rec["provenance"]
            for k in ("seed", "split_file", "split_sha256", "dataset_sha256",
                      "git_commit", "git_dirty", "image_digest"):
                self.assertIn(k, prov)
            self.assertEqual(prov["seed"], 60077)
            self.assertEqual(prov["split_sha256"], hashlib.sha256(split.read_bytes()).hexdigest())
            self.assertEqual(prov["dataset_sha256"],
                             hashlib.sha256(dataset.read_bytes()).hexdigest())
            self.assertIn(prov["git_dirty"], (True, False, None))

    def test_report_record_refuses_null_provenance(self):
        # for_report=True with missing provenance (no seed/split/dataset, git env unset) must raise
        # before any write.
        rec = tracking.build_run_record("A", "xgboost", "classifier", seed=None, for_report=True)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run.json"
            with self.assertRaises(ValueError):
                tracking.log_run(rec, json_path=str(out), use_mlflow=False)
            self.assertFalse(out.exists())   # nothing written

    def test_report_record_writes_when_provenance_complete(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            split = Path(d) / "split_v2.json"; split.write_text('{"train":["r01"]}')
            dataset = Path(d) / "ds.parquet"; dataset.write_bytes(b"x")
            os.environ["WARDWATCH_GIT_COMMIT"] = "deadbeef"
            os.environ["WARDWATCH_GIT_DIRTY"] = ""            # empty porcelain => clean (not null)
            os.environ["WARDWATCH_IMAGE_DIGEST"] = "sha256:abc"
            try:
                rec = tracking.build_run_record(
                    "A", "xgboost", "classifier", seed=0, for_report=True,
                    split_path=str(split), dataset_path=str(dataset))
                self.assertEqual(rec["provenance"]["git_dirty"], False)   # empty => clean, non-null
                out = Path(d) / "run.json"
                tracking.log_run(rec, json_path=str(out), use_mlflow=False)   # must not raise
                self.assertTrue(out.exists())
            finally:
                for k in ("WARDWATCH_GIT_COMMIT", "WARDWATCH_GIT_DIRTY", "WARDWATCH_IMAGE_DIGEST"):
                    os.environ.pop(k, None)

    def test_dirty_env_porcelain_nonempty_is_dirty(self):
        import os
        os.environ["WARDWATCH_GIT_DIRTY"] = " M README.md"
        try:
            self.assertTrue(tracking.run_provenance()["git_dirty"])
        finally:
            os.environ.pop("WARDWATCH_GIT_DIRTY", None)

    def test_json_written_without_mlflow(self):
        with tempfile.TemporaryDirectory() as d:
            rec = tracking.build_run_record("B", "autoencoder", "anomaly", seed=1)
            out = Path(d) / "run.json"
            tracking.log_run(rec, json_path=str(out), use_mlflow=False)
            self.assertTrue(out.exists())
            loaded = json.loads(out.read_text())
            self.assertEqual(loaded["model"], "autoencoder")
            self.assertIn("provenance", loaded)


if __name__ == "__main__":
    unittest.main()
