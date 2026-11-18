"""Tests for src/experiments/run_tree_report.py (report-grade XGBoost/RF runs for Experiments A, B, C) on the
synthetic frames of tests/pipeline/synth_c.py. The key test runs src/models/train_tree.py itself on the same
data (its module paths pointed at a temp dir) and checks the wrapper reproduces its per-seed numbers exactly,
so the wrapper changes the records, not the protocol."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import run_tree_report as rt
from src.models import train_tree as tt
from tests.pipeline.synth_c import make_synthetic

ENV = {"WARDWATCH_GIT_COMMIT": "0" * 40, "WARDWATCH_GIT_DIRTY": "false",
       "WARDWATCH_IMAGE_DIGEST": "sha256:" + "a" * 64}
SEEDS = (0, 1)
GRIDS = {"xgboost": {"max_depth": [2, 3], "n_estimators": [20], "learning_rate": [0.3]},
         "random_forest": {"n_estimators": [20], "max_depth": [None, 4], "min_samples_leaf": [1]}}
TT_GLOBALS = ("V2", "DATASET", "SPLIT", "OUT_DIR", "MODEL_DIR", "CIC_DIR", "GRIDS", "SEEDS")


def make_env(root, paths, **kw):
    root = Path(root)
    return rt.Env(dataset=paths["tb_windows"], fl=paths["tb_fl"], split=paths["split_path"], runs=root / "runs",
                  model_dir=root / "models", pred_dir=root / "preds", cic_dir=Path(paths["cic_windows"]).parent,
                  for_report=True, seeds=SEEDS, **kw)


def run_train_tree(paths, out, model, experiment):
    """train_tree.main on the synthetic data, unchanged code, module paths patched for the call."""
    saved = {k: getattr(tt, k) for k in TT_GLOBALS}
    tb_dir = Path(paths["tb_windows"]).parent
    try:
        tt.V2, tt.DATASET, tt.SPLIT = tb_dir, Path(paths["tb_windows"]), Path(paths["split_path"])
        tt.OUT_DIR, tt.MODEL_DIR = out / "expA", out / "models"
        tt.CIC_DIR, tt.GRIDS, tt.SEEDS = Path(paths["cic_windows"]).parent, GRIDS, SEEDS
        argv = sys.argv
        sys.argv = ["train_tree", "--model", model, "--experiment", experiment, "--no-mlflow"]
        try:
            tt.main()
        finally:
            sys.argv = argv
    finally:
        for k, v in saved.items():
            setattr(tt, k, v)
    sub = {"A": out / "expA", "B": tb_dir / "runs" / "expB", "C": tb_dir / "runs" / "expC"}[experiment]
    return (json.loads((sub / f"{model}_details.json").read_text()),
            json.loads((sub / f"{model}.json").read_text()))


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old = {k: os.environ.get(k) for k in ENV}
        os.environ.update(ENV)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.paths = make_synthetic(cls.root / "data")
        cls._grids = tt.GRIDS
        tt.GRIDS = GRIDS                                    # the wrapper reads train_tree's grid
        cls.env = make_env(cls.root / "out", cls.paths)
        for m in rt.MODELS:
            rt.run_a(m, cls.env)
            rt.run_b(m, cls.env)
            rt.run_c(m, cls.env)

    @classmethod
    def tearDownClass(cls):
        tt.GRIDS = cls._grids
        for k, v in cls._old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        cls._tmp.cleanup()

    def recs(self, exp, pattern):
        """Per-seed records only (the glob also matches _selected/_summary/_details files)."""
        import re
        return [json.loads(p.read_text()) for p in sorted((self.env.runs / f"exp{exp}").glob(pattern))
                if re.search(r"_s\d+\.json$", p.name) or "_grid_" in p.name]


class TestRecords(Base):
    def test_counts_names_and_provenance(self):
        classes = sorted(pd.read_parquet(self.paths["tb_windows"]).query("label == 'attack'")["attack_class"].unique())
        for m in rt.MODELS:
            a = self.recs("A", f"{m}_s*.json")
            b = self.recs("B", f"{m}_*_s*.json")
            c = self.recs("C", f"{m}_*_astrained_s*.json")
            self.assertEqual(len(a), len(SEEDS))
            self.assertEqual(len(b), len(classes) * len(SEEDS))
            self.assertEqual(len(c), 2 * len(SEEDS))
            for r in a + b + c:
                self.assertTrue(r["for_report"])
                self.assertEqual(r["role"], "classifier")
                self.assertEqual(r["provenance"]["git_commit"], "0" * 40)
                self.assertIn("n_jobs", r["params"])
            grid = self.recs("A", f"{m}_grid_*.json")
            self.assertEqual(len(grid), 2)
            self.assertFalse(any(r["for_report"] for r in grid))
            self.assertTrue((self.env.runs / "expA" / f"{m}_selected.json").exists())

    def test_both_views_and_macro_f1(self):
        for m in rt.MODELS:
            for r in self.recs("A", f"{m}_s*.json"):
                for k in ("test_all_f1", "test_noatk_f1", "val_all_f1", "test_recall_attacker",
                          "test_macro_f1", "test_macro_f1_traintree"):
                    self.assertIn(k, r["metrics"])
                self.assertTrue(0.0 <= r["metrics"]["test_macro_f1"] <= 1.0)
                self.assertIn("labels=range(9)", r["params"]["macro_f1_definition"])
            for r in self.recs("C", f"{m}_*_astrained_s*.json"):
                for k in ("tgt_test_all_f1", "tgt_test_noatk_f1", "src_test_all_f1", "val_noatk_f1"):
                    self.assertIn(k, r["metrics"])

    def test_predictions_layout(self):
        for m in rt.MODELS:
            for s in SEEDS:
                p = pd.read_parquet(self.env.runs / "expA" / "predictions" / f"{m}_s{s}.parquet")
                r = json.loads((self.env.runs / "expA" / f"{m}_s{s}.json").read_text())
                self.assertEqual(sorted(p["split"].unique()), ["test", "val"])
                self.assertIn("host_role", p.columns)
                np.testing.assert_array_equal(p["pred"], (p["score"] >= r["params"]["threshold"]).astype(int))
            self.assertTrue((self.env.pred_dir / f"A_{m}.parquet").exists())
            c = pd.read_parquet(self.env.runs / "expC" / "predictions" / f"{m}_tb2cic_astrained_s0.parquet")
            self.assertEqual(set(c["domain"]), {"val", "src_test", "tgt_test"})
            self.assertEqual(set(c.loc[c["domain"] == "tgt_test", "dataset"]), {"cic"})
        self.assertTrue((self.env.model_dir / "xgboost_binary_seed0.json").exists())
        self.assertTrue((self.env.model_dir / "random_forest_binary_seed0.joblib").exists())

    def test_b_withheld_class_never_in_train_or_val(self):
        for r in self.recs("B", "xgboost_*_s0.json"):
            X = r["params"]["held_out_class"]
            p = pd.read_parquet(self.env.runs / "expB" / "predictions" / f"xgboost_{X}_s0.parquet")
            val = p[p["split"] == "val"]
            self.assertFalse(((val["label"] == "attack") & (val["attack_class"] == X)).any())
            test_cls = set(p.loc[(p["split"] == "test") & (p["label"] == "attack"), "attack_class"])
            self.assertEqual(test_cls, {X})

    def test_c_tags_and_source_threshold(self):
        for r in self.recs("C", "random_forest_*_astrained_s*.json"):
            self.assertIn("cic_files", r["tags"])
            self.assertFalse(any("ddos" in s.lower() for v in r["tags"]["cic_files"].values() for s in v))
            self.assertEqual(r["params"]["variant"], "astrained")


class TestEquivalenceWithTrainTree(Base):
    """The wrapper must reproduce train_tree's per-seed numbers exactly (same functions, same data)."""

    def test_a(self):
        for m in rt.MODELS:
            det, agg = run_train_tree(self.paths, self.root / f"tt_{m}", m, "A")
            for i, s in enumerate(SEEDS):
                r = json.loads((self.env.runs / "expA" / f"{m}_s{s}.json").read_text())
                self.assertEqual(r["metrics"]["test_all_f1"], det["per_seed"][i]["f1"])
                self.assertEqual(r["params"]["threshold"], det["per_seed"][i]["threshold"])
                self.assertEqual(r["metrics"]["test_all_fpr"], det["per_seed"][i]["fpr"])
            ours = np.mean([json.loads((self.env.runs / "expA" / f"{m}_s{s}.json").read_text())["metrics"]
                            ["test_macro_f1_traintree"] for s in SEEDS])
            self.assertAlmostEqual(ours, agg["metrics"]["macro_f1_mean"], places=12)

    def test_b(self):
        for m in rt.MODELS:
            det, _ = run_train_tree(self.paths, self.root / f"ttb_{m}", m, "B")
            for X, d in det.items():
                for i, s in enumerate(SEEDS):
                    r = json.loads((self.env.runs / "expB" / f"{m}_{X}_s{s}.json").read_text())
                    self.assertEqual(r["metrics"]["test_all_f1"], d["per_seed"][i]["f1"], (m, X, s))
                    self.assertEqual(r["params"]["threshold"], d["per_seed"][i]["threshold"])

    def test_c(self):
        for m in rt.MODELS:
            det, _ = run_train_tree(self.paths, self.root / f"ttc_{m}", m, "C")
            for key, tag in (("testbed->cic", "tb2cic"), ("cic->testbed", "cic2tb")):
                src = key.split("->")[0]
                for i, s in enumerate(SEEDS):
                    r = json.loads((self.env.runs / "expC" / f"{m}_{tag}_astrained_s{s}.json").read_text())
                    self.assertEqual(r["metrics"]["tgt_test_all_f1"], det["per_seed"][key][i]["f1"], (m, tag, s))
                    self.assertEqual(r["params"]["threshold"], det["thresholds"][src][i])


class TestGuards(unittest.TestCase):
    def setUp(self):
        self._old = {k: os.environ.get(k) for k in ENV}
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.paths = make_synthetic(self.root / "data")
        self._grids = tt.GRIDS
        tt.GRIDS = GRIDS

    def tearDown(self):
        tt.GRIDS = self._grids
        for k, v in self._old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self._tmp.cleanup()

    def test_ddos_file_stops_c_before_training(self):
        os.environ.update(ENV)
        p = Path(self.paths["cic_windows"])
        df = pd.read_parquet(p)
        df.loc[df["run_id"] == "Benign_train", "run_id"] = "TCP_IP-DDoS-SYN1_train"
        df.to_parquet(p, index=False)
        env = make_env(self.root / "out", self.paths)
        with self.assertRaises(SystemExit):
            rt.run_c("xgboost", env)
        self.assertEqual(list((env.runs / "expC").glob("*.json")) if (env.runs / "expC").exists() else [], [])

    def test_for_report_refuses_missing_provenance(self):
        for k in ENV:
            os.environ.pop(k, None)
        env = make_env(self.root / "out", self.paths)
        with self.assertRaises((SystemExit, AssertionError, ValueError, RuntimeError)):
            rt.run_a("xgboost", env)
        self.assertFalse((env.runs / "expA").exists() and any((env.runs / "expA").glob("*_s*.json")))


if __name__ == "__main__":
    unittest.main()
