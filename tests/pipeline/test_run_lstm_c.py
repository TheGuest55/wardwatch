"""End-to-end tests of the LSTM Experiment C runner on synthetic CIC-like frames (src/experiments/run_lstm_c.py)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import run_lstm_c as rc
from src.models import lstm
from src.models import sequence_c as sc
from tests.pipeline.synth_c import make_synthetic

ENV = {"WARDWATCH_GIT_COMMIT": "0" * 40, "WARDWATCH_GIT_DIRTY": "false", "WARDWATCH_IMAGE_DIGEST": "sha256:" + "a" * 64}


def _selected(root, k=12, h=32):
    p = Path(root) / "selected_config.json"
    p.write_text(json.dumps({"k": k, "hidden": h}))
    return p


def _smoke(root, paths, tag):
    runs, models = Path(root) / f"runs_{tag}", Path(root) / f"models_{tag}"
    rc.run_smoke(paths, 1, runs, models, selected=_selected(root))
    return runs, models


class TestRunnerC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old = {k: os.environ.get(k) for k in ENV}
        os.environ.update(ENV)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = cls._tmp.name
        cls.paths = make_synthetic(Path(cls.root) / "a")
        cls.runs, cls.models = _smoke(cls.root, cls.paths, "a")

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cls._tmp.cleanup()

    def _rec(self, name):
        return json.loads((self.runs / f"{name}.json").read_text())

    def test_smoke_writes_two_complete_report_records(self):
        for tag in ("tb2cic", "cic2tb"):
            r = self._rec(f"C_{tag}_s0")
            self.assertTrue(r["for_report"])
            self.assertEqual(r["experiment"], "C")
            self.assertEqual([k for k in ("seed", "split_sha256", "dataset_sha256", "git_commit", "git_dirty",
                                          "image_digest") if r["provenance"].get(k) is None], [])
            p = r["params"]
            self.assertEqual((p["k"], p["hidden"], p["n_features"], p["use_presence"]), (12, 32, 48, True))
            self.assertEqual(p["threshold_rule"], "source-val-max-F1 (as trained)")
            self.assertEqual(sorted(p["shared_classes"]), sorted(sc.SHARED))
            t = r["tags"]
            for key in ("session_parquet_sha256", "cic_windows_sha256", "cic_feature_list_sha256",
                        "cic_extract_report_sha256", "sequence_counts", "cic_files", "row_table_source"):
                self.assertIn(key, t)
            self.assertIn("Benign_train", t["cic_files"]["train"])
            self.assertTrue(all("ddos" not in s.lower() for v in t["cic_files"].values() for s in v))
            for pre in ("val", "src_test", "tgt_test"):
                for view in ("all", "noatk"):
                    for m in ("f1", "precision", "recall", "fpr", "pr_auc"):
                        self.assertIn(f"{pre}_{view}_{m}", r["metrics"])
                self.assertIn(f"{pre}_all_recall_syn_flood", r["metrics"])

    def test_predictions_cover_three_domains_with_five_classes_and_curves(self):
        for tag, src, tgt in (("tb2cic", "testbed", "cic"), ("cic2tb", "cic", "testbed")):
            f = pd.read_parquet(self.runs / "predictions" / f"C_{tag}_s0.parquet")
            self.assertEqual(set(f["domain"]), {"val", "src_test", "tgt_test"})
            self.assertEqual(set(f[f.domain == "tgt_test"]["dataset"]), {tgt})
            self.assertEqual(set(f[f.domain != "tgt_test"]["dataset"]), {src})
            for c in sc.C_CLASSES:
                self.assertIn(f"p_{c}", f.columns)
            np.testing.assert_allclose(f[[f"p_{c}" for c in sc.C_CLASSES]].sum(axis=1), 1.0, atol=1e-5)
            self.assertTrue(set(f["attack_class"].dropna()) <= set(sc.SHARED))
            c = json.loads((self.runs / "curves" / f"C_{tag}_s0.json").read_text())
            self.assertEqual(c["classes"], sc.C_CLASSES)
        self.assertTrue((self.runs / "summary.md").exists())

    def test_artefact_scaler_is_the_source_scaler_and_model_has_five_outputs(self):
        for tag, src in (("tb2cic", "testbed"), ("cic2tb", "cic")):
            art = lstm.load_artefact(self.models / f"C_{tag}_s0.pt")
            want = sc.fit_source_scaler(sc.load_domain(src, self.paths))
            np.testing.assert_array_equal(art["scaler"].mean_, want.mean_)
            np.testing.assert_array_equal(art["scaler"].scale_, want.scale_)
            self.assertEqual(art["class_order"], sc.C_CLASSES)
            self.assertEqual(len(art["feature_names"]), 48)
            self.assertEqual(art["model"].head.out_features, 5)

    def test_rerun_skips_existing_records(self):
        ctx = rc.make_ctx(self.paths, 1, self.runs, self.models, smoke=True, epochs=1,
                          selected=_selected(self.root))
        self.assertEqual(set(rc.run_all(ctx, seeds=(0,)).values()), {"skipped"})

    def test_target_perturbation_leaves_threshold_and_source_metrics_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            p2 = make_synthetic(Path(t) / "a")
            c = pd.read_parquet(p2["cic_windows"])
            feats = sc.load_domain("cic", p2).feats
            c[feats] = c[feats] * 3.0 + 50.0                       # every target row, train/val/test
            c.to_parquet(p2["cic_windows"], index=False)
            ctx = rc.make_ctx(p2, 1, Path(t) / "runs", Path(t) / "models", smoke=True, epochs=1,
                              selected=_selected(t))
            rc.run_one(ctx, "testbed", "cic", "tb2cic", 0)
            new = json.loads((Path(t) / "runs" / "C_tb2cic_s0.json").read_text())
        old = self._rec("C_tb2cic_s0")
        self.assertEqual(new["params"]["threshold"], old["params"]["threshold"])
        self.assertEqual(new["params"]["best_epoch"], old["params"]["best_epoch"])
        for k, v in old["metrics"].items():
            if k.startswith(("val_", "src_test_")) and isinstance(v, float) and v == v:
                self.assertEqual(new["metrics"][k], v, k)

    def test_ddos_file_stops_the_run(self):
        with tempfile.TemporaryDirectory() as t:
            p2 = make_synthetic(Path(t) / "a")
            c = pd.read_parquet(p2["cic_windows"])
            c["run_id"] = c["run_id"].replace({"Benign_test": "DDoS-ICMP_Flood_test"})
            c.to_parquet(p2["cic_windows"], index=False)
            with self.assertRaises(SystemExit):
                rc.make_ctx(p2, 1, Path(t) / "r", Path(t) / "m", selected=_selected(t))

    def test_config_other_than_frozen_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(SystemExit):
                rc.load_selected(_selected(t, k=6, h=64))

    def test_check_runs_and_reports_clean(self):
        self.assertEqual(rc.check(self.paths), 0)

    def test_summary_has_selection_score_and_row_table(self):
        s = (self.runs / "summary.md").read_text()
        self.assertIn("selection score", s)
        self.assertIn("dropped by context rule", s)
        self.assertIn("CIC files", s)


if __name__ == "__main__":
    unittest.main()
