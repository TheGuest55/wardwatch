"""Tests for the AE/IF fixes: per-seed report records, both metric views, predictions in two formats,
architecture read from A's selection, Experiment C source scaler + threshold-refit variant, on
synthetic frames (tests/pipeline/synth_c.py)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import autoencoder as ae
from src.models import metrics as mm
from src.models import predictions
from src.models import sequence_c as sc
from src.models import train_autoencoder as ta
from tests.pipeline.synth_c import make_synthetic

ENV = {"WARDWATCH_GIT_COMMIT": "0" * 40, "WARDWATCH_GIT_DIRTY": "false",
       "WARDWATCH_IMAGE_DIGEST": "sha256:" + "a" * 64}
SEEDS = (0, 1)
AE_GRID = [{"hidden": (16, 8), "bottleneck": 4}, {"hidden": (16, 8), "bottleneck": 2}]
IF_GRID = [{"n_estimators": 10, "max_samples": 1.0}]


def make_env(root, paths, **kw):
    root = Path(root)
    return ta.Env(dataset=paths["tb_windows"], fl=paths["tb_fl"], split=paths["split_path"], runs=root / "runs",
                  model_dir=root / "models", pred_dir=root / "preds", cic_dir=Path(paths["cic_windows"]).parent,
                  for_report=True, seeds=SEEDS, ae_cfg={"epochs": 2}, ae_grid=AE_GRID, if_grid=IF_GRID, **kw)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old = {k: os.environ.get(k) for k in ENV}
        os.environ.update(ENV)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.paths = make_synthetic(cls.root / "data")
        cls.env = make_env(cls.root / "out", cls.paths)
        cls.ds = ta._load(cls.env, scale=True)
        for m in ("autoencoder", "isolation_forest"):
            ta.run_a(m, cls.env)
            ta.run_b(m, cls.ds, cls.ds.feature_names, env=cls.env)
        ta.run_c("autoencoder", cls.env)

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        cls._tmp.cleanup()

    def recs(self, exp, pattern):
        return [json.loads(p.read_text()) for p in sorted((self.env.runs / f"exp{exp}").glob(pattern))]


class TestRecords(Base):
    def test_counts_and_provenance(self):
        classes = {"syn_flood", "arp_mitm", "bruteforce", "recon_scan", "mqtt_flood"}
        for m in ("autoencoder", "isolation_forest"):
            a = self.recs("A", f"{m}_s[0-9].json")
            b = self.recs("B", f"{m}_*_s[0-9].json")
            self.assertEqual(len(a), len(SEEDS))
            self.assertEqual(len(b), len(classes) * len(SEEDS))
            self.assertEqual({r["params"]["held_out_class"] for r in b}, classes)
            for r in a + b:
                self.assertTrue(r["for_report"])
                self.assertEqual([k for k in ("seed", "split_sha256", "dataset_sha256", "git_commit",
                                              "git_dirty", "image_digest") if r["provenance"][k] is None], [])
            grid = self.recs("A", f"{m}_grid_*.json")
            self.assertTrue(grid and not any(r["for_report"] for r in grid))
        c = self.recs("C", "autoencoder_*_s[0-9].json")
        self.assertEqual(len(c), 2 * 2 * len(SEEDS))
        self.assertTrue(all(r["for_report"] for r in c))

    def test_aggregate_files_are_summaries_not_records(self):
        self.assertFalse((self.env.runs / "expA" / "autoencoder.json").exists())
        S = json.loads((self.env.runs / "expA" / "autoencoder_summary.json").read_text())
        recs = self.recs("A", "autoencoder_s[0-9].json")
        self.assertAlmostEqual(S["all"]["f1"][0], np.mean([r["metrics"]["test_all_f1"] for r in recs]))
        self.assertAlmostEqual(S["all"]["f1"][1], np.std([r["metrics"]["test_all_f1"] for r in recs]))
        B = json.loads((self.env.runs / "expB" / "isolation_forest_summary.json").read_text())
        rb = self.recs("B", "isolation_forest_*_s[0-9].json")
        per_seed = [np.mean([r["metrics"]["test_all_f1"] for r in rb if r["params"]["seed"] == s]) for s in SEEDS]
        self.assertAlmostEqual(B["b_score"]["all"]["f1"][0], np.mean(per_seed))

    def test_for_report_without_provenance_is_refused_before_training(self):
        old = {k: os.environ.pop(k, None) for k in ("WARDWATCH_IMAGE_DIGEST", "ANALYSIS_IMAGE")}
        try:
            env = make_env(self.root / "refused", self.paths)
            with self.assertRaises(ValueError):
                ta.run_a("autoencoder", env)
            self.assertFalse((self.root / "refused" / "runs" / "expA").exists() and
                             any((self.root / "refused" / "runs" / "expA").glob("*.json")))
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v


class TestViews(Base):
    def test_both_views_and_masked_metrics_equal_binary_metrics(self):
        r = self.recs("A", "autoencoder_s0.json")[0]
        for view in ("all", "noatk"):
            for m in ("f1", "precision", "recall", "fpr", "pr_auc"):
                self.assertIn(f"test_{view}_{m}", r["metrics"])
        self.assertIn("test_recall_attacker", r["metrics"])
        self.assertIn("test_recall_broker", r["metrics"])
        f = pd.read_parquet(self.env.runs / "expA" / "predictions" / "autoencoder_s0.parquet")
        t = f[(f.split == "test") & (f.host_role != "attacker")]
        thr = r["params"]["threshold"]
        want = mm.binary_metrics(t.y_bin.to_numpy(), y_score=t.score.to_numpy(), y_pred=ae.predict(t.score.to_numpy(), thr))
        self.assertAlmostEqual(r["metrics"]["test_noatk_f1"], want["f1"])
        self.assertAlmostEqual(r["metrics"]["test_noatk_pr_auc"], want["pr_auc"])
        self.assertEqual(r["metrics"]["test_noatk_n"], len(t))
        a = f[(f.split == "test") & (f.host_role == "attacker") & (f.y_bin == 1)]
        self.assertAlmostEqual(r["metrics"]["test_recall_attacker"], float((a.score >= thr).mean()))


class TestPredictions(Base):
    def test_per_run_layout_and_save_predictions(self):
        f = pd.read_parquet(self.env.runs / "expB" / "predictions" / "autoencoder_arp_mitm_s1.parquet")
        self.assertEqual(list(f.columns), ta.PRED_COLS)
        self.assertEqual(set(f.split), {"val", "test"})
        self.assertTrue(set(f[f.split == "test"].attack_class.dropna()) <= {"arp_mitm"})   # benign + X only
        self.assertNotIn("arp_mitm", set(f[f.split == "val"].attack_class.dropna()))       # X withheld from val
        for m in ("autoencoder", "isolation_forest"):
            name, df = predictions.load_predictions(self.env.pred_dir / f"A_{m}.parquet")
            self.assertEqual(len(df), len(self.ds.meta("test")))
        pc = pd.read_parquet(self.env.runs / "expC" / "predictions" / "autoencoder_tb2cic_refit_s0.parquet")
        self.assertEqual(set(pc.domain), {"val", "src_test", "tgt_test"})


class TestArchitecture(Base):
    def test_b_and_c_need_a_selection(self):
        env = make_env(self.root / "noA", self.paths)
        with self.assertRaises(SystemExit):
            ta.run_b("autoencoder", self.ds, self.ds.feature_names, env=env)
        with self.assertRaises(SystemExit):
            ta.run_c("autoencoder", env)

    def test_b_uses_the_selected_architecture(self):
        sel = json.loads((self.env.runs / "expA" / "autoencoder_selected.json").read_text())["params"]
        for r in self.recs("B", "autoencoder_*_s0.json"):
            self.assertEqual(r["params"]["bottleneck"], sel["bottleneck"])
            self.assertEqual(list(r["params"]["hidden"]), list(sel["hidden"]))

    def test_model_trained_once_per_seed_is_reproducible(self):
        ds = self.ds
        p = {"hidden": (16, 8), "bottleneck": 4}
        a = ta._fit_detector("autoencoder", p, 0, ds.normal_only("train"), ds.normal_only("val"), {"epochs": 2})[0]
        b = ta._fit_detector("autoencoder", p, 0, ds.normal_only("train"), ds.normal_only("val"), {"epochs": 2})[0]
        np.testing.assert_array_equal(a(ds.X("test")), b(ds.X("test")))


class TestC(Base):
    def _doms(self, paths):
        env = make_env(self.root / "tmp_c", paths)
        return ta.load_c_domains(env)

    def test_source_scaler_trio(self):
        doms, _ = self._doms(self.paths)
        sc_ = ta.source_scalers(doms)
        tb, cic = doms["testbed"], doms["cic"]
        raw = cic.X("test")
        want = (raw - tb.X("train").mean(axis=0)) / tb.X("train").std(axis=0)
        np.testing.assert_allclose(sc_["testbed"].transform(raw), want, rtol=1e-9, atol=1e-9)
        self.assertGreater(float(np.abs(sc_["cic"].mean_ - sc_["testbed"].mean_).mean()), 1.0)   # refit would differ
        with tempfile.TemporaryDirectory() as t:
            p2 = make_synthetic(Path(t))
            c = pd.read_parquet(p2["cic_windows"])
            c[tb.feature_names] += 77.0
            c.to_parquet(p2["cic_windows"], index=False)
            d = pd.read_parquet(p2["tb_windows"])
            d.loc[d.split != "train", tb.feature_names] += 55.0
            d.to_parquet(p2["tb_windows"], index=False)
            doms2, _ = self._doms(p2)
            s2 = ta.source_scalers(doms2)
        np.testing.assert_array_equal(s2["testbed"].mean_, sc_["testbed"].mean_)
        np.testing.assert_array_equal(s2["testbed"].scale_, sc_["testbed"].scale_)

    def test_refit_uses_only_target_benign_train_val(self):
        doms, _ = self._doms(self.paths)
        scaler = ta.source_scalers(doms)["testbed"]
        score_fn = lambda X: np.asarray(X)[:, 0]
        base = ta.refit_threshold(score_fn, scaler, doms["cic"], 0.05, 0)
        with tempfile.TemporaryDirectory() as t:
            def variant(edit):
                p2 = make_synthetic(Path(t) / str(len(os.listdir(t))))
                c = pd.read_parquet(p2["cic_windows"])
                feats = doms["cic"].feature_names
                edit(c, feats)
                c.to_parquet(p2["cic_windows"], index=False)
                d2, _ = self._doms(p2)
                return ta.refit_threshold(score_fn, scaler, d2["cic"], 0.05, 0)
            same_test = variant(lambda c, f: c.__setitem__(f, c[f].where(c.split != "test", c[f] + 99.0)))
            same_atk = variant(lambda c, f: c.__setitem__(f, c[f].where(c.label != "attack", c[f] + 99.0)))
            diff = variant(lambda c, f: c.__setitem__(f, c[f].where(~((c.label == "benign") & (c.split != "test")), c[f] + 99.0)))
        self.assertEqual(same_test, base)
        self.assertEqual(same_atk, base)
        self.assertNotEqual(diff[0], base[0])
        n_avail = int(((doms["cic"].y("train") == 0).sum()) + (doms["cic"].y("val") == 0).sum())
        self.assertEqual(base[1], min(1000, n_avail))

    def test_variants_share_model_scaler_and_source_numbers(self):
        for tag in ("tb2cic", "cic2tb"):
            for s in SEEDS:
                a = json.loads((self.env.runs / "expC" / f"autoencoder_{tag}_astrained_s{s}.json").read_text())
                r = json.loads((self.env.runs / "expC" / f"autoencoder_{tag}_refit_s{s}.json").read_text())
                self.assertEqual(a["params"]["threshold"], r["params"]["threshold"])
                for k, v in a["metrics"].items():
                    if k.startswith(("val_", "src_test_")) and isinstance(v, float) and v == v:
                        self.assertEqual(r["metrics"][k], v, k)
                for k in ("n_refit", "fpr_src", "thr_refit"):
                    self.assertIn(k, r["params"])
                self.assertEqual(r["params"]["target_threshold"], r["params"]["thr_refit"])
                self.assertIn("cic_files", r["tags"])
        S = json.loads((self.env.runs / "expC" / "autoencoder_summary.json").read_text())
        self.assertIsNotNone(S["selection_score_astrained_f1"])

    def test_ddos_stem_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p2 = make_synthetic(Path(t))
            c = pd.read_parquet(p2["cic_windows"])
            c["run_id"] = c["run_id"].replace({"Benign_test": "DDoS-ICMP_Flood_test"})
            c.to_parquet(p2["cic_windows"], index=False)
            with self.assertRaises(SystemExit):
                self._doms(p2)


if __name__ == "__main__":
    unittest.main()
