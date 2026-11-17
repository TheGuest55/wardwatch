"""Tests for the frame-size sensitivity variant (Experiment C, nosize), the XGBoost latency path
(src/experiments/tree_latency.py) and the selection-rule bootstrap (src/models/bootstrap_selection.py)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import run_lstm_c as rc
from src.experiments import run_tree_report as rt
from src.experiments import tree_latency as tl
from src.models import bootstrap_selection as bs
from src.models import calibrate_explain as ce
from src.models import data as mdata
from src.models import sequence_c as sc
from src.models import train_autoencoder as ta
from src.models import train_tree as tt
from tests.pipeline.synth_c import make_synthetic

ENV = {"WARDWATCH_GIT_COMMIT": "0" * 40, "WARDWATCH_GIT_DIRTY": "false",
       "WARDWATCH_IMAGE_DIGEST": "sha256:" + "a" * 64}
GRIDS = {"xgboost": {"max_depth": [3], "n_estimators": [20], "learning_rate": [0.3]},
         "random_forest": {"n_estimators": [20], "max_depth": [None], "min_samples_leaf": [1]}}


class _EnvCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old = {k: os.environ.get(k) for k in ENV}
        os.environ.update(ENV)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.paths = make_synthetic(cls.root / "data")
        cls._grids = tt.GRIDS
        tt.GRIDS = GRIDS

    @classmethod
    def tearDownClass(cls):
        tt.GRIDS = cls._grids
        for k, v in cls._old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        cls._tmp.cleanup()

    @classmethod
    def env(cls, **kw):
        p = cls.paths
        return rt.Env(dataset=p["tb_windows"], fl=p["tb_fl"], split=p["split_path"], runs=cls.root / "runs",
                      model_dir=cls.root / "models", pred_dir=cls.root / "preds",
                      cic_dir=Path(p["cic_windows"]).parent, for_report=True, seeds=(0, 1),
                      ae_cfg={"epochs": 2}, ae_grid=[{"hidden": (8,), "bottleneck": 3}],
                      if_grid=[{"n_estimators": 10, "max_samples": 1.0}], **kw)


class TestNosize(_EnvCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.main, cls.nosize = cls.env(), cls.env(size_variant="nosize")
        rt.run_c("xgboost", cls.main)
        rt.run_c("xgboost", cls.nosize)
        ta.run_a("isolation_forest", cls.main)
        ta.run_c("isolation_forest", cls.main)
        ta.run_c("isolation_forest", cls.nosize)

    def _recs(self, d, pat):
        return [json.loads(p.read_text()) for p in sorted(d.glob(pat))]

    def test_trees_nosize_drop_exactly_the_size_features_into_their_own_dir(self):
        main = self._recs(self.root / "runs" / "expC", "xgboost_*_astrained_s*.json")
        ns = self._recs(self.root / "runs" / "expC_nosize", "xgboost_*_astrained_s*.json")
        self.assertEqual((len(main), len(ns)), (4, 4))
        for r in ns:
            self.assertEqual(r["tags"]["sensitivity"], "nosize")
            self.assertEqual(sorted(r["tags"]["dropped_features"]), sorted(tt.SIZE_FEATURES))
            self.assertEqual(r["params"]["size_variant"], "nosize")
            self.assertEqual(r["params"]["n_features"], main[0]["params"]["n_features"] - len(tt.SIZE_FEATURES))
        for r in main:
            self.assertNotIn("sensitivity", r["tags"])
            self.assertEqual(r["params"]["size_variant"], "none")

    def test_if_nosize_tagged_and_main_summary_unaffected(self):
        ns = self._recs(self.root / "runs" / "expC_nosize", "isolation_forest_*_s*.json")
        self.assertEqual(len(ns), 8)                     # 2 directions x 2 variants x 2 seeds
        self.assertTrue(all(r["tags"]["sensitivity"] == "nosize" for r in ns))
        S = json.loads((self.root / "runs" / "expC" / "isolation_forest_summary.json").read_text())
        self.assertEqual(S["n_records"], 8)
        self.assertNotIn("sensitivity", S)

    def test_summary_refit_sentence_only_when_refit_records_exist(self):
        tree = (self.root / "runs" / "expC" / "xgboost_summary.md").read_text()
        iso = (self.root / "runs" / "expC" / "isolation_forest_summary.md").read_text()
        self.assertNotIn("refit variant is reported", tree)
        self.assertIn("no threshold-refit variant", tree)
        self.assertIn("refit variant is reported alongside and is NOT used for selection", iso)
        self.assertIn("SENSITIVITY", (self.root / "runs" / "expC_nosize" / "xgboost_summary.md").read_text())

    def test_size_variant_refused_outside_c(self):
        with self.assertRaises(SystemExit):
            rt.main(["--model", "xgboost", "--experiment", "A", "--size-variant", "nosize", "--no-mlflow"])
        with self.assertRaises(SystemExit):
            ta.main(["--model", "isolation_forest", "--experiment", "B", "--size-variant", "nosize"])

    def test_lstm_nosize_context_and_record(self):
        full = sc.load_pair(self.paths)
        sel = self.root / "selected_config.json"
        sel.write_text(json.dumps({"k": 12, "hidden": 32}))
        ctx = rc.make_ctx(self.paths, 1, self.root / "lstm_ns", self.root / "lstm_ns_m", smoke=True, epochs=1,
                          selected=sel, size_variant="nosize")
        self.assertEqual(len(ctx.doms["testbed"].feats), len(full["testbed"].feats) - len(tt.SIZE_FEATURES))
        self.assertFalse(set(tt.SIZE_FEATURES) & set(ctx.doms["cic"].feats))
        rec = rc.run_one(ctx, "testbed", "cic", "tb2cic", 0)
        self.assertEqual(rec["tags"]["sensitivity"], "nosize")
        self.assertEqual(rec["params"]["size_variant"], "nosize")
        self.assertEqual(rec["params"]["n_features"], len(ctx.doms["testbed"].feats))
        rc.write_summary(ctx.runs_dir)
        self.assertIn("SENSITIVITY", (ctx.runs_dir / "summary.md").read_text())


class TestTreeLatency(_EnvCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.e = cls.env()
        rt.run_a("xgboost", cls.e)
        cls._model_dir = ce.MODEL_DIR
        ce.MODEL_DIR = Path(cls.e.model_dir)
        df, fl = mdata.load_dataset(cls.paths["tb_windows"], cls.paths["tb_fl"])
        cls.ds = mdata.Dataset(df, fl, scale=False)
        model = ce.load_model("xgboost")                       # calibrate_explain's own steps, validation only
        s_va = model.predict_proba(cls.ds.X("val"))[:, 1]
        cal = ce.Calibrator("platt").fit(s_va, cls.ds.y("val"))
        lo, hi = ce.confidence_cutoffs(cls.ds.y("val"), cal.predict(s_va))
        import joblib
        joblib.dump(cal, ce.MODEL_DIR / "xgboost_calibrator.joblib")
        (ce.MODEL_DIR / "xgboost_triage_config.json").write_text(json.dumps(
            {"model": "xgboost", "calibration": "platt", "lo": lo, "hi": hi, "features": cls.ds.feature_names}))
        cls.pred_file = cls.e.runs / "expA" / "predictions" / "xgboost_s0.parquet"

    @classmethod
    def tearDownClass(cls):
        ce.MODEL_DIR = cls._model_dir
        super().tearDownClass()

    def test_record_from_raw_input_with_identity_and_equivalence(self):
        out = self.root / "latency_xgboost.json"
        rec = tl.run("xgboost", self.ds.X("test"), self.ds.meta("test"), out, n_calls=200, n_equiv=50,
                     pred_file=self.pred_file, for_report=True, split_path=self.paths["split_path"],
                     dataset_path=self.paths["tb_windows"])
        m = rec["metrics"]
        self.assertTrue(out.exists() and rec["for_report"] and rec["tags"]["latency"])
        self.assertGreaterEqual(m["n"], 200)
        self.assertEqual((m["equivalence_decision_flips"], m["equivalence_band_flips"]), (0, 0))
        self.assertLessEqual(m["equivalence_max_scaled_diff"], 1e-6)
        self.assertLessEqual(m["identity_max_abs_diff"], 1e-6)
        self.assertIn("shap_p95_ms", m)
        self.assertIn("SHAP timed separately", rec["tags"]["timed_path"])

        anomaly = self.root / "latency_if.json"
        anomaly.write_text(json.dumps({"for_report": True, "model": "isolation_forest", "tags": {"latency": True},
                                       "metrics": {"p95_ms": 3.0, "median_ms": 2.0},
                                       "provenance": {"git_commit": "1" * 40}}))
        pr = tl.pair(out, anomaly, self.root / "pair.json")
        self.assertAlmostEqual(pr["pair_p95_sum_ms"], m["p95_ms"] + 3.0)

    def test_a_different_model_fails_the_identity_check(self):
        p = pd.read_parquet(self.pred_file)
        p.loc[p["split"] == "test", "score"] = p.loc[p["split"] == "test", "score"] * 0.9 + 0.05
        bad = self.root / "bad.parquet"
        p.to_parquet(bad)
        with self.assertRaises(AssertionError):
            tl.run("xgboost", self.ds.X("test"), self.ds.meta("test"), None, n_calls=200, n_equiv=20,
                   pred_file=bad, split_path=self.paths["split_path"], dataset_path=self.paths["tb_windows"])


# ------------------------------------------------------------------------------------ bootstrap
def _rows(rng, dataset, sessions, classes, runs_per_class=2):
    """Benign windows for 3 hosts per session + attacker windows grouped in runs."""
    out = []
    for si, s in enumerate(sessions):
        t = np.arange(0, 3600, 5.0) + si * 10_000
        for h in ("h1", "h2", "h3"):
            out.append(pd.DataFrame({"session": s, "host_mac": h, "window_start_epoch": t, "label": "benign",
                                     "run_id": None, "attack_class": None}))
        for ci, c in enumerate(classes):
            for r in range(runs_per_class):
                if r % len(sessions) != si:
                    continue
                t0 = t[0] + 300 * (ci * runs_per_class + r) + 50
                out.append(pd.DataFrame({"session": s, "host_mac": "atk", "window_start_epoch": t0 + 5.0 * np.arange(12),
                                         "label": "attack", "run_id": f"{dataset}-{c}-{r}", "attack_class": c}))
    d = pd.concat(out, ignore_index=True)
    d["dataset"] = dataset
    d["host_role"] = np.where(d["host_mac"] == "atk", "attacker", "device")
    return d


def _f1(y, p):
    tp, fp, fn = ((y == 1) & (p == 1)).sum(), ((y == 0) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum()
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def _write(rec_path, pred_path, frame, f1_key, tags=None, with_pred=True, threshold=0.5):
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    y = (frame["label"] == "attack").to_numpy().astype(int)
    p = (frame["score"].to_numpy() >= threshold).astype(int)
    f = frame.copy()
    if with_pred:
        f["pred"] = p
    f.to_parquet(pred_path)
    sub = f[f["split"] == "test"] if "domain" not in f else f[f["domain"] == "tgt_test"]
    yy = (sub["label"] == "attack").to_numpy().astype(int)
    pp = (sub["score"].to_numpy() >= threshold).astype(int)
    rec_path.write_text(json.dumps({"for_report": True, "params": {"threshold": threshold},
                                    "metrics": {f1_key: float(_f1(yy, pp))}, "tags": tags or {},
                                    "provenance": {"git_commit": "0" * 40}}))
    return y


class TestBootstrapSelection(unittest.TestCase):
    CLASSES = ("ca", "cb")
    SEEDS = (0, 1)
    MODELS = ("xgboost", "random_forest", "lstm")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self._tmp.name)
        self._saved = (bs.SEEDS, bs.B_CLASSES)
        bs.SEEDS, bs.B_CLASSES = self.SEEDS, self.CLASSES
        rng = np.random.default_rng(0)
        self.tb = _rows(rng, "testbed", ["te1", "te2"], self.CLASSES)
        self.cic = _rows(rng, "cic", ["Benign_test", "Atk_test"], ("ca",))

    def tearDown(self):
        bs.SEEDS, bs.B_CLASSES = self._saved
        self._tmp.cleanup()

    def _score(self, d, m, s, noise=0.35):
        rng = np.random.default_rng(hash((m, s)) % 2**32)
        base = (d["label"] == "attack").to_numpy() * 0.6 + rng.uniform(0, 0.55, len(d))
        return base if m != "lstm" else base * (1 - noise) + noise * rng.uniform(0, 1, len(d))

    def _build(self, identical=False, variants=("none",)):
        for m in self.MODELS:
            mm = "xgboost" if identical and m == "random_forest" else m
            for s in self.SEEDS:
                for X in self.CLASSES:
                    d = self.tb[(self.tb["label"] == "benign") | (self.tb["attack_class"] == X)].copy()
                    d["split"] = "test"
                    d["score"] = self._score(d, mm, s)
                    if m == "lstm":
                        d = d.drop(columns="dataset")
                    _write(*bs.b_paths(self.runs, m, X, s), d, "test_all_f1", with_pred=m != "lstm")
                for v in variants:
                    for tag, tgt in (("tb2cic", self.cic), ("cic2tb", self.tb)):
                        d = tgt.copy()
                        d["split"], d["domain"] = "test", "tgt_test"
                        if m == "lstm":
                            d = d.iloc[20:]                       # the LSTM's context rule drops rows
                            d = d.assign(session=d["session"] + "@test")
                        d["score"] = self._score(d, mm, s + (7 if v != "none" else 0))
                        tags = {"sensitivity": v} if v != "none" else {}
                        _write(*bs.c_paths(self.runs, m, tag, s, v), d, "tgt_test_all_f1", tags=tags,
                               with_pred=m != "lstm")

    def test_point_estimates_match_records_and_cis_bracket_them(self):
        self._build(variants=("none", "nosize"))
        res = bs.run(self.runs, n_boot=200, models=self.MODELS)
        for k, v in res["intervals"].items():
            self.assertLessEqual(v["lo"], v["point"] + 1e-12, k)
            self.assertGreaterEqual(v["hi"], v["point"] - 1e-12, k)
        self.assertIn("C_nosize/xgboost", res["intervals"])
        self.assertIn("sel/xgboost-lstm", res["differences"])
        self.assertEqual(res["variants"], ["none", "nosize"])

    def test_identical_models_give_zero_difference(self):
        self._build(identical=True)
        res = bs.run(self.runs, n_boot=100, models=self.MODELS, variants=("none",))
        for st in ("B", "C", "sel"):
            d = res["differences"][f"{st}/xgboost-random_forest"]
            self.assertEqual((d["diff"], d["lo"], d["hi"]), (0.0, 0.0, 0.0))

    def test_point_mismatch_stops(self):
        self._build()
        rp, _ = bs.b_paths(self.runs, "xgboost", "ca", 0)
        r = json.loads(rp.read_text())
        r["metrics"]["test_all_f1"] += 0.01
        rp.write_text(json.dumps(r))
        with self.assertRaises(SystemExit):
            bs.run(self.runs, n_boot=10, models=self.MODELS, variants=("none",))

    def test_a_sensitivity_record_in_the_main_dir_stops(self):
        self._build()
        rp, _ = bs.c_paths(self.runs, "random_forest", "tb2cic", 1)
        r = json.loads(rp.read_text())
        r["tags"]["sensitivity"] = "nosize"
        rp.write_text(json.dumps(r))
        with self.assertRaises(SystemExit):
            bs.run(self.runs, n_boot=10, models=self.MODELS, variants=("none",))

    def test_block_weights_equal_explicit_row_resampling(self):
        self._build()
        tables = bs.collect(self.runs, self.MODELS)
        U, blocks = bs.universe(tables)
        bs.attach(tables, U)
        reps = bs.representatives(U)
        s1 = bs.BlockSampler(U["block"].to_numpy(), U["stratum"].to_numpy(), seed=3)
        s2 = bs.BlockSampler(U["block"].to_numpy(), U["stratum"].to_numpy(), seed=3)
        for _ in range(5):
            rows = s1.draw()
            w = bs.block_weights(s2, reps)
            for key in [("B", "none", "lstm", "cb", 1), ("C", "none", "lstm", "cic2tb", 0),
                        ("C", "none", "xgboost", "tb2cic", 1)]:
                t = tables[key]
                pos = U.reset_index().set_index(bs.KEYS)["index"]
                uidx = pos.reindex(pd.MultiIndex.from_frame(t["df"][bs.KEYS])).to_numpy()
                row_of = {u: i for i, u in enumerate(uidx)}
                pick = np.array([row_of[u] for u in rows if u in row_of], dtype=int)
                self.assertAlmostEqual(bs.f1_w(t, w), _f1(t["y"][pick], t["pred"][pick]), places=12)


if __name__ == "__main__":
    unittest.main()
