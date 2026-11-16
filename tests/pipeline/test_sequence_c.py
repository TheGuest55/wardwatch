"""Tests for the Experiment-C domain loader (src/models/sequence_c.py) on synthetic CIC-like frames."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import common
from src.models import data as mdata
from src.models import sequence as seq
from src.models import sequence_c as sc
from tests.pipeline.synth_c import SHIFT, make_synthetic

K = 4


class TestSequenceC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.paths = make_synthetic(cls._tmp.name)
        cls.dom = sc.load_pair(cls.paths)
        cls.tb, cls.cic = cls.dom["testbed"], cls.dom["cic"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ---- scaler
    def test_source_scaler_is_train_only_and_exact(self):
        scaler = sc.fit_source_scaler(self.tb)
        raw = pd.read_parquet(self.paths["tb_windows"])
        keep = raw[(raw["split"] == "train") & ((raw["label"] == "benign") |
                                                (raw["attack_class"].isin(sc.SHARED)))]
        keep = keep[~keep["label"].isin(("victim", "boundary"))]
        X = keep[self.tb.feats].to_numpy(float)
        np.testing.assert_allclose(scaler.mean_, X.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(scaler.scale_, X.std(axis=0), rtol=1e-10)

    def test_target_is_transformed_with_the_source_scaler_exactly(self):
        scaler = sc.fit_source_scaler(self.tb)
        s = sc.domain_sequences(self.cic, scaler, K)["test"]
        m = s.meta.iloc[-1]
        raw = self.cic.df[(self.cic.df.session == m.session) & (self.cic.df.host_mac == m.host_mac)
                          & (self.cic.df.window_start_epoch == m.window_start_epoch)]
        want = (raw[self.cic.feats].to_numpy(float)[0] - scaler.mean_) / scaler.scale_
        np.testing.assert_allclose(s.X[-1, -1, :-1], want, rtol=1e-5, atol=1e-5)
        self.assertEqual(s.X[-1, -1, -1], 1.0)
        self.assertGreater(float(s.X[:, :, :-1][s.X[:, :, -1] == 1].mean()), SHIFT * 0.5)  # still shifted

    def test_scaler_ignores_target_and_source_val_test(self):
        a = sc.fit_source_scaler(self.tb)
        with tempfile.TemporaryDirectory() as t:
            p2 = make_synthetic(t, seed=0)
            for key in ("tb_windows",):
                d = pd.read_parquet(p2[key])
                d.loc[d["split"] != "train", self.tb.feats] += 123.0
                d.to_parquet(p2[key], index=False)
            c = pd.read_parquet(p2["cic_windows"])
            c[self.cic.feats] += 77.0
            c.to_parquet(p2["cic_windows"], index=False)
            b = sc.fit_source_scaler(sc.load_domain("testbed", p2))
        np.testing.assert_array_equal(a.mean_, b.mean_)
        np.testing.assert_array_equal(a.scale_, b.scale_)

    def test_a_scaler_refit_on_the_target_would_differ(self):
        src = sc.fit_source_scaler(self.tb)
        tgt = sc.fit_source_scaler(self.cic)
        self.assertGreater(float(np.abs(src.mean_ - tgt.mean_).mean()), SHIFT * 0.5)

    # ---- width and features
    def test_width_is_shared_features_plus_presence(self):
        n_shared = len(common.FEATURES) - len(common.TESTBED_ONLY_FEATURES)
        self.assertEqual(len(self.tb.feats), n_shared)
        for d in (self.tb, self.cic):
            s = sc.domain_sequences(d, sc.fit_source_scaler(self.tb), K)
            self.assertEqual(s["train"].X.shape[-1], n_shared + 1)
            self.assertFalse(set(self.tb.feats) & set(common.TESTBED_ONLY_FEATURES))
            self.assertFalse(set(d.feats) & (set(common.META_COLUMNS) | set(common.METADATA_EXTRAS)))

    def test_feature_list_equals_the_trees_domain_list(self):
        from src.models import train_tree as tt
        self.assertEqual(self.tb.feats, tt._domain(self.paths["tb_windows"], self.paths["tb_fl"]).feature_names)
        self.assertEqual(self.cic.feats, tt._domain(self.paths["cic_windows"], self.paths["cic_fl"]).feature_names)

    def test_mismatched_feature_lists_raise_systemexit(self):
        with tempfile.TemporaryDirectory() as t:
            p2 = make_synthetic(t)
            fl = json.loads(Path(p2["cic_fl"]).read_text())
            fl["features"] = fl["features"][:-1] + ["pkts_total"]       # drop one, duplicate another
            fl["features"] = [f for f in fl["features"] if f != "mqtt_pkts"] + ["mqtt_pkts"]  # reorder
            Path(p2["cic_fl"]).write_text(json.dumps(fl))
            with self.assertRaises(SystemExit):
                sc.load_pair(p2)

    # ---- cut and context rules
    def test_no_sequence_crosses_train_val_inside_a_cic_file_and_val_is_padded(self):
        s = sc.domain_sequences(self.cic, sc.fit_source_scaler(self.tb), K)
        for sp in ("train", "val"):
            self.assertTrue(s[sp].meta.session.str.endswith("@" + sp).all())
        va = s["val"]
        first = va.meta.groupby(["session", "host_mac"])["window_start_epoch"].transform("min")
        i = int(np.flatnonzero(va.meta.window_start_epoch == first)[0])
        self.assertEqual(va.X[i, :-1, -1].sum(), 0)           # only the target present: left padded
        # train windows of the same file never carry val windows: tensor values differ by segment
        tr_epochs = self.cic.df[self.cic.df.split == "train"].groupby("session")["window_start_epoch"].max()
        va_epochs = self.cic.df[self.cic.df.split == "val"].groupby("session")["window_start_epoch"].min()
        for sess_t, e_t in tr_epochs.items():
            self.assertLess(e_t, va_epochs[sess_t.replace("@train", "@val")])
        self.assertGreater(sc.padded_share(s["val"]), 0.0)

    def test_unmapped_and_nonshared_classes_absent_from_every_context_and_target(self):
        scaler = sc.fit_source_scaler(self.tb)
        for dom in (self.tb, self.cic):
            s = sc.domain_sequences(dom, scaler, K)
            bad_cls = sc.NONSHARED[dom.name]
            bad = dom.df[dom.df["exclusion_class"].isin(bad_cls)]
            self.assertGreater(len(bad), 0, dom.name)
            # identity-free check: rebuild from a frame where each bad window is marked by a huge value
            marked = dom.df.copy()
            marked.loc[bad.index, dom.feats[0]] = 1e9
            sm = seq.build_sequences(marked, scaler, dom.feats, K, classes=sc.C_CLASSES, drop_classes=bad_cls)
            for sp in seq.SPLITS:
                self.assertTrue((sm[sp].X[:, :, 0] < 1e6).all(), (dom.name, sp))
                names = np.array(sc.C_CLASSES)[sm[sp].y_class]
                self.assertTrue(set(names) <= set(sc.C_CLASSES))
                self.assertTrue(sm[sp].meta.attack_class.dropna().isin(sc.SHARED).all())
                self.assertTrue((sm[sp].y_class >= 0).all())

    def test_row_table_reconciles(self):
        scaler = sc.fit_source_scaler(self.tb)
        for dom in (self.tb, self.cic):
            sets = sc.domain_sequences(dom, scaler, K)
            for r in sc.row_table(dom, sets):
                self.assertEqual(r["tree_rows"] - r["dropped_by_context_rule"], r["lstm_targets"])
        tb_rows = sc.row_table(self.tb, sc.domain_sequences(self.tb, scaler, K))
        self.assertGreater(sum(r["victim"] for r in tb_rows), 0)

    # ---- DDoS rule
    def test_ddos_stem_is_rejected_and_stems_are_reported(self):
        stems = sc.check_cic_files(self.cic.df)
        self.assertIn("Benign_train", stems["train"])
        self.assertIn("Benign_test", stems["test"])
        bad = self.cic.df.copy()
        bad["run_id"] = bad["run_id"].replace({"Benign_test": "DDoS-SYN_Flood_test"})
        with self.assertRaises(SystemExit):
            sc.check_cic_files(bad)


if __name__ == "__main__":
    unittest.main()
