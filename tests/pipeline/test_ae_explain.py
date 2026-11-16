"""Tests for the per-alert AE explanation (src/models/ae_explain.py, src/experiments/ae_explain_run.py)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.experiments import ae_explain_run as run
from src.models import ae_explain as ex
from src.models import autoencoder as ae
from src.models import data as mdata
from src.models import train_autoencoder as ta
from tests.pipeline.synth_c import make_synthetic


class Shift(torch.nn.Module):
    """Reconstructs every feature perfectly except column 3, which is off by +1 (scaled units)."""
    def forward(self, x):
        y = x.clone()
        y[:, 3] += 1.0
        return y


class TestExplain(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.feats = [f"f{i}" for i in range(6)]
        self.raw = rng.normal(5, 2, (50, 6))
        self.scaler = StandardScaler().fit(self.raw)

    def test_errors_sum_to_the_anomaly_score_and_perfect_features_have_zero_share(self):
        r = ex.explain(Shift(), self.scaler, self.raw[0], self.feats)
        self.assertAlmostEqual(r["total_error"] / 6, r["score"])
        self.assertAlmostEqual(r["score"], r["score_check"], places=6)
        by = {d["feature"]: d for d in r["features"]}
        self.assertAlmostEqual(by["f3"]["share"], 1.0)
        self.assertTrue(all(by[f]["share"] == 0.0 for f in self.feats if f != "f3"))
        self.assertAlmostEqual(sum(d["share"] for d in r["features"]), 1.0)
        self.assertEqual(r["features"][0]["feature"], "f3")
        # expected value = reconstruction mapped back to raw units
        self.assertAlmostEqual(by["f3"]["expected_value"], self.raw[0, 3] + self.scaler.scale_[3], places=5)
        self.assertAlmostEqual(by["f3"]["raw_value"], self.raw[0, 3])
        self.assertEqual(len(r["top"]), 5)

    def test_score_equals_ae_anomaly_scores_for_a_trained_model(self):
        ae.set_determinism(0, 1)
        m = ae.AutoEncoder(6, hidden=(8,), bottleneck=3)
        r = ex.explain(m, self.scaler, self.raw[7], self.feats)
        want = ae.anomaly_scores(m, self.scaler.transform(self.raw[7:8]))[0]
        self.assertAlmostEqual(r["score"], want, places=6)

    def test_run_on_lstm_alert_windows_and_unflagged_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            p = make_synthetic(Path(t) / "d")
            env = ta.Env(dataset=p["tb_windows"], fl=p["tb_fl"], split=p["split_path"], runs=Path(t) / "r",
                         model_dir=Path(t) / "m", seeds=(0,), ae_cfg={"epochs": 2})
            ds = ta._load(env, scale=True)
            fit = ta._fit_detector("autoencoder", {"hidden": (8,), "bottleneck": 3}, 0,
                                   ds.normal_only("train"), ds.normal_only("val"), {"epochs": 2})
            model = fit[2]
            thr = ae.pick_threshold(ds.y("val"), ae.anomaly_scores(model, ds.X("val")))
            ta._save_ae(env, model, thr, ds.feature_names, ds.scaler)
            raw = ta._load(env, scale=False)
            m = raw.meta("test")
            a = m[(m.attack_class == "syn_flood") & (m.label == "attack")].iloc[0]
            b = m[(m.label == "benign")].iloc[0]                       # pretend an LSTM alert on a quiet row
            lj = Path(t) / "lstm.json"
            ent = lambda r: {"status": "ok", "session": r.session, "host_mac": r.host_mac,
                             "window_start_epoch": int(r.window_start_epoch)}
            lj.write_text(json.dumps({"classes": {"syn_flood": ent(a), "arp_mitm": ent(b),
                                                  "pump_cmd": {"status": "no_alert"}}}))
            out = run.run(env.model_dir / "autoencoder_seed0.pt", raw, lj, Path(t) / "out.json")
        c = out["classes"]
        self.assertEqual(c["pump_cmd"], {"status": "no_lstm_alert"})
        self.assertEqual(c["syn_flood"]["lstm_alert"]["window_start_epoch"], int(a.window_start_epoch))
        self.assertAlmostEqual(sum(d["share"] for d in c["syn_flood"]["lstm_alert"]["features"]), 1.0)
        self.assertIn(c["arp_mitm"]["lstm_alert"]["flagged"], (True, False))
        if not c["arp_mitm"]["lstm_alert"]["flagged"]:
            self.assertIn("best_true_positive", c["arp_mitm"])


if __name__ == "__main__":
    unittest.main()
