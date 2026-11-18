"""Seed-0 multiclass test predictions from the saved model (real data, read-only): identity, schema, no write on a mismatch."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import multiclass_predictions as mp
from src.models import save_multiclass as smc
from src.models import train_tree as tt


class TestMulticlassPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f, cls.metrics = mp.score_test()

    def test_macro_f1_equals_the_recorded_seed0_value(self):
        diff = smc.check_identity(self.metrics)                       # SystemExit if not equal
        self.assertTrue(all(d <= smc.TOL for d in diff.values()), diff)

    def test_schema_probabilities_and_rows(self):
        f = self.f
        classes = json.loads(smc.CLASSES.read_text())["classes"]
        self.assertEqual(set(f["split"]), {"test"})
        for c in classes:
            self.assertIn(f"p_{c}", f.columns)
        P = f[[f"p_{c}" for c in classes]].to_numpy()
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(f["score"], 1.0 - f["p_benign"])
        np.testing.assert_array_equal(f["pred"], (np.array(classes)[P.argmax(axis=1)] != "benign").astype(int))
        b = pd.read_parquet(tt.OUT_DIR / "predictions" / "xgboost_s0.parquet")
        b = b[b["split"] == "test"].reset_index(drop=True)
        keys = ["session", "host_mac", "window_start_epoch"]
        self.assertTrue(f[keys].equals(b[keys]))                       # the same test rows as the binary seed-0 file
        self.assertEqual(list(f["y_bin"]), list(b["y_bin"]))

    def test_main_writes_file_and_meta_then_stops_without_writing_on_a_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "x.parquet"
            mp.main(["--out", str(out)])
            self.assertTrue(out.exists())
            meta = json.loads((Path(t) / mp.META.name).read_text())
            self.assertEqual((meta["seed"], meta["derived"]), (0, True))
            out2 = Path(t) / "y.parquet"
            orig = mp.score_test
            mp.score_test = lambda *a, **k: (self.f, {k2: v + 1e-3 for k2, v in self.metrics.items()})
            try:
                with self.assertRaises(SystemExit):
                    mp.main(["--out", str(out2)])
            finally:
                mp.score_test = orig
            self.assertFalse(out2.exists())


if __name__ == "__main__":
    unittest.main()
