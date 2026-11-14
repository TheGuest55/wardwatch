"""Block bootstrap (src/models/bootstrap_ci.py) and the shared prediction format."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.bootstrap_ci import (BlockSampler, bootstrap, make_blocks, metrics_at,
                                     paired_diff, same_rows)
from src.models.predictions import load_predictions, save_predictions


def _table(n_runs_per_class=3, run_len=20, n_benign=600, seed=0, recall=0.8, fp_rate=0.01):
    """Synthetic test split: 2 classes x n runs of attack windows + benign windows over 2 sessions."""
    rng = np.random.default_rng(seed)
    rows = []
    t = 0
    for cls in ("syn_flood", "pump_cmd"):
        for r in range(n_runs_per_class):
            for _ in range(run_len):
                rows.append(("r05" if r % 2 else "r07", float(t), "aa", "attack", cls, f"{cls}-{r}"))
                t += 5
    for i in range(n_benign):
        rows.append(("r05" if i % 2 else "r07", float(5 * i), f"h{i % 7}", "benign", None, None))
    df = pd.DataFrame(rows, columns=["session", "window_start_epoch", "host_mac", "label",
                                     "attack_class", "run_id"])
    df["y"] = (df["label"] == "attack").astype(int)
    hit = rng.uniform(size=len(df))
    df["pred"] = np.where(df["y"] == 1, hit < recall, hit < fp_rate).astype(int)
    df["score"] = df["pred"] * 0.5 + rng.uniform(0, 0.5, len(df))
    return df


class TestBlocks(unittest.TestCase):
    def test_attack_rows_block_by_run_and_benign_by_time_chunk(self):
        df = _table()
        blocks, strata = make_blocks(df, benign_block_s=300)
        att = df["label"] == "attack"
        self.assertEqual(len(set(blocks[att.to_numpy()])), 6)               # 2 classes x 3 runs
        self.assertEqual(set(strata[att.to_numpy()]), {"attack:syn_flood", "attack:pump_cmd"})
        self.assertTrue(all(b.startswith("benign:") for b in blocks[~att.to_numpy()]))

    def test_attack_without_run_id_rejected(self):
        df = _table()
        df.loc[0, "run_id"] = None
        with self.assertRaises(ValueError):
            make_blocks(df)

    def test_resample_keeps_whole_blocks_and_strata_sizes(self):
        df = _table()
        blocks, strata = make_blocks(df)
        s = BlockSampler(blocks, strata, seed=1)
        i = s.draw()
        drawn = pd.Series(blocks[i]).value_counts()
        full = pd.Series(blocks).value_counts()
        for b, n in drawn.items():                       # each block drawn k times = k x its size
            self.assertEqual(n % full[b], 0)
        att_runs = {b for b in blocks[i] if b.startswith("run:pump_cmd")}
        self.assertTrue(att_runs <= {f"run:pump_cmd-{r}" for r in range(3)})
        n_att = (df["y"].to_numpy()[i] == 1).sum()
        self.assertEqual(n_att, 2 * 3 * 20)             # stratified: same number of runs per class


class TestMetricsAndIntervals(unittest.TestCase):
    def test_metrics_at(self):
        m = metrics_at([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["fpr"], 0.5)

    def test_interval_contains_point_and_is_deterministic(self):
        df = _table()
        a, _, nb = bootstrap(df, n_boot=200, seed=0)
        b, _, _ = bootstrap(df, n_boot=200, seed=0)
        self.assertEqual(a, b)
        for k in ("f1", "recall", "precision"):
            self.assertLessEqual(a[k]["lo"], a[k]["point"] + 1e-9)
            self.assertGreaterEqual(a[k]["hi"], a[k]["point"] - 1e-9)
        self.assertEqual(nb["attack:syn_flood"], 3)

    def test_identical_models_paired_diff_is_zero(self):
        df = _table()
        r = paired_diff(df, df.copy(), n_boot=100)
        self.assertEqual((r["f1"]["lo"], r["f1"]["hi"], r["f1"]["diff"]), (0.0, 0.0, 0.0))

    def test_clearly_better_model_ci_excludes_zero(self):
        a = _table(n_runs_per_class=6, recall=0.99, seed=0)
        b = a.copy()
        rng = np.random.default_rng(5)
        miss = (b["y"] == 1) & (rng.uniform(size=len(b)) < 0.5)
        b.loc[miss, "pred"] = 0
        r = paired_diff(a, b, n_boot=300)
        self.assertGreater(r["recall"]["lo"], 0)

    def test_paired_requires_same_rows(self):
        a = _table()
        with self.assertRaises(ValueError):
            paired_diff(a, a.iloc[1:].reset_index(drop=True))
        self.assertFalse(same_rows(a, a.iloc[::-1]))


class TestPredictionFormat(unittest.TestCase):
    def test_roundtrip(self):
        df = _table()
        with tempfile.TemporaryDirectory() as d:
            p = save_predictions(df, df["y"], df["score"], df["pred"], "A", "toy", out_dir=Path(d))
            name, back = load_predictions(p)
        self.assertEqual(name, "A_toy")
        self.assertTrue(np.array_equal(back["pred"].to_numpy(), df["pred"].to_numpy()))


if __name__ == "__main__":
    unittest.main()
