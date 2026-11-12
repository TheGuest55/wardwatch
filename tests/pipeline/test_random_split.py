"""Random-split reference (src/models/random_split.py): the split helper and leakage diagnostics."""
import unittest

import numpy as np
import pandas as pd

from src.models.random_split import (attack_run_in_train, neighbour_in_train,
                                     random_split_indices)


class TestRandomSplitIndices(unittest.TestCase):
    def setUp(self):
        self.strata = np.array(["benign"] * 900 + ["syn_flood"] * 60 + ["pump_cmd"] * 40)

    def test_sizes_disjoint_and_complete(self):
        tr, va, te = random_split_indices(self.strata, (500, 250, 250), seed=0)
        self.assertEqual((len(tr), len(va), len(te)), (500, 250, 250))
        allidx = np.concatenate([tr, va, te])
        self.assertEqual(len(set(allidx)), 1000)

    def test_stratified(self):
        tr, va, te = random_split_indices(self.strata, (500, 250, 250), seed=0)
        for part, n in ((tr, 500), (te, 250)):
            share = (self.strata[part] == "pump_cmd").mean()
            self.assertAlmostEqual(share, 40 / 1000, delta=0.01)

    def test_deterministic_per_seed(self):
        a = random_split_indices(self.strata, (500, 250, 250), seed=3)
        b = random_split_indices(self.strata, (500, 250, 250), seed=3)
        c = random_split_indices(self.strata, (500, 250, 250), seed=4)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(np.array_equal(a[2], c[2]))

    def test_bad_sizes_rejected(self):
        with self.assertRaises(ValueError):
            random_split_indices(self.strata, (500, 250, 249), seed=0)


def _meta(rows):
    return pd.DataFrame(rows, columns=["session", "host_mac", "window_start_epoch", "label",
                                       "run_id"])


class TestLeakageDiagnostics(unittest.TestCase):
    def test_neighbour_found_either_side_only_same_host_and_session(self):
        train = _meta([("r01", "aa", 100.0, "benign", None),
                       ("r01", "bb", 200.0, "benign", None)])
        test = _meta([("r01", "aa", 105.0, "benign", None),    # 100 in train -> neighbour
                      ("r01", "bb", 195.0, "benign", None),    # 200 in train -> neighbour
                      ("r01", "cc", 105.0, "benign", None),    # other host -> no
                      ("r02", "aa", 105.0, "benign", None),    # other session -> no
                      ("r01", "aa", 120.0, "benign", None)])   # 15 s away -> no
        self.assertAlmostEqual(neighbour_in_train(train, test), 2 / 5)

    def test_disjoint_sessions_have_zero_overlap(self):
        train = _meta([("r01", "aa", 100.0, "attack", "run1")])
        test = _meta([("r05", "aa", 105.0, "attack", "run9")])
        self.assertEqual(neighbour_in_train(train, test), 0.0)
        self.assertEqual(attack_run_in_train(train, test), 0.0)

    def test_attack_run_overlap_counts_attack_rows_only(self):
        train = _meta([("r01", "aa", 100.0, "attack", "run1")])
        test = _meta([("r01", "aa", 300.0, "attack", "run1"),
                       ("r01", "aa", 400.0, "attack", "run2"),
                       ("r01", "aa", 500.0, "benign", "run1")])
        self.assertAlmostEqual(attack_run_in_train(train, test), 1 / 2)


if __name__ == "__main__":
    unittest.main()
