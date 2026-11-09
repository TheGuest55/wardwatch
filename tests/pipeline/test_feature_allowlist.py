"""Pipeline test for review N6: features are an explicit allow-list (common.FEATURES), disjoint from
metadata, and the dataset builder refuses drift — a missing allow-listed feature or a stray emitted
column is an error, not silently treated as a feature or dropped."""
import unittest

import pandas as pd

from src import common, dataset


def _full_df():
    cols = {f: [0.0] for f in common.FEATURES}
    cols.update({
        "session": ["r01"], "window_start_iso": ["2026-11-09T00:00:00+00:00"],
        "window_start_epoch": [1762646400], "host_mac": ["52:54:00:00:00:01"],
        "host_ip": ["172.28.10.11"], "host_role": ["monitor"], "label": ["benign"],
        "attack_class": [""], "run_id": [""], "is_boundary": [False], "split": ["train"],
        "touches_attacker": [False], "dry_run": [False],
    })
    return pd.DataFrame(cols)


class TestFeatureAllowlist(unittest.TestCase):
    def test_disjoint_and_no_dupes(self):
        common.assert_feature_metadata_disjoint()  # must not raise

    def test_feature_columns_are_allowlist_in_order(self):
        feats = dataset._feature_columns(_full_df())
        self.assertEqual(feats, list(common.FEATURES))

    def test_missing_feature_is_error(self):
        df = _full_df().drop(columns=["nf_mean_flow_age"])
        with self.assertRaises(SystemExit):
            dataset._feature_columns(df)

    def test_unknown_column_is_error(self):
        df = _full_df()
        df["mystery_leak_col"] = 1.0
        with self.assertRaises(SystemExit):
            dataset._feature_columns(df)


if __name__ == "__main__":
    unittest.main()
