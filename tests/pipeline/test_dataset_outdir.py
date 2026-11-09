"""Pipeline test for the dataset/datacheck output-directory parameterisation.

The v2 corpus must build into its own directory (data/processed/v2) so the v1 parquet at
data/processed/ stays byte-unchanged as review evidence. These tests pin that `build(out_dir=...)`
writes only under the given directory and never touches common.PROCESSED_DIR, and that datacheck's
per-session reader honours an explicit in-dir.
"""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import common, dataset
from src.datacheck import check


def _tiny_session_df(session):
    """A minimal valid host-window frame: the metadata columns build() needs plus one feature."""
    return pd.DataFrame({
        "session": [session, session],
        "window_start_iso": ["2026-11-09T00:00:00+00:00", "2026-11-09T00:00:05+00:00"],
        "window_start_epoch": [1762646400, 1762646405],
        "host_mac": ["52:54:00:00:00:01", "52:54:00:00:00:02"],
        "host_ip": ["172.28.10.11", "172.28.10.66"],
        "host_role": ["monitor", "attacker"],
        "label": ["normal", "attack"],
        "attack_class": ["", "recon_scan"],
        "run_id": ["", "x-recon_scan-01"],
        "is_boundary": [False, False],
        "foo": [1.0, 2.0],
    })


class TestDatasetOutDir(unittest.TestCase):
    def test_writes_to_out_dir_not_processed_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out_dir = tmp / "v2"
            processed = tmp / "processed_v1"  # stand-in for common.PROCESSED_DIR; must stay empty
            processed.mkdir()

            split = tmp / "split_v2.json"
            split.write_text(json.dumps({"train": ["a"], "val": [], "test": ["b"]}))

            orig_proc, orig_extract, orig_discover = (
                common.PROCESSED_DIR, dataset.extract.extract_session, dataset.discover_sessions)
            common.PROCESSED_DIR = processed
            dataset.extract.extract_session = lambda s: (_tiny_session_df(s), {})
            dataset.discover_sessions = lambda include_dry: ["a", "b"]
            try:
                dataset.build(str(split), force=True, include_dry=False, out_dir=out_dir)

                # Outputs land under out_dir …
                self.assertTrue((out_dir / "testbed_windows.parquet").exists())
                self.assertTrue((out_dir / "feature_list.json").exists())
                self.assertTrue((out_dir / "sessions" / "a.parquet").exists())
                self.assertTrue((out_dir / "sessions" / "b.parquet").exists())
                # … and PROCESSED_DIR is never written.
                self.assertEqual(list(processed.iterdir()), [])

                fl = json.loads((out_dir / "feature_list.json").read_text())
                self.assertIn("foo", fl["features"])
                self.assertNotIn("session", fl["features"])  # metadata excluded
            finally:
                common.PROCESSED_DIR = orig_proc
                dataset.extract.extract_session = orig_extract
                dataset.discover_sessions = orig_discover

    def test_default_out_dir_is_processed_dir(self):
        """Back-compat: omitting out_dir still targets common.PROCESSED_DIR."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            split = tmp / "s.json"
            split.write_text(json.dumps({"train": ["a"], "val": [], "test": ["b"]}))
            orig_proc, orig_extract, orig_discover = (
                common.PROCESSED_DIR, dataset.extract.extract_session, dataset.discover_sessions)
            common.PROCESSED_DIR = tmp / "default"
            dataset.extract.extract_session = lambda s: (_tiny_session_df(s), {})
            dataset.discover_sessions = lambda include_dry: ["a", "b"]
            try:
                dataset.build(str(split), force=True, include_dry=False)  # no out_dir
                self.assertTrue((common.PROCESSED_DIR / "testbed_windows.parquet").exists())
            finally:
                common.PROCESSED_DIR = orig_proc
                dataset.extract.extract_session = orig_extract
                dataset.discover_sessions = orig_discover


class TestDatacheckInDir(unittest.TestCase):
    def test_per_session_parquet_reads_in_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "v2"
            (in_dir / "sessions").mkdir(parents=True)
            _tiny_session_df("r01").to_parquet(in_dir / "sessions" / "r01.parquet", index=False)
            # Reads from the explicit in-dir …
            self.assertIsNotNone(check.per_session_parquet("r01", in_dir))
            # … and does not fall back to a different (empty) dir.
            self.assertIsNone(check.per_session_parquet("r01", Path(tmp) / "empty"))


if __name__ == "__main__":
    unittest.main()
