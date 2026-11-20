"""scripts/make_sanitised_gt.py and the ground-truth lookup in src/models/sequence.py."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.models import sequence as seq
from tests.needs import needs

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("make_sanitised_gt", ROOT / "scripts" / "make_sanitised_gt.py")
msg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msg)

RAW_LINE = {"run_id": "r1-spoof_vitals-01", "session": "r1", "class": "spoof_vitals", "stage": "credential abuse",
            "start_utc": "2026-11-09T00:14:05.000Z", "end_utc": "2026-11-09T00:14:39.263Z", "src_ip": "172.28.10.66",
            "src_mac": "c6:e6:a8:5f:29:af", "targets": ["x"], "params": {"user": "SECRETUSERNAME"}, "outcome": "success",
            "notes": "to bed2 as bed2-monitor with SECRETPASSWORD123"}


def make_raw(root, creds=("SECRETPASSWORD123",), notes=None):
    raw = root / "sessions"
    for s, split in (("r1", "train"), ("r2", "test")):
        (raw / s).mkdir(parents=True)
        rec = dict(RAW_LINE, session=s, run_id=f"{s}-spoof_vitals-01")
        if notes is not None:
            rec["class"] = notes
        (raw / s / "ground_truth.jsonl").write_text(json.dumps(rec) + "\n\n")
        (raw / s / "stolen_creds.json").write_text(json.dumps({"bed2-monitor": {"password": creds[0]}}))
    (raw / "split_v2.json").write_text(json.dumps({"train": ["r1"], "val": [], "test": ["r2"]}))
    (root / "accounts.json").write_text(json.dumps({"bed2-monitor": "SECRETPASSWORD123"}))
    return raw


class TestMake(unittest.TestCase):
    def test_keeps_only_the_labelling_fields_and_drops_notes_params_targets(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            raw = make_raw(t)
            msg.build(raw, t / "out", t / "accounts.json")
            text = (t / "out" / "r1" / "ground_truth.jsonl").read_text()
            rec = json.loads(text.splitlines()[0])
            self.assertEqual(set(rec), {"run_id", "session", "class", "start_utc", "end_utc", "src_ip", "src_mac"})
            for bad in ("SECRETPASSWORD123", "SECRETUSERNAME", "bed2-monitor", "params", "notes"):
                self.assertNotIn(bad, text)

    def test_structured_victims_are_kept(self):
        rec = dict(RAW_LINE, victims=[{"ip": "172.28.10.21", "mac": "aa:bb"}])
        self.assertEqual(msg.sanitise_line(rec)["victims"], [{"ip": "172.28.10.21", "mac": "aa:bb"}])

    def test_a_credential_in_the_output_stops_the_build_without_printing_it(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            raw = make_raw(t, notes="SECRETPASSWORD123")            # the class field would carry the password
            with self.assertRaises(SystemExit) as e:
                msg.build(raw, t / "out", t / "accounts.json")
            self.assertIn("credential", str(e.exception))
            self.assertNotIn("SECRETPASSWORD123", str(e.exception))
            self.assertFalse((t / "out").exists())                    # nothing written

    def test_a_session_without_raw_ground_truth_gets_an_empty_file_only_if_all_its_windows_are_benign(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            raw = make_raw(t)
            (raw / "r2" / "ground_truth.jsonl").unlink()
            sd = t / "proc"
            sd.mkdir()
            pd.DataFrame({"label": ["benign", "benign"]}).to_parquet(sd / "r2.parquet")
            msg.build(raw, t / "out", t / "accounts.json", sd)
            self.assertEqual((t / "out" / "r2" / "ground_truth.jsonl").read_text(), "")
            pd.DataFrame({"label": ["benign", "attack"]}).to_parquet(sd / "r2.parquet")
            with self.assertRaises(SystemExit) as e:
                msg.build(raw, t / "out2", t / "accounts.json", sd)
            self.assertIn("no ground truth", str(e.exception))
            (sd / "r2.parquet").unlink()
            with self.assertRaises(SystemExit):
                msg.build(raw, t / "out3", t / "accounts.json", sd)

    def test_only_sessions_of_the_split_are_written(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            raw = make_raw(t)
            (raw / "x9").mkdir()
            (raw / "x9" / "ground_truth.jsonl").write_text("{}")
            self.assertEqual(msg.build(raw, t / "out", t / "accounts.json"), ["r1", "r2"])
            self.assertFalse((t / "out" / "x9").exists())


class TestLookup(unittest.TestCase):
    def _tree(self, t, raw_sessions, san_sessions):
        t = Path(t)
        (t / "raw").mkdir()
        (t / "raw" / "split_v2.json").write_text(json.dumps({"train": ["a"], "val": ["b"], "test": []}))
        for s in raw_sessions:
            (t / "raw" / s).mkdir()
            (t / "raw" / s / "ground_truth.jsonl").write_text("{}")
        san = t / "san"
        for s in san_sessions:
            (san / s).mkdir(parents=True)
            (san / s / "ground_truth.jsonl").write_text("{}")
        seq.V2_GT_DIR = san
        return t / "raw" / "split_v2.json", t

    def setUp(self):
        self._orig = seq.V2_GT_DIR

    def tearDown(self):
        seq.V2_GT_DIR = self._orig

    def test_raw_is_preferred_when_complete_then_sanitised_then_raw(self):
        with tempfile.TemporaryDirectory() as t:
            sp, t = self._tree(t, ["a", "b"], ["a", "b"])
            self.assertEqual(seq.default_ground_truth_dir(sp), t / "raw")
        with tempfile.TemporaryDirectory() as t:
            sp, t = self._tree(t, ["a"], ["a", "b"])                  # raw incomplete
            self.assertEqual(seq.default_ground_truth_dir(sp), t / "san")
        with tempfile.TemporaryDirectory() as t:
            sp, t = self._tree(t, ["a"], ["a"])                       # neither complete
            self.assertEqual(seq.default_ground_truth_dir(sp), t / "raw")


@needs("GROUND_TRUTH", "DATASET", "CREDS")
class TestRealEquality(unittest.TestCase):
    def test_labels_and_exclusion_classes_are_identical_from_raw_and_sanitised_ground_truth(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            msg.build(ROOT / "data/raw/sessions", t / "gt", ROOT / "data/raw/creds/accounts.json", ROOT / "data/processed/v2/sessions")
            sp = ROOT / "data/raw/sessions/split_v2.json"
            a = seq.load_sessions(ROOT / "data/processed/v2/sessions", sp, ROOT / "data/raw/sessions")
            b = seq.load_sessions(ROOT / "data/processed/v2/sessions", sp, t / "gt")
            self.assertGreater((a["label"] == "boundary").sum(), 0)
            self.assertTrue(a["exclusion_class"].equals(b["exclusion_class"]))
            self.assertTrue(a[["label", "attack_class", "run_id"]].equals(b[["label", "attack_class", "run_id"]]))


if __name__ == "__main__":
    unittest.main()
