"""src/reproduce.py: the comparison and its tolerances, the --out guard, the entry points it reuses, and that the user manual's commands
refer to files, make targets and modules that exist."""
import re
import tempfile
import unittest
from pathlib import Path

from src import reproduce as rp
from tests.needs import needs

ROOT = Path(__file__).resolve().parents[2]


ENTRY = {"experiment": "A", "model": "xgboost", "tolerance": 0.005,
         "metrics": {"test_all_f1": 0.900, "test_noatk_f1": 0.950, "test_all_pr_auc": 0.93, "test_noatk_pr_auc": 0.96,
                     "test_all_fpr_at_recall_90": 0.0175, "test_noatk_fpr_at_recall_90": 0.0}}


def good():
    return dict(ENTRY["metrics"])


class TestCompare(unittest.TestCase):
    def test_every_metric_of_the_entry_is_compared_with_the_entrys_tolerance(self):
        rows = rp.compare("x", good(), ENTRY)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["tolerance"] for r in rows}, {0.005})
        self.assertTrue(all(r["status"] == "PASS" for r in rows))
        near = good()
        near["test_all_pr_auc"] += 0.004
        self.assertTrue(all(r["status"] == "PASS" for r in rp.compare("x", near, ENTRY)))
        far = good()
        far["test_all_pr_auc"] += 0.006
        far["test_noatk_f1"] -= 0.0001
        res = {r["metric"]: r["status"] for r in rp.compare("x", far, ENTRY)}
        self.assertEqual(res["test_all_pr_auc"], "FAIL")
        self.assertEqual(res["test_noatk_f1"], "PASS")
        loose = {**ENTRY, "tolerance": 0.02}                                           # a different entry tolerance is used as given
        self.assertTrue(all(r["status"] == "PASS" for r in rp.compare("x", far, loose)))

    def test_a_missing_reproduced_value_or_a_missing_run_is_a_fail(self):
        got = good()
        del got["test_all_fpr_at_recall_90"]
        self.assertEqual([r["metric"] for r in rp.compare("x", got, ENTRY) if r["status"] == "FAIL"], ["test_all_fpr_at_recall_90"])
        self.assertTrue(all(r["status"] == "FAIL" for r in rp.compare("x", None, ENTRY)))

    def test_tolerances_per_model(self):
        for m in ("xgboost", "random_forest", "isolation_forest", "autoencoder", "rule_baseline"):
            self.assertEqual(rp.TOLERANCES[rp.MODEL_TOL[m]], 0.005, m)
        self.assertEqual(rp.TOLERANCES[rp.MODEL_TOL["lstm"]], 0.02)

    def test_report_writes_a_file_and_returns_the_verdict(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertTrue(rp.report(rp.compare("x", good(), ENTRY), t, "A", 1.0))
            self.assertTrue((Path(t) / "reproduce_A.json").exists())
            bad = good()
            bad["test_all_f1"] = 0.5
            self.assertFalse(rp.report(rp.compare("x", bad, ENTRY), t, "B", 1.0))


def fake_expected(dataset_sha="a" * 64):
    runs = {}
    for m in rp.A_MODELS:
        runs[rp.entry_key("A", m)] = {**ENTRY, "model": m, "experiment": "A"}
    for m in rp.B_MODELS:
        for c in rp.B_CLASSES:
            runs[rp.entry_key("B", m, c)] = {**ENTRY, "model": m, "experiment": "B", "class": c}
    runs[rp.lstm_key(12, 32)] = {**ENTRY, "model": "lstm", "experiment": "LSTM", "k": 12, "hidden": 32, "tolerance": 0.02}
    return {"generated_utc": "x", "generator": rp.GENERATOR, "dataset_sha256": dataset_sha, "split_sha256": "b" * 64, "runs": runs}


class TestExpectedFile(unittest.TestCase):
    def test_schema_of_the_committed_file(self):
        d = rp.load_expected()
        for k in ("generated_utc", "generator", "notes", "dataset_sha256", "split_sha256", "runs"):
            self.assertIn(k, d)
        self.assertEqual(d["generator"], "python -m src.reproduce --write-expected")
        self.assertIn("SHA-256 of the run record file", d["notes"])
        self.assertRegex(d["dataset_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(d["split_sha256"], r"^[0-9a-f]{64}$")
        for key, e in d["runs"].items():
            for f in ("experiment", "model", "record_path", "run_id", "git_commit", "tolerance", "metrics"):
                self.assertIn(f, e, key)
            self.assertRegex(e["run_id"], r"^[0-9a-f]{64}$")
            self.assertEqual(set(e["metrics"]), set(rp.METRICS), key)
            self.assertTrue(all(isinstance(v, float) for v in e["metrics"].values()), key)
            self.assertEqual(e["tolerance"], rp.TOLERANCES[rp.MODEL_TOL[e["model"]]], key)
            self.assertFalse(e["record_path"].startswith("/"), key)
            if e["experiment"] == "B":
                self.assertIn(e["class"], rp.B_CLASSES)
            if e["experiment"] == "LSTM":
                self.assertEqual((e["k"], e["hidden"]), (12, 32))

    def test_every_run_the_targets_compare_has_an_entry(self):
        d = rp.load_expected()
        self.assertEqual({e["model"] for _, e in rp.runs_for(d, "A")}, set(rp.A_MODELS))
        self.assertEqual({(e["model"], e["class"]) for _, e in rp.runs_for(d, "B")},
                         {(m, c) for m in rp.B_MODELS for c in rp.B_CLASSES})
        self.assertEqual(len(rp.runs_for(d, "LSTM")), 1)
        self.assertEqual(len(d["runs"]), len(rp.A_MODELS) + len(rp.B_MODELS) * len(rp.B_CLASSES) + 1)

    def test_missing_expected_file_is_a_clear_error(self):
        with self.assertRaises(SystemExit):
            rp.load_expected("definitely/not/here.json")

    @needs("RESULTS", "MODELS")
    def test_expected_json_equals_the_run_records_on_full_data(self):
        import json
        d = rp.load_expected()
        sha = lambda p: rp.sha256_file(p)
        for key, e in d["runs"].items():
            rec_path = ROOT / e["record_path"]
            rec = json.loads(rec_path.read_text())
            self.assertTrue(rec["for_report"], key)
            self.assertEqual(e["git_commit"], rec["provenance"]["git_commit"], key)
            self.assertEqual(e["run_id"], sha(rec_path), key)
            self.assertEqual(d["dataset_sha256"], rec["provenance"]["dataset_sha256"], key)
            self.assertEqual(d["split_sha256"], rec["provenance"]["split_sha256"], key)
            for m, v in e["metrics"].items():
                self.assertEqual(v, float(rec["metrics"][m]), (key, m))
        self.assertEqual(d["dataset_sha256"], sha(rp.DATASET_FILE))
        self.assertEqual(d["split_sha256"], sha(rp.SPLIT_FILE))


class TestWriteExpected(unittest.TestCase):
    def _records(self, root, for_report=True, dataset="d" * 64):
        import json
        def rec(path, f1):
            p = Path(root) / path
            p.parent.mkdir(parents=True, exist_ok=True)
            m = {k: f1 for k in rp.METRICS}
            p.write_text(json.dumps({"for_report": for_report, "metrics": m,
                                     "params": {"threads": 5}, "provenance": {"git_commit": "c" * 40, "dataset_sha256": dataset,
                                                                               "split_sha256": "s" * 64}}))
        for m in rp.A_MODELS:
            rec(f"expA/{m}_s0.json", 0.9)
        for m in rp.B_MODELS:
            for c in rp.B_CLASSES:
                rec(f"expB/{m}_{c}_s0.json", 0.5)
        rec("lstm/A_k12_h32_s0.json", 0.95)
        (Path(root) / "lstm" / "selected_config.json").write_text(json.dumps({"k": 12, "hidden": 32}))

    def test_writes_the_file_from_report_grade_records_and_refuses_otherwise(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            self._records(t / "repo" / "runs")
            doc = rp.write_expected(records=t / "repo" / "runs", out=t / "repo" / "reproduce" / "expected.json", repo_root=t / "repo",
                                    dataset_file=t / "nope.parquet")
            self.assertEqual(len(doc["runs"]), 12)
            self.assertEqual(doc["runs"]["A/xgboost"]["record_path"], "runs/expA/xgboost_s0.json")
            self.assertEqual(doc["runs"]["LSTM/A_k12_h32_s0"]["k"], 12)
            (t / "repo" / "runs" / "expA" / "xgboost_s0.json").unlink()                 # a missing record stops it
            with self.assertRaises(SystemExit):
                rp.write_expected(records=t / "repo" / "runs", out=t / "x.json", repo_root=t / "repo", dataset_file=t / "nope")
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            self._records(t / "repo" / "runs", for_report=False)                        # not report-grade
            with self.assertRaises(SystemExit):
                rp.write_expected(records=t / "repo" / "runs", out=t / "x.json", repo_root=t / "repo", dataset_file=t / "nope")


class TestDatasetCheck(unittest.TestCase):
    def test_wrong_dataset_fails_before_any_training_and_the_right_one_passes(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "testbed_windows.parquet"
            f.write_bytes(b"some dataset bytes")
            good_sha = rp.sha256_file(f)
            self.assertEqual(rp.check_dataset(fake_expected(good_sha), f), good_sha)
            with self.assertRaises(SystemExit) as e:
                rp.check_dataset(fake_expected("0" * 64), f)
            self.assertIn("wrong dataset", str(e.exception))
            with self.assertRaises(SystemExit):
                rp.check_dataset(fake_expected(good_sha), Path(t) / "absent.parquet")

    def test_main_stops_on_a_wrong_dataset_without_training(self):
        with tempfile.TemporaryDirectory() as t:
            ex = Path(t) / "expected.json"
            import json
            ex.write_text(json.dumps(fake_expected("0" * 64)))
            orig = rp.DATASET_FILE
            rp.DATASET_FILE = Path(t) / "x.parquet"
            rp.DATASET_FILE.write_bytes(b"not the dataset")
            try:
                with self.assertRaises(SystemExit) as e:
                    rp.main(["A", "--expected", str(ex), "--out", str(Path(t) / "out")])
            finally:
                rp.DATASET_FILE = orig
            self.assertIn("wrong dataset", str(e.exception))
            self.assertFalse((Path(t) / "out" / "runs").exists())                      # nothing was trained


class TestGroundTruthCheck(unittest.TestCase):
    def test_lstm_target_says_what_is_missing_before_doing_any_work(self):
        import json
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "split_v2.json").write_text(json.dumps({"train": ["a", "b"], "val": ["c"], "test": ["d"]}))
            (t / "a").mkdir()
            (t / "a" / "ground_truth.jsonl").write_text("{}")
            with self.assertRaises(SystemExit) as e:
                rp.check_ground_truth(t / "split_v2.json")
            msg = str(e.exception)
            self.assertIn("['b', 'c', 'd']", msg)
            self.assertIn("USER_MANUAL Level 3", msg)
            for s in "bcd":
                (t / s).mkdir()
                (t / s / "ground_truth.jsonl").write_text("{}")
            rp.check_ground_truth(t / "split_v2.json")                                  # all present: no error


class TestNeverReadsRuns(unittest.TestCase):
    def test_loading_comparing_and_checking_open_nothing_under_a_runs_folder(self):
        import builtins
        seen = []
        real_open = builtins.open

        def spy(file, *a, **k):
            seen.append(str(file))
            return real_open(file, *a, **k)
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "testbed_windows.parquet"
            f.write_bytes(b"x")
            builtins.open = spy
            try:
                d = fake_expected(rp.sha256_file(f))
                rp.check_dataset(d, f)
                rp.compare("x", good(), d["runs"]["A/xgboost"])
                rp.load_expected()
            finally:
                builtins.open = real_open
        self.assertFalse([p for p in seen if "/runs/" in p.replace("\\", "/") or p.replace("\\", "/").endswith("/runs")], seen)

    def test_the_only_code_that_names_the_run_records_is_write_expected_and_the_out_guard(self):
        import re
        src = (ROOT / "src" / "reproduce.py").read_text()
        body = src.split("def write_expected(")[0]
        self.assertNotIn("selected_config.json", body)
        self.assertNotIn("--recorded", src)
        # the target functions read only what they wrote under --out
        for fn in ("run_A", "run_B", "run_lstm"):
            code = src.split(f"def {fn}(")[1].split("\ndef ")[0]
            self.assertNotIn("RECORDED", code)


class TestOutGuard(unittest.TestCase):
    def test_refuses_recorded_results_and_their_parents_accepts_a_separate_folder(self):
        with tempfile.TemporaryDirectory() as t:
            rec = Path(t) / "v2" / "runs"
            (rec / "expA").mkdir(parents=True)
            for bad in (rec, rec / "expA", Path(t) / "v2" / "runs" / "new", rec.parent):
                with self.assertRaises(SystemExit, msg=str(bad)):
                    rp.check_out(bad, rec)
            self.assertEqual(rp.check_out(Path(t) / "v2" / "repro" / "A", rec), (Path(t) / "v2" / "repro" / "A").resolve())

    def test_default_out_is_not_runs_and_is_accepted(self):
        self.assertNotEqual(rp.OUT_DEFAULT.name, "runs")
        rp.check_out(rp.OUT_DEFAULT)                                                  # does not raise


class TestWiring(unittest.TestCase):
    def test_targets_and_entry_points_exist(self):
        mk = (ROOT / "Makefile").read_text()
        for t in ("reproduce-A", "reproduce-B", "reproduce-lstm"):
            self.assertRegex(mk, rf"(?m)^{t}:")
        sh = (ROOT / "scripts" / "analysis.sh").read_text()
        self.assertIn("reproduce)", sh)
        self.assertIn("src.reproduce", sh)
        import inspect
        from src.experiments import run_lstm, run_rule_baseline, run_tree_report
        from src.models import train_autoencoder as ta
        for fn in (run_tree_report.run_a, run_tree_report.run_b, ta.run_a, ta.run_b, run_rule_baseline.run_a, run_rule_baseline.run_b):
            self.assertIsNotNone(fn)
        for fn in (run_tree_report.run_b, ta.run_b, run_rule_baseline.run_b):
            self.assertIn("only", inspect.signature(fn).parameters)                   # the reproduce-B class filter
        self.assertTrue(callable(run_lstm.run_one) and callable(run_lstm.load_context))

    def test_runs_directory_is_never_an_output_of_the_reproduce_code(self):
        src = (ROOT / "src" / "reproduce.py").read_text()
        self.assertNotIn("for_report=True", src)
        self.assertIn("for_report=False", src)


class TestManualReferences(unittest.TestCase):
    """Every make target, script, module and docker folder the manual or the README tells the reader to use exists."""

    def _text(self):
        return (ROOT / "USER_MANUAL.md").read_text() + "\n" + (ROOT / "README.md").read_text()

    def test_make_targets_exist(self):
        mk = (ROOT / "Makefile").read_text()
        targets = set(re.findall(r"(?m)^([A-Za-z][\w-]*):", mk))
        for t in set(re.findall(r"`make ([a-z][\w-]*)", self._text())):
            self.assertIn(t, targets, f"manual or README mentions `make {t}` but the Makefile has no such target")

    def test_scripts_modules_and_folders_exist(self):
        text = self._text()
        for p in set(re.findall(r"scripts/[\w./-]+\.(?:sh|py)", text)):
            self.assertTrue((ROOT / p).exists(), p)
        for m in set(re.findall(r"python -m (src\.[\w.]+)", text)):
            self.assertTrue((ROOT / (m.replace(".", "/") + ".py")).exists(), m)
        for d in ("docker/dashboard", "services/analysis"):
            self.assertTrue((ROOT / d / "Dockerfile").exists(), d)
        # the dashboard image tag in the manual is the tag the Makefile builds
        self.assertIn("wardwatch-dashboard:0.1.0", (ROOT / "Makefile").read_text())

    def test_level_one_and_two_quick_start_commands_are_present_in_both_shells(self):
        m = (ROOT / "USER_MANUAL.md").read_text()
        self.assertIn("## Quick start", m)
        head = m.split("## 1. Requirements")[0]
        self.assertGreaterEqual(head.count("```bash"), 2)
        self.assertGreaterEqual(head.count("```powershell"), 2)
        self.assertIn("submission package", head)                                      # the primary route: no GitHub access needed
        self.assertIn("If you have access to the private repository", head)


if __name__ == "__main__":
    unittest.main()
