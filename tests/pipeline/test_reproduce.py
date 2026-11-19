"""src/reproduce.py: the comparison and its tolerances, the --out guard, the entry points it reuses, and that the user manual's commands
refer to files, make targets and modules that exist."""
import re
import tempfile
import unittest
from pathlib import Path

from src import reproduce as rp

ROOT = Path(__file__).resolve().parents[2]


class TestCompare(unittest.TestCase):
    def test_pass_fail_and_missing_values(self):
        want = {"test_all_f1": 0.900, "test_noatk_f1": 0.950}
        ok = rp.compare("x", {"test_all_f1": 0.903, "test_noatk_f1": 0.949}, want, 0.005)
        self.assertEqual([r["status"] for r in ok], ["PASS", "PASS"])
        bad = rp.compare("x", {"test_all_f1": 0.910, "test_noatk_f1": 0.950}, want, 0.005)
        self.assertEqual([r["status"] for r in bad], ["FAIL", "PASS"])
        self.assertAlmostEqual(bad[0]["diff"], 0.010)
        edge = rp.compare("x", {"test_all_f1": 0.905, "test_noatk_f1": 0.950}, want, 0.005 + 1e-12)
        self.assertEqual(edge[0]["status"], "PASS")                                   # the bound itself passes
        miss = rp.compare("x", {"test_all_f1": 0.9}, want, 0.005)
        self.assertEqual([r["status"] for r in miss], ["PASS", "FAIL"])                # an absent reproduced metric is a FAIL
        self.assertEqual(rp.compare("x", {"test_all_f1": 0.9}, None, 0.005)[0]["status"], "FAIL")   # no recorded run: FAIL

    def test_tolerances_per_model(self):
        for m in ("xgboost", "random_forest", "isolation_forest", "autoencoder", "rule_baseline"):
            self.assertEqual(rp.TOLERANCES[rp.MODEL_TOL[m]], 0.005, m)
        self.assertEqual(rp.TOLERANCES[rp.MODEL_TOL["lstm"]], 0.02)

    def test_report_writes_a_file_and_returns_the_verdict(self):
        rows = rp.compare("x", {"test_all_f1": 0.9, "test_noatk_f1": 0.95}, {"test_all_f1": 0.9, "test_noatk_f1": 0.95}, 0.005)
        with tempfile.TemporaryDirectory() as t:
            self.assertTrue(rp.report(rows, t, "A", 1.0))
            self.assertTrue((Path(t) / "reproduce_A.json").exists())
            bad = rp.compare("x", {"test_all_f1": 0.5, "test_noatk_f1": 0.95}, {"test_all_f1": 0.9, "test_noatk_f1": 0.95}, 0.005)
            self.assertFalse(rp.report(bad, t, "B", 1.0))


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
        self.assertEqual(head.count("```bash"), 2)
        self.assertEqual(head.count("```powershell"), 2)


if __name__ == "__main__":
    unittest.main()
