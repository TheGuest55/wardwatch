"""The Calibrator must be picklable from a `python -m` run of calibrate_explain (module __main__)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.models import calibrate_explain

ROOT = Path(__file__).resolve().parents[2]

# Executes calibrate_explain's code as __main__ (argv --help exits inside parse_args, so the module
# namespace is grabbed there), fits a Calibrator from that namespace and pickles it.
CHILD = r"""
import argparse, runpy, sys
import joblib, numpy as np
t = sys.argv[1]
sys.argv = ["calibrate_explain", "--help"]
holder = {}
orig = argparse.ArgumentParser.parse_args
def grab(self, *a, **k):
    holder["ns"] = sys._getframe(1).f_globals
    return orig(self, *a, **k)
argparse.ArgumentParser.parse_args = grab
try:
    runpy.run_module("src.models.calibrate_explain", run_name="__main__")
except SystemExit:
    pass
ns = holder["ns"]
assert ns["__name__"] == "__main__"
C = ns["Calibrator"]
for m in ("isotonic", "platt"):
    c = C(m).fit(np.load(t + "/s.npy"), np.load(t + "/y.npy"))
    joblib.dump(c, t + "/" + m + ".joblib")
"""


class TestCalibratorPickle(unittest.TestCase):
    def test_class_lives_in_the_importable_module(self):
        from src.models import calibrator as cal_mod
        self.assertEqual(cal_mod.Calibrator.__module__, "src.models.calibrator")
        self.assertIs(calibrate_explain.Calibrator, cal_mod.Calibrator)
        self.assertEqual(calibrate_explain.Calibrator.__module__, "src.models.calibrator")

    def test_pickle_made_under_main_loads_in_another_process(self):
        rng = np.random.default_rng(0)
        score = rng.uniform(0, 1, 500)
        y = (rng.uniform(0, 1, 500) < score).astype(int)
        grid = rng.uniform(0, 1, 200)
        with tempfile.TemporaryDirectory() as t:
            np.save(Path(t) / "s.npy", score)
            np.save(Path(t) / "y.npy", y)
            r = subprocess.run([sys.executable, "-c", CHILD, t], cwd=ROOT, capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": str(ROOT)})
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            for m in ("isotonic", "platt"):
                loaded = joblib.load(Path(t) / f"{m}.joblib")
                want = calibrate_explain.Calibrator(m).fit(score, y).predict(grid)
                np.testing.assert_array_equal(loaded.predict(grid), want)


if __name__ == "__main__":
    unittest.main()
