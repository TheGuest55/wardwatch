"""Score calibrator, in its own importable module.

calibrate_explain is run as `python -m src.models.calibrate_explain`, so a class defined in it is
pickled as `__main__.Calibrator` and cannot be loaded by any other process (triage, latency, tests).
Defining the class here makes a pickle always refer to `src.models.calibrator.Calibrator`.
EPS and _logit are copies of the ones in calibrate_explain (kept there too, used elsewhere).
"""
import numpy as np

EPS = 1e-6


def _logit(s):
    s = np.clip(np.asarray(s, dtype=float), EPS, 1 - EPS)
    return np.log(s / (1 - s))


class Calibrator:
    """Maps a raw classifier score to a calibrated attack probability. method: platt | isotonic."""

    def __init__(self, method):
        if method not in ("platt", "isotonic"):
            raise ValueError(method)
        self.method = method
        self._m = None

    def fit(self, score, y):
        y = np.asarray(y).astype(int)
        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self._m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(score)[:, None], y)
        else:
            from sklearn.isotonic import IsotonicRegression
            self._m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                np.asarray(score, dtype=float), y)
        return self

    def predict(self, score):
        if self.method == "platt":
            return self._m.predict_proba(_logit(score)[:, None])[:, 1]
        return self._m.predict(np.asarray(score, dtype=float))
