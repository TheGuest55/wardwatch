"""The non-ML rule baseline (CLAUDE.md): rate thresholds plus a wildcard-subscription rule, OR-ed.

Rules (a window is an alert if ANY rule fires):
  * five rate rules, one per feature f in RATE_FEATURES (mqtt_connect_rate, mqtt_publish_rate, tcp_syn_rate,
    arp_reply_rate, mqtt_connack_failures):  fire if  x_f >= t_f;
  * the wildcard rule, fixed, no threshold:  fire if  topic_wildcard_subs >= 1.
A rate rule with t_f = +inf is OFF. (">=" rather than ">" so that the score below is consistent with the verdict.)

Thresholds are fitted on VALIDATION rows only. Candidates per feature are the unique POSITIVE validation values
(capped at MAX_CANDIDATES by quantiles, values actually present) plus "off". A deterministic coordinate ascent
visits the features in the fixed RATE_FEATURES order, setting each t_f to the candidate that maximises the
validation F1 of the OR with the other thresholds held fixed (ties go to the HIGHER threshold, "off" being the
highest, i.e. the fewest alerts), and repeats until a full pass changes nothing. It starts with every rate rule off.
fit() refuses anything but validation rows.

Score (for PR-AUC / ROC-AUC and the shared prediction files): score = max over the rules of their strength, where a
rate rule's strength is x_f / t_f and the wildcard rule's is 1 + topic_wildcard_subs when it fires (else 0); OFF rules
contribute 0. pred = (score >= 1), exactly the OR. It is a rule-strength score, NOT a probability. fired() returns
which rules fired per window, for later triage use.
"""
import numpy as np

RATE_FEATURES = ("mqtt_connect_rate", "mqtt_publish_rate", "tcp_syn_rate", "arp_reply_rate", "mqtt_connack_failures")
WILDCARD = "topic_wildcard_subs"
RULE_FEATURES = RATE_FEATURES + (WILDCARD,)
MAX_CANDIDATES = 200
INF = float("inf")


def _f1(y, pred):
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def candidates(values, max_candidates=MAX_CANDIDATES):
    """Unique positive finite values of a feature on validation, thinned to at most max_candidates by quantiles
    (always values that occur), sorted ascending."""
    u = np.unique(np.asarray(values, dtype=float))
    u = u[np.isfinite(u) & (u > 0)]
    if len(u) > max_candidates:
        idx = np.unique(np.round(np.linspace(0, len(u) - 1, max_candidates)).astype(int))
        u = u[idx]
    return u


class RuleBaseline:
    def __init__(self, thresholds=None, features=None):
        self.thresholds = dict(thresholds) if thresholds else {f: INF for f in RATE_FEATURES}
        self.features = list(features) if features is not None else None

    # ---------------------------------------------------------------- columns
    def _cols(self, X, feats=None):
        feats = list(feats if feats is not None else self.features)
        X = np.asarray(X, dtype=float)
        miss = [f for f in RULE_FEATURES if f not in feats]
        if miss:
            raise ValueError(f"rule features missing from the feature list: {miss}")
        return {f: X[:, feats.index(f)] for f in RULE_FEATURES}

    # ---------------------------------------------------------------- fit
    def fit(self, X, y, feats, split):
        """Fit the rate thresholds on VALIDATION rows. `split` is the split name of every row (a str or an array);
        anything other than "val" raises ValueError, so test rows can never reach the fit."""
        s = np.unique(np.asarray(split, dtype=object).astype(str)) if not isinstance(split, str) else [split]
        if list(s) != ["val"]:
            raise ValueError(f"the rule thresholds are fitted on validation rows only; got split(s) {list(s)}")
        y = np.asarray(y).astype(int)
        if len(y) != len(X):
            raise ValueError("X and y differ in length")
        if y.sum() == 0:
            raise ValueError("validation has no attack rows")
        self.features = list(feats)
        cols = self._cols(X, feats)
        wild = cols[WILDCARD] >= 1
        t = {f: INF for f in RATE_FEATURES}
        fire = {f: np.zeros(len(y), bool) for f in RATE_FEATURES}
        cand = {f: candidates(cols[f]) for f in RATE_FEATURES}

        def total(skip, extra):
            out = wild.copy()
            for f in RATE_FEATURES:
                out |= extra if f == skip else fire[f]
            return out

        for _ in range(50):
            changed = False
            for f in RATE_FEATURES:
                best_t, best_f1 = INF, _f1(y, total(f, np.zeros(len(y), bool)))     # "off" first = the highest threshold
                for c in cand[f][::-1]:                                                # then descending: ties keep the higher t
                    f1 = _f1(y, total(f, cols[f] >= c))
                    if f1 > best_f1 + 1e-12:
                        best_t, best_f1 = float(c), f1
                if best_t != t[f]:
                    t[f], changed = best_t, True
                    fire[f] = cols[f] >= best_t if np.isfinite(best_t) else np.zeros(len(y), bool)
            if not changed:
                break
        self.thresholds = t
        self.val_f1_ = _f1(y, total(None, np.zeros(len(y), bool)))
        return self

    # ---------------------------------------------------------------- apply
    def fired(self, X, feats=None):
        """Boolean matrix (n, 6) of the rules that fired, columns in RULE_FEATURES order."""
        cols = self._cols(X, feats)
        out = np.zeros((len(cols[WILDCARD]), len(RULE_FEATURES)), bool)
        for j, f in enumerate(RATE_FEATURES):
            t = self.thresholds[f]
            if np.isfinite(t):
                out[:, j] = cols[f] >= t
        out[:, -1] = cols[WILDCARD] >= 1
        return out

    def score(self, X, feats=None):
        """Rule-strength score (not a probability): max over the rules of x_f / t_f (rate rules that are on) and
        1 + topic_wildcard_subs (wildcard rule, when it fires). pred = score >= 1."""
        cols = self._cols(X, feats)
        s = np.zeros(len(cols[WILDCARD]))
        for f in RATE_FEATURES:
            t = self.thresholds[f]
            if np.isfinite(t):
                s = np.maximum(s, cols[f] / t)
        w = cols[WILDCARD]
        return np.maximum(s, np.where(w >= 1, 1.0 + w, 0.0))

    def predict(self, X, feats=None):
        return (self.score(X, feats) >= 1.0).astype(int)

    def rules_that_fired(self, X, feats=None):
        """Per window, the list of rule names that fired (for triage / explanations)."""
        F = self.fired(X, feats)
        return [[RULE_FEATURES[j] for j in np.flatnonzero(r)] for r in F]

    def describe(self):
        return {**{f: (None if not np.isfinite(t) else float(t)) for f, t in self.thresholds.items()},
                WILDCARD: ">= 1 (fixed)"}
