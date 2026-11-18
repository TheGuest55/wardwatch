"""The six triage cells (./CLAUDE.md "Forensic triage") for one host window.

    detector (IF) | classifier (XGBoost band)      | verdict                        | cell
    anomalous     | confident attack               | Known attack (type named)      | known
    anomalous     | unsure                         | Possible novel attack          | novel
    anomalous     | confident benign               | Detectors disagree → review    | review_anomaly_only
    normal        | confident attack               | Detectors disagree → review    | review_classifier_only
    normal        | unsure                         | Low priority, logged           | low_priority
    normal        | confident benign               | No alert                       | no_alert

The verdict strings are CLAUDE.md's, exactly. The two review cells share one verdict string, so the extra `cell`
field keeps them apart. The first four are ALERTING (they open or extend an incident); "Low priority, logged" is
counted but never becomes an incident.

host_role (and every other identity: MAC, IP) is DISPLAY-ONLY. Nothing in this package lets a role influence a
verdict, an attribution or a grouping decision; tests/triage/test_roles_display_only.py permutes roles and
asserts nothing else changes.
"""
import numpy as np

KNOWN = "Known attack (type named)"
NOVEL = "Possible novel attack"
REVIEW = "Detectors disagree → review"
LOW = "Low priority, logged"
NO_ALERT = "No alert"

VERDICTS = (KNOWN, NOVEL, REVIEW, LOW, NO_ALERT)
CELLS = ("known", "novel", "review_anomaly_only", "review_classifier_only", "low_priority", "no_alert")
CELL_VERDICT = {"known": KNOWN, "novel": NOVEL, "review_anomaly_only": REVIEW,
                "review_classifier_only": REVIEW, "low_priority": LOW, "no_alert": NO_ALERT}
ALERTING_CELLS = ("known", "novel", "review_anomaly_only", "review_classifier_only")
# most severe first; the incident verdict is the most severe cell among its alerts
SEVERITY = {c: i for i, c in enumerate(ALERTING_CELLS)}
BAND_BENIGN, BAND_UNSURE, BAND_ATTACK = 0, 1, 2     # calibrate_explain.band()

_TABLE = {(True, BAND_ATTACK): "known", (True, BAND_UNSURE): "novel", (True, BAND_BENIGN): "review_anomaly_only",
          (False, BAND_ATTACK): "review_classifier_only", (False, BAND_UNSURE): "low_priority",
          (False, BAND_BENIGN): "no_alert"}


def cell(anomalous, band):
    return _TABLE[(bool(anomalous), int(band))]


def cells(anomalous, band):
    """Vectorised cell(): arrays of anomaly flags and bands -> object array of cell names."""
    a, b = np.asarray(anomalous).astype(bool), np.asarray(band).astype(int)
    out = np.empty(len(a), dtype=object)
    for (an, bd), name in _TABLE.items():
        out[(a == an) & (b == bd)] = name
    return out


def verdict_of(c):
    return CELL_VERDICT[c]


def is_alerting(c):
    return c in SEVERITY
