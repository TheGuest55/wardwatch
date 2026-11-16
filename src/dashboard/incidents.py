"""Incident helpers (format: docs/dashboard-notes.md, file: live/incidents.jsonl)."""
import hashlib
from pathlib import Path

import pandas as pd

from . import config

# the six verdicts of the triage table (CLAUDE.md), in display order
VERDICTS = ("Known attack", "Possible novel attack", "Detectors disagree: review",
            "Low priority, logged", "No alert")
ALIASES = {"detectors disagree → review": "Detectors disagree: review",
           "detectors disagree -> review": "Detectors disagree: review",
           "detectors disagree": "Detectors disagree: review"}


def canonical(v):
    s = str(v or "").strip()
    low = s.lower()
    for k, c in ALIASES.items():
        if low.startswith(k):
            return c
    for c in VERDICTS:
        if low.startswith(c.lower()):
            return c
    return s or "unknown"


def verdict_counts(df, truth_col="attack_class"):
    """Incidents per verdict; and, when incidents carry ground truth, per verdict x true class."""
    v = df["verdict"].map(canonical)
    counts = v.value_counts().reindex(list(VERDICTS), fill_value=0)
    extra = v[~v.isin(VERDICTS)].value_counts()
    counts = pd.concat([counts, extra]).rename("incidents")
    cross = pd.crosstab(df[truth_col].fillna("benign"), v) if truth_col in df else None
    return counts, cross


def verify_hash(slice_path, expected, root=None):
    """(status, detail). Only reads files under the data root, never under a 'raw' directory."""
    if not slice_path or not expected:
        return "unverified", "no slice path or no recorded hash"
    root = (Path(root) if root is not None else config.data_root()).resolve()
    p = Path(slice_path)
    p = (p if p.is_absolute() else root / p).resolve()
    if root not in p.parents or "raw" in p.relative_to(root).parts:
        return "refused", "slice path is outside the data root (or under raw/)"
    if not p.is_file():
        return "unverified", f"slice file not found ({p.name}); showing the recorded hash only"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return ("verified", h.hexdigest()) if h.hexdigest() == str(expected).lower() else \
        ("MISMATCH", f"file hashes to {h.hexdigest()}")
