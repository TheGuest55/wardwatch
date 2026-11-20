"""Skip a test ONLY when the data it needs is absent, with a reason that says what is missing and where the manual explains it.

A test that needs data is never weakened: with the data present it runs exactly as before. The data sets, by what a reader has:

  DATASET   Level 2: the v2 dataset release (data/processed/v2/testbed_windows.parquet, feature_list.json, sessions/*.parquet,
            data/raw/sessions/split_v2.json)
  RESULTS   Level 1 / 2: the results zip's runs/ folder (recorded run records, predictions, incidents)
  GROUND_TRUTH  Level 3: data/raw/sessions/<S>/ground_truth.jsonl (not in the dataset release)
  RAW       Level 3: the raw session PCAPs (data/raw/sessions/<S>/pcap), never published
  MODELS    Level 3: the trained model files (data/processed/v2/models/...), never published
  CREDS     Level 3: data/raw/creds/accounts.json (lab credentials), never published
  CIC_CSV   section 5: CICIoMT2024's feature CSVs (not redistributed)

`needs("RAW", "DATASET")` is a unittest decorator (class or function); `missing(...)` returns the list of absent data sets.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = "data/processed/v2"
SETS = {
    "DATASET": ([f"{V2}/testbed_windows.parquet", f"{V2}/feature_list.json", f"{V2}/sessions/r05.parquet",
                 "data/raw/sessions/split_v2.json"], "the v2 dataset release", "USER_MANUAL Level 2"),
    "RESULTS": ([f"{V2}/runs/expA/xgboost_s0.json", f"{V2}/runs/lstm/selected_config.json"],
                "the recorded runs/ from the results zip", "USER_MANUAL Level 2"),
    "GROUND_TRUTH": (["data/raw/sessions/r01/ground_truth.jsonl", "data/raw/sessions/r05/ground_truth.jsonl"],
                     "the raw session ground truth files (not published)", "USER_MANUAL Level 3"),
    "RAW": (["data/raw/sessions/r05/pcap"], "the raw session PCAPs (not published)", "USER_MANUAL Level 3"),
    "MODELS": ([f"{V2}/models/expA/xgboost_binary_seed0.json"], "the trained model files (not published)", "USER_MANUAL Level 3"),
    "CREDS": (["data/raw/creds/accounts.json"], "the lab credentials file (not published)", "USER_MANUAL Level 3"),
    "CIC_CSV": (["data/raw/ciciomt2024/WiFi_and_MQTT/attacks/csv/train"], "CICIoMT2024's feature CSVs (not redistributed)",
                "USER_MANUAL section 5"),
}


def _present(name):
    return all((ROOT / p).exists() for p in SETS[name][0])


def missing(*names):
    return [n for n in names if not _present(n)]


def reason(*names):
    miss = missing(*names)
    if not miss:
        return ""
    return "needs " + " and ".join(SETS[n][1] for n in miss) + ": see " + "; ".join(sorted({SETS[n][2] for n in miss}))


def needs(*names):
    """unittest.skipUnless over the named data sets; the skip message names what is missing and the manual section."""
    return unittest.skipUnless(not missing(*names), reason(*names) or "data present")
