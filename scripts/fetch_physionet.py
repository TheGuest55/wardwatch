#!/usr/bin/env python3
"""Download the BIDMC numerics CSVs named in config/ward.json into data/raw/physionet/.

Standard-library only; uses the allowed host physionet.org. These numerics files are plain
CSV (Time, HR, PULSE, RESP, SpO2), so no wfdb dependency is needed. If a download fails the
monitor/wearable fall back to a clearly-labelled synthetic generator at run time; this script
only warns and continues so `make up` is never blocked by the network.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "ward.json"
DEST = ROOT / "data" / "raw" / "physionet"
BASE = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv"


def fetch(record):
    name = f"bidmc_{record}_Numerics.csv"
    out = DEST / name
    if out.exists() and out.stat().st_size > 0:
        return True
    url = f"{BASE}/{name}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  WARN {name}: download failed ({exc}); monitor will use synthetic fallback")
        return False
    tmp = out.with_suffix(".csv.tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    print(f"  ok   {name} ({len(data)} bytes)")
    return True


def main():
    cfg = json.loads(CONFIG.read_text())
    DEST.mkdir(parents=True, exist_ok=True)
    records = sorted(set(cfg["bidmc_records"].values()))
    print(f"fetching {len(records)} BIDMC records to {DEST}")
    ok = sum(fetch(r) for r in records)
    print(f"{ok}/{len(records)} records available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
