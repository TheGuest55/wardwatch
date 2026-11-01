import csv
import itertools
import json
import math
import os
import random
import time
from datetime import datetime, timezone

from .client import build_client, connect

DEFAULT_CSV = "/data/bidmc/bidmc_01_Numerics.csv"


def bidmc_rows(path):
    """Loads HR and SpO2 from a BIDMC numerics file, skipping incomplete rows."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            try:
                hr, spo2 = float(row["HR"]), float(row["SpO2"])
            except (TypeError, ValueError):
                continue
            if not (math.isnan(hr) or math.isnan(spo2)):
                rows.append((hr, spo2))
    return rows


def synthetic_rows():
    for t in itertools.count():
        hr = 75 + 5 * math.sin(t / 30) + random.gauss(0, 1)
        spo2 = min(100.0, 97 + random.gauss(0, 0.5))
        yield round(hr, 1), round(spo2, 1)


def vitals_source():
    """Returns (iterator of (hr, spo2), source label). Falls back to synthetic data."""
    path = os.environ.get("BIDMC_CSV", DEFAULT_CSV)
    try:
        rows = bidmc_rows(path)
    except OSError as exc:
        print(f"BIDMC unavailable ({exc}); using synthetic vitals", flush=True)
        return synthetic_rows(), "synthetic"
    if not rows:
        print(f"BIDMC file {path} has no usable rows; using synthetic vitals", flush=True)
        return synthetic_rows(), "synthetic"
    print(f"replaying {len(rows)} rows from {path}", flush=True)
    return itertools.cycle(rows), "bidmc"


def synthetic_bp():
    # BIDMC numerics have no blood pressure, so it is always synthetic.
    return round(random.gauss(120, 4)), round(random.gauss(78, 3))


def jitter_delay():
    """Realism hook, off by default and not implemented in this phase."""
    return 0.0


def maybe_reboot():
    """Realism hook, off by default and not implemented in this phase."""


def run():
    bed = os.environ.get("BED", "1")
    jitter = os.environ.get("REALISM_JITTER") == "1"
    reboot = os.environ.get("REALISM_REBOOT") == "1"
    topic = f"ward1/bed{bed}/monitor/vitals"
    source_iter, source = vitals_source()

    client = build_client("patient-monitor")
    connect(client)
    client.loop_start()

    next_send = time.monotonic()
    for hr, spo2 in source_iter:
        sys_bp, dia_bp = synthetic_bp()
        payload = {
            "device": "patient-monitor",
            "bed": int(bed),
            "ts": datetime.now(timezone.utc).isoformat(),
            "hr": hr,
            "spo2": spo2,
            "bp_sys": sys_bp,
            "bp_dia": dia_bp,
            "vitals_source": source,
            "bp_source": "synthetic",
        }
        client.publish(topic, json.dumps(payload), qos=0)
        if reboot:
            maybe_reboot()
        next_send += 1.0 + (jitter_delay() if jitter else 0.0)
        time.sleep(max(0.0, next_send - time.monotonic()))
