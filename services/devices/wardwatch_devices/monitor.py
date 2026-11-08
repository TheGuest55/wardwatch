import csv
import itertools
import math
import os
import random

from payloads import monitor_vitals

from .base import Device

DEFAULT_CSV = "/data/physionet/bidmc_01_Numerics.csv"


def bidmc_rows(path):
    """Loads HR and SpO2 from a BIDMC numerics file, skipping incomplete rows."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            try:
                hr, spo2 = float(row["HR"]), float(row["SpO2"])
            except (TypeError, ValueError, KeyError):
                continue
            if not (math.isnan(hr) or math.isnan(spo2)):
                rows.append((hr, spo2))
    return rows


def synthetic_rows(rng):
    for t in itertools.count():
        hr = 75 + 5 * math.sin(t / 30) + rng.gauss(0, 1)
        spo2 = min(100.0, 97 + rng.gauss(0, 0.5))
        yield round(hr, 1), round(spo2, 1)


def vitals_source(rng):
    """Returns (iterator of (hr, spo2), source label). Falls back to synthetic data."""
    path = os.environ.get("BIDMC_CSV", DEFAULT_CSV)
    try:
        rows = bidmc_rows(path)
    except OSError as exc:
        print(f"BIDMC unavailable ({exc}); using synthetic vitals", flush=True)
        return synthetic_rows(rng), "synthetic"
    if not rows:
        print(f"BIDMC file {path} has no usable rows; using synthetic vitals", flush=True)
        return synthetic_rows(rng), "synthetic"
    # Seeded random start offset so beds sharing a record aren't phase-locked.
    start = rng.randrange(len(rows))
    print(f"replaying {len(rows)} rows from {path} (start offset {start})", flush=True)
    return itertools.cycle(rows[start:] + rows[:start]), "bidmc"


class Monitor(Device):
    def __init__(self):
        super().__init__("patient-monitor")
        self.bp_rng = random.Random(self.noise.rng.random())
        self.source_iter, self.source = vitals_source(self.noise.rng)

    def payload(self):
        hr, spo2 = next(self.source_iter)
        # Shared builder (services/shared/payloads.py) fixes field order + representation so the
        # spoof_vitals attack, which uses the same builder, is not separable by a payload artefact.
        return monitor_vitals(self.bed, hr, spo2,
                              self.bp_rng.gauss(120, 4), self.bp_rng.gauss(78, 3),
                              vitals_source=self.source)


def run():
    m = Monitor()
    m.run_periodic(1.0, m.payload, f"ward1/bed{m.bed}/monitor/vitals")
