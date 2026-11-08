"""Seeded, application-layer timing realism: per-device sampling-rate drift, per-publish
jitter, and occasional reboots. All off unless enabled by env. NET_ADMIN is deliberately
not used — this is application-layer only (see REPORT_NOTES [DATA] deviation)."""
import hashlib
import json
import os
import random
from datetime import datetime, timezone


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def device_seed_rng(seed, session_id, username):
    """Per-(session, device) RNG. Review B3: the old seed was `SEED ^ hash(username)` with no
    session id, so every session recorded at the same base seed replayed identical reboots, nurse
    commands and BIDMC offsets — train/val/test benign traffic were not independent. Folding the
    session id in (mirrors the attacker's timing.session_rng) makes benign behaviour differ per
    session while staying reproducible for a given (seed, session, device)."""
    h = hashlib.sha256(f"{seed}:{session_id}:{username}".encode()).hexdigest()[:16]
    return random.Random(int(h, 16))


def _seeded_rng(username):
    return device_seed_rng(os.environ.get("SEED", "0"),
                           os.environ.get("SESSION_ID", "").strip(), username)


class Noise:
    def __init__(self, username, host_ip):
        self.username = username
        self.host_ip = host_ip
        self.rng = _seeded_rng(username)
        self.jitter = os.environ.get("REALISM_JITTER", "0") == "1"
        self.reboot = os.environ.get("REALISM_REBOOT", "0") == "1"
        drift = float(os.environ.get("SAMPLING_DRIFT", "0") or 0)
        # one fixed drift factor per device for the whole run
        self.rate_factor = 1.0 + self.rng.uniform(-drift, drift) if drift else 1.0
        self.reboot_per_hour = float(os.environ.get("REBOOT_PER_HOUR", "0") or 0)
        sid = os.environ.get("SESSION_ID", "").strip()
        self.session_dir = f"/data/sessions/{sid}" if sid else ""

    def interval(self, base):
        """Scale a base interval by the device's fixed drift and optional per-tick jitter."""
        iv = base * self.rate_factor
        if self.jitter:
            iv += self.rng.uniform(-0.1, 0.1) * base
        return max(0.01, iv)

    def due_for_reboot(self, interval):
        if not (self.reboot and self.reboot_per_hour > 0):
            return False
        return self.rng.random() < self.reboot_per_hour * interval / 3600.0

    def offline_seconds(self):
        return self.rng.uniform(10, 60)

    def _emit(self, rec):
        line = json.dumps(rec)
        print(f"BENIGN_EVENT {line}", flush=True)
        if not self.session_dir:
            return
        # One file per device avoids concurrent-append races; the session runner merges them.
        d = os.path.join(self.session_dir, "benign_events")
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{self.username}.jsonl"), "a") as f:
                f.write(line + "\n")
        except OSError as exc:
            print(f"could not write benign event: {exc}", flush=True)

    def log_event(self, kind, ts=None, **fields):
        """Record a benign event (reboot, reconnect, nurse_command, ...) with a 'ts' field."""
        self._emit({"host": self.username, "ip": self.host_ip, "event": kind,
                    "ts": ts or utcnow(), **fields})

    def log_reboot(self, start_iso, end_iso, offline_s):
        self.log_event("reboot", ts=start_iso, start_utc=start_iso, end_utc=end_iso,
                       offline_s=round(offline_s, 2))
