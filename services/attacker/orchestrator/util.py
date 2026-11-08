"""Small helpers for the attack orchestrator: UTC timestamps and bounded subprocess runs."""
import subprocess
import time
from datetime import datetime, timezone


def utc_iso():
    """UTC with millisecond precision, e.g. 2026-11-02T07:00:00.123Z. Used for ground-truth run
    start/end timestamps (kept in `...Z` form; iso_to_epoch parses both Z and +00:00). A SINGLE
    clock read — the old version read the clock twice and could log ~1 s early at a second rollover
    (review m1). Attack *payload* timestamps come from the shared builder (payloads.utcnow)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def utc_iso_at(epoch):
    """Format a given epoch (seconds) in the same `...mmmZ` form as utc_iso(). Used to record a
    run's start_utc at the aligned 5 s grid boundary (review M7)."""
    dt = datetime.fromtimestamp(epoch, timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def monotonic():
    return time.monotonic()


def run(cmd, timeout, check=False):
    """Run a command with a hard timeout. Returns (rc, stdout, stderr). A timeout is not an
    error here — attack tools are expected to be stopped by the timeout."""
    try:
        p = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), (e.stderr or "")


def popen(cmd):
    """Start a background process (for tools that must run concurrently, e.g. a sniffer)."""
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
