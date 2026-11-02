"""Small helpers for the attack orchestrator: UTC timestamps and bounded subprocess runs."""
import subprocess
import time
from datetime import datetime, timezone


def utc_iso():
    """UTC with millisecond precision, e.g. 2026-11-02T07:00:00.123Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


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
