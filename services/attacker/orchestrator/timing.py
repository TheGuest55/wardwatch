"""Session timing: per-session seeding and attack-gap drawing, shared by the orchestrator
(which executes it) and scripts/session_timeline.py (which previews it on the host). Standard
library only so the host can import it without the attacker image's dependencies."""
import hashlib
import math
import random
import time

# Nominal durations (seconds) for classes whose runtime isn't a fixed param, used only for the
# timeline estimate. Actual runtime varies (e.g. bruteforce depends on guesses).
NOMINAL = {"recon_scan": 90, "syn_flood": 30, "mqtt_flood": 30, "bruteforce": 15,
           "arp_mitm": 90, "eavesdrop": 180, "spoof_vitals": 60, "pump_cmd": 60}


def session_rng(base_seed, session_id):
    """Deterministic per-session RNG: reproducible, but different per session id."""
    h = hashlib.sha256(f"{base_seed}:{session_id}".encode()).hexdigest()[:16]
    return random.Random(int(h, 16))


def _bounds(schedule):
    if "gap_min" in schedule and "gap_max" in schedule:
        return float(schedule["gap_min"]), float(schedule["gap_max"])
    g = float(schedule.get("gap_seconds", 180))
    return g, g


def draw_gaps(rng, schedule, n):
    """n seeded inter-run gaps in [gap_min, gap_max] (random, so timing isn't periodic)."""
    lo, hi = _bounds(schedule)
    return [round(rng.uniform(lo, hi), 1) for _ in range(max(0, n))]


def choose_spoof_mode(rng):
    """Seeded choice of the spoof_vitals sub-mode (review M6: don't always run reuse). True = reuse
    the victim's client id (evicts the real device); False = connect with a second client id (the
    harder-to-detect variant). Per-session rng, so the mode varies across sessions and is logged."""
    return rng.random() < 0.5


def align_to_grid(window=5.0, lead=0.05, clock=time.time, sleep=time.sleep):
    """Block until just past the next window boundary, and return that boundary (epoch seconds) for
    the caller to record as the run's start_utc.

    Review M7: a run whose start_utc falls *inside* a window makes that window straddle the interval
    boundary, so the labeller drops it (is_boundary) — losing the eavesdrop wildcard SUBSCRIBE and
    every short-abuse onset. Recording start_utc AT the boundary (and letting the first packets land
    a few ms into the new window) makes the onset window fully-inside instead. Does not change the
    labelling rule."""
    now = clock()
    boundary = (math.floor(now / window) + 1) * window
    sleep(max(0.0, boundary + lead - now))
    return boundary


def run_duration(cls, params):
    return params.get("duration") or params.get("timeout") or NOMINAL.get(cls, 30)


def compute_timeline(schedule, base_seed, session_id, pre_roll, post_roll):
    """Return (rows, total_seconds, normal_seconds) where rows = [(class, start, end), ...].
    Uses the same seeded gaps the orchestrator will draw, so the preview matches the run."""
    runs = schedule.get("runs", [])
    gaps = draw_gaps(session_rng(base_seed, session_id), schedule, len(runs) - 1)
    t, rows = float(pre_roll), []
    for i, run in enumerate(runs):
        if i > 0:
            t += gaps[i - 1]
        dur = run_duration(run["class"], run.get("params", {}))
        rows.append((run["class"], t, t + dur))
        t += dur
    total = t + post_roll
    normal = pre_roll + sum(gaps) + post_roll
    return rows, total, normal
