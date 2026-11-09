"""Session timing: per-session seeding and attack-gap drawing, shared by the orchestrator
(which executes it) and scripts/session_timeline.py (which previews it on the host). Standard
library only so the host can import it without the attacker image's dependencies."""
import hashlib
import math
import random
import re
import time

# Nominal durations (seconds) for classes whose runtime isn't a fixed param, used only for the
# timeline estimate. Actual runtime varies (e.g. bruteforce depends on guesses).
NOMINAL = {"recon_scan": 90, "syn_flood": 30, "mqtt_flood": 30, "bruteforce": 15,
           "arp_mitm": 150, "eavesdrop": 180, "spoof_vitals": 60, "pump_cmd": 60}


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


def session_index(session_id):
    """Integer index of a session from its id: the trailing number (``r06`` -> 6), else a stable
    hash of the id so alternation is always deterministic from the id alone (no external counter)."""
    m = re.search(r"(\d+)$", session_id or "")
    if m:
        return int(m.group(1))
    return int(hashlib.sha256((session_id or "").encode()).hexdigest(), 16)


def choose_spoof_mode(session_id):
    """Deterministic spoof_vitals sub-mode, alternating by session index (review M6 + N1: the mode
    must not be a random coin flip that can leave a whole batch on one mode). True = reuse the
    victim's client id (evicts the real device); False = second client id (harder to detect). Even
    index -> reuse, odd -> second_client_id, so a batch of consecutive sessions alternates both. A
    schedule can still pin the mode via params.client_id_reuse (used by the override + smoke/tests).
    Deterministic from the session id alone."""
    return session_index(session_id) % 2 == 0


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


def resolve_duration(rng, params):
    """If params carry a `duration_range` [lo, hi], draw a concrete seeded integer `duration` from it
    and drop the range; otherwise return params unchanged. Randomised durations within the spec
    ranges keep runs non-periodic while staying >= one clean 5 s window (review: randomise durations)."""
    p = dict(params)
    rng_bounds = p.pop("duration_range", None)
    if rng_bounds:
        lo, hi = int(rng_bounds[0]), int(rng_bounds[1])
        p["duration"] = rng.randint(lo, hi)
    return p


def run_duration(cls, params):
    if "duration_range" in params:          # timeline preview: use the midpoint estimate
        lo, hi = params["duration_range"]
        return (lo + hi) / 2
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
