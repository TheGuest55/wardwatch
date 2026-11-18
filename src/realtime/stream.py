"""Replay a recorded session through the incremental window builder at a chosen speed.

    packets -> WindowBuilder.add -> (timer) close_before(window end) -> on_tick(rows) callback

`speed` 1.0 replays in real time (packet timestamps are honoured against a wall clock), 2.0 twice as fast,
0 as fast as possible (no sleeping: the same sequence of ticks, used by tests and the equivalence check).

A window is closed by a TIMER, not by the next packet: at replay time (window end + grace) every window that
ended at or before (close time - grace) is finalized and handed to `on_tick`, even if no packet has arrived
since. A packet that arrives after its window was closed is counted by the builder as `late_packets`.

Measured here (all with perf_counter): per-packet ingest cost, and replay lag (how far the wall clock is
behind the packet clock when a packet is ingested; positive = behind). The per-tick scoring cost is measured
by the on_tick consumer (src/realtime/pipeline.py).
"""
import itertools
import time
from array import array
from pathlib import Path

from .. import common
from ..features import packets

GRACE_S = 1.0


def pcap_paths(session, sessions_dir=None):
    d = Path(sessions_dir or common.SESSIONS_DIR) / session / "pcap"
    paths = sorted(d.glob("*.pcap"))
    if not paths:
        raise SystemExit(f"no pcaps under {d}")
    return paths


def session_packets(session, sessions_dir=None, max_seconds=None, start_offset=0.0):
    """Packet records of a session in file order. start_offset / max_seconds (relative to the first packet)
    cut an excerpt; both are packet-time seconds."""
    it = itertools.chain.from_iterable(packets.iter_packets(p) for p in pcap_paths(session, sessions_dir))
    if not max_seconds and not start_offset:
        return it
    return _excerpt(it, start_offset, max_seconds)


def _excerpt(it, start_offset, max_seconds):
    t0 = None
    for rec in it:
        if t0 is None:
            t0 = rec["ts"]
        rel = rec["ts"] - t0
        if rel < start_offset:
            continue
        if max_seconds and rel >= start_offset + max_seconds:
            return
        yield rec


class ReplayStats:
    def __init__(self):
        self.ingest_s = array("d")       # seconds per builder.add call
        self.lag_s = array("d")          # wall clock behind packet clock (s), sampled at each packet
        self.n_ticks = 0
        self.wall_s = 0.0
        self.packet_span_s = 0.0
        self.n_packets = 0

    def summary(self):
        import numpy as np
        ing = np.frombuffer(self.ingest_s, dtype=float) if len(self.ingest_s) else np.zeros(1)
        lag = np.frombuffer(self.lag_s, dtype=float) if len(self.lag_s) else np.zeros(1)
        return {"n_packets": self.n_packets, "n_ticks": self.n_ticks, "wall_s": self.wall_s,
                "packet_span_s": self.packet_span_s,
                "ingest_us_mean": float(ing.mean() * 1e6), "ingest_us_p95": float(np.percentile(ing, 95) * 1e6),
                "ingest_us_max": float(ing.max() * 1e6),
                "lag_s_max": float(lag.max()), "lag_s_p95": float(np.percentile(lag, 95))}


def replay(packet_iter, builder, on_tick, speed=1.0, grace=GRACE_S, clock=time.perf_counter, sleep=time.sleep,
           stats=None):
    """Drive `builder` from `packet_iter`. on_tick(rows, ctx) is called once per closed window time slot that
    has rows, in window order; ctx = {"window_end": epoch, "replay_now": packet-clock epoch at call time}.
    Returns ReplayStats. The builder is flushed at the end (windows still open when the capture ends)."""
    stats = stats or ReplayStats()
    W = common.WINDOW_SECONDS
    state = {"t0": None, "wall0": None}

    def replay_now():
        return state["t0"] + (clock() - state["wall0"]) * speed if speed else None

    def sleep_until(t):                      # t: packet-clock epoch
        if not speed:
            return
        target = state["wall0"] + (t - state["t0"]) / speed
        d = target - clock()
        if d > 0:
            sleep(d)

    def tick(close_time):
        rows = builder.close_before(close_time - grace)
        if rows:
            stats.n_ticks += 1
            on_tick(rows, {"window_end": close_time - grace,
                           "replay_now": replay_now() if speed else close_time})

    next_close = None
    last_ts = None
    for rec in packet_iter:
        ts = rec["ts"]
        if state["t0"] is None:
            state["t0"], state["wall0"] = ts, clock()
            next_close = common.window_start(ts) + W + grace
        while next_close <= ts:               # timer-driven close of every window that has run out
            sleep_until(next_close)
            tick(next_close)
            next_close += W
        sleep_until(ts)
        t_in = clock()
        builder.add(rec)
        stats.ingest_s.append(clock() - t_in)
        if speed:
            stats.lag_s.append((clock() - state["wall0"]) * speed - (ts - state["t0"]))
        stats.n_packets += 1
        last_ts = ts
    if state["t0"] is None:
        return stats
    stats.packet_span_s = last_ts - state["t0"]
    final_end = common.window_start(last_ts) + W
    while next_close - grace <= final_end:    # capture ended: let the timers run the last windows out
        sleep_until(next_close)
        tick(next_close)
        next_close += W
    rest = builder.flush()
    if rest:
        stats.n_ticks += 1
        on_tick(rest, {"window_end": final_end, "replay_now": replay_now() if speed else final_end})
    stats.wall_s = clock() - state["wall0"]
    return stats
