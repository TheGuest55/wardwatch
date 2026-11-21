"""Playback of a recorded alert stream, for readers who have no PCAPs or models.

    python -m src.realtime.playback --data-root DIR --session r05 [--speed 10] [--source FILE]

Reads <DIR>/live/recorded/alerts_stream_<session>.jsonl (a recording of one real replay, see run_e --record) and writes it to
<DIR>/live/alerts_stream.jsonl progressively, each event delayed by (its window ts - the previous ts) / SPEED seconds, so the dashboard's
"Live alerts" page shows the alerts arriving. The header keeps the recorded provenance and gets mode = "playback".
Standard library only: it runs in the dashboard image.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _epoch(ts):
    """Window ts as the stream writes it (ISO, UTC: '...+00:00', also accepted '...Z')."""
    d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


def load(source):
    lines = [json.loads(l) for l in Path(source).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines or "header" not in lines[0]:
        raise SystemExit(f"[playback] {source}: first line is not a stream header")
    return lines[0]["header"], lines[1:]


def delays(events, speed):
    """Seconds to wait before each event: the gap between consecutive window ts, divided by speed (never negative)."""
    out, prev = [], None
    for e in events:
        t = _epoch(e["ts"]) if e.get("ts") else prev
        out.append(0.0 if prev is None or t is None else max(0.0, (t - prev) / speed))
        if t is not None:
            prev = t
    return out


def play(source, dest, speed=10.0, sleep=time.sleep):
    head, events = load(source)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    new = dict(head)
    new.update({"mode": "playback", "playback_speed": speed, "playback_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    with open(dest, "w", encoding="utf-8") as f:
        f.write(json.dumps({"header": new}, sort_keys=True) + "\n")
        f.flush()
        for d, e in zip(delays(events, speed), events):
            if d:
                sleep(d)
            f.write(json.dumps(e, sort_keys=True) + "\n")
            f.flush()
    return len(events)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--session", default="r05")
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--source", default=None, help="recording (default: <data-root>/live/recorded/alerts_stream_<session>.jsonl)")
    a = ap.parse_args(argv)
    if a.speed <= 0:
        raise SystemExit("[playback] --speed must be positive")
    root = Path(a.data_root)
    src = Path(a.source) if a.source else root / "live" / "recorded" / f"alerts_stream_{a.session}.jsonl"
    if not src.exists():
        raise SystemExit(f"[playback] no recording at {src} (the results zip carries live/recorded/)")
    n = play(src, root / "live" / "alerts_stream.jsonl", a.speed)
    print(f"[playback] {n} events from {src.name} written to {root / 'live' / 'alerts_stream.jsonl'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
