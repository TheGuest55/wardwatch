#!/usr/bin/env python3
"""Preview a session's attack timeline without running anything.

  session_timeline.py <schedule.json> <session-id> <pre_roll_s> <post_roll_s>

Uses the same seeded gaps the orchestrator will draw (orchestrator/timing.py), so the printed
start/end of each run and the total normal minutes match the actual session. Host-side, stdlib
only — it does not touch the lab."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "attacker"))
from orchestrator import timing  # noqa: E402  (stdlib-only module)


def mmss(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def main():
    sched_path, sid, pre, post = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    cfg = json.loads((ROOT / "config" / "ward.json").read_text())
    sched = json.loads(Path(sched_path).read_text())
    rows, total, normal = timing.compute_timeline(sched, cfg["seed"], sid, pre, post)

    print(f"session {sid}  schedule={sched_path}  seed={cfg['seed']}")
    print(f"  {mmss(0)}  pre-roll start ({pre/60:.0f} min normal)")
    for cls, s, e in rows:
        print(f"  {mmss(s)}-{mmss(e)}  {cls}")
    print(f"  {mmss(total - post)}  post-roll ({post/60:.0f} min normal) -> {mmss(total)} end")
    print(f"TOTAL {total/60:.1f} min | normal {normal/60:.1f} min | "
          f"attack runs {len(rows)}")


if __name__ == "__main__":
    main()
