"""Schedule runner: executes attack classes in order, enforces a quiet gap between runs,
measures real start/end times, and appends one ground-truth line per run."""
import json
import time

from . import timing, util
from .attacks import REGISTRY


class Runner:
    def __init__(self, ctx, schedule):
        self.ctx = ctx
        self.runs = schedule.get("runs", [])
        # Seeded random gaps, drawn up front (first use of ctx.rng) so the timeline preview in
        # scripts/session_timeline.py matches exactly. Timing is non-periodic by design.
        self.gaps = timing.draw_gaps(ctx.rng, schedule, len(self.runs) - 1)
        self.gt_path = ctx.out_dir / "ground_truth.jsonl"

    def _record(self, line):
        with open(self.gt_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    def run_all(self):
        for i, entry in enumerate(self.runs):
            cls = entry["class"]
            if cls not in REGISTRY:
                raise SystemExit(f"unknown attack class: {cls}")
            handler = REGISTRY[cls]
            params = entry.get("params", {})
            run_id = entry.get("run_id") or f"{self.ctx.session_id}-{cls}-{i + 1:02d}"

            if i > 0:
                time.sleep(self.gaps[i - 1])  # seeded quiet normal traffic between runs

            print(f"[{run_id}] {cls} starting", flush=True)
            start = util.utc_iso()
            t0 = util.monotonic()
            try:
                result = handler(self.ctx, params) or {}
                outcome = result.get("outcome", "success")
            except Exception as exc:  # a failed attack is data, not a crash
                result, outcome = {"notes": f"exception: {exc}"}, "fail"
            end = util.utc_iso()

            self._record({
                "run_id": run_id,
                "session": self.ctx.session_id,
                "class": cls,
                "stage": handler.STAGE,
                "start_utc": start,
                "end_utc": end,
                "src_ip": self.ctx.src_ip,
                "src_mac": self.ctx.src_mac,
                "targets": result.get("targets", []),
                "params": {**params, **result.get("params_out", {})},
                "outcome": outcome,
                "notes": result.get("notes", ""),
                **({"creds_source": result["creds_source"]} if "creds_source" in result else {}),
            })
            print(f"[{run_id}] {cls} {outcome} in {util.monotonic() - t0:.1f}s", flush=True)
