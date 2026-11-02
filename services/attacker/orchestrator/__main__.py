"""Entry point: python -m orchestrator --schedule <yaml> --session <id> --out <dir> [--seed N]

Runs the scheduled attack classes against the lab and writes ground truth. Invoked inside the
attacker container by the session runner."""
import argparse
import os

import yaml

from .context import Context
from .runner import Runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    args = ap.parse_args()

    with open(args.schedule) as f:
        schedule = yaml.safe_load(f)

    ctx = Context(args.session, args.out, args.seed)
    Runner(ctx, schedule).run_all()


if __name__ == "__main__":
    main()
