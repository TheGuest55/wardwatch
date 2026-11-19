#!/usr/bin/env python3
"""Build the dashboard results bundle (see src/dashboard/bundle.py). Run from the repo root, e.g. in the analysis image:

    python scripts/make_results_bundle.py --out bundle_out
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dashboard import bundle, config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(config.data_root()), help="processed corpus root (default: %(default)s)")
    ap.add_argument("--out", default="bundle_out")
    ap.add_argument("--creds", default=str(Path("data/raw/creds/accounts.json")),
                    help="credentials file, read in memory for the scan only (default: %(default)s)")
    a = ap.parse_args(argv)
    r = bundle.build(a.data_root, a.out, a.creds)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
