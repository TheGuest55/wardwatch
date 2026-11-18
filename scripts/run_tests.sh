#!/usr/bin/env bash
# Run the regression tests.
#   all (default) - the full suite: scripts/analysis.sh test (see there). `make test` is the alias.
#   generation    - only tests/generation, on the host python (stdlib-only; paho-free modules).
# Usage: scripts/run_tests.sh [generation|all]
set -euo pipefail
cd "$(dirname "$0")/.."

TIER="${1:-all}"
# Paho-free source roots so `import payloads`, `wardwatch_devices.noise`, `orchestrator.timing`,
# `record_provenance` work (plus the repo root, on sys.path via cwd, for `src.*`).
export PYTHONPATH="services/shared:services/devices:services/attacker:scripts${PYTHONPATH:+:$PYTHONPATH}"

run_generation() {
  echo "== generation tests (host python) =="
  python3 -m unittest discover -s tests/generation -p 'test_*.py' -v
}

# The one-command full suite lives in scripts/analysis.sh test: pipeline, realtime, triage and generation in the analysis
# image, dashboard in the dashboard image, a per-folder and TOTAL summary, non-zero on any failure, skip or empty folder.
run_all() {
  PYTHONPATH="" scripts/analysis.sh test
}

case "$TIER" in
  generation) run_generation ;;                 # host python only, fast (the same 50 tests also run inside `all`)
  pipeline|all) run_all ;;
  *) echo "usage: run_tests.sh [generation|all]" >&2; exit 2 ;;
esac
