#!/usr/bin/env bash
# Run the regression tests. Two tiers:
#   generation/  - stdlib-only (payload builders, seeding, timing, grid-align, arp_mitm detection).
#                  Run on the host python; imports only paho-free modules.
#   pipeline/    - need pandas/scapy/sklearn; run inside the pinned analysis image via analysis.sh.
# Usage: scripts/run_tests.sh [generation|pipeline|all]   (default all)
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

run_pipeline() {
  if [ -d tests/pipeline ] && ls tests/pipeline/test_*.py >/dev/null 2>&1; then
    echo "== pipeline tests (analysis image) =="
    PYTHONPATH="" scripts/analysis.sh test
  else
    echo "== pipeline tests: none yet (skipped) =="
  fi
}

case "$TIER" in
  generation) run_generation ;;
  pipeline)   run_pipeline ;;
  all)        run_generation; run_pipeline ;;
  *) echo "usage: run_tests.sh [generation|pipeline|all]" >&2; exit 2 ;;
esac
