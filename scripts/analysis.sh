#!/usr/bin/env bash
# Run a Task 4-5 analysis step inside the pinned wardwatch-analysis image (Python 3.11).
#
# This is NOT a lab command: it uses `docker run` (never compose), touches no lab container, and
# mounts data/raw READ-ONLY so captured evidence can never be modified. Writes land in
# data/processed/ and docs/. Keep --cpus modest while a capture session is recording.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${ANALYSIS_IMAGE:-wardwatch-analysis:0.2.0}"
CMD="${1:-}"
shift || true

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[analysis] building $IMAGE (first run only)…" >&2
  docker build -t "$IMAGE" services/analysis >&2
fi

run() {
  docker run --rm \
    --network none \
    --cpus "${ANALYSIS_CPUS:-1}" \
    -v "$PWD":/work \
    -v "$PWD/data/raw":/work/data/raw:ro \
    -w /work "$IMAGE" "$@"
}

case "$CMD" in
  features)
    SESSION="${1:-}"
    [ -n "$SESSION" ] || { echo "usage: analysis.sh features <session>" >&2; exit 2; }
    run python -m src.features.extract --session "$SESSION"
    ;;
  dataset)   run python -m src.dataset "$@" ;;
  datacheck) run python -m src.datacheck.check "$@" ;;
  test)      run sh -c "python -m unittest discover -s tests/pipeline -p 'test_*.py' -v \
                && python -m unittest discover -s tests/realtime -t . -p 'test_*.py' -v \
                && python -m unittest discover -s tests/triage -t . -p 'test_*.py' -v" ;;
  # Triage / real time / Experiment E (docs in the README "Triage and real time" section)
  check-session) run python -m src.realtime.check_session --session "$@" ;;
  triage)    run python -m src.triage.offline "$@" ;;
  merge-live) run python -m src.realtime.merge_live "$@" ;;
  replay)
    SESSION="${1:-}"
    [ -n "$SESSION" ] || { echo "usage: analysis.sh replay <session> [--speed N] [--for-report]" >&2; exit 2; }
    shift
    run python -m src.experiments.run_e --session "$SESSION" "$@"
    ;;
  shell)     run bash ;;
  *)
    echo "usage: analysis.sh {features <session>|dataset|datacheck|test|check-session <s>..|triage|merge-live|replay <session>|shell}" >&2
    exit 2
    ;;
esac
