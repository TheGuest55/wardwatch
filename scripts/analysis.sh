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

# ---- one-command full suite ------------------------------------------------------------------------------------
# Every tests/<dir> that holds test_*.py must be listed here (the guard below refuses to run otherwise), so a new
# test folder can never be silently left out. pipeline, realtime, triage and generation run in the analysis image;
# dashboard (pytest) runs in the dashboard image. Every folder runs even after one fails; a per-folder line and a
# TOTAL line are printed; the exit is non-zero if any folder failed, skipped a test, or ran 0 tests.
SUITE_DIRS="pipeline realtime triage generation dashboard"
DASH_IMAGE="${DASHBOARD_IMAGE:-wardwatch-dashboard:0.1.0}"

suite_guard() {
  local d name missing=""
  for d in tests/*/; do
    name="$(basename "$d")"
    ls "$d"test_*.py >/dev/null 2>&1 || continue
    case " $SUITE_DIRS " in *" $name "*) ;; *) missing="$missing $name" ;; esac
  done
  if [ -n "$missing" ]; then
    echo "[test] tests/<dir> with test_*.py is not in SUITE_DIRS of scripts/analysis.sh:$missing" >&2
    return 1
  fi
}

# stdin: the folder's output. Prints "<ran> <skipped>" from a unittest ("Ran N tests", "skipped=K") or a pytest
# ("N passed, K skipped") summary.
suite_counts() {
  local out ran skipped
  out="$(cat)"
  ran="$(printf '%s\n' "$out" | sed -n 's/^Ran \([0-9][0-9]*\) test.*/\1/p' | tail -1 || true)"
  skipped="$(printf '%s\n' "$out" | grep -E '^(OK|FAILED) \(' | grep -Eo 'skipped=[0-9]+' | tail -1 | cut -d= -f2 || true)"
  if [ -z "$ran" ]; then      # pytest: the last summary line, e.g. "51 passed, 2 skipped in 4.59s"
    local line
    line="$(printf '%s\n' "$out" | grep -E '[0-9]+ (passed|failed|error|errors|skipped)' | tail -1 || true)"
    ran="$(printf '%s\n' "$line" | grep -Eo '[0-9]+ (passed|failed|errors?|skipped)' | awk '{s+=$1} END {print s+0}' || true)"
    skipped="$(printf '%s\n' "$line" | grep -Eo '[0-9]+ skipped' | awk '{print $1}' || true)"
  fi
  echo "${ran:-0} ${skipped:-0}"
}

run_suite_dir() {
  case "$1" in
    pipeline)   run python -m unittest discover -s tests/pipeline -p 'test_*.py' -v ;;
    realtime)   run python -m unittest discover -s tests/realtime -t . -p 'test_*.py' -v ;;
    triage)     run python -m unittest discover -s tests/triage -t . -p 'test_*.py' -v ;;
    generation) run env PYTHONPATH=services/shared:services/devices:services/attacker:scripts \
                  python -m unittest discover -s tests/generation -p 'test_*.py' -v ;;
    dashboard)
      if ! docker image inspect "$DASH_IMAGE" >/dev/null 2>&1; then
        echo "[test] image $DASH_IMAGE is missing: build it with make dashboard-image" >&2
        return 127
      fi
      docker run --rm --network none -v "$PWD":/work:ro -w /work -e PYTHONDONTWRITEBYTECODE=1 "$DASH_IMAGE" \
        python -m pytest tests/dashboard -q -p no:cacheprovider ;;
  esac
}

full_suite() {
  suite_guard || exit 2
  local d tmp rc ran skipped status failed=0 t_ran=0 t_skip=0 lines=""
  tmp="$(mktemp)"
  for d in $SUITE_DIRS; do
    echo "== tests/$d =="
    set +e
    run_suite_dir "$d" 2>&1 | tee "$tmp"
    rc=${PIPESTATUS[0]}
    set -e
    read -r ran skipped < <(suite_counts < "$tmp") || true
    ran="${ran:-0}"; skipped="${skipped:-0}"
    status=OK
    if [ "$rc" -ne 0 ]; then status="FAILED (exit $rc)"
    elif [ "$ran" -eq 0 ]; then status="FAILED (ran 0 tests)"
    elif [ "$skipped" -gt 0 ]; then status="FAILED ($skipped skipped)"
    fi
    case "$status" in OK) ;; *) failed=1 ;; esac
    t_ran=$((t_ran + ran)); t_skip=$((t_skip + skipped))
    lines="$lines$(printf '%-12s ran %-5s skipped %-3s %s' "$d" "$ran" "$skipped" "$status")"$'\n'
  done
  rm -f "$tmp"
  echo
  echo "== test summary =="
  printf '%s' "$lines"
  printf '%-12s ran %-5s skipped %-3s %s\n' TOTAL "$t_ran" "$t_skip" "$([ "$failed" -eq 0 ] && echo OK || echo FAILED)"
  return "$failed"
}

case "$CMD" in
  features)
    SESSION="${1:-}"
    [ -n "$SESSION" ] || { echo "usage: analysis.sh features <session>" >&2; exit 2; }
    run python -m src.features.extract --session "$SESSION"
    ;;
  dataset)   run python -m src.dataset "$@" ;;
  datacheck) run python -m src.datacheck.check "$@" ;;
  test)      full_suite ;;
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
    echo "usage: analysis.sh {features <session>|dataset|datacheck|test (the full suite)|check-session <s>..|triage|merge-live|replay <session>|shell}" >&2
    exit 2
    ;;
esac
