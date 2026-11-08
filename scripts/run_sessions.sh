#!/usr/bin/env bash
# Run several sessions back to back for unattended / overnight recording:
#   make sessions IDS="s01 s02 s03"
# Each token may carry a distinct base seed as "id:seed" (e.g. "r01:10007"); a bare id uses the
# config/env seed. Distinct per-session seeds keep benign traffic independent across train/val/test.
# Stops at the first failure (set -e) and never overwrites an existing session, so it is safe
# to resume by re-invoking with the ids that have not run yet.
set -euo pipefail
cd "$(dirname "$0")/.."

[ "$#" -ge 1 ] || { echo "usage: run_sessions.sh <id[:seed]> [<id[:seed]> ...]" >&2; exit 2; }

for tok in "$@"; do
  id="${tok%%:*}"
  seed=""
  [ "$tok" = "$id" ] || seed="${tok#*:}"
  if [ -e "data/raw/sessions/$id" ]; then
    echo "== skip $id (already exists)"
    continue
  fi
  echo "== session $id${seed:+ (seed $seed)}"
  if [ -n "$seed" ]; then
    SEED="$seed" bash scripts/run_session.sh "$id"
  else
    bash scripts/run_session.sh "$id"
  fi
done
echo "== all requested sessions done"
