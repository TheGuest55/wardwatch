#!/usr/bin/env bash
# Run several sessions back to back for unattended / overnight recording:
#   make sessions IDS="s01 s02 s03"
# Stops at the first failure (set -e) and never overwrites an existing session, so it is safe
# to resume by re-invoking with the ids that have not run yet.
set -euo pipefail
cd "$(dirname "$0")/.."

[ "$#" -ge 1 ] || { echo "usage: run_sessions.sh <id> [<id> ...]" >&2; exit 2; }

for id in "$@"; do
  if [ -e "data/raw/sessions/$id" ]; then
    echo "== skip $id (already exists)"
    continue
  fi
  echo "== session $id"
  bash scripts/run_session.sh "$id"
done
echo "== all requested sessions done"
