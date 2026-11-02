#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cmd="docker compose exec -T bed1-monitor python - < scripts/smoke/mqtt_check.py"
if out=$(docker compose exec -T bed1-monitor python - < scripts/smoke/mqtt_check.py 2>&1); then
  record S4 "Broker auth and ACLs" PASS "$cmd" "$(tail -1 <<<"$out")"
else
  record S4 "Broker auth and ACLs" FAIL "$cmd" "$(tail -1 <<<"$out" | cut -c1-200)"
fi
