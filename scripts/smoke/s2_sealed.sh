#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cmd="docker compose exec -T attacker curl -sS -m 5 https://example.com"
out=$($cmd 2>&1); rc=$?
# Exit 126/127 would mean curl itself did not run, which proves nothing about the network.
if [ "$rc" -ne 0 ] && [ "$rc" -lt 126 ]; then
  record S2 "Lab network sealed" PASS "$cmd" "curl exit $rc: $(tail -1 <<<"$out" | cut -c1-100)"
else
  record S2 "Lab network sealed" FAIL "$cmd" "curl exit $rc (reached the internet, or curl missing)"
fi
