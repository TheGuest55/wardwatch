#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cmd="docker run --rm hello-world:linux"
if out=$($cmd 2>&1) && grep -q "Hello from Docker" <<<"$out"; then
  record S1 "Docker works" PASS "$cmd" "output contains 'Hello from Docker!'"
else
  record S1 "Docker works" FAIL "$cmd" "$(tail -1 <<<"$out" | cut -c1-150)"
fi
