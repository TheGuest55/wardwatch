#!/usr/bin/env bash
# Runs S1-S7 and writes the status table. S8 and S9 need a person, so they are run
# on their own (scripts/smoke/s8_host_files.sh, s9_port.sh) and recorded by hand.
source "$(dirname "$0")/common.sh"
status=0
for t in s1_docker s2_sealed s3_mqtt s4_acl s5_capture s6_flood s7_arp; do
  bash "scripts/smoke/$t.sh" || status=1
done
mkdir -p docs
{
  echo "# Environment status"
  echo
  echo "Last automated run: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "S8 (files reach the host) and S9 (port publishing) need manual confirmation;"
  echo "see scripts/smoke/s8_host_files.sh and s9_port.sh."
  echo
  echo "| # | Test | Result | Command | Evidence |"
  echo "|---|---|---|---|---|"
  cat "$RESULTS"/S*.row 2>/dev/null
} > docs/environment-status.md
exit $status
