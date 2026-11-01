#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cmd="docker compose logs --since 60s nurse-station"
start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
sleep 60
stop=$(date -u +%Y-%m-%dT%H:%M:%SZ)
n=$(docker compose logs --no-log-prefix --since "$start" --until "$stop" nurse-station \
    | grep -c '^ward1/bed1/monitor/vitals ')
if [ "$n" -ge 50 ]; then
  record S3 "MQTT end to end" PASS "$cmd" "$n monitor messages received by nurse station in 60 s"
else
  record S3 "MQTT end to end" FAIL "$cmd" "$n monitor messages in 60 s (need at least 50)"
fi
