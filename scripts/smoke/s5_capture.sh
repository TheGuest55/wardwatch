#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
filter='mqtt && ip.addr == 172.28.10.31 && ip.addr == 172.28.10.10'
cmd="tshark over data/raw/pcap/*.pcap, filter: $filter (60 s window)"
start=$(now); sleep 60; stop=$(now)
sleep 2  # let the capture flush the last packets to disk
n=$(pcap_count "$start" "$stop" "$filter")
if [ "${n:-0}" -ge 50 ]; then
  record S5 "Capture sees bridge traffic" PASS "$cmd" "$n MQTT packets between .31 and .10 in 60 s"
else
  record S5 "Capture sees bridge traffic" FAIL "$cmd" "${n:-0} MQTT packets in 60 s (need at least 50)"
fi
