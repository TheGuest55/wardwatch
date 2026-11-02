#!/bin/sh
# Rotating capture on the lab bridge. Writes into the current session's dir when SESSION_ID is
# set (data/raw/sessions/<id>/pcap), else data/raw/pcap. Rotates hourly or at 100 MB; filenames
# use UTC because TZ=UTC is set in compose.
set -eu
if [ -n "${SESSION_ID:-}" ]; then
  outdir="/data/sessions/${SESSION_ID}/pcap"
else
  outdir="/data/pcap"
fi
mkdir -p "$outdir"
until [ -e /sys/class/net/br-ward ]; do
  echo "waiting for br-ward"
  sleep 1
done
exec tcpdump -i br-ward -n -U -s 0 -Z root -G 3600 -C 100 -w "${outdir}/ward_%Y%m%dT%H%M%SZ.pcap"
