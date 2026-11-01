#!/bin/sh
# Rotates hourly or at 100 MB. Filenames use UTC because TZ=UTC is set in compose.
set -eu
until [ -e /sys/class/net/br-ward ]; do
  echo "waiting for br-ward"
  sleep 1
done
exec tcpdump -i br-ward -n -U -s 0 -Z root -G 3600 -C 100 -w '/data/pcap/ward_%Y%m%dT%H%M%SZ.pcap'
