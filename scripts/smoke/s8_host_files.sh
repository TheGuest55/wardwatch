#!/usr/bin/env bash
# Needs a person: confirms the pcap written in the VM is visible on the host mount.
source "$(dirname "$0")/common.sh"
latest=$(ls -t data/raw/pcap/*.pcap 2>/dev/null | head -1)
echo "On the host, run:  ls -la $latest"
echo "It should exist with a non-zero size."
