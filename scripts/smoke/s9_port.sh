#!/usr/bin/env bash
# Needs a person: serves a test page on 8501 for 5 minutes so the host browser can reach it.
source "$(dirname "$0")/common.sh"
echo "Open http://localhost:8501 in your browser. The server stops after 5 minutes."
echo "If the port is not published yet, on the host run: sbx ports wardwatch --publish 8501:8501/tcp"
exec timeout 300 python3 -m http.server 8501 --bind 0.0.0.0 --directory scripts/smoke
