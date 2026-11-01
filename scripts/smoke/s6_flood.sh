#!/usr/bin/env bash
# S6: confirm the capture records a rate-limited flood and that resource limits hold.
# The traffic is bounded three ways (B1 rule 5): interval -i u1000 (~1000 pps),
# packet count -c 10000, and a 10 s timeout. Target is the lab broker only; the
# network is internal, so nothing leaves the VM.
source "$(dirname "$0")/common.sh"
cmd="timeout 10 hping3 -S -p 1883 -i u1000 -c 10000 172.28.10.10"
filter='ip.src == 172.28.10.66 && ip.dst == 172.28.10.10 && tcp.flags.syn == 1 && tcp.flags.ack == 0'
start=$(now)
docker compose exec -T attacker sh -c "$cmd" >/dev/null 2>&1 &
flood=$!
sleep 5
stats=$(docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' \
        wardwatch-attacker-1 wardwatch-broker-1)
t0=$(date +%s%N); docker ps >/dev/null; t1=$(date +%s%N)
latency_ms=$(( (t1 - t0) / 1000000 ))
wait $flood
stop=$(( $(now) + 1 ))
sleep 2
n=$(pcap_count "$start" "$stop" "$filter")
pps=$(( ${n:-0} / 10 ))
evidence="$n SYNs in pcap (~$pps pps over 10 s); docker ps took ${latency_ms} ms during flood; ${stats//$'\n'/; }"
if [ "${n:-0}" -ge 5000 ] && [ "${n:-0}" -le 11000 ] && [ "$latency_ms" -lt 5000 ]; then
  record S6 "Rate-limited flood" PASS "$cmd" "$evidence"
else
  record S6 "Rate-limited flood" FAIL "$cmd" "$evidence"
fi
