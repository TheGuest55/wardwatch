# Shared helpers for the smoke tests. Source this from each script.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RESULTS=data/interim/smoke
mkdir -p "$RESULTS"

# record ID NAME RESULT COMMAND EVIDENCE -> writes a table row, prints, returns PASS/FAIL.
record() {
  printf '| %s | %s | %s | `%s` | %s |\n' "$1" "$2" "$3" "$4" "${5//|/\\|}" > "$RESULTS/$1.row"
  echo "$1 $3: $5"
  [ "$3" = PASS ]
}

now() { date -u +%s; }

# pcap_count SINCE UNTIL FILTER: counts matching packets across the capture files.
pcap_count() {
  docker compose exec -T capture sh -c '
    for f in /data/pcap/*.pcap; do
      tshark -r "$f" -Y "frame.time_epoch >= $0 && frame.time_epoch < $1 && ($2)" 2>/dev/null
    done | wc -l' "$1" "$2" "$3" | tr -d '[:space:]'
}

mac_of() { docker compose exec -T "$1" cat /sys/class/net/eth0/address | tr -d '\r\n'; }
