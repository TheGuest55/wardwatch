#!/usr/bin/env bash
# Extract CICIoMT2024 Wi-Fi/MQTT windows for Experiment C (DoS-only, single-attacker captures).
#
#   bash scripts/extract_cic_dos.sh          # stage the 19 PCAPs and run extract_cic (3 workers)
#   bash scripts/extract_cic_dos.sh --list   # dry run: check and print the staged list, extract nothing
#   make cic                                  # same as the first line
#   DATA_DIR=/path/to/data bash scripts/extract_cic_dos.sh   # data folder elsewhere (e.g. from a git worktree)
#
# Why this wrapper exists:
#   * CIC ships its PCAPs as WiFi_and_MQTT/attacks/pcap/{train,test}/<Class>_{train,test}.pcap, next to
#     DDoS captures. DDoS files have several attacking addresses; our labelling keys on ONE attacker
#     address, so they are EXCLUDED (REPORT_NOTES 2026-11-16). The 19 files used are listed explicitly
#     below: a missing file fails the run, and nothing else (never a DDoS file) is staged.
#   * data/raw is evidence and stays read-only: it is mounted :ro, the 19 files are symlinked into a
#     container-only staging dir (/stage/{train,test}), and extract_cic.main() runs with RAW_DIR pointed
#     there. src/features/extract_cic.py is not modified.
#   * Outputs: data/processed/cic/ (cic_windows.parquet, feature_list.json, extract_report.json,
#     sessions/), plus pcap_sha256.txt (hash of every staged PCAP) and extract.log.
set -euo pipefail

TRAIN="Benign ARP_Spoofing Recon-OS_Scan Recon-Port_Scan Recon-VulScan TCP_IP-DoS-SYN1 TCP_IP-DoS-SYN2 TCP_IP-DoS-SYN3 TCP_IP-DoS-SYN4 MQTT-DoS-Connect_Flood MQTT-DoS-Publish_Flood"
TEST="Benign ARP_Spoofing Recon-OS_Scan Recon-Port_Scan Recon-VulScan TCP_IP-DoS-SYN MQTT-DoS-Connect_Flood MQTT-DoS-Publish_Flood"

stage() {   # runs INSIDE the container; /cic is the CIC pcap folder (read-only)
  mkdir -p /stage/train /stage/test
  for s in $TRAIN; do f=/cic/train/${s}_train.pcap; [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; ln -s "$f" /stage/train/; done
  for s in $TEST;  do f=/cic/test/${s}_test.pcap;   [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; ln -s "$f" /stage/test/;  done
  case "$(ls /stage/train /stage/test)" in *DDoS*) echo "DDoS file staged: refusing" >&2; exit 1;; esac
  echo "[stage] train=$(ls /stage/train | wc -l) test=$(ls /stage/test | wc -l) (DoS only; DDoS excluded)"
}

if [ "${1:-}" = "--in-container" ]; then
  shift
  stage
  if [ "${1:-}" = "--list" ]; then
    ls -l /stage/train /stage/test
    exit 0
  fi
  mkdir -p data/processed/cic
  (cd /stage && sha256sum train/*.pcap test/*.pcap) > data/processed/cic/pcap_sha256.txt
  echo "[stage] sha256 written for $(wc -l < data/processed/cic/pcap_sha256.txt) pcaps"
  python -c "
import sys
from pathlib import Path
import src.features.extract_cic as e
e.RAW_DIR = Path('/stage')
sys.argv = ['extract_cic', '--workers', '${CIC_WORKERS:-3}']
e.main()
"
  echo "[done] exit 0"
  exit 0
fi

cd "$(dirname "$0")/.."
IMAGE="${ANALYSIS_IMAGE:-wardwatch-analysis:0.2.0}"
DATA_DIR="$(cd "${DATA_DIR:-$PWD/data}" && pwd)"
PCAP="$DATA_DIR/raw/ciciomt2024/WiFi_and_MQTT/attacks/pcap"
[ -d "$PCAP" ] || { echo "CIC pcaps not found at $PCAP" >&2; exit 1; }

run() {
  docker run --rm --network none --cpus "${ANALYSIS_CPUS:-3}" \
    -v "$PWD":/work -v "$DATA_DIR/raw":/work/data/raw:ro -v "$DATA_DIR/processed":/work/data/processed \
    -v "$PCAP":/cic:ro \
    -w /work "$IMAGE" bash scripts/extract_cic_dos.sh --in-container "$@"
}

if [ "${1:-}" = "--list" ]; then
  run --list
else
  mkdir -p "$DATA_DIR/processed/cic"
  run 2>&1 | tee "$DATA_DIR/processed/cic/extract.log"
fi
