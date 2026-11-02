#!/usr/bin/env bash
# Run one session end to end: bring the ward up tagged with this session id (so capture and
# benign events land in the session dir), pre-roll normal traffic, run the attack schedule,
# post-roll, stop capture, then merge events and write the manifest.
#
# A session is self-contained and NEVER overwrites an existing one (B1 rule 2). Dry-run
# sessions (id "s00" or any id starting with "t") use the short schedule, are flagged
# dry_run=true in config.json, and are excluded from the dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

ID="${1:-}"
[ -n "$ID" ] || { echo "usage: run_session.sh <session-id>" >&2; exit 2; }
SDIR="data/raw/sessions/$ID"
if [ -e "$SDIR" ]; then
  echo "session $ID already exists at $SDIR; refusing to overwrite" >&2
  exit 1
fi

# Three modes by id: dry (s00, t*) short smoke schedule; normal-only (n*) long attack-free
# capture for the autoencoder; attack (everything else, e.g. s01) the full schedule.
DRY=false
case "$ID" in
  t*|s00) MODE=dry;   DRY=true;;
  n*)     MODE=normal;;
  *)      MODE=attack;;
esac
if [ "$MODE" = dry ]; then
  SCHED="${SCHEDULE:-schedules/smoke.json}"; PRE_ROLL="${PRE_ROLL:-15}"; POST_ROLL="${POST_ROLL:-10}"
  # Dry runs only: shorten the nurse's command/reconnect cadence (from config) so a short
  # session exercises them. Other sessions leave these unset and use the built-in cadence.
  export NURSE_CMD_INTERVAL="$(python3 -c "import json;print(json.load(open('config/ward.json')).get('dry_run',{}).get('nurse_cmd_interval',''))")"
  export NURSE_RECONNECT_MIN="$(python3 -c "import json;print(json.load(open('config/ward.json')).get('dry_run',{}).get('reconnect_min',120))")"
  export NURSE_RECONNECT_MAX="$(python3 -c "import json;print(json.load(open('config/ward.json')).get('dry_run',{}).get('reconnect_max',240))")"
elif [ "$MODE" = normal ]; then
  SCHED="${SCHEDULE:-schedules/normal_only.json}"; PRE_ROLL="${PRE_ROLL:-7200}"; POST_ROLL="${POST_ROLL:-0}"
else
  # ~90 min attack session: 10 min pre-roll, 8 runs with seeded 6-10 min gaps, 5 min post-roll.
  SCHED="${SCHEDULE:-schedules/default.json}"; PRE_ROLL="${PRE_ROLL:-600}"; POST_ROLL="${POST_ROLL:-300}"
fi
SEED="$(python3 -c "import json;print(json.load(open('config/ward.json'))['seed'])")"

mkdir -p "$SDIR"
python3 - "$ID" "$DRY" "$SCHED" "$SEED" > "$SDIR/config.json" <<'PY'
import json, sys
cid, dry, sched, seed = sys.argv[1], sys.argv[2] == "true", sys.argv[3], int(sys.argv[4])
cfg = json.load(open("config/ward.json"))
json.dump({"session": cid, "dry_run": dry, "schedule": sched, "seed": seed,
           "beds": cfg["beds"], "noise": cfg["noise"], "images": cfg["images"]},
          sys.stdout, indent=2)
PY
cp "$SCHED" "$SDIR/schedule.json"

echo "[$ID] bringing ward up (dry_run=$DRY, schedule=$SCHED)"
SESSION_ID="$ID" docker compose up -d --build --remove-orphans >/dev/null
echo "[$ID] pre-roll normal traffic: ${PRE_ROLL}s"
sleep "$PRE_ROLL"

echo "[$ID] running attack schedule"
docker compose exec -T -e SEED="$SEED" attacker \
  python -m orchestrator --schedule "/schedules/$(basename "$SCHED")" \
  --session "$ID" --out "/data/sessions/$ID" --seed "$SEED"

echo "[$ID] post-roll normal traffic: ${POST_ROLL}s"
sleep "$POST_ROLL"

docker compose stop capture >/dev/null    # finalise the pcap before hashing
python3 scripts/merge_session.py "$SDIR"
echo "[$ID] complete"
