#!/usr/bin/env bash
# Build the shared v2 dataset bundle for teammates. Includes ONLY the processed v2 corpus and the
# v2 split — what teammates need to train models. Deliberately leaves out:
#   - runs/ and mlruns/  — run records; a pre-made one would make a FAILED training step look like
#                          it worked, so teammates must generate their own.
#   - the raw PCAPs under data/raw/sessions/<id>/ — gigabytes, not needed for model work (training
#                          reads the processed parquet, not the captures).
#   - anything with credentials — .env, the generated compose, the broker password file (none are
#                          under the paths added below, so they cannot slip in).
#
# Portable (uses Python's zipfile, no `zip` binary needed).
# Usage:  bash scripts/make_dataset_bundle.sh [output.zip]     (default: wardwatch-v2-dataset.zip)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-wardwatch-v2-dataset.zip}"

python3 - "$OUT" <<'PY'
import os, sys, zipfile
out = sys.argv[1]
INCLUDE = [
    "data/processed/v2/testbed_windows.parquet",
    "data/processed/v2/feature_list.json",
    "data/processed/v2/sessions",            # per-session processed parquets (NOT the raw pcaps)
    "data/raw/sessions/split_v2.json",
]
EXCLUDE_DIRS = {"runs", "mlruns", "__pycache__"}

missing = [p for p in INCLUDE if not os.path.exists(p)]
if missing:
    sys.exit("missing (build v2 first, see README): " + ", ".join(missing))

if os.path.exists(out):
    os.remove(out)

def add(zf, path):
    if os.path.isfile(path):
        zf.write(path)
        return
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            zf.write(os.path.join(root, f))

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in INCLUDE:
        add(zf, p)
    names = sorted(zf.namelist())

# Safety assertions: nothing excluded must be in the bundle.
bad = [n for n in names if "/runs/" in n or "/mlruns/" in n or n.endswith(".pcap")
       or n.endswith(".env") or "accounts" in n or "docker-compose" in n]
if bad:
    sys.exit("REFUSING: excluded files slipped into the bundle: " + ", ".join(bad[:5]))

size = os.path.getsize(out)
print(f"wrote {out} ({size/1_048_576:.1f} MiB), {len(names)} files:")
for n in names:
    print("  " + n)
PY
