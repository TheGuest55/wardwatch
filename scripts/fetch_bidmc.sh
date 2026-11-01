#!/bin/sh
# Downloads one BIDMC numerics record. Without it the monitor uses labelled synthetic vitals.
set -eu
dir=data/raw/bidmc
file=bidmc_01_Numerics.csv
mkdir -p "$dir"
[ -s "$dir/$file" ] && exit 0
if curl -fsS -m 60 -o "$dir/$file.tmp" "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/$file"; then
  mv "$dir/$file.tmp" "$dir/$file"
else
  rm -f "${dir:?}/${file:?}.tmp"
  echo "BIDMC download failed; the monitor will use synthetic vitals" >&2
fi
