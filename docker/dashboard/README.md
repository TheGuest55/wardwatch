# wardwatch-dashboard:0.1.0

Read-only results dashboard (Streamlit, port 8501). Separate from `wardwatch-analysis:0.2.0`: no torch, and
the analysis image's digest (recorded in every run record) is never touched. Pinned in `requirements.txt`.

It only reads. The repo and the data are mounted read-only; it never trains, never recomputes a headline
metric, never reads `data/raw`, makes no network calls.

## Build (once)

bash:
```
docker build -t wardwatch-dashboard:0.1.0 docker/dashboard
```
PowerShell:
```
docker build -t wardwatch-dashboard:0.1.0 docker/dashboard
```

## Run, then open http://localhost:8501

The data root is `data/processed/v2` of the checkout you run from (set `DATA` to point elsewhere, for example
the main checkout when you are in a git worktree, which has no `data/` of its own).

bash:
```
DATA="${DATA:-$PWD/data/processed/v2}"
docker run --rm --name wardwatch-dashboard -p 8501:8501 \
  --read-only --tmpfs /tmp --tmpfs /root --cap-drop ALL --security-opt no-new-privileges \
  -v "$PWD:/work:ro" -v "$DATA:/data:ro" -e WARDWATCH_DATA_ROOT=/data \
  wardwatch-dashboard:0.1.0
```
PowerShell:
```
$Data = if ($env:DATA) { $env:DATA } else { "$PWD/data/processed/v2" }
docker run --rm --name wardwatch-dashboard -p 8501:8501 `
  --read-only --tmpfs /tmp --tmpfs /root --cap-drop ALL --security-opt no-new-privileges `
  -v "${PWD}:/work:ro" -v "${Data}:/data:ro" -e WARDWATCH_DATA_ROOT=/data `
  wardwatch-dashboard:0.1.0
```
Stop with Ctrl+C. (`--network none` cannot be combined with a published port.)

## Tests (all pages render, loaders, headline rules)

bash:
```
docker run --rm --network none -v "$PWD:/work:ro" -w /work -e PYTHONDONTWRITEBYTECODE=1 \
  wardwatch-dashboard:0.1.0 python -m pytest tests/dashboard -q -p no:cacheprovider
```
PowerShell:
```
docker run --rm --network none -v "${PWD}:/work:ro" -w /work -e PYTHONDONTWRITEBYTECODE=1 `
  wardwatch-dashboard:0.1.0 python -m pytest tests/dashboard -q -p no:cacheprovider
```

## Where it looks
`WARDWATCH_DATA_ROOT` (default `/work/data/processed/v2` in the image): `runs/**.json` run records
(any layout), `runs/<model>/predictions/*.parquet`, `runs/lstm/explanations.json`,
`runs/calibration/<model>_details.json`, `runs/expA/autoencoder_details.json`, `runs/bootstrap/*.json`,
`testbed_windows.parquet`, `live/incidents.jsonl`. Optional `WARDWATCH_RESULTS_ROOT` for `results/runs`.
Anything missing shows "not available" with the reason.
