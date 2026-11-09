# WardWatch

ICT3214 Coursework 2: medical IoT digital-twin intrusion detection and forensic triage.

An isolated Docker lab with a simulated hospital ward (MQTT broker, devices, nurse station), traffic capture and an attacker container.

## What's inside

A Docker Compose lab on one internal (no internet route) network, `ward_net`:

- **broker** — Mosquitto 2.x, anonymous access off, one account per device, ACLs per topic.
- **patient-monitor** — publishes heart rate, SpO2 and blood pressure once a second as JSON, replayed from a PhysioNet BIDMC record (synthetic fallback if unavailable).
- **nurse-station** — subscribes to the ward and logs every message.
- **capture** — rotating `tcpdump` on the lab bridge, writing pcaps to `data/raw/pcap/`.
- **attacker** — a toolbox image (nmap, hping3, arpspoof, tcpdump, paho-mqtt); idle until later phases.

## Requirements

- Docker with Compose v2. No special sandbox is needed.
- Internet access for the first build, to download base images and packages.

## Quick start

Docker builds the images from this repo, so there are no prebuilt images to download.

```bash
git clone <repo-url> wardwatch
cd wardwatch
cp .env.example .env    # then set your own test passwords
docker compose up -d --build
```

Stop the lab with:

```bash
docker compose down
```

With `make` installed:

```bash
make up      # fetch sample data, build images, start the lab
make smoke   # run the S1-S7 checks, writing docs/environment-status.md
make down    # stop the lab (never touches data/)
```

Captured traffic is written to `data/raw/pcap/`. Two checks need you to confirm by
hand: `scripts/smoke/s8_host_files.sh` (the pcap is visible on the host) and
`scripts/smoke/s9_port.sh` (a test page on port 8501 loads in your browser).

## Quickstart for teammates (analysis — no lab needed)

Work with the shared **v2** dataset and the model harness **without running the lab**. Pick your OS
column and run the one command for your platform at each step. The commands use the pinned analysis
image; an optional no-Docker (venv) route is at the end.

- **Windows:** PowerShell + Docker Desktop + Git for Windows. No `make`, no `bash`.
- **Linux / macOS:** bash + Docker Engine or Docker Desktop. `make` optional.

### 1. Clone

Same on every OS:

```bash
git clone <repo-url> wardwatch
cd wardwatch
```

### 2. Get the shared v2 dataset

It is shared out-of-band — `data/` is gitignored, so the dataset is **not** in the repo. Create the
folders, then copy the shared files into them.

| Windows (PowerShell) | Linux / macOS (bash) |
|---|---|
| `New-Item -ItemType Directory -Force data\processed\v2, data\raw\sessions \| Out-Null` | `mkdir -p data/processed/v2 data/raw/sessions` |

Copy the shared files so the layout is exactly:

```
data/processed/v2/testbed_windows.parquet
data/processed/v2/feature_list.json
data/processed/v2/sessions/*.parquet
data/raw/sessions/split_v2.json
```

### 3. Build the analysis image

Same on every OS:

```bash
docker build -t wardwatch-analysis:0.2.0 services/analysis
```

### 4. Run the tests (pipeline + harness, inside the image)

**Windows (PowerShell)** — via the wrapper (allow `.ps1` for this session first), or the plain command:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force ; .\scripts\analysis.ps1 test
# or, no wrapper:
docker run --rm -v ${PWD}:/work -w /work wardwatch-analysis:0.2.0 python -m unittest discover -s tests/pipeline -p "test_*.py"
```

**Linux / macOS (bash)** — via the wrapper, or the plain command:

```bash
bash scripts/analysis.sh test
# or, no wrapper:
docker run --rm -v "$PWD":/work -w /work wardwatch-analysis:0.2.0 python -m unittest discover -s tests/pipeline -p "test_*.py"
```

### 5. Load the dataset with the harness (quick check it reads)

**Windows (PowerShell):**

```powershell
docker run --rm -v ${PWD}:/work -w /work wardwatch-analysis:0.2.0 python -c "from src.models.data import load_dataset, Dataset; df, fl = load_dataset(); ds = Dataset(df, fl, scale=False); print(len(df), 'rows,', len(ds.feature_names), 'features; train', len(ds.y('train')), 'test', len(ds.y('test')))"
```

**Linux / macOS (bash):**

```bash
docker run --rm -v "$PWD":/work -w /work wardwatch-analysis:0.2.0 python -c "from src.models.data import load_dataset, Dataset; df, fl = load_dataset(); ds = Dataset(df, fl, scale=False); print(len(df), 'rows,', len(ds.feature_names), 'features; train', len(ds.y('train')), 'test', len(ds.y('test')))"
```

Expected: `102152 rows, 51 features; train 53164 test 24840`.

### 6. Train a toy model and check the run record

Train + log (same image, one command per OS):

```powershell
# Windows (PowerShell)
docker run --rm -v ${PWD}:/work -w /work wardwatch-analysis:0.2.0 python examples/quickstart_train.py
```

```bash
# Linux / macOS (bash)
docker run --rm -v "$PWD":/work -w /work wardwatch-analysis:0.2.0 python examples/quickstart_train.py
```

It trains a shallow XGBoost on the train split, prints the fixed binary metrics on the test split,
and writes the run record to `data/processed/v2/runs/quickstart.json`. Check it:

| Windows (PowerShell) | Linux / macOS (bash) |
|---|---|
| `Get-Content data\processed\v2\runs\quickstart.json` | `cat data/processed/v2/runs/quickstart.json` |

This is a smoke test (untuned, one seed) that your harness + dataset are wired up — **not** a
reportable result.

### Notes

- `scripts/analysis.ps1` mirrors `scripts/analysis.sh` (same `docker run` flags; repo mounted at
  `/work`, `data/raw` read-only): `features <session> | dataset | datacheck | test | shell`. Override
  the image with `$env:ANALYSIS_IMAGE` (PowerShell) or `ANALYSIS_IMAGE=…` (bash).
- If PowerShell errors on `${PWD}` in a `-v` flag, use `-v "$($PWD.Path):/work"`.
- `data/` is gitignored, so nothing you place or generate here is committed.

### Optional: no-Docker route (venv)

For running the harness and training models without Docker. Needs **Python 3.11** (the image uses
3.11.9); it installs the pinned stack (CPU-only torch) from `requirements-models.txt`. This covers
steps 5–6 (load the dataset, train a model); feature extraction from pcaps and the full test suite
also need scapy and stay in Docker.

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements-models.txt
$env:PYTHONPATH="." ; python examples/quickstart_train.py
```

**Linux / macOS (bash):**

```bash
python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements-models.txt
PYTHONPATH=. python examples/quickstart_train.py
```

## Safety

Only run this lab on a machine you control. The lab network has no route to the internet, and the attack tools are only meant to target the lab's own containers. The capture container uses host networking so it can listen on the lab's bridge.
