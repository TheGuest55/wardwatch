# WardWatch

ICT3214 Coursework 2: medical IoT digital-twin intrusion detection and forensic triage.

An isolated Docker lab with a simulated hospital ward (MQTT broker, devices, nurse station), traffic
capture and an attacker container. Most people only need the **model-work quickstart** below — the
lab itself is **not needed** for working with the shared dataset (it is documented further down).

## What's inside

A Docker Compose lab on one internal (no internet route) network, `ward_net`: a Mosquitto **broker**
(per-device accounts, topic ACLs), **patient-monitor / pump / wearable** devices publishing JSON
vitals, a **nurse-station** subscriber, a **capture** container (rotating `tcpdump` → `data/raw/pcap/`),
and an **attacker** toolbox (nmap, hping3, arpspoof, paho-mqtt). The attacker runs scripted attacks
with ground-truth timing; the captures become the labelled dataset the models train on.

## Quickstart for teammates (analysis — Docker only, no lab)

Work with the shared **v2** dataset and the model harness on your own machine, using Docker directly
— Docker Desktop on **Windows** or **macOS**, Docker Engine on **Linux**.

**You do not need the lab, the attacker container, or any traffic recording for model work.** Do
**not** run `make up` or `make sessions` — those start the ward and capture, which is unrelated to
training on the shared dataset. Everything below is just `docker build` and `docker run`.

The analysis containers run with `--network none` and mount `data/raw` **read-only** — the same
isolation as our setup, so the captured evidence can't be modified and nothing reaches the network.
The `docker` commands are **identical in PowerShell and bash** (they use `${PWD}` and `$(...)`, which
both shells expand); the only shell-specific lines are creating folders and viewing a file, called
out inline.

**Requirements:** Docker (Desktop on Windows/macOS, Engine on Linux) **and git**. Confirm:

```
docker --version
git --version
```

### 1. Clone

```
git clone https://github.com/TheGuest55/wardwatch.git wardwatch
cd wardwatch
```

### 2. Get the shared v2 dataset

The dataset is shared out-of-band (`data/` is gitignored, so it is **not** in the repo).

- **Download it from the shared drive:** <ADD-SHARED-DRIVE-LINK-HERE>
- The bundle contains **only** `data/processed/v2/` and `data/raw/sessions/split_v2.json` — **never**
  `runs/` or `mlruns/` (run records), no raw PCAPs, and nothing with credentials. (Whoever builds the
  bundle uses `scripts/make_dataset_bundle.sh`, which enforces exactly this.)

Create the folders (this is the one place the shells differ):

- PowerShell: `New-Item -ItemType Directory -Force data\processed\v2, data\raw\sessions`
- bash: `mkdir -p data/processed/v2 data/raw/sessions`

Unzip the bundle into the repo so the layout is exactly:

```
data/processed/v2/testbed_windows.parquet
data/processed/v2/feature_list.json
data/processed/v2/sessions/*.parquet
data/raw/sessions/split_v2.json
```

> The repo already tracks `data/raw/sessions/split.json` — that is the **v1** split, kept as review
> evidence. Teammates use **`split_v2.json`** only; don't edit or rely on `split.json`.

### 3. Build the analysis image (≈ 4 GB, ≈ 6 min; needs network for this step only)

Pass the commit as a build arg so the image is **labelled with the code it was built from**, and
`--pull` so a stale base image can't sneak in. Same command in both shells:

```
docker build --pull --build-arg "GIT_COMMIT=$(git rev-parse HEAD)" -t wardwatch-analysis:0.2.0 services/analysis
```

Verify the image was really built from your checkout (catches a stale image reused under the same
tag) — this label must equal your `HEAD`:

```
docker image inspect --format "{{index .Config.Labels \"org.opencontainers.image.revision\"}}" wardwatch-analysis:0.2.0
git rev-parse HEAD
```

### 4. Run the tests (pipeline + harness, in the image)

```
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m unittest discover -s tests/pipeline -p "test_*.py"
```

Expected: it ends with `OK` (all pipeline + harness tests pass).

### 5. Load the dataset with the harness

```
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -c "from src.models.data import load_dataset; df, fl = load_dataset(); print('rows', len(df), '| features', fl['n_features']); [print(f'{s}:', int((df['split']==s).sum()), 'rows', sorted(df[df['split']==s]['session'].unique())) for s in ('train','val','test')]"
```

Expected:

```
rows 102152 | features 51
train: 53164 rows ['n02', 'r01', 'r02', 'r03']
val: 24148 rows ['r04', 'r06']
test: 24840 rows ['r05', 'r07']
```

### 6. Train a toy model and check the run record

Pass the run **provenance** from the host (code commit, dirty flag, image id) so the run record is
fully populated — these `-e "VAR=$(...)"` forms expand in both PowerShell and bash:

```
docker run --rm --network none -e "WARDWATCH_GIT_COMMIT=$(git rev-parse HEAD)" -e "WARDWATCH_GIT_DIRTY=$(git status --porcelain)" -e "WARDWATCH_IMAGE_DIGEST=$(docker image inspect --format "{{.Id}}" wardwatch-analysis:0.2.0)" -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python examples/quickstart_train.py
```

It trains a shallow XGBoost on the train split, prints the fixed binary metrics on the **test** split
(F1 ≈ 0.88, ROC-AUC ≈ 0.99), and writes the run record to `data/processed/v2/runs/quickstart.json`
with a non-null `git_commit`, `git_dirty` and `image_digest`. View it:

- PowerShell: `Get-Content data\processed\v2\runs\quickstart.json`
- bash: `cat data/processed/v2/runs/quickstart.json`

This is a smoke test (untuned, one seed) that your harness and dataset are wired up — **not** a
reportable result. (Report-grade runs set `for_report=True`; the harness then **refuses** to write a
record if any provenance field is null, so pass the `-e` vars above.)

### Notes

- **mlflow / `mlruns/`:** `examples/quickstart_train.py` also logs to MLflow, which creates a local
  `mlruns/` folder (gitignored). You can ignore or delete it. If you ever see a GitPython
  *"Bad git executable"* message, it is **harmless** — git is intentionally not in the image; run
  provenance comes from the `WARDWATCH_*` env vars above, and the image sets `GIT_PYTHON_REFRESH=quiet`.
- If PowerShell errors on `${PWD}` in a `-v` flag, use `"$($PWD.Path):/work"` (and likewise for the
  `data/raw` mount).
- `data/` is gitignored, so nothing you place or generate here is committed.

## Running the lab (not needed for model work)

This section is only for generating new captures with the ward — teammates doing **model work can
skip it entirely**. Docker builds the images from this repo, so there are no prebuilt images.

```bash
git clone https://github.com/TheGuest55/wardwatch.git wardwatch
cd wardwatch
cp .env.example .env    # then set your own test passwords
docker compose up -d --build
docker compose down     # stop the lab
```

With `make` installed:

```bash
make up      # fetch sample data, build images, start the lab
make smoke   # run the S1-S7 checks, writing docs/environment-status.md
make down    # stop the lab (never touches data/)
```

Captured traffic is written to `data/raw/pcap/`. Two checks need manual confirmation:
`scripts/smoke/s8_host_files.sh` (the pcap is visible on the host) and `scripts/smoke/s9_port.sh`
(a test page on port 8501 loads in your browser).

## Safety

Only run this lab on a machine you control. The lab network has no route to the internet, and the attack tools are only meant to target the lab's own containers. The capture container uses host networking so it can listen on the lab's bridge.
