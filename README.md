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

## Quickstart for teammates (analysis — Docker only, no lab)

For working with the shared **v2** dataset and the model harness on your own machine, using Docker
directly — Docker Desktop on **Windows** or **macOS**, Docker Engine on **Linux**.

**You do not need the lab, the attacker container, or any traffic recording for model work.** Do
**not** run `make up` or `make sessions` — those start the ward and capture, which is unrelated to
training on the shared dataset. Everything below is just `docker build` and `docker run`.

The analysis containers run with `--network none` and mount `data/raw` read-only — the same isolation
as our setup, so the captured evidence can't be modified and nothing reaches the network. The
`docker` commands are identical in **PowerShell and bash**; the only shell differences are creating
folders (step 3) and viewing a file (step 7), noted inline.

**1. Install Docker** (Docker Desktop on Windows/macOS, Docker Engine on Linux), then confirm:

```
docker --version
```

**2. Clone the repo:**

```
git clone <repo-url> wardwatch
cd wardwatch
```

**3. Put the shared v2 dataset in place.** It is shared out-of-band — `data/` is gitignored, so the
dataset is **not** in the repo. Create the folders (this is the one place the shells differ):

- PowerShell: `mkdir data\processed\v2, data\raw\sessions`
- bash: `mkdir -p data/processed/v2 data/raw/sessions`

Then copy the shared files in so the layout is exactly:

```
data/processed/v2/testbed_windows.parquet
data/processed/v2/feature_list.json
data/processed/v2/sessions/*.parquet
data/raw/sessions/split_v2.json
```

**4. Build the analysis image** (exact tag `0.2.0`; needs network for this step only):

```
docker build -t wardwatch-analysis:0.2.0 services/analysis
```

**5. Run the tests** (pipeline + harness, in the image):

```
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m unittest discover -s tests/pipeline -p "test_*.py"
```

**6. Load the dataset with the harness** (quick check it reads):

```
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -c "from src.models.data import load_dataset, Dataset; df, fl = load_dataset(); ds = Dataset(df, fl, scale=False); print(len(df), 'rows,', len(ds.feature_names), 'features; train', len(ds.y('train')), 'test', len(ds.y('test')))"
```

Expected: `102152 rows, 51 features; train 53164 test 24840`.

**7. Train a toy model and check the run record:**

```
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python examples/quickstart_train.py
```

It trains a shallow XGBoost on the train split, prints the fixed binary metrics on the test split,
and writes the run record to `data/processed/v2/runs/quickstart.json`. View it:

- PowerShell: `Get-Content data\processed\v2\runs\quickstart.json`
- bash: `cat data/processed/v2/runs/quickstart.json`

This is a smoke test (untuned, one seed) that your harness and dataset are wired up — **not** a
reportable result.

> If PowerShell errors on `${PWD}` in a `-v` flag, use `"$($PWD.Path):/work"` (and likewise for the
> `data/raw` mount). `data/` is gitignored, so nothing you place or generate here is committed.

## Safety

Only run this lab on a machine you control. The lab network has no route to the internet, and the attack tools are only meant to target the lab's own containers. The capture container uses host networking so it can listen on the lab's bridge.
