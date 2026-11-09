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

For teammates who just want to work with the shared **v2** dataset and the model harness. Windows +
Docker Desktop, in **PowerShell**. You do **not** need to run the lab, capture traffic, or have `make`.
One command per step.

1. Clone the repo (then `cd` in):

   ```powershell
   git clone <repo-url> wardwatch
   cd wardwatch
   ```

2. Drop in the shared **v2** dataset. It is shared out-of-band (it is gitignored, so it is *not* in
   the repo). Create the folders:

   ```powershell
   New-Item -ItemType Directory -Force data\processed\v2, data\raw\sessions | Out-Null
   ```

   then copy the shared files into place so the layout is exactly:

   ```
   data\processed\v2\testbed_windows.parquet
   data\processed\v2\feature_list.json
   data\processed\v2\sessions\*.parquet
   data\raw\sessions\split_v2.json
   ```

3. Build the analysis image:

   ```powershell
   docker build -t wardwatch-analysis:0.2.0 services/analysis
   ```

4. Run the tests (pipeline + harness, inside the image):

   ```powershell
   docker run --rm -v ${PWD}:/work -w /work wardwatch-analysis:0.2.0 python -m unittest discover -s tests/pipeline -p "test_*.py"
   ```

5. Load the dataset through the harness (quick check it reads):

   ```powershell
   docker run --rm -v ${PWD}:/work -w /work wardwatch-analysis:0.2.0 python -c "from src.models.data import load_dataset, Dataset; df, fl = load_dataset(); ds = Dataset(df, fl, scale=False); print(len(df), 'rows,', len(ds.feature_names), 'features; train', len(ds.y('train')), 'test', len(ds.y('test')))"
   ```

6. Train a toy model and log a run:

   ```powershell
   docker run --rm -v ${PWD}:/work -w /work wardwatch-analysis:0.2.0 python examples/quickstart_train.py
   ```

   This trains a shallow XGBoost on the train split, prints the fixed binary metrics on the test
   split, and writes a run record to `data\processed\v2\runs\quickstart.json`. It is a smoke test that
   your harness + dataset are wired up — **not** a reportable result (untuned, uncalibrated, one seed).

> If PowerShell errors on `${PWD}` in the `-v` flag, use `-v "$($PWD.Path):/work"` instead. The
> `data\` folder is gitignored, so nothing you place or generate here is committed.

## Safety

Only run this lab on a machine you control. The lab network has no route to the internet, and the attack tools are only meant to target the lab's own containers. The capture container uses host networking so it can listen on the lab's bridge.
