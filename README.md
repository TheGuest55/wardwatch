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
both shells expand); the only shell-specific lines are downloading/extracting the dataset (step 2)
and viewing a file (step 6), called out inline.

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

The dataset is published as a GitHub release (it is gitignored, so it is **not** in the repo).
Release page: <https://github.com/TheGuest55/wardwatch/releases/tag/v2-dataset>

Download the bundle into the **repo root** and extract it there. The archive carries the `data/...`
paths, so extracting at the repo root puts every file in the right place (no `mkdir` needed). From
the repo root:

- bash:

  ```bash
  curl -L -o wardwatch-v2-dataset.zip "https://github.com/TheGuest55/wardwatch/releases/download/v2-dataset/wardwatch-v2-dataset.zip"
  unzip -o wardwatch-v2-dataset.zip
  ```

- PowerShell:

  ```powershell
  Invoke-WebRequest -Uri "https://github.com/TheGuest55/wardwatch/releases/download/v2-dataset/wardwatch-v2-dataset.zip" -OutFile wardwatch-v2-dataset.zip
  Expand-Archive -Path wardwatch-v2-dataset.zip -DestinationPath . -Force
  ```

Afterwards the layout must be exactly:

```
data/processed/v2/testbed_windows.parquet
data/processed/v2/feature_list.json
data/processed/v2/sessions/*.parquet
data/raw/sessions/split_v2.json
```

The bundle (built by `scripts/make_dataset_bundle.sh`) contains **only** these — **never** `runs/` or
`mlruns/` (run records), no raw PCAPs, and nothing with credentials.

> The download URL above is the release **asset** (`/releases/download/<tag>/<file>`), not the release
> **page** (`/releases/tag/<tag>`) — fetching the page URL would save HTML, not the zip. If the
> uploaded asset has a different filename, use that name in both commands.
>
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
- The repo is **private**, so the plain `curl` / `Invoke-WebRequest` in step 2 saves a 9-byte
  "Not Found" file instead of the zip. Use the GitHub CLI (logged in as a collaborator) instead:
  `gh release download v2-dataset -R TheGuest55/wardwatch -p wardwatch-v2-dataset.zip`.

## Tree-classifier experiments (XGBoost + Random Forest)

Everything below runs in the analysis image. **`RUN`** means the full step-6 prefix, with the three
`-e WARDWATCH_*` provenance flags, up to and including `wardwatch-analysis:0.2.0`. Add
`--for-report` only on a clean tree (`git status` empty). Run the steps in order: each one reads the
previous step's outputs under `data/processed/v2/`.

| # | Step | Command (after `RUN`) | Writes |
|---|---|---|---|
| A | Experiment A (time split) + seed-0 models | `python -m src.models.train_tree --model xgboost --for-report` (and `--model random_forest`) | `runs/expA/`, `models/expA/` |
| B / C | Leave-one-attack-out / cross-dataset | `python -m src.models.train_tree --model xgboost --experiment B` (or `C`) | `runs/expB/`, `runs/expC/` |
| 1 | Random-split reference (proposal §3.4) | `python -m src.models.random_split --model xgboost --seeds 5 --save-predictions --for-report` | `runs/random_split/`, `runs/predictions/random_*.parquet` |
| 2 | Calibration, confidence bands, SHAP | `python -m src.models.calibrate_explain --model xgboost --for-report` | `runs/calibration/`, `models/expA/*_calibrator.joblib`, `*_triage_config.json` |
| 3 | Block-bootstrap 95% CIs | `python -m src.models.bootstrap_ci --export-a xgboost --export-a random_forest --pred A_xgboost --pred A_random_forest --pred random_xgboost --pred random_random_forest --tag xgb_rf_time_random --for-report` | `runs/bootstrap/`, `runs/predictions/A_*.parquet` |
| 4 | Report figures (PNG 300 dpi + PDF) | `python -m src.models.figures` | `figures/` |
| D | Benchmark on CICIoMT2024's own CSVs | `python -m src.models.exp_d --models xgboost random_forest --for-report` (leakage check: add `--drop IAT --seeds 1`) | `runs/expD/` |

Steps A, B/C, 1 and 2 repeat with `--model random_forest`. Step D is the long one: measured on a
12-core laptop, XGBoost with 3 seeds takes ≈ 1.6 h (most of it on the 19-class task) and RF ≈ 25 min.
The other steps take minutes.

**Using the outputs in other modules**

- *Triage / dashboard:* `from src.models.calibrate_explain import CalibratedDetector`, then
  `d = CalibratedDetector.load("xgboost")`. `d.probability(X)` returns the calibrated attack
  probability, `d.band(X)` returns 0 = confident benign, 1 = unsure, 2 = confident attack, and
  `d.explain(x)` returns the top-3 reasons in plain language.
- *Comparing another model (autoencoder, IF, LSTM, rules):* save its Experiment A test predictions
  with `src.models.predictions.save_predictions(meta, y, score, pred, "A", "<model>")`, then run
  `python -m src.models.bootstrap_ci --pred A_xgboost --pred A_<model>` for paired CIs.

**Experiment D data (not in the dataset bundle).** CIC requires a registration form, so download by
hand from <https://www.unb.ca/cic/datasets/iomt-dataset-2024.html> → `WiFI_and_MQTT/attacks/CSV`:
all 51 `train/` and all 21 `test/` files (≈ 2.2 GB) into `data/raw/ciciomt2024_csv/{train,test}/`.
The server serves **one download per login session at a time** and pauses often, so download the
files one by one rather than in parallel. `exp_d` refuses to run unless all 19 classes are present.
Memory: the full set needs ≈ 4 GB in Docker. Random Forest *without IAT* on the 6- and 19-class
tasks needs more than the default 8 GB and was not run.

**Headline results** (details and every number's run record are in each `runs/…/*.json`):

- *Random vs time split:* shuffling windows (71% of test windows have a neighbour in train; 100% of
  test attacks come from a run seen in training) does **not** raise F1: XGBoost 0.896 → 0.900,
  RF 0.918 → 0.911, CIs overlap. The time split is already in-distribution (every attack class in
  every session), so experiments B and C are the real generalisation tests.
- *Calibration:* isotonic (chosen by cross-fitting the two validation sessions) cuts XGBoost's test
  ECE from 0.0176 to 0.0013. Misses concentrate in eavesdrop, pump_cmd and spoof_vitals. A calibrated
  score plus SHAP explanation takes 33.6 ms per window (p95), inside the 50 ms budget. RF takes
  83 ms / 181 ms, over budget.
- *CIs (Experiment A, paired):* XGBoost − RF F1 = −0.021 [−0.032, −0.011], but PR-AUC is equal
  (+0.000 [−0.007, +0.008]): the two models separate attacks equally well and differ only in where
  the validation threshold lands.
- *Experiment D (CIC official split):* matches the published best (Dadkhah et al., Table 8) on 2 and
  6 classes: XGBoost macro-F1 0.983 / 0.946 and RF 0.984 / 0.949, vs 0.982 / 0.942 published. RF
  reaches 0.902 on 19 classes vs 0.907 published. **However, CIC's `IAT` column is a capture
  timestamp**: it is near-constant within each attack class and identical across train and test.
  Without it, XGBoost multiclass macro-F1 falls to 0.822 (6 classes) and 0.703 (19 classes),
  accuracy ≈ 0.80. Binary detection is robust (0.970). The published multiclass numbers are
  therefore inflated by a recording artefact, not attack behaviour.

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
