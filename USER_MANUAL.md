# WardWatch user manual

Three levels, from "just look at the results" to "regenerate the data":

| Level | What you do | What you need | Time |
|---|---|---|---|
| **1. View** | open the dashboard on the published results | Docker, git, the results zip | about 10 min |
| **2. Reproduce** | retrain models from the published v2 dataset and compare with the recorded results | Docker, git, the v2 dataset zip, the analysis image (about 4 GB) | about 30 min plus the image build |
| **3. Regenerate** | record new lab sessions, extract features, rebuild the dataset | Docker, hours of lab time, the BIDMC vitals download | many hours (written, not tested end to end) |

Every command is given for **bash** (Linux, macOS, Git Bash, WSL) and **PowerShell** (Windows). Commands inside a code block marked
`docker` are identical in both shells.

---

## Quick start

> Replace the `<...>` placeholders with the names on the GitHub release pages (see "Downloads" under each level). Run everything from the
> repository root. The results and dataset downloads are release assets of <https://github.com/TheGuest55/wardwatch/releases>
> (tags `results-v1` and `v2-dataset`).

### Level 1 in 5 commands (view the results)

bash:
```bash
git clone -c core.autocrlf=input https://github.com/TheGuest55/wardwatch.git && cd wardwatch
curl -L -o results.zip "https://github.com/TheGuest55/wardwatch/releases/download/results-v1/<RESULTS_ASSET>.zip"
unzip results.zip -d results
docker build -t wardwatch-dashboard:0.1.0 docker/dashboard
docker run --rm --name wardwatch-dashboard -p 8501:8501 --read-only --tmpfs /tmp --tmpfs /root --cap-drop ALL --security-opt no-new-privileges -v "$PWD:/work:ro" -v "$PWD/results:/data:ro" -e WARDWATCH_DATA_ROOT=/data wardwatch-dashboard:0.1.0
```
PowerShell:
```powershell
git clone -c core.autocrlf=input https://github.com/TheGuest55/wardwatch.git; cd wardwatch
Invoke-WebRequest -Uri "https://github.com/TheGuest55/wardwatch/releases/download/results-v1/<RESULTS_ASSET>.zip" -OutFile results.zip
Expand-Archive results.zip -DestinationPath results
docker build -t wardwatch-dashboard:0.1.0 docker/dashboard
docker run --rm --name wardwatch-dashboard -p 8501:8501 --read-only --tmpfs /tmp --tmpfs /root --cap-drop ALL --security-opt no-new-privileges -v "${PWD}:/work:ro" -v "${PWD}/results:/data:ro" -e WARDWATCH_DATA_ROOT=/data wardwatch-dashboard:0.1.0
```
**Expected:** the container prints `You can now view your Streamlit app in your browser.` Open <http://localhost:8501>: the Overview page shows
a headline table with six models (XGBoost, Random Forest, autoencoder, Isolation Forest, LSTM, rule baseline). Stop with Ctrl+C.

### Level 2 in 5 commands (reproduce)

bash:
```bash
git clone -c core.autocrlf=input https://github.com/TheGuest55/wardwatch.git && cd wardwatch
curl -L -o v2.zip "https://github.com/TheGuest55/wardwatch/releases/download/v2-dataset/wardwatch-v2-dataset.zip" && unzip -o v2.zip
docker build --pull --build-arg "GIT_COMMIT=$(git rev-parse HEAD)" -t wardwatch-analysis:0.2.0 services/analysis
bash scripts/analysis.sh test
bash scripts/analysis.sh reproduce A
```
PowerShell:
```powershell
git clone -c core.autocrlf=input https://github.com/TheGuest55/wardwatch.git; cd wardwatch
Invoke-WebRequest -Uri "https://github.com/TheGuest55/wardwatch/releases/download/v2-dataset/wardwatch-v2-dataset.zip" -OutFile v2.zip; Expand-Archive v2.zip -DestinationPath . -Force
docker build --pull --build-arg "GIT_COMMIT=$(git rev-parse HEAD)" -t wardwatch-analysis:0.2.0 services/analysis
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m unittest discover -s tests/pipeline -p "test_*.py"
docker run --rm --network none --cpus 5 -e ANALYSIS_CPUS=5 -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m src.reproduce A
```
**Expected:** the tests end with `OK` (the analysis tests that need the full data set are marked as skipped on a fresh clone with only the
dataset; see "Tests per level"). The reproduce command prints a table of recorded versus reproduced F1 and ends with
`RESULT A: PASS (10 comparisons, ...)`. Reproduction compares against the recorded results in `data/processed/v2/runs/`, which the results zip (Level 1) provides:
unzip the results zip's `runs/` into `data/processed/v2/runs/` first (see Level 2).

---

## 1. Requirements

- **Docker** (Docker Desktop on Windows or macOS, Docker Engine on Linux) with at least **8 GB of memory** for Docker (**10 GB** for Experiment D).
  In Docker Desktop: Settings, Resources. Disk: about 4 GB for the analysis image, 1 GB for the dashboard image, about 1 GB for the v2 dataset, up to
  10 GB with CIC data. Confirm with `docker --version` and `docker info`.
- **git**, with line endings kept as LF: clone with `-c core.autocrlf=input` as above (or run `git config --global core.autocrlf input` before cloning). Windows
  checkouts with CRLF break the shell scripts (`bash\r: command not found`; see Troubleshooting).
- **bash** is only needed for `scripts/analysis.sh` and `make` shortcuts; PowerShell users can use the `docker run` forms given below.
- No Python is needed on the host: everything runs in the images.

## 2. Level 1: view the results

**Downloads.** Go to <https://github.com/TheGuest55/wardwatch/releases/tag/results-v1> and download the asset `<RESULTS_ASSET>.zip`
(about 316 MB). Check it is the published file:

bash: `sha256sum results.zip`   PowerShell: `Get-FileHash results.zip -Algorithm SHA256`

The expected SHA-256 is printed on the release page. (To be filled in once the release check is done: `<RESULTS_SHA256>`.)

**Unzip, build, run.** The zip holds `runs/`, `live/incidents.jsonl`, `testbed_windows.parquet`, `MANIFEST.sha256` and `BUNDLE.json`; unzip it into a folder called
`results`. Verify the files against the manifest (bash): `( cd results && sha256sum -c MANIFEST.sha256 )`. Build the dashboard image once
(`docker build -t wardwatch-dashboard:0.1.0 docker/dashboard`, about 1 GB) and run it as in the Quick start. The dashboard only reads: the repo and the data are
mounted read-only, there is no network access and nothing is recomputed.

**Port 8501 busy, or inside the `sbx` sandbox.** Use another host port (`-p 8502:8501`) and open that one. In an `sbx` sandbox the port must also be published from the
host: `sbx ports <sandbox-name> --publish 8501:8501`.

**What each page shows** (the sidebar switches every table between "All rows" and "Attacker host excluded"):

| Page | Shows |
|---|---|
| Overview | dataset summary, one headline row per model (A F1, PR-AUC, FPR; B and C scores), the selection rule and its winners (XGBoost as classifier, Isolation Forest as anomaly detector) |
| Model comparison | Experiments A, B (leave one attack out), C (cross-dataset), the frame-size sensitivity, Experiment D (indicative, CIC's own features), bootstrap confidence intervals |
| Curves and confusion | precision-recall, ROC and confusion matrices derived from saved predictions (the multiclass matrix is "derived, seed 0 only" for the trees) |
| Calibration | reliability diagrams and ECE for XGBoost and Random Forest |
| Explanations | LSTM occlusion, TreeSHAP and autoencoder reconstruction-error explanations |
| Incidents | the 184 incidents of the two replayed sessions, with a session filter, timeline, top features and the evidence hash |
| Triage and Experiment E | the six-cell verdict tables, grouping accuracy and its diagnosis, withheld attacks, and the real-time replay latency |

**"Evidence file not included in this bundle".** Each incident points at a PCAP slice and its SHA-256. The slices hold the lab's plaintext MQTT credentials, so they
are never published; the page shows the recorded hash and this message instead. This is expected, not an error.

## 3. Level 2: reproduce the results from the v2 dataset

**Downloads.** From <https://github.com/TheGuest55/wardwatch/releases/tag/v2-dataset> download `<DATASET_ASSET>.zip` (the Quick start uses the asset name
`wardwatch-v2-dataset.zip`) and extract it **at the repository root**: it carries the `data/...` paths. Afterwards these must exist:

```
data/processed/v2/testbed_windows.parquet
data/processed/v2/feature_list.json
data/processed/v2/sessions/*.parquet
data/raw/sessions/split_v2.json
```
The dataset contains no run records and no raw PCAPs. The recorded results to compare with come from the results zip: extract its `runs/` folder to
`data/processed/v2/runs/`.
(Release hashes to check: `<DATASET_SHA256>` for the zip; to be filled in once the release check is done.)

**Build the analysis image** (about 6 minutes, about 4 GB; the only step that needs the network):
`docker build --pull --build-arg "GIT_COMMIT=$(git rev-parse HEAD)" -t wardwatch-analysis:0.2.0 services/analysis`

**Tests.** `bash scripts/analysis.sh test` runs every test folder and prints a per-folder and TOTAL line. See "Tests per level" for the expected counts.

**Reproduce.** Each command retrains with the existing training code, writes **only** to `data/processed/v2/repro/` (never to `runs/`), compares test F1 (all rows and
attacker host excluded) with the recorded run, prints PASS or FAIL with the tolerance, and exits non-zero on FAIL.

| Command (bash) | PowerShell | What it retrains | Time |
|---|---|---|---|
| `bash scripts/analysis.sh reproduce A` (or `make reproduce-A`) | `docker run ... python -m src.reproduce A` | Experiment A, seed 0: XGBoost, Random Forest, Isolation Forest, autoencoder, rule baseline | about 2 min |
| `bash scripts/analysis.sh reproduce B` (or `make reproduce-B`) | `... python -m src.reproduce B` | Experiment B, seed 0, two held-out classes (recon_scan = easy, spoof_vitals = hard): XGBoost, Isolation Forest, rule baseline | about 1 min |
| `bash scripts/analysis.sh reproduce lstm` (or `make reproduce-lstm`) | `... python -m src.reproduce lstm` | the LSTM, seed 0, the selected configuration (k=12, hidden 32) | about 2 min |

PowerShell form of any of them (replace `A` by `B` or `lstm`):
```powershell
docker run --rm --network none --cpus 5 -e ANALYSIS_CPUS=5 -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m src.reproduce A
```
**Times** were measured on one machine with 5 CPUs (Docker `--cpus 5`): A 106-114 s, B 57-62 s, LSTM 105-114 s.

**Tolerance.** The pass bound is an absolute difference of 0.005 in test F1 for the tree models, Isolation Forest, autoencoder and rule baseline, and 0.02 for the
LSTM. It was set from a measurement: reproducing each target twice on one machine gave results that were identical to each other and to the recorded run (every one of
the 424 compared metrics of A, 492 of B and 46-50 of the LSTM, maximum difference 0). The bounds are therefore safety margins for other hardware (about the seed-to-seed
standard deviation of those models, 0.002-0.014, and for the LSTM about its 0.011; CPU torch is not guaranteed to be repeatable across CPUs). A FAIL is a finding to
report, never a number to loosen.

## 4. Level 3: regenerate the data (written from the Makefile and scripts; not simulated)

Level 3 needs the lab (`docker compose`), captures real-time traffic for hours and a network connection for the BIDMC vitals download. It is documented
here from the code; it has not been re-run end to end for this manual.

1. **Generate the lab files** from `config/ward.json`: `make gen` (writes docker-compose.yml, broker ACLs and accounts, `.env`).
2. **Bring the ward up and download the BIDMC vitals** (PhysioNet records 01, 05, 15 and 41 into `data/raw/physionet/`, the only network use): `make up`. Check the ward:
   `make smoke` runs the S1-S7 checks.
3. **Record a session** end to end (pre-roll normal traffic, the scripted attacks with ground-truth timing, post-roll, manifest with SHA-256 of every file):
   `make session ID=<id>` or several back to back, `make sessions IDS="<id1> <id2>"`. Ids starting with `t` or `s00` are short dry runs excluded from the dataset; a session
   never overwrites an existing one, and the runner refuses to start if `data/raw/pcap` changed in the last 5 minutes (another capture may be running).
   A full attack session takes about 80-90 minutes.
4. **Extract features** for a session in the analysis container: `make features SESSION=<id>` (one row per host per 5 s window, labelled from the ground truth).
5. **Build the dataset** with the v2 split: `make dataset SPLIT=data/raw/sessions/split_v2.json OUT=data/processed/v2 FORCE=1`, then check it:
   `make datacheck IN=data/processed/v2 REPORT=datacheck-v2.md`.

PowerShell users: `make` needs a Windows port; use WSL or Git Bash for this level, or run the underlying commands from the Makefile.

## 5. Experiments C, D and E need external data or extra steps

- **C (cross-dataset, testbed to CICIoMT2024)** needs the 19 single-attacker (DoS) PCAPs of the CICIoMT2024 Wi-Fi/MQTT set. They are not redistributed:
  register on the University of New Brunswick CIC portal (the CICIoMT2024 dataset page; `<CIC_PORTAL_LINK>`, to be confirmed: the files were served from cicresearch.ca after registration), download the Wi-Fi/MQTT PCAPs into
  `data/raw/ciciomt2024/WiFi_and_MQTT/attacks/pcap/`, and run `make cic` (extracts only the 19 DoS files; the DDoS files are excluded because several attackers cannot be
  labelled from one attacker address). Hashes of the files used are in `data/processed/cic/pcap_sha256.txt` once extracted.
- **D (indicative benchmark on CIC's own CSVs)** needs the 72 feature CSVs under `data/raw/ciciomt2024/WiFi_and_MQTT/attacks/csv/{train,test}/` (51 and 21 files). Run
  `python -m src.models.exp_d --models xgboost --for-report` in the analysis image with at least 10 GB of Docker memory (a full XGBoost run is several hours on 5 CPUs;
  `--train-subsample 0.25` is the documented out-of-memory fallback).
- **E (replay of a held-out session through the deployed pair)** needs the raw session PCAPs, which are not published; it cannot be reproduced from the release.

## 6. Tests per level

(To be filled in from the measured skip counts: step 4.)

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `bash\r: No such file or directory` or `$'\r': command not found` | the scripts were checked out with CRLF line endings. Re-clone with `git clone -c core.autocrlf=input ...` (or `git config --global core.autocrlf input`, then `git rm --cached -r . && git reset --hard`). |
| a training run ends with `Killed` or exit code 137 | Docker ran out of memory. Raise Docker's memory (Settings, Resources) to 8 GB, or 10 GB for Experiment D. |
| `port is already allocated` or the page does not open | another process uses 8501: use `-p 8502:8501`. Inside an `sbx` sandbox, publish the port from the host (`sbx ports <name> --publish 8501:8501`). |
| `docker build` takes a long time or fills the disk | the analysis image is about 4 GB and takes about 6 minutes; free space with `docker builder prune`. |
| the tests report data files not found | the v2 dataset or the results `runs/` folder is not at the path in Level 2; the layout must match exactly. |
| `scripts/analysis.sh` says `no such image` | the script builds `wardwatch-analysis:0.2.0` itself the first time it is missing; or build it as in Level 2. |
| a worktree or clone shows no `data/` | `data/` is not in git. Download the releases as above. |

## 8. Where the numbers come from

Every number in the report and the dashboard comes from a **run record**: a JSON file under `data/processed/v2/runs/` written by the training code with `tracking.build_run_record`.
A record that feeds the report has `for_report: true`; the headline tables use only those, for experiments A, B, C and D, and exclude smoke runs, latency timings, sensitivity
analyses and records without a seed. Each record carries its provenance: `seed`, the split file and its SHA-256, the dataset SHA-256, the git commit, whether the tree was clean
(`git_dirty`) and the analysis image digest. The dashboard never recomputes a metric: curves and confusion matrices are drawn from saved predictions and labelled as derived. The
results bundle contains exactly the files the dashboard reads (`BUNDLE.json` records its commit, `MANIFEST.sha256` every file's hash).
