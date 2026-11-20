# WardWatch user manual

Three levels, from "just look at the results" to "regenerate the data":

| Level | What you do | What you need | Time |
|---|---|---|---|
| **1. View** | open the dashboard on the published results | Docker, git, the results zip | about 10 min |
| **2. Reproduce** | retrain models from the published v2 dataset and compare with `reproduce/expected.json` (in the repository) | Docker, git, the v2 dataset zip, the analysis image (about 4 GB); **not** the results zip | about 30 min plus the image build |
| **3. Regenerate** | record new lab sessions, extract features, rebuild the dataset | Docker, hours of lab time, the BIDMC vitals download | many hours (written, not tested end to end) |

Every command is given for **bash** (Linux, macOS, Git Bash, WSL) and **PowerShell** (Windows). Commands inside a code block marked
`docker` are identical in both shells.

---

## Quick start

**Your starting point is the submission package** (no GitHub account or access is needed): a folder that holds the code and a `release/` folder with the v2 dataset zip.
The results zip is larger and is **not** in the package: download it from the shared-drive link `<RESULTS_LINK>` (Level 1 only) and save it in the `wardwatch-submission` folder.

```
wardwatch-submission/
  code/       the repository (this manual is code/USER_MANUAL.md)
  release/    wardwatch-v2-dataset.zip (the v2 dataset, Level 2)
  <RESULTS_ZIP>   the results zip, downloaded from <RESULTS_LINK> (Level 1)
```
(`<RESULTS_ZIP>` stands for the name of the results zip you downloaded; the dataset zip is `wardwatch-v2-dataset.zip`. The repository is private, so `git clone` and the GitHub release download links only work for people who
have access: those commands are in the box "If you have access to the private repository" at the end of this section.)

Run every command from the folder named in the first line of the block. **Check each zip against its hash before you use it**:
the results zip must have SHA-256 `<RESULTS_ZIP_SHA256>` and the dataset zip `e2d3ea557eb48554ac263e63dfd3be0e548e8dec8662baff531b9cfc2b14d87e` (the results hash is filled in when that zip is published; see "Downloads" under Level 1 and
Level 2). bash: `sha256sum <file>`. PowerShell: `Get-FileHash <file> -Algorithm SHA256`.

### Level 1 in 5 commands (view the results)

bash:
```bash
cd wardwatch-submission/code
sha256sum ../<RESULTS_ZIP>
unzip ../<RESULTS_ZIP> -d results
docker build -t wardwatch-dashboard:0.1.0 docker/dashboard
docker run --rm --name wardwatch-dashboard -p 8501:8501 --read-only --tmpfs /tmp --tmpfs /root --cap-drop ALL --security-opt no-new-privileges -v "$PWD:/work:ro" -v "$PWD/results:/data:ro" -e WARDWATCH_DATA_ROOT=/data wardwatch-dashboard:0.1.0
```
PowerShell:
```powershell
cd wardwatch-submission/code
Get-FileHash ../<RESULTS_ZIP> -Algorithm SHA256
Expand-Archive ../<RESULTS_ZIP> -DestinationPath results
docker build -t wardwatch-dashboard:0.1.0 docker/dashboard
docker run --rm --name wardwatch-dashboard -p 8501:8501 --read-only --tmpfs /tmp --tmpfs /root --cap-drop ALL --security-opt no-new-privileges -v "${PWD}:/work:ro" -v "${PWD}/results:/data:ro" -e WARDWATCH_DATA_ROOT=/data wardwatch-dashboard:0.1.0
```
**Expected:** the first command prints the results hash `<RESULTS_ZIP_SHA256>` (compare it). The last prints `You can now view your Streamlit app in your browser.` Open <http://localhost:8501>: the Overview page shows
a headline table with six models (XGBoost, Random Forest, autoencoder, Isolation Forest, LSTM, rule baseline). Stop with Ctrl+C.

### Level 2 in 5 commands (reproduce)

bash:
```bash
cd wardwatch-submission/code
sha256sum ../release/wardwatch-v2-dataset.zip && unzip -o ../release/wardwatch-v2-dataset.zip
docker build --pull -t wardwatch-analysis:0.2.0 services/analysis
bash scripts/analysis.sh test
bash scripts/analysis.sh reproduce A
```
PowerShell:
```powershell
cd wardwatch-submission/code
Get-FileHash ../release/wardwatch-v2-dataset.zip -Algorithm SHA256; Expand-Archive ../release/wardwatch-v2-dataset.zip -DestinationPath . -Force
docker build --pull -t wardwatch-analysis:0.2.0 services/analysis
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 sh -c "python -m unittest discover -s tests/pipeline -p 'test_*.py'; python -m unittest discover -s tests/realtime -t . -p 'test_*.py'; python -m unittest discover -s tests/triage -t . -p 'test_*.py'; PYTHONPATH=services/shared:services/devices:services/attacker:scripts python -m unittest discover -s tests/generation -p 'test_*.py'"
docker run --rm --network none --cpus 5 -e ANALYSIS_CPUS=5 -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m src.reproduce A
```
(The dataset zip extracts `data/processed/v2/...` and `data/raw/sessions/split_v2.json` into the `code` folder. The package has no `.git` folder, so the image's commit label stays `unknown`; that does not affect anything.)

**Expected.** With the code and the v2 dataset only, `bash scripts/analysis.sh test` ends with this summary (the dashboard folder needs the dashboard image from Level 1; the
skips are tests that need data that is not published, each skip message says which; section 6):

```
pipeline     ran 301   skipped 13  OK (13 skipped: data absent, see USER_MANUAL)
realtime     ran 18    skipped 1   OK (1 skipped: data absent, see USER_MANUAL)
triage       ran 44    skipped 8   OK (8 skipped: data absent, see USER_MANUAL)
generation   ran 50    skipped 0   OK
dashboard    ran 90    skipped 4   OK (4 skipped: data absent, see USER_MANUAL)
TOTAL        ran 503   skipped 26  OK
```
The PowerShell form runs the four analysis-image folders only: each unittest run ends with `OK` (with `skipped=...` where data is absent; 301 + 18 + 44 + 50 tests).

`reproduce A` retrains five models and compares **30 numbers** (test F1, PR-AUC and FPR at 90% recall, for all rows and with the attacker host excluded) with
`reproduce/expected.json`, a file that ships with the repository. Before it trains anything it checks that your `testbed_windows.parquet` is the file the results were
made on (SHA-256 in `expected.json`) and stops with `FAIL: wrong dataset` otherwise. Expected: a table in which every row ends `PASS` and the last line
`RESULT A: PASS (30 comparisons, ...)` (about 3 minutes on 5 CPUs). `reproduce B` ends `RESULT B: PASS (36 comparisons, ...)` (about 1 minute).
The results zip is **not** needed for Level 2.

<details>
<summary>If you have access to the private repository (git clone and release download commands)</summary>

bash:
```bash
git clone -c core.autocrlf=input https://github.com/TheGuest55/wardwatch.git && cd wardwatch
curl -L -o v2.zip "https://github.com/TheGuest55/wardwatch/releases/download/v2-dataset/wardwatch-v2-dataset.zip"
```
PowerShell:
```powershell
git clone -c core.autocrlf=input https://github.com/TheGuest55/wardwatch.git; cd wardwatch
Invoke-WebRequest -Uri "https://github.com/TheGuest55/wardwatch/releases/download/v2-dataset/wardwatch-v2-dataset.zip" -OutFile v2.zip
```
Then follow the Level 2 block above with `v2.zip` in place of `../release/wardwatch-v2-dataset.zip` (check the same hash); the results zip for Level 1 always comes from `<RESULTS_LINK>`. Release page of the dataset:
<https://github.com/TheGuest55/wardwatch/releases/tag/v2-dataset>. Without access they return "404 Not Found": that is
normal for a private repository; use the submission package.
</details>

---

## 1. Requirements

- **Docker** (Docker Desktop on Windows or macOS, Docker Engine on Linux) with at least **8 GB of memory** for Docker (**10 GB** for Experiment D).
  In Docker Desktop: Settings, Resources. Disk: **at least 15 GB free for Docker** (the images are about 5.3 GB: 4 GB analysis, 1.3 GB dashboard; the rest is build cache and room for the tests,
  which write a temporary 300 MB file), about 1 GB for the v2 dataset, up to 10 GB with CIC data. Confirm with `docker --version` and `docker info`.
- **Internet access for the image builds.** `docker build` of the analysis image downloads packages from Debian and PyPI, and the dashboard image from PyPI; the base image comes from Docker Hub.
  Everything after the builds runs with the network off.
- **git** is needed only for the private-repository route (it is not needed for the submission package). Keep line endings LF: clone with `-c core.autocrlf=input` (or run
  `git config --global core.autocrlf input` before cloning). Windows checkouts with CRLF break the shell scripts (`bash\r: command not found`; see Troubleshooting). When you unzip the
  submission package on Windows, use a tool that keeps the files as they are (Expand-Archive does).
- **bash** is only needed for `scripts/analysis.sh` and `make` shortcuts; PowerShell users can use the `docker run` forms given below.
- No Python is needed on the host: everything runs in the images.

## 2. Level 1: view the results

**Downloads.** The results zip is for **viewing only** (Level 1); Level 2 does not use it. It is about 316 MB and is **not** in the submission package: download it from the shared-drive link
`<RESULTS_LINK>` in your browser and save it in the `wardwatch-submission` folder (next to `code/`). Check it is the published file:

bash: `sha256sum <RESULTS_ZIP>`   PowerShell: `Get-FileHash <RESULTS_ZIP> -Algorithm SHA256`

The expected SHA-256 is `<RESULTS_ZIP_SHA256>` (to be filled in when the zip is published; `BUNDLE.json` inside the zip records the commit and build date it was made from).

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

**Downloads.** The dataset zip is `release/wardwatch-v2-dataset.zip` in the submission package (with access to the private repository: the asset of the `v2-dataset` release). Extract it **at the root of the
`code` folder**: it carries the `data/...` paths. Afterwards these must exist:

```
data/processed/v2/testbed_windows.parquet
data/processed/v2/feature_list.json
data/processed/v2/sessions/*.parquet
data/processed/v2/ground_truth/<session>/ground_truth.jsonl     (sanitised: class, run id, interval, attacker ip/mac)
data/raw/sessions/split_v2.json
```
The dataset contains no run records and no raw PCAPs. **Level 2 needs nothing else**: not the results zip and not the run records. What your run is compared
with is `reproduce/expected.json`, committed in the repository (see below).
(Release hashes to check: `e2d3ea557eb48554ac263e63dfd3be0e548e8dec8662baff531b9cfc2b14d87e` for the zip. The file inside it that matters, `testbed_windows.parquet`, must have the
SHA-256 `58b3e6b6341521d9a4a58a393679766dfa05e2b92363125f6fb5b721d87e29a6`; `reproduce` checks this itself before training.)

**Build the analysis image** (about 6 minutes, about 4 GB; the only step that needs the network):
`docker build --pull -t wardwatch-analysis:0.2.0 services/analysis` (from a git checkout you may add `--build-arg "GIT_COMMIT=$(git rev-parse HEAD)"` to label the image with the commit)

**Tests.** `bash scripts/analysis.sh test` runs every test folder and prints a per-folder and TOTAL line. See "Tests per level" for the expected counts.

**Reproduce.** Each command retrains with the existing training code, writes **only** to `data/processed/v2/repro/` (never to `runs/`), compares test F1, PR-AUC and
FPR at 90% recall (all rows and attacker host excluded: 6 numbers per run) with `reproduce/expected.json`, prints PASS or FAIL with the tolerance, and exits non-zero on FAIL.

**Where the expected numbers come from.** `reproduce/expected.json` is generated by the authors with `python -m src.reproduce --write-expected` on the full data and committed. It holds the
SHA-256 of the dataset and of the split the results were made on, and one entry per compared run: the run record's path, commit and SHA-256 (the records carry no MLflow run id, so the file hash
identifies the record exactly), the tolerance, and the headline metrics, which are exactly the numbers of the dashboard's headline columns. Before training, `reproduce` stops with `FAIL: wrong dataset`
if your `testbed_windows.parquet` has a different SHA-256. It never reads `runs/`.

| Command (bash) | PowerShell | What it retrains | Time |
|---|---|---|---|
| `bash scripts/analysis.sh reproduce A` (or `make reproduce-A`) | the PowerShell form below, with `A` | Experiment A, seed 0: XGBoost, Random Forest, Isolation Forest, autoencoder, rule baseline | about 3 min (30 comparisons) |
| `bash scripts/analysis.sh reproduce B` (or `make reproduce-B`) | `... python -m src.reproduce B` | Experiment B, seed 0, two held-out classes (recon_scan = easy, spoof_vitals = hard): XGBoost, Isolation Forest, rule baseline | about 1 min (36 comparisons) |
| `bash scripts/analysis.sh reproduce lstm` (or `make reproduce-lstm`) | `... python -m src.reproduce lstm` | the LSTM, seed 0, the configuration in expected.json (k=12, hidden 32). It labels boundary windows from the **sanitised ground truth** in the dataset zip (class, run id, interval and attacker address only; the raw files name lab accounts and are not published), or from the raw files when they are present: both give identical labels (tested) | about 2 min |

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

`bash scripts/analysis.sh test` runs five test folders. A test that needs data **skips only when that data is absent**, and says what is missing and which
level of this manual provides it; with the data present the test runs unchanged. On a checkout **without** the full data the skips are reported ("data absent")
and are not a failure; on a **full-data** checkout (raw session PCAPs and trained models present) any skip fails the run, so a full-data suite is 503 run, 0 skipped.

Measured (19 Nov, `bash scripts/analysis.sh test`, wardwatch-analysis:0.2.0 and wardwatch-dashboard:0.1.0):

| What you have | pipeline | realtime | triage | generation | dashboard | TOTAL run | skipped |
|---|---|---|---|---|---|---|---|
| fresh clone, no data | 301 run, 14 skipped | 18 run, 2 skipped | 44 run, 8 skipped | 50 run, 0 | 90 run, 4 skipped | 503 | 28 |
| plus the v2 dataset (Level 2) | 301 run, 13 skipped | 18 run, 1 skipped | 44 run, 8 skipped | 50 run, 0 | 90 run, 4 skipped | 503 | 26 |
| plus the results zip (Level 1; adds no tests) | 301 run, 13 skipped | 18 run, 1 skipped | 44 run, 8 skipped | 50 run, 0 | 90 run, 4 skipped | 503 | 26 |
| full data (the authors' checkout: raw PCAPs, models, credentials file) | 301 run, 0 skipped | 18 run, 0 skipped | 44 run, 0 skipped | 50 run, 0 | 90 run, 0 skipped | 503 | 0 |

The results zip adds no running tests: the tests that read recorded runs also need the trained models or the raw PCAPs (Level 3 material, not published). The 26
skipped tests at Level 2 need, by their skip messages: the raw session PCAPs, the raw session ground truth, the trained model files, the lab credentials file (the
bundle build tests) or CICIoMT2024's feature CSVs (section 5). Run `python -m unittest ... -v` (inside the image) to see every skip message.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `bash\r: No such file or directory` or `$'\r': command not found` | the scripts were checked out with CRLF line endings. Re-clone with `git clone -c core.autocrlf=input ...` (or `git config --global core.autocrlf input`, then `git rm --cached -r . && git reset --hard`). |
| a training run ends with `Killed` or exit code 137 | Docker ran out of memory. Raise Docker's memory (Settings, Resources) to 8 GB, or 10 GB for Experiment D. |
| `port is already allocated` or the page does not open | another process uses 8501: use `-p 8502:8501`. Inside an `sbx` sandbox, publish the port from the host (`sbx ports <name> --publish 8501:8501`). |
| `docker build` takes a long time or fills the disk | the analysis image is about 4 GB and takes about 6 minutes; free space with `docker builder prune`. |
| `No space left on device` (in a build, a test or a training run) | Docker's own disk is full: keep at least 15 GB free for Docker (Docker Desktop: Settings, Resources, Disk image size). `docker builder prune` removes the build cache; `docker system df` shows what uses the space. |
| the tests report data files not found | the v2 dataset is not at the path in Level 2; the layout must match exactly. |
| `reproduce` prints `FAIL: wrong dataset` | your `testbed_windows.parquet` is not the file the expected numbers were made on: download the v2 dataset release again and extract it at the repository root. |
| `reproduce lstm` stops with `cannot run the LSTM target` | the sanitised ground truth `data/processed/v2/ground_truth/<session>/ground_truth.jsonl` is missing: it is part of the v2 dataset zip, so extract that zip again at the root of the `code` folder (an older copy of the zip lacks it). |
| the tests print `skipped` | normal without the full data: each skip names what is missing and the level that provides it (section 6). |
| `scripts/analysis.sh` says `no such image` | the script builds `wardwatch-analysis:0.2.0` itself the first time it is missing; or build it as in Level 2. |
| a worktree or clone shows no `data/` | `data/` is not in git. Download the releases as above. |

## 8. Where the numbers come from

Every number in the report and the dashboard comes from a **run record**: a JSON file under `data/processed/v2/runs/` written by the training code with `tracking.build_run_record`.
A record that feeds the report has `for_report: true`; the headline tables use only those, for experiments A, B, C and D, and exclude smoke runs, latency timings, sensitivity
analyses and records without a seed. Each record carries its provenance: `seed`, the split file and its SHA-256, the dataset SHA-256, the git commit, whether the tree was clean
(`git_dirty`) and the analysis image digest. The dashboard never recomputes a metric: curves and confusion matrices are drawn from saved predictions and labelled as derived. The
results bundle contains exactly the files the dashboard reads (`BUNDLE.json` records its commit, `MANIFEST.sha256` every file's hash).
