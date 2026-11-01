# environment.md — WardWatch: Environment Setup

Project: ICT3214 Coursework 2 — *Medical IoT digital-twin intrusion detection and forensic triage*.
This file has two parts:

- **Part A — Host setup.** Done by the human, once, on the real machine. Claude Code cannot do these steps from inside the sandbox.
- **Part B — Instructions for Claude Code.** Everything Claude Code builds in this phase, the rules it follows, and when it must stop and ask.

Scope of this phase: a working, isolated lab environment that passes the smoke test, plus the repo skeleton, broker, one simulated device, capture, and the report-notes system. Attacks, feature extraction and ML come in later briefs.

---

## Part A — Host setup (human only)

### A1. Make a dedicated project folder

The sandbox mounts the project folder **read-write**, and the agent can read, write and delete anything inside it, including hidden files, build scripts and Git hooks. That folder is the entire blast radius, so:

- Create a **new, empty** folder used for nothing else, e.g. `~/labs/wardwatch`.
- **Never** run `sbx` from your home directory, Desktop, Downloads, or any folder with other work in it. `sbx run` mounts whatever directory you are in.
- **Not** inside OneDrive, iCloud Drive, Dropbox or Google Drive. Docker warns against cloud-synced folders (slow), and a sync client would also propagate any deletion everywhere.
- Free disk space: about 100 GB (captures, CICIoMT2024, Docker images).

```bash
mkdir -p ~/labs/wardwatch && cd ~/labs/wardwatch
git init
mkdir -p docs
# copy this file in as docs/environment.md
```

Create a **private** GitHub repo and add it as the remote. **Push only from the host**, never from the sandbox. The remote is your backup that the agent cannot touch.

### A2. Install Docker Sandboxes

```bash
# macOS
brew install docker/tap/sbx
# Windows
winget install Docker.sbx
sbx login
```

MicroVM isolation is supported on macOS and Windows. If your machine runs Linux, check the current install page first; otherwise use a full Ubuntu VM instead.

### A3. Lock down the network policy

On first start, `sbx` asks you to pick a default network policy. Pick the most restrictive option offered. Then review what is allowed:

```bash
sbx policy ls
```

The defaults can include broad wildcards (for example `*.googleapis.com`) that cover far more than AI APIs. Remove anything this project doesn't need (check `sbx policy rm --help` for the exact syntax on your version).

Allow only what the build needs:

```bash
# Python packages
sbx policy allow network pypi.org
sbx policy allow network files.pythonhosted.org
sbx policy allow network download.pytorch.org
# Docker Hub image pulls
sbx policy allow network registry-1.docker.io
sbx policy allow network auth.docker.io
sbx policy allow network production.cloudflare.docker.com
# Source / release downloads
sbx policy allow network github.com
sbx policy allow network objects.githubusercontent.com
# Vital-sign data
sbx policy allow network physionet.org
```

Anything else the agent tries to reach will show up as an approval request rather than silently succeeding. Review them with `sbx policy approval ls` and approve only what you recognise. Debian/Ubuntu package mirrors used inside Dockerfiles (e.g. `deb.debian.org`) will appear this way the first time.

**Download CICIoMT2024 on the host**, not from the sandbox, straight into `~/labs/wardwatch/data/raw/ciciomt2024/`. Only the Wi-Fi/MQTT part is needed.

### A4. Credentials

- Sign Claude Code in with `/login` inside the sandbox. The host-side proxy injects credentials into outbound requests, so raw values never enter the VM.
- **Do not** give the sandbox GitHub tokens, SSH keys, or cloud credentials. It doesn't need them: you push from the host.
- **Do not** put secrets in files inside the project folder. The agent can read everything there. The only "secrets" in this project are throwaway MQTT test passwords, which are fine in `.env` (gitignored).

### A5. Create the sandbox

```bash
cd ~/labs/wardwatch
sbx create --name wardwatch --cpus 6 --memory 12g --skills=off \
  --publish 3000:3000 --publish 8501:8501 claude .
sbx run --name wardwatch
```

- `--cpus` / `--memory`: cap what the lab can take from your machine. Leave at least 2 cores and 4 GB for yourself. Adjust to your hardware.
- `--skills=off`: no shared skills store mounted, so nothing written here can affect other sandboxes.
- `--publish`: 3000 = Grafana, 8501 = Streamlit, reachable from your browser at `localhost`. Add more later with `sbx ports wardwatch --publish <host>:<sandbox>`.
- The sandbox's private Docker storage defaults to 10 GB. If image builds run out of space, recreate the sandbox with a larger `DOCKER_SANDBOXES_DOCKER_SIZE` (captures and datasets live in the project folder, not there).
- Do **not** register local MCP servers for this sandbox. Local MCP servers run on the host with host permissions, outside the sandbox boundary.

Then tell Claude Code: *"Read docs/environment.md and carry out Part B."*

### A6. Daily hygiene (2 minutes)

After each working session:

```bash
cd ~/labs/wardwatch
git status && git diff --stat            # what changed
ls .git/hooks | grep -v '\.sample$'      # hooks don't show in git diff; this should print nothing
git log --oneline -5
git push                                  # from the host
rsync -a data/raw/ ~/labs/wardwatch-backup/raw/   # back up captures outside the mounted folder
```

Also skim `docs/REPORT_NOTES.md` for anything the agent changed, dropped or worked around.

Rules for you:

- Only run project code **inside the sandbox** (`sbx exec -it wardwatch bash`). If you ever run it on the host, review `Makefile`, `scripts/`, `.vscode/`, `.claude/` and `.git/hooks` first, since the agent can edit them.
- Never run `claude` directly on the host inside this folder.
- To wipe the lab environment: `sbx rm wardwatch` deletes the VM, its images and packages. Files in the project folder stay on your host.

---

## Part B — Instructions for Claude Code

You are running inside a Docker Sandboxes microVM with your own Docker Engine and sudo. The project folder is a live read-write mount of the human's real folder. Work only inside it.

### B1. Hard rules

1. **Stay inside the project folder.** Never write outside it except for normal tool caches inside the VM.
2. **Never delete or overwrite anything under `data/raw/`** without asking. Captures take hours to regenerate.
3. **Never touch `.git/hooks/`, `.git/config` or Git history** (`rebase`, `reset --hard`, `push --force`, `filter-branch`). Normal `git add` / `git commit` is fine. Do not configure Git remotes or credentials.
4. **Lab networking:**
   - All lab services sit on one Docker network, `ward_net`, with `internal: true` (no route out, not even to the internet).
   - No service uses `privileged: true`.
   - No service uses `network_mode: host`, **except the capture container**, which needs to listen on the bridge interface. Inside this sandbox, "host" is the microVM, not the human's machine.
   - Add `NET_ADMIN` / `NET_RAW` capabilities only to the attacker and capture containers.
   - Every service has `cpus` and `mem_limit` set.
5. **Rate-limit every traffic generator.** No unbounded floods: hping3 floods use an explicit interval (e.g. `-i u1000` ≈ 1,000 packets/s) and a timeout.
6. **Pin versions:** image tags (no `latest`), Python packages in `requirements.txt` with exact versions, Python 3.11.
7. **Never fake success.** If something doesn't work, don't simplify the task, loosen a check, or skip a step quietly. Log it in `docs/REPORT_NOTES.md` (see B5) and, if it changes the plan, stop and ask.
8. **Secrets:** throwaway MQTT test passwords go in `.env` (gitignored) with a committed `.env.example`. Nothing else secret belongs in the repo.
9. **Commit** after each completed task, small and descriptive (`env: add mosquitto with ACLs`).

### B2. Stop and ask the human when

- any smoke test in B4 fails and you would need to change the architecture (not just fix a bug);
- you need a network domain that isn't allowed (they approve it on the host);
- you want to delete or move anything in `data/`;
- a planned component turns out to be impossible in this environment;
- disk usage of `data/` passes 50 GB.

### B3. Repo layout to create

```
wardwatch/
├── README.md                 # what this is, quick start
├── .gitignore                # data/, .env, *.pcap, *.pcapng, .venv/, __pycache__/, mlruns/, *.parquet
├── .env.example
├── Makefile                  # make up / down / smoke / capture / logs / clean (never deletes data/raw)
├── docker-compose.yml
├── requirements.txt          # analysis environment, pinned
├── services/
│   ├── broker/               # mosquitto.conf, acl, passwd generation script
│   ├── devices/              # simulator package + Dockerfile (one image, role chosen by env var)
│   ├── capture/              # Dockerfile + rotating tcpdump entrypoint
│   └── attacker/             # Dockerfile with nmap, hping3, dsniff (arpspoof), paho-mqtt; NO attacks yet
├── scripts/
│   └── smoke/                # one script per smoke test, plus run_all.sh
├── src/                      # analysis code (later phases)
├── notebooks/                # exploration only; anything in the report must come from src/
├── docs/
│   ├── environment.md        # this file
│   ├── environment-status.md # smoke-test results table
│   └── REPORT_NOTES.md       # running log for the report (B5)
└── data/                     # gitignored
    ├── raw/                  # pcaps, orchestrator logs, ciciomt2024/
    ├── interim/
    └── processed/
```

### B4. Tasks for this phase, in order

**Task 1 — Skeleton.** Create the layout above, `.gitignore`, `.env.example`, `README.md` stub, `docs/REPORT_NOTES.md` from the template in B5. Commit.

**Task 2 — Network and broker.**

- `ward_net`: bridge driver, `internal: true`, subnet `172.28.10.0/24`, bridge interface named `br-ward` (driver option `com.docker.network.bridge.name`). Fixed IPs make labelling exact later:

  | Service | IP |
  |---|---|
  | broker (Mosquitto 2.x) | 172.28.10.10 |
  | nurse-station | 172.28.10.20 |
  | patient-monitor | 172.28.10.31 |
  | infusion-pump | 172.28.10.32 |
  | wearable | 172.28.10.33 |
  | attacker | 172.28.10.66 |

- Mosquitto: `allow_anonymous false`, one username/password per device, ACLs so each device can only publish to its own topics (`ward1/bed<N>/<device>/...`) and only the nurse station can publish to `ward1/+/pump/cmd`. Logging to stdout with connect/disconnect events.

**Task 3 — One device simulator.** A Python package (paho-mqtt) runnable as any role via environment variables. For this phase, implement the **patient monitor** only:

- publishes heart rate, SpO2 and blood pressure about once per second, JSON payloads with a device timestamp;
- values replayed from PhysioNet BIDMC if downloadable through the allowlist, otherwise a clearly-labelled synthetic generator, with the choice logged in REPORT_NOTES;
- reconnect with backoff; realism hooks stubbed (jitter, occasional reboot) but off by default.

Add the nurse-station subscriber that logs everything it receives. Commit.

**Task 4 — Capture.** Capture container with `network_mode: host`, `NET_RAW` + `NET_ADMIN`, running `tcpdump -i br-ward` into rotating files under `data/raw/pcap/` (e.g. 100 MB per file), filenames with UTC timestamps. Commit.

**Task 5 — Attacker image (tools only).** Build the attacker container with nmap, hping3, arpspoof (dsniff package), tcpdump and paho-mqtt. No attack scripts yet. Commit.

**Task 6 — Smoke tests.** One script each in `scripts/smoke/`, plus `run_all.sh` / `make smoke`. Record PASS/FAIL, the exact command, and evidence (counts, output lines) in `docs/environment-status.md`.

| # | Test | Pass criteria |
|---|---|---|
| S1 | Docker works in the sandbox | `docker run --rm hello-world` succeeds |
| S2 | Lab network is sealed | from a container on `ward_net`, `curl -m 5 https://example.com` **fails** |
| S3 | MQTT end to end | nurse station receives patient-monitor messages for 60 s |
| S4 | Broker auth and ACLs | wrong password is rejected; monitor publishing to the pump command topic is denied |
| S5 | Capture sees bridge traffic | a 60 s pcap contains MQTT packets between monitor (.31) and broker (.10) (count with tshark or scapy) |
| S6 | Rate-limited flood | 10 s hping3 SYN flood from attacker to broker at ~1,000 pps appears in the pcap; sandbox stays responsive; `docker stats` shows limits respected |
| S7 | ARP spoofing works | arpspoof from attacker between monitor and broker changes the monitor's ARP entry for .10 to the attacker's MAC (`ip neigh`), visible in the pcap; entries restored afterwards |
| S8 | Files reach the host | pcap written under `data/raw/` is visible from the host (ask the human to confirm with `ls`) |
| S9 | Port publishing | a test server on sandbox port 8501 loads in the human's browser at `http://localhost:8501` (ask the human to confirm) |

If S5, S6 or S7 fails after reasonable debugging, **stop and report**: the fallback is a full VM, and that is the human's decision.

**Task 7 — Wrap-up.** Update `README.md` quick start (`make up`, `make smoke`, `make down`), finish `docs/environment-status.md`, add a REPORT_NOTES entry summarising the phase, commit, and tell the human what to review.

### B5. Report notes — `docs/REPORT_NOTES.md`

This file feeds the final report. The report is limited to **8 pages excluding references and appendix**, so notes should be factual and short; detail goes to the appendix.

Rules:

- Append an entry for **every** decision, success, failure, workaround, and number that might appear in the report. Failures matter as much as successes.
- Never edit or delete past entries. Corrections are new entries that reference the old one.
- Every number must say where it came from (script, command, file). No number without a source.
- Tag each entry with the report section it belongs to.

Template (create the file with this header, then append entries):

```markdown
# REPORT_NOTES

Report limit: 8 pages excl. references and appendix.
Section tags: [DATA] dataset & testbed · [PREP] pre-processing · [MODEL] models · [EVAL] evaluation & results ·
[FORENSICS] triage & incidents · [RT] real-time & dashboard · [LIMIT] limitations · [APPENDIX] detail for appendix ·
[ENV] environment (usually appendix / user manual)

## Page budget (target)
| Section | Pages |
|---|---|
| Title, abstract, introduction | 1 |
| Related work | 1 |
| Dataset & testbed, system diagram | 1.5 |
| Pre-processing & features | 0.5 |
| Models & evaluation design | 1 |
| Results | 2 |
| Discussion, limitations, conclusion | 1 |

## Log

### 2026-11-02 · [ENV] · decision
**What:** …
**Why:** …
**Evidence:** command / file / output
**Report impact:** one line on where this goes (main text or appendix)
```

Entry types: `decision`, `success`, `failure`, `workaround`, `metric`, `deviation` (anything that differs from the proposal).

### B6. Definition of done for this phase

- `make up` brings up broker, patient monitor, nurse station, capture and an idle attacker; `make down` stops them without touching `data/`.
- S1–S9 all PASS in `docs/environment-status.md`, or failures are reported to the human with evidence.
- `docs/REPORT_NOTES.md` has an entry for each task.
- Everything committed; `git status` clean.
