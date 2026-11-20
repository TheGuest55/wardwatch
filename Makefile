.PHONY: reproduce-A reproduce-B reproduce-lstm check-session triage replay gen up down smoke session sessions features dataset datacheck test logs clean cic dashboard-image dashboard-test dashboard

# Regenerate docker-compose.yml, broker ACLs/accounts and .env from config/ward.json.
gen:
	python3 scripts/gen_lab.py

up: gen
	python3 scripts/fetch_physionet.py
	mkdir -p data/raw/pcap data/raw/sessions
	docker compose up -d --build --remove-orphans

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f --tail=50

smoke:
	bash scripts/smoke/run_all.sh

# One session end to end (Task 3): make session ID=s01
session: gen
	bash scripts/run_session.sh "$(ID)"

# Several sessions back to back for unattended recording (Task 3): make sessions IDS="s01 s02"
sessions: gen
	bash scripts/run_sessions.sh $(IDS)

# Feature extraction / dataset / checks (Tasks 4-5) run in the analysis container.
features:
	bash scripts/analysis.sh features "$(SESSION)"

# Build a labelled dataset. Defaults target data/processed/ (v1, back-compat). For a separate
# corpus pass a split file + output dir, e.g.:
#   make dataset SPLIT=data/raw/sessions/split_v2.json OUT=data/processed/v2 FORCE=1
dataset:
	bash scripts/analysis.sh dataset \
	  $(if $(SPLIT),--split $(SPLIT),) $(if $(OUT),--out-dir $(OUT),) $(if $(FORCE),--force,)

# Data-quality + leakage report. Point at a corpus's output dir and a report path, e.g.:
#   make datacheck IN=data/processed/v2 REPORT=docs/data-report-v2.md
datacheck:
	bash scripts/analysis.sh datacheck \
	  $(if $(IN),--in-dir $(IN),) $(if $(REPORT),--out $(REPORT),)

# Full regression suite (alias of scripts/analysis.sh test): pipeline, realtime, triage, generation, dashboard. make test [TIER=generation]
test:
	bash scripts/run_tests.sh "$(or $(TIER),all)"

# Triage (offline, held-out sessions) and the live replay (Experiment E). Needs the model artefacts from
# the README pipeline (A, calibration, save_multiclass) and `make check-session` for the peer sidecars.
#   make check-session SESSIONS="n02 r01 r02 r03 r04 r05 r06 r07"
#   make triage
#   make replay SESSION=r05 [ARGS="--speed 1 --for-report"]
check-session:
	bash scripts/analysis.sh check-session $(or $(SESSIONS),r05)

triage:
	bash scripts/analysis.sh triage $(ARGS)

replay:
	@test -n "$(SESSION)" || { echo "usage: make replay SESSION=r05 [ARGS=...]"; exit 2; }
	bash scripts/analysis.sh replay $(SESSION) $(ARGS)

# Reproduce recorded results from the v2 dataset (USER_MANUAL Level 2). Output only under data/processed/v2/repro/.
#   make reproduce-A      Experiment A, seed 0: XGBoost, RF, IF, autoencoder, rule baseline
#   make reproduce-B      Experiment B, seed 0, two held-out classes (recon_scan, spoof_vitals): XGBoost, IF, rule baseline
#   make reproduce-lstm   LSTM, seed 0, k=12 hidden=32
reproduce-A:
	bash scripts/analysis.sh reproduce A

reproduce-B:
	bash scripts/analysis.sh reproduce B

reproduce-lstm:
	bash scripts/analysis.sh reproduce lstm

# CICIoMT2024 windows for Experiment C: the 19 single-attacker (DoS) PCAPs only, DDoS excluded.
# Dry run (check and list the staged files, extract nothing): bash scripts/extract_cic_dos.sh --list
cic:
	bash scripts/extract_cic_dos.sh

# Removes containers and locally built images. Never deletes data/.
clean:
	docker compose down --rmi local --remove-orphans

# Read-only results dashboard (see docker/dashboard/README.md). make dashboard [DATA=path/to/data/processed/v2]
dashboard-image:
	docker build -t wardwatch-dashboard:0.1.0 docker/dashboard

dashboard-test:
	docker run --rm --network none -v "$(CURDIR):/work:ro" -w /work -e PYTHONDONTWRITEBYTECODE=1 \
	  wardwatch-dashboard:0.1.0 python -m pytest tests/dashboard -q -p no:cacheprovider

dashboard:
	docker run --rm --name wardwatch-dashboard -p 8501:8501 \
	  --read-only --tmpfs /tmp --tmpfs /root --cap-drop ALL --security-opt no-new-privileges \
	  -v "$(CURDIR):/work:ro" -v "$(or $(DATA),$(CURDIR)/data/processed/v2):/data:ro" \
	  -e WARDWATCH_DATA_ROOT=/data wardwatch-dashboard:0.1.0
