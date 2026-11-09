.PHONY: gen up down smoke session sessions features dataset datacheck test logs clean

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

# Regression tests: generation (host, stdlib) + pipeline (analysis image). make test [TIER=generation]
test:
	bash scripts/run_tests.sh "$(or $(TIER),all)"

# Removes containers and locally built images. Never deletes data/.
clean:
	docker compose down --rmi local --remove-orphans
