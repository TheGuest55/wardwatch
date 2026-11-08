.PHONY: gen up down smoke session sessions features dataset datacheck test logs clean

# Regenerate docker-compose.yml, broker ACLs/accounts and .env from config/ward.json.
gen:
	python3 scripts/gen_lab.py

up: gen
	python3 scripts/fetch_physionet.py
	mkdir -p data/raw/pcap data/raw/sessions
	docker compose up -d --build --remove-orphans

down:
	docker compose down

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

dataset:
	bash scripts/analysis.sh dataset

datacheck:
	bash scripts/analysis.sh datacheck

# Regression tests: generation (host, stdlib) + pipeline (analysis image). make test [TIER=generation]
test:
	bash scripts/run_tests.sh "$(or $(TIER),all)"

# Removes containers and locally built images. Never deletes data/.
clean:
	docker compose down --rmi local --remove-orphans
