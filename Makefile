.PHONY: up down smoke capture logs clean

up:
	sh scripts/fetch_bidmc.sh
	mkdir -p data/raw/pcap
	docker compose up -d --build

down:
	docker compose down

smoke:
	bash scripts/smoke/run_all.sh

capture:
	docker compose up -d capture

logs:
	docker compose logs -f --tail=50

# Removes containers and locally built images. Never deletes data/.
clean:
	docker compose down --rmi local --remove-orphans
