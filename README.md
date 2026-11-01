# WardWatch

ICT3214 Coursework 2: medical IoT digital-twin intrusion detection and forensic triage.

An isolated Docker lab with a simulated hospital ward (MQTT broker, devices, nurse station), traffic capture and an attacker container.

## Status

**In progress: the commands below don't work yet.** The lab services, `docker-compose.yml` and `Makefile` haven't been added. This notice will be removed once the smoke tests pass.

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

With `make` installed, you can use `make up`, `make smoke` and `make down` instead.

Captured traffic is written to `data/raw/pcap/`.

## Safety

Only run this lab on a machine you control. The lab network has no route to the internet, and the attack tools are only meant to target the lab's own containers. The capture container uses host networking so it can listen on the lab's bridge.
