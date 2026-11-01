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

## Safety

Only run this lab on a machine you control. The lab network has no route to the internet, and the attack tools are only meant to target the lab's own containers. The capture container uses host networking so it can listen on the lab's bridge.
