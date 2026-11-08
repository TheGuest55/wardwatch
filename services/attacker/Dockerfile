# Build context is ./services (set in the generated compose) so this image can COPY the shared
# payload builder that the devices also use (review B2 single source of truth).
FROM python:3.11.18-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends nmap hping3 dsniff tcpdump iproute2 curl \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir paho-mqtt==2.1.0
COPY shared/payloads.py /app/payloads.py
COPY attacker/orchestrator /app/orchestrator
WORKDIR /app
CMD ["sleep", "infinity"]
