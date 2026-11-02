FROM python:3.11.18-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends nmap hping3 dsniff tcpdump iproute2 curl \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir paho-mqtt==2.1.0 pyyaml==6.0.2
COPY orchestrator /app/orchestrator
WORKDIR /app
CMD ["sleep", "infinity"]
