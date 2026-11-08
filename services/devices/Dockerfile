# Build context is ./services (set in the generated compose) so this image can COPY the shared
# payload builder that the attacker also uses (review B2 single source of truth).
FROM python:3.11.18-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 && rm -rf /var/lib/apt/lists/*
COPY devices/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/payloads.py /app/payloads.py
COPY devices/wardwatch_devices /app/wardwatch_devices
WORKDIR /app
CMD ["python", "-u", "-m", "wardwatch_devices"]
