"""Shared device lifecycle: connect with backoff, publish on a seeded schedule, and simulate
reboots. Roles supply payloads and (optionally) subscriptions; the timing realism lives in
noise.Noise."""
import json
import time

from .client import build_client, connect
from .noise import Noise, utcnow


class Device:
    def __init__(self, role):
        import os
        self.role = role
        self.username = os.environ["MQTT_USERNAME"]
        self.bed = int(os.environ.get("BED", "0"))
        self.host_ip = os.environ.get("HOST_IP", "")
        self.noise = Noise(self.username, self.host_ip)
        self.client = build_client(self.username)
        self._bind()

    def _bind(self):
        """Attach callbacks to self.client. Overridden by roles that subscribe."""

    def start(self):
        connect(self.client)
        self.client.loop_start()

    def reboot(self):
        """Go offline for a seeded interval, then reconnect with a fresh client."""
        start = utcnow()
        self.client.loop_stop()
        try:
            self.client.disconnect()
        except Exception:
            pass
        off = self.noise.offline_seconds()
        time.sleep(off)
        self.noise.log_reboot(start, utcnow(), off)
        self.client = build_client(self.username)
        self._bind()
        self.start()

    def publish(self, topic, payload, qos=0):
        self.client.publish(topic, json.dumps(payload), qos=qos)

    def run_periodic(self, base_interval, make_payload, topic):
        """Publish make_payload() to topic every base_interval seconds (drift + jitter applied)."""
        self.start()
        next_send = time.monotonic()
        while True:
            iv = self.noise.interval(base_interval)
            if self.noise.due_for_reboot(iv):
                self.reboot()
                next_send = time.monotonic()
            self.publish(topic, make_payload())
            next_send += iv
            time.sleep(max(0.0, next_send - time.monotonic()))
