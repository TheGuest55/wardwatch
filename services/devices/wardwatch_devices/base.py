"""Shared device lifecycle: connect with backoff, publish on a seeded schedule, and simulate
reboots. Roles supply payloads and (optionally) subscriptions; the timing realism lives in
noise.Noise."""
import json
import time

from .client import build_client, connect
from .noise import Noise, utcnow


def _reschedule(noise):
    """Monotonic deadline for the next quick reconnect, or None if disabled."""
    rc = noise.next_reconnect_in()
    return time.monotonic() + rc if rc else None


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

    def reconnect(self):
        """Quick reconnect (not an outage): drop and re-establish the session so a fresh CONNECT
        (and any subscriptions) reappear on the wire. Enables in-path credential capture (review M8)
        and keeps wildcard SUBSCRIBE recurring. Logged as a benign 'reconnect' event."""
        self.client.loop_stop()
        try:
            self.client.disconnect()
        except Exception:
            pass
        self.client = build_client(self.username)
        self._bind()
        self.start()
        self.noise.log_event("reconnect")

    def publish(self, topic, payload, qos=0):
        self.client.publish(topic, json.dumps(payload), qos=qos)

    def run_periodic(self, base_interval, make_payload, topic):
        """Publish make_payload() to topic every base_interval seconds (drift + jitter applied),
        with occasional reboots and seeded quick reconnects."""
        self.start()
        next_send = time.monotonic()
        rc = self.noise.next_reconnect_in()
        next_reconnect = time.monotonic() + rc if rc else None
        while True:
            iv = self.noise.interval(base_interval)
            if self.noise.due_for_reboot(iv):
                self.reboot()
                next_send = time.monotonic()
                next_reconnect = _reschedule(self.noise)
            elif next_reconnect is not None and time.monotonic() >= next_reconnect:
                self.reconnect()
                next_send = time.monotonic()
                next_reconnect = _reschedule(self.noise)
            self.publish(topic, make_payload())
            next_send += iv
            time.sleep(max(0.0, next_send - time.monotonic()))
