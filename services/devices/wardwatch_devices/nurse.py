"""Nurse station: subscribes to the whole ward and logs it, sends legitimate pump commands on
a seeded schedule, and reconnects periodically so benign ward1/# SUBSCRIBE packets recur
through the session (not just once at startup)."""
import json
import os
import time

from .base import Device
from .noise import utcnow


class Nurse(Device):
    def __init__(self):
        self.beds = int(os.environ.get("BEDS", "1"))
        super().__init__("nurse-station")
        # a few commands per bed per hour by default; dry runs shorten these via config.
        env_iv = os.environ.get("NURSE_CMD_INTERVAL", "").strip()
        self.cmd_interval = float(env_iv) if env_iv else 3600.0 / max(1, self.beds * 3)
        self.reconnect_min = float(os.environ.get("NURSE_RECONNECT_MIN", "120"))
        self.reconnect_max = float(os.environ.get("NURSE_RECONNECT_MAX", "240"))

    def _bind(self):
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"connected: {reason_code}", flush=True)
        if not reason_code.is_failure:
            client.subscribe("ward1/#", qos=0)

    def _on_message(self, client, userdata, msg):
        print(f"{msg.topic} {msg.payload.decode(errors='replace')}", flush=True)

    def reconnect(self):
        """Quick reconnect (not an outage) so a fresh SUBSCRIBE appears on the wire."""
        self.client.loop_stop()
        try:
            self.client.disconnect()
        except Exception:
            pass
        self.start()
        self.noise.log_event("reconnect")
        print("nurse reconnected", flush=True)

    def send_command(self):
        bed = self.noise.rng.randint(1, self.beds)
        if self.noise.rng.random() < 0.5:
            cmd = {"action": "set_rate", "rate": self.noise.rng.choice([75, 100, 125, 150])}
        else:
            cmd = {"action": "ack_alarm"}
        cmd["ts"] = utcnow()
        self.publish(f"ward1/bed{bed}/pump/cmd", cmd)
        self.noise.log_event("nurse_command", bed=bed, action=cmd["action"],
                             rate=cmd.get("rate"))
        print(f"sent command to bed{bed}: {cmd}", flush=True)

    def run(self):
        self.start()
        now = time.monotonic()
        next_cmd = now + self.noise.rng.uniform(0, self.cmd_interval)
        next_reconnect = now + self.noise.rng.uniform(self.reconnect_min, self.reconnect_max)
        while True:
            time.sleep(1.0)
            now = time.monotonic()
            if now >= next_cmd:
                self.send_command()
                next_cmd = now + self.noise.interval(self.cmd_interval)
            if now >= next_reconnect:
                self.reconnect()
                next_reconnect = now + self.noise.rng.uniform(self.reconnect_min,
                                                              self.reconnect_max)


def run():
    Nurse().run()
