"""Infusion pump: publishes telemetry ~0.5 Hz and applies valid commands from the nurse
station on its own command topic. Commands: {"action":"set_rate","rate":<ml/h>} and
{"action":"ack_alarm"}."""
import json

from .base import Device
from .noise import utcnow


class Pump(Device):
    def __init__(self):
        self.rate = 100.0          # ml/h
        self.volume = 500.0        # ml remaining
        self.alarm = "none"
        super().__init__("infusion-pump")

    def _bind(self):
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if not reason_code.is_failure:
            client.subscribe(f"ward1/bed{self.bed}/pump/cmd", qos=0)

    def _on_message(self, client, userdata, msg):
        try:
            cmd = json.loads(msg.payload.decode())
        except (ValueError, UnicodeDecodeError):
            return
        action = cmd.get("action")
        if action == "set_rate" and isinstance(cmd.get("rate"), (int, float)):
            self.rate = max(0.0, min(1200.0, float(cmd["rate"])))  # clamp to a safe range
        elif action == "ack_alarm":
            self.alarm = "none"
        print(f"applied command: {cmd}", flush=True)

    def payload(self):
        self.volume = max(0.0, self.volume - self.rate / 3600.0 * 2.0)  # ~2 s per tick
        if self.volume < 50 and self.alarm == "none":
            self.alarm = "low_volume"
        return {
            "device": "infusion-pump", "bed": self.bed, "ts": utcnow(),
            "rate_ml_h": round(self.rate, 1),
            "volume_ml": round(self.volume, 1),
            "alarm": self.alarm,
        }


def run():
    p = Pump()
    p.run_periodic(2.0, p.payload, f"ward1/bed{p.bed}/pump/telemetry")
