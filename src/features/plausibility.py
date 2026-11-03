"""Payload plausibility (testbed-only feature family).

For each vital-bearing topic we track the most recent value of each numeric field and score how big
a jump a new reading is relative to a per-field plausible step. Spoofed-vitals and tampered pump
commands show up as large normalised jumps; benign jitter stays near zero. The topic string is used
only to route a reading to its per-stream history and to read the right fields — it is not a
feature and is never emitted.

Scale = the largest step we treat as fully plausible for one reading. A jump is normalised as
abs(delta)/scale, so ~1.0 is borderline and >1.0 is implausible.
"""
import json

# field -> plausible single-step magnitude
SCALES = {
    "hr": 20.0, "spo2": 6.0, "bp_sys": 25.0, "bp_dia": 18.0,
    "rate_ml_h": 60.0, "volume_ml": 50.0,
}
IMPLAUSIBLE_AT = 1.0  # normalised jump above this counts as an implausible reading


class PlausibilityTracker:
    """Per-stream running state. Call update() in timestamp order; it returns the normalised jump
    for that reading (0.0 when there is no field to score or no prior value)."""

    def __init__(self):
        self._last = {}  # stream_key -> {field: value}

    @staticmethod
    def _stream_key(topic):
        # One history per device stream (topic identifies the stream; not retained as a feature).
        return topic or ""

    def update(self, topic, app_payload):
        """Score one PUBLISH. Returns (max_normalised_jump, implausible_bool)."""
        fields = self._decode(app_payload)
        if not fields:
            return 0.0, False
        key = self._stream_key(topic)
        last = self._last.setdefault(key, {})
        max_jump = 0.0
        for name, val in fields.items():
            scale = SCALES.get(name)
            if scale is None or not isinstance(val, (int, float)):
                continue
            if name in last:
                jump = abs(float(val) - float(last[name])) / scale
                if jump > max_jump:
                    max_jump = jump
            last[name] = val
        return max_jump, (max_jump > IMPLAUSIBLE_AT)

    @staticmethod
    def _decode(app_payload):
        """Return the numeric vital fields in a publish payload, if it is our JSON telemetry.
        Handles monitor/pump payloads and wearable batches (scored on the last sample)."""
        if not app_payload:
            return {}
        try:
            obj = json.loads(app_payload.decode("utf-8", "replace"))
        except (ValueError, AttributeError):
            return {}
        if not isinstance(obj, dict):
            return {}
        out = {}
        for name in SCALES:
            if name in obj and isinstance(obj[name], (int, float)):
                out[name] = obj[name]
        # wearable batches carry samples[]; score the final sample's hr/spo2
        samples = obj.get("samples")
        if isinstance(samples, list) and samples and isinstance(samples[-1], dict):
            for name in ("hr", "spo2"):
                v = samples[-1].get(name)
                if isinstance(v, (int, float)):
                    out[name] = v
        return out
