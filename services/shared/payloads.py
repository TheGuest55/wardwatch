"""Canonical MQTT application payloads for the ward, shared by the real devices AND the attacker.

Review finding B2: the attacker's payloads used a different timestamp format (24-char `...mmmZ`)
and integer vitals, so `spoof_vitals`/`pump_cmd` traffic occupied a byte-size (and per-field
representation) range the benign devices never produce — a single-threshold tell for the two
"hardest" stolen-credential classes. The fix is a single source of truth for payload construction,
built into both the devices and attacker images from `services/shared`.

These builders fix the field set, field ORDER and value REPRESENTATION (float precision for vitals,
int types for rates/bp, one timestamp format) so an attacker reusing them cannot emit a per-field
token the real device never produces. `json.dumps(...)` with default separators is the serialiser
on both sides. Standard library only (importable on the host and in both images).
"""
from datetime import datetime, timezone


def utcnow():
    """UTC timestamp, a SINGLE clock read (the old util.utc_iso read the clock twice — minor m1),
    ISO-8601 with offset exactly as the devices emit, e.g. 2026-11-08T07:00:00.123456+00:00."""
    return datetime.now(timezone.utc).isoformat()


def _vital(x):
    """Coerce a vital to the device's representation: a float rounded to 1 decimal. So an attacker
    passing an int (e.g. hr=220) serialises as `220.0`, inside the monitor's float token space."""
    return round(float(x), 1)


def monitor_vitals(bed, hr, spo2, bp_sys, bp_dia, vitals_source="bidmc",
                   bp_source="synthetic", ts=None):
    """Patient-monitor vitals payload. hr/spo2 are always 1-dp floats; bed/bp are ints — identical
    representation whether a real monitor or the spoof_vitals attack builds it. Field order matches
    the monitor exactly."""
    return {
        "device": "patient-monitor",
        "bed": int(bed),
        "ts": ts or utcnow(),
        "hr": _vital(hr),
        "spo2": _vital(spo2),
        "bp_sys": int(round(bp_sys)),
        "bp_dia": int(round(bp_dia)),
        "vitals_source": vitals_source,
        "bp_source": bp_source,
    }


def pump_command(action, rate=None, ts=None):
    """Nurse / attacker pump command. `set_rate` carries an int rate; `ack_alarm` carries none.
    Field order (action, [rate,] ts) matches the nurse station exactly."""
    cmd = {"action": action}
    if rate is not None:
        cmd["rate"] = int(rate)
    cmd["ts"] = ts or utcnow()
    return cmd
