"""Attack-class registry. Each handler takes (ctx, params), is bounded by a timeout, sources
from the fixed attacker address, and returns a dict (outcome/notes/targets, plus creds_source
for abuse classes). All run only against the lab's own hosts on the sealed internal network."""
from . import (arp_mitm, bruteforce, eavesdrop, mqtt_flood, pump_cmd, recon,
               spoof_vitals, syn_flood)
from .creds import pick_creds  # noqa: F401 (re-exported for convenience)

REGISTRY = {
    "recon_scan": recon.recon_scan,
    "syn_flood": syn_flood.syn_flood,
    "mqtt_flood": mqtt_flood.mqtt_flood,
    "bruteforce": bruteforce.bruteforce,
    "arp_mitm": arp_mitm.arp_mitm,
    "eavesdrop": eavesdrop.eavesdrop,
    "spoof_vitals": spoof_vitals.spoof_vitals,
    "pump_cmd": pump_cmd.pump_cmd,
}
