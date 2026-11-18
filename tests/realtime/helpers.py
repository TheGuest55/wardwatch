"""Synthetic packet records for the streaming tests (same shape as packets.iter_packets)."""
import random

BCAST = "ff:ff:ff:ff:ff:ff"
MACS = {"a": "02:00:00:00:00:0a", "b": "02:00:00:00:00:0b", "c": "02:00:00:00:00:0c", "br": "02:00:00:00:00:10"}
IPS = {"a": "10.0.0.11", "b": "10.0.0.12", "c": "10.0.0.13", "br": "10.0.0.10"}
IP_ROLES = {"10.0.0.10": "broker", "10.0.0.11": "monitor", "10.0.0.12": "pump"}
T0 = 3_000_000.0


def tcp(ts, s, d, sport=40000, dport=1883, syn=False, ack=True, fin=False, rst=False, mqtt=None, length=80):
    r = {"ts": float(ts), "length": length, "eth_src": MACS[s], "eth_dst": MACS[d], "ip_src": IPS[s],
         "ip_dst": IPS[d], "sport": sport, "dport": dport, "l3": "ip", "l4": "tcp", "syn": syn, "ack": ack,
         "rst": rst, "fin": fin}
    if mqtt:
        r["mqtt"] = mqtt
    return r


def arp_reply(ts, s, psrc_key, pdst_key, hw=None):
    return {"ts": float(ts), "length": 60, "eth_src": MACS[s], "eth_dst": BCAST, "ip_src": IPS[psrc_key],
            "ip_dst": IPS[pdst_key], "sport": None, "dport": None, "l3": "arp", "l4": None, "arp_op": 2,
            "arp_psrc": IPS[psrc_key], "arp_pdst": IPS[pdst_key], "arp_hwsrc": MACS[hw or s]}


def connect(client_id):
    return [{"type": 1, "type_name": "connect", "client_id": client_id, "payload_size": 0}]


def synthetic_packets(seed=0, seconds=400):
    """~seconds of mixed traffic: steady flows, a flow closed by FIN then reopened on the same 4-tuple, an idle gap
    longer than FLOW_IDLE_TIMEOUT on one flow, long-lived flows spanning many windows, window-boundary packets,
    two live connections claiming one MQTT client id, and an ARP IP->MAC conflict. Timestamps are non-decreasing."""
    rnd = random.Random(seed)
    pk = []
    t = T0 + 0.3
    while t < T0 + seconds:
        pk.append(tcp(t, "a", "br", sport=40001, mqtt=None))
        pk.append(tcp(t + 0.01, "br", "a", sport=1883, dport=40001))
        if rnd.random() < 0.5:
            pk.append(tcp(t + 0.02, "b", "br", sport=40002))
        t += rnd.uniform(0.5, 2.5)
    # exact boundaries (window starts are multiples of 5)
    base = (int(T0) // 5) * 5
    for k in range(1, 6):
        pk.append(tcp(base + 5 * k, "c", "br", sport=40100 + k))
        pk.append(tcp(base + 5 * k - 0.000001, "c", "br", sport=40100 + k))
    # FIN closes the flow, the next packet reopens it
    pk.append(tcp(T0 + 50, "c", "br", sport=40200, fin=True))
    pk.append(tcp(T0 + 52, "c", "br", sport=40200))
    pk.append(tcp(T0 + 53, "c", "br", sport=40200))
    # idle for > FLOW_IDLE_TIMEOUT (120 s) then more traffic: a new flow life
    pk.append(tcp(T0 + 60, "b", "br", sport=40300))
    pk.append(tcp(T0 + 200, "b", "br", sport=40300))
    # client-id reuse: two different live connections CONNECT with the same id
    pk.append(tcp(T0 + 120, "a", "br", sport=41000, mqtt=connect("dup")))
    pk.append(tcp(T0 + 121, "b", "br", sport=41001, mqtt=connect("dup")))
    # ARP: br answers for its IP, then c claims the same IP (conflict in that window)
    pk.append(arp_reply(T0 + 150, "br", "br", "a"))
    pk.append(arp_reply(T0 + 152, "c", "br", "a"))
    pk.sort(key=lambda r: r["ts"])
    return pk
