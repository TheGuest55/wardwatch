"""Read a pcap into lightweight per-packet records with scapy, decoding MQTT on TCP/1883.

One pass, streamed (PcapReader) so large session pcaps don't load whole into memory. Each record
is a plain dict of primitives; nothing here keys off host identity — that mapping happens in
windows.py. Raw topics/client-ids are carried only so windows.py can derive non-identifying
booleans from them; they never reach the feature table.
"""
from scapy.utils import PcapReader
from scapy.layers.l2 import Ether, ARP
from scapy.layers.inet import IP, TCP, UDP, ICMP

from . import mqtt

MQTT_PORT = 1883


def _tcp_flag_names(flags):
    return {
        "syn": bool(flags & 0x02),
        "ack": bool(flags & 0x10),
        "rst": bool(flags & 0x04),
        "fin": bool(flags & 0x01),
    }


def iter_packets(pcap_path):
    """Yield one dict per packet. Keys always present: ts (epoch float), length (int), l3
    ('ip'|'arp'|'other'), l4 ('tcp'|'udp'|'icmp'|'other'|None), eth_src, eth_dst, ip_src, ip_dst,
    sport, dport. TCP packets add syn/ack/rst/fin. ARP packets add arp_op (1 req/2 reply),
    arp_psrc, arp_pdst, arp_hwsrc. MQTT (TCP/1883 with payload) adds mqtt: list of control-packet
    dicts from mqtt.parse_segment."""
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if not pkt.haslayer(Ether):
                continue
            eth = pkt[Ether]
            rec = {
                "ts": float(pkt.time),
                "length": len(pkt),
                "eth_src": eth.src,
                "eth_dst": eth.dst,
                "ip_src": None, "ip_dst": None,
                "sport": None, "dport": None,
                "l3": "other", "l4": None,
            }

            if pkt.haslayer(ARP):
                arp = pkt[ARP]
                rec.update({
                    "l3": "arp",
                    "arp_op": int(arp.op),
                    "arp_psrc": arp.psrc, "arp_pdst": arp.pdst,
                    "arp_hwsrc": arp.hwsrc,
                    "ip_src": arp.psrc, "ip_dst": arp.pdst,
                })
                yield rec
                continue

            if pkt.haslayer(IP):
                ip = pkt[IP]
                rec["l3"] = "ip"
                rec["ip_src"], rec["ip_dst"] = ip.src, ip.dst
                if pkt.haslayer(TCP):
                    tcp = pkt[TCP]
                    rec["l4"] = "tcp"
                    rec["sport"], rec["dport"] = int(tcp.sport), int(tcp.dport)
                    rec.update(_tcp_flag_names(int(tcp.flags)))
                    payload = bytes(tcp.payload)
                    if payload and (rec["sport"] == MQTT_PORT or rec["dport"] == MQTT_PORT):
                        parsed = mqtt.parse_segment(payload)
                        if parsed:
                            rec["mqtt"] = parsed
                elif pkt.haslayer(UDP):
                    udp = pkt[UDP]
                    rec["l4"] = "udp"
                    rec["sport"], rec["dport"] = int(udp.sport), int(udp.dport)
                elif pkt.haslayer(ICMP):
                    rec["l4"] = "icmp"
                else:
                    rec["l4"] = "other"

            yield rec
