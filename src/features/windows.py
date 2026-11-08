"""Aggregate per-packet records into one feature row per host (MAC) per 5-second window.

A host's row covers every packet it sends or receives in the window (./CLAUDE.md granularity).
All columns here are model-input features EXCEPT the identity/time fields (host_mac, host_ip,
window_*), which windows.py emits as metadata for labelling and forensics. Derived MQTT features
keep only non-identifying signal (type counts/rates, wildcard/depth/is-command booleans,
client-id *reuse* count) — never raw topics or client-id strings.
"""
import numpy as np

from .. import common
from .plausibility import PlausibilityTracker
from . import mqtt

BROADCAST = "ff:ff:ff:ff:ff:ff"


def _is_groupcast(mac):
    if mac == BROADCAST:
        return True
    try:
        return bool(int(mac.split(":")[0], 16) & 1)  # LSB of first octet = multicast
    except (ValueError, IndexError):
        return False


class _Acc:
    """Mutable accumulator for one (mac, window)."""
    __slots__ = (
        "pkts_sent", "pkts_recv", "bytes_sent", "bytes_recv", "sizes", "ts",
        "peers", "peer_ips", "contacted_ports",
        "n_tcp", "n_udp", "n_arp", "n_icmp", "n_other",
        "tcp_syn", "tcp_synack", "tcp_rst", "tcp_fin",
        "mqtt_counts", "mqtt_connect", "mqtt_publish", "mqtt_payloads",
        "connack_fail", "clientid_reuse", "wildcard_subs", "topic_depth_max", "is_command_pub",
        "arp_req", "arp_reply", "arp_gratuitous",
        "vital_max_jump", "vital_jump_count", "implausible",
        "touches_attacker", "ip_counts", "flows",
    )

    def __init__(self):
        self.pkts_sent = self.pkts_recv = 0
        self.bytes_sent = self.bytes_recv = 0
        self.sizes, self.ts = [], []
        self.peers, self.peer_ips, self.contacted_ports = set(), set(), set()
        self.n_tcp = self.n_udp = self.n_arp = self.n_icmp = self.n_other = 0
        self.tcp_syn = self.tcp_synack = self.tcp_rst = self.tcp_fin = 0
        self.mqtt_counts = {}
        self.mqtt_connect = self.mqtt_publish = 0
        self.mqtt_payloads = []
        self.connack_fail = 0
        self.clientid_reuse = 0
        self.wildcard_subs = 0
        self.topic_depth_max = 0
        self.is_command_pub = 0
        self.arp_req = self.arp_reply = self.arp_gratuitous = 0
        self.vital_max_jump = 0.0
        self.vital_jump_count = 0
        self.implausible = 0
        self.touches_attacker = False
        self.ip_counts = {}
        self.flows = set()        # distinct transport flow keys active in this (mac, window)


def _topic_depth(topic):
    return topic.count("/") + 1 if topic else 0


def _flow_key(rec):
    """Direction-agnostic transport-flow key for an IPv4 TCP/UDP packet, or None. Endpoints are
    sorted so both directions map to one flow (like NFStream's bidirectional flows). ARP/ICMP and
    non-IP packets have no flow (NFStream ignores ARP; the L2/ARP features cover that signal)."""
    if rec["l3"] != "ip" or rec["l4"] not in ("tcp", "udp"):
        return None
    a = (rec.get("ip_src"), rec.get("sport"))
    b = (rec.get("ip_dst"), rec.get("dport"))
    lo, hi = (a, b) if a <= b else (b, a)
    return (rec["l4"], lo, hi)


def build_rows(packet_iter, session, attacker_mac, attacker_ip, ip_roles):
    """Consume per-packet dicts (in capture order) and return (rows, conflict_windows).

    rows: list of per-(mac,window) feature+metadata dicts. conflict_windows: set of window_start
    epochs in which an IP was seen mapped to more than one MAC (joined onto rows as ip_mac_conflict).
    """
    acc = {}                      # (mac, win) -> _Acc
    clientid_owner = {}           # client_id -> first MAC that used it
    ip_to_mac = {}               # running IP->MAC map (latest), for conflict detection
    conflict_windows = set()
    flow_first_seen = {}          # flow_key -> epoch of its first packet (causal; never the future)
    plaus = PlausibilityTracker()

    def get(mac, win):
        key = (mac, win)
        a = acc.get(key)
        if a is None:
            a = acc[key] = _Acc()
        return a

    for rec in packet_iter:
        win = common.window_start(rec["ts"])
        src, dst = rec["eth_src"], rec["eth_dst"]
        size = rec["length"]
        peer_ip_for_src = rec.get("ip_dst")
        peer_ip_for_dst = rec.get("ip_src")
        attacker_peer = (
            src == attacker_mac or dst == attacker_mac
            or rec.get("ip_src") == attacker_ip or rec.get("ip_dst") == attacker_ip
        )

        # ---- causal transport-flow tracking (review B1) ----
        # Record each flow's first-seen time and mark it active in this window for both endpoints.
        # Flow age is computed at finalize as (window_end - first_seen), so it only ever reflects
        # elapsed time up to the window end — never the flow's whole future lifetime.
        fkey = _flow_key(rec)
        if fkey is not None and fkey not in flow_first_seen:
            flow_first_seen[fkey] = rec["ts"]

        # ---- sender side ----
        a = get(src, win)
        a.pkts_sent += 1
        a.bytes_sent += size
        a.sizes.append(size)
        a.ts.append(rec["ts"])
        if not _is_groupcast(dst):
            a.peers.add(dst)
        if peer_ip_for_src:
            a.peer_ips.add(peer_ip_for_src)
        if rec.get("ip_src"):
            a.ip_counts[rec["ip_src"]] = a.ip_counts.get(rec["ip_src"], 0) + 1
        if attacker_peer:
            a.touches_attacker = True
        if fkey is not None:
            a.flows.add(fkey)

        l3, l4 = rec["l3"], rec["l4"]
        if l3 == "arp":
            a.n_arp += 1
            op = rec.get("arp_op")
            if op == 1:
                a.arp_req += 1
            elif op == 2:
                a.arp_reply += 1
                if rec.get("arp_psrc") and rec.get("arp_psrc") == rec.get("arp_pdst"):
                    a.arp_gratuitous += 1
        elif l4 == "tcp":
            a.n_tcp += 1
            if rec.get("dport"):
                a.contacted_ports.add(rec["dport"])
            syn, ack = rec.get("syn"), rec.get("ack")
            if syn and ack:
                a.tcp_synack += 1
            elif syn:
                a.tcp_syn += 1
            if rec.get("rst"):
                a.tcp_rst += 1
            if rec.get("fin"):
                a.tcp_fin += 1
            _accumulate_mqtt(a, rec, src, clientid_owner, plaus)
        elif l4 == "udp":
            a.n_udp += 1
            if rec.get("dport"):
                a.contacted_ports.add(rec["dport"])
        elif l4 == "icmp":
            a.n_icmp += 1
        else:
            a.n_other += 1

        # ---- receiver side (unicast only) ----
        if not _is_groupcast(dst):
            b = get(dst, win)
            b.pkts_recv += 1
            b.bytes_recv += size
            b.sizes.append(size)
            b.ts.append(rec["ts"])
            b.peers.add(src)
            if peer_ip_for_dst:
                b.peer_ips.add(peer_ip_for_dst)
            if attacker_peer:
                b.touches_attacker = True
            if fkey is not None:
                b.flows.add(fkey)

        # ---- L2 conflict tracking (IP mapped to >1 MAC within a window) ----
        claim_ip, claim_mac = None, None
        if l3 == "arp" and rec.get("arp_op") == 2:
            claim_ip, claim_mac = rec.get("arp_psrc"), rec.get("arp_hwsrc") or src
        elif l3 == "ip" and rec.get("ip_src"):
            claim_ip, claim_mac = rec["ip_src"], src
        if claim_ip and claim_mac:
            prev = ip_to_mac.get(claim_ip)
            if prev is not None and prev != claim_mac:
                conflict_windows.add(win)
            ip_to_mac[claim_ip] = claim_mac

    rows = [_finalize(mac, win, a, session, ip_roles, win in conflict_windows, flow_first_seen)
            for (mac, win), a in acc.items()]
    return rows, conflict_windows


def _accumulate_mqtt(a, rec, src_mac, clientid_owner, plaus):
    for m in rec.get("mqtt", []):
        name = m["type_name"]
        a.mqtt_counts[name] = a.mqtt_counts.get(name, 0) + 1
        if m["type"] == mqtt.CONNECT:
            a.mqtt_connect += 1
            cid = m.get("client_id")
            if cid:
                owner = clientid_owner.get(cid)
                if owner is None:
                    clientid_owner[cid] = src_mac
                elif owner != src_mac:
                    a.clientid_reuse += 1
        elif m["type"] == mqtt.CONNACK:
            if m.get("return_code"):  # non-zero = connection refused
                a.connack_fail += 1
        elif m["type"] == mqtt.PUBLISH:
            a.mqtt_publish += 1
            a.mqtt_payloads.append(m.get("payload_size", 0))
            topic = m.get("topic") or ""
            a.topic_depth_max = max(a.topic_depth_max, _topic_depth(topic))
            if topic.endswith("/cmd") or "/pump/cmd" in topic:
                a.is_command_pub += 1
            jump, bad = plaus.update(topic, m.get("app_payload", b""))
            if jump > a.vital_max_jump:
                a.vital_max_jump = jump
            if bad:
                a.vital_jump_count += 1
                a.implausible = 1
        elif m["type"] == mqtt.SUBSCRIBE:
            for tf in m.get("topics", []):
                a.topic_depth_max = max(a.topic_depth_max, _topic_depth(tf))
                if "+" in tf or "#" in tf:
                    a.wildcard_subs += 1


def _finalize(mac, win, a, session, ip_roles, conflict, flow_first_seen):
    sizes = np.array(a.sizes, dtype=float) if a.sizes else np.array([0.0])
    tss = sorted(a.ts)
    iats = np.diff(tss) if len(tss) > 1 else np.array([0.0])
    pkts_total = a.pkts_sent + a.pkts_recv
    host_ip = max(a.ip_counts, key=a.ip_counts.get) if a.ip_counts else None
    mqtt_total = sum(a.mqtt_counts.values())
    payloads = np.array(a.mqtt_payloads, dtype=float) if a.mqtt_payloads else np.array([0.0])

    # Causal flow features (review B1): computed from packets seen up to this window's end only.
    win_end = win + common.WINDOW_SECONDS
    nf_active_flows = len(a.flows)
    nf_new_flows = sum(1 for k in a.flows
                       if common.window_start(flow_first_seen[k]) == win)
    ages = [max(0.0, win_end - flow_first_seen[k]) for k in a.flows]
    nf_mean_flow_age = float(np.mean(ages)) if ages else 0.0

    return {
        # --- metadata (never model inputs) ---
        "session": session,
        "window_start_epoch": win,
        "window_start_iso": common.epoch_to_iso(win),
        "host_mac": mac,
        "host_ip": host_ip,
        "host_role": ip_roles.get(host_ip, "unknown"),
        "touches_attacker": bool(a.touches_attacker),  # used by labeller, dropped before modelling
        # --- features: volume / size / timing ---
        "pkts_total": pkts_total,
        "pkts_sent": a.pkts_sent,
        "pkts_recv": a.pkts_recv,
        "bytes_total": a.bytes_sent + a.bytes_recv,
        "bytes_sent": a.bytes_sent,
        "bytes_recv": a.bytes_recv,
        "mean_pkt_size": float(sizes.mean()),
        "std_pkt_size": float(sizes.std()),
        "min_pkt_size": float(sizes.min()),
        "max_pkt_size": float(sizes.max()),
        "mean_iat": float(iats.mean()),
        "std_iat": float(iats.std()),
        "unique_peers": len(a.peers),
        "unique_peer_ips": len(a.peer_ips),
        "unique_contacted_ports": len(a.contacted_ports),
        # --- features: protocol mix ---
        "frac_tcp": a.n_tcp / pkts_total if pkts_total else 0.0,
        "frac_udp": a.n_udp / pkts_total if pkts_total else 0.0,
        "frac_arp": a.n_arp / pkts_total if pkts_total else 0.0,
        "frac_icmp": a.n_icmp / pkts_total if pkts_total else 0.0,
        "frac_other": a.n_other / pkts_total if pkts_total else 0.0,
        "tcp_syn": a.tcp_syn,
        "tcp_synack": a.tcp_synack,
        "tcp_rst": a.tcp_rst,
        "tcp_fin": a.tcp_fin,
        "tcp_syn_rate": a.tcp_syn / common.WINDOW_SECONDS,
        # --- features: MQTT ---
        "mqtt_pkts": mqtt_total,
        "mqtt_connect": a.mqtt_connect,
        "mqtt_publish": a.mqtt_publish,
        "mqtt_subscribe": a.mqtt_counts.get("subscribe", 0),
        "mqtt_connack": a.mqtt_counts.get("connack", 0),
        "mqtt_pingreq": a.mqtt_counts.get("pingreq", 0),
        "mqtt_disconnect": a.mqtt_counts.get("disconnect", 0),
        "mqtt_connect_rate": a.mqtt_connect / common.WINDOW_SECONDS,
        "mqtt_publish_rate": a.mqtt_publish / common.WINDOW_SECONDS,
        "mqtt_mean_payload": float(payloads.mean()),
        "mqtt_max_payload": float(payloads.max()),
        "mqtt_connack_failures": a.connack_fail,
        "mqtt_clientid_reuse": a.clientid_reuse,
        "topic_wildcard_subs": a.wildcard_subs,
        "topic_max_depth": a.topic_depth_max,
        "topic_is_command": a.is_command_pub,
        # --- features: L2 / ARP ---
        "arp_requests": a.arp_req,
        "arp_replies": a.arp_reply,
        "arp_gratuitous": a.arp_gratuitous,
        "arp_reply_rate": a.arp_reply / common.WINDOW_SECONDS,
        "ip_mac_conflict": 1 if conflict else 0,
        # --- features: payload plausibility (testbed only) ---
        "vital_max_jump": float(a.vital_max_jump),
        "vital_jump_count": a.vital_jump_count,
        "implausible_vital": a.implausible,
        # --- features: causal flow stats (review B1; live-computable, no future lifetime) ---
        "nf_active_flows": nf_active_flows,
        "nf_new_flows": nf_new_flows,
        "nf_mean_flow_age": nf_mean_flow_age,
    }
