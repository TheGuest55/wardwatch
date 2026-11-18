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
MQTT_PORT = 1883
# A transport flow ends on FIN/RST; a later packet on the same 4-tuple is a NEW flow.
# It also expires after this many idle seconds, so a long-lived flow's age never grows
# into a host-lifetime / capture-start proxy (review m-new-1, same spirit as B1/M5).
FLOW_IDLE_TIMEOUT = 120.0


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
        "touches_attacker", "ip_counts", "flows", "flow_fs", "peer_pkts",
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
        self.flow_fs = {}         # flow_key -> its first-seen epoch AS OF this window (reset-aware)
        self.peer_pkts = {}       # peer mac -> [sent, recv] unicast packets (metadata; WindowBuilder with_peers)


def _topic_depth(topic):
    return topic.count("/") + 1 if topic else 0


def _mqtt_conn_key(rec):
    """Canonical (client_endpoint, broker_endpoint) key for a TCP/1883 packet, direction-agnostic,
    so both halves of a connection map to one key. None if the packet is not MQTT-port TCP."""
    sp, dp = rec.get("sport"), rec.get("dport")
    si, di = rec.get("ip_src"), rec.get("ip_dst")
    if dp == MQTT_PORT:
        return ((si, sp), (di, dp))
    if sp == MQTT_PORT:
        return ((di, dp), (si, sp))
    return None


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


class WindowBuilder:
    """Incremental per-(mac, 5 s window) feature builder; build_rows() is "add every packet, then flush".

    add(rec) takes one packet record (capture order); close_before(t) finalizes and returns the rows
    of every window that ends at or before epoch t; flush() returns whatever is still open.
    Rows within one call keep first-touch order, so build_rows' output is unchanged.

    A packet whose window was already closed is LATE: it is counted in `late_packets` (never silently
    dropped) and still processed, so it shows up as an extra row for that window on the next close/flush
    and in `late_windows`. Nothing here depends on the clock: the caller decides when to close.

    with_peers=True adds `peer_pkts` to each row: {peer_mac: [sent, received]} over UNICAST packets, the
    host's packets to/from each peer in the window. It is METADATA for source attribution only, never a
    model input (not in common.FEATURES); build_rows leaves it out so its rows are unchanged.
    """

    def __init__(self, session, attacker_mac, attacker_ip, ip_roles, with_peers=False):
        self.session, self.attacker_mac, self.attacker_ip = session, attacker_mac, attacker_ip
        self.ip_roles, self.with_peers = ip_roles, with_peers
        self.acc = {}                 # (mac, win) -> _Acc, in first-touch order
        # Live client-id ownership by TCP connection (review M5): a CONNECT whose client id is currently
        # held by a *different* live connection is reuse. Ownership is released on DISCONNECT/FIN/RST, so
        # a benign device reconnecting after its previous session closed is not counted.
        self.active_cid = {}          # client_id -> owning connection key
        self.conn_cid = {}            # connection key -> client_id it currently holds
        self.ip_to_mac = {}           # running IP->MAC map (latest), for conflict detection
        self.conflict_windows = set()  # windows with ANY conflict (for the extract-level count)
        self.conflict_macs = {}       # win -> set of MACs involved in a conflicting IP claim (review M2)
        # Flow lifecycle (review m-new-1): first-seen is reset when a flow is reopened — i.e. after a
        # FIN/RST closed it, or after >FLOW_IDLE_TIMEOUT idle — so nf_mean_flow_age stays bounded and a
        # reopened 4-tuple counts as a genuinely new flow.
        self.flow_first_seen = {}     # flow_key -> epoch its CURRENT life began (causal; never future)
        self.flow_last_seen = {}      # flow_key -> epoch of its most recent packet
        self.flow_closed = {}         # flow_key -> True once FIN/RST seen (next packet reopens)
        self.plaus = PlausibilityTracker()
        self.closed_to = None         # highest t passed to close_before
        self.late_packets = 0
        self.late_windows = set()
        self.n_packets = 0

    def _get(self, mac, win):
        key = (mac, win)
        a = self.acc.get(key)
        if a is None:
            a = self.acc[key] = _Acc()
        return a

    def add(self, rec):
        acc_get = self._get
        win = common.window_start(rec["ts"])
        self.n_packets += 1
        if self.closed_to is not None and win + common.WINDOW_SECONDS <= self.closed_to:
            self.late_packets += 1
            self.late_windows.add(win)
        attacker_mac, attacker_ip = self.attacker_mac, self.attacker_ip
        src, dst = rec["eth_src"], rec["eth_dst"]
        size = rec["length"]
        peer_ip_for_src = rec.get("ip_dst")
        peer_ip_for_dst = rec.get("ip_src")
        attacker_peer = (
            src == attacker_mac or dst == attacker_mac
            or rec.get("ip_src") == attacker_ip or rec.get("ip_dst") == attacker_ip
        )

        # ---- causal transport-flow tracking (review B1 + m-new-1) ----
        # Track each flow's current-life first-seen, resetting it when the flow reopens (a prior
        # FIN/RST closed it, or it was idle > FLOW_IDLE_TIMEOUT). Age is computed per window from the
        # first-seen captured on the accumulator, so it only reflects elapsed time up to that
        # window's end and never a flow's whole (or future) lifetime.
        fkey = _flow_key(rec)
        if fkey is not None:
            prev_last = self.flow_last_seen.get(fkey)
            reopened = (fkey not in self.flow_first_seen or self.flow_closed.get(fkey, False)
                        or (prev_last is not None and rec["ts"] - prev_last > FLOW_IDLE_TIMEOUT))
            if reopened:
                self.flow_first_seen[fkey] = rec["ts"]
                self.flow_closed[fkey] = False
            self.flow_last_seen[fkey] = rec["ts"]
            # A FIN/RST closes the flow AFTER this packet (the packet still belongs to it).
            if rec["l4"] == "tcp" and (rec.get("fin") or rec.get("rst")):
                self.flow_closed[fkey] = True

        # ---- sender side ----
        a = acc_get(src, win)
        a.pkts_sent += 1
        a.bytes_sent += size
        a.sizes.append(size)
        a.ts.append(rec["ts"])
        if not _is_groupcast(dst):
            a.peers.add(dst)
            if self.with_peers:
                pp = a.peer_pkts.setdefault(dst, [0, 0])
                pp[0] += 1
        if peer_ip_for_src:
            a.peer_ips.add(peer_ip_for_src)
        if rec.get("ip_src"):
            a.ip_counts[rec["ip_src"]] = a.ip_counts.get(rec["ip_src"], 0) + 1
        if attacker_peer:
            a.touches_attacker = True
        if fkey is not None:
            a.flows.add(fkey)
            a.flow_fs[fkey] = self.flow_first_seen[fkey]

        # Protocol/flag counts cover every packet the host sends OR receives (./CLAUDE.md
        # granularity; review M4). Sender side also records MQTT message counts and the stateful
        # MQTT signals (client-id ownership, payload plausibility).
        _count_proto_flags(a, rec, is_sender=True)
        if rec["l4"] == "tcp":
            # Release client-id ownership when its connection closes (FIN/RST), either direction.
            if rec.get("rst") or rec.get("fin"):
                conn = _mqtt_conn_key(rec)
                held_cid = self.conn_cid.pop(conn, None) if conn else None
                if held_cid is not None and self.active_cid.get(held_cid) == conn:
                    del self.active_cid[held_cid]
            _count_mqtt(a, rec)
            _stateful_mqtt(a, rec, self.active_cid, self.conn_cid, self.plaus)

        # ---- receiver side (unicast only) ----
        if not _is_groupcast(dst):
            b = acc_get(dst, win)
            b.pkts_recv += 1
            b.bytes_recv += size
            b.sizes.append(size)
            b.ts.append(rec["ts"])
            b.peers.add(src)
            if self.with_peers:
                pp = b.peer_pkts.setdefault(src, [0, 0])
                pp[1] += 1
            if peer_ip_for_dst:
                b.peer_ips.add(peer_ip_for_dst)
            if attacker_peer:
                b.touches_attacker = True
            if fkey is not None:
                b.flows.add(fkey)
                b.flow_fs[fkey] = self.flow_first_seen[fkey]
            # Count received protocol/flags and MQTT messages too (review M4): a subscriber's or
            # eavesdropper's received PUBLISHes, and failed CONNACKs a client receives, are otherwise
            # invisible. Plausibility and client-id ownership stay sender-only (not re-scored here).
            _count_proto_flags(b, rec, is_sender=False)
            if rec["l4"] == "tcp":
                _count_mqtt(b, rec)

        # ---- L2 conflict tracking (IP mapped to >1 MAC within a window) ----
        claim_ip, claim_mac = None, None
        l3 = rec["l3"]
        if l3 == "arp" and rec.get("arp_op") == 2:
            claim_ip, claim_mac = rec.get("arp_psrc"), rec.get("arp_hwsrc") or src
        elif l3 == "ip" and rec.get("ip_src"):
            claim_ip, claim_mac = rec["ip_src"], src
        if claim_ip and claim_mac:
            prev = self.ip_to_mac.get(claim_ip)
            if prev is not None and prev != claim_mac:
                self.conflict_windows.add(win)
                # Only the old and new owners of the disputed IP are "involved" (review M2); the
                # flag is a host property, not a window-wide one, so bystanders stay unflagged.
                self.conflict_macs.setdefault(win, set()).update({prev, claim_mac})
            self.ip_to_mac[claim_ip] = claim_mac

    def _row(self, mac, win, a):
        row = _finalize(mac, win, a, self.session, self.ip_roles,
                        mac in self.conflict_macs.get(win, ()), self.attacker_mac)
        if self.with_peers:
            row["peer_pkts"] = a.peer_pkts
        return row

    def close_before(self, t):
        """Finalize and return the rows of every window whose end is <= t (epoch seconds)."""
        if self.closed_to is None or t > self.closed_to:
            self.closed_to = t
        done = [k for k in self.acc if k[1] + common.WINDOW_SECONDS <= t]
        rows = [self._row(mac, win, self.acc.pop((mac, win))) for mac, win in done]
        for w in {k[1] for k in done}:
            if not any(k[1] == w for k in self.acc):
                self.conflict_macs.pop(w, None)
        return rows

    def flush(self):
        """Finalize and return every window still open."""
        rows = [self._row(mac, win, a) for (mac, win), a in self.acc.items()]
        self.acc = {}
        return rows


def build_rows(packet_iter, session, attacker_mac, attacker_ip, ip_roles):
    """Consume per-packet dicts (in capture order) and return (rows, conflict_windows).

    rows: list of per-(mac,window) feature+metadata dicts. conflict_windows: set of window_start
    epochs in which an IP was seen mapped to more than one MAC (joined onto rows as ip_mac_conflict).
    """
    wb = WindowBuilder(session, attacker_mac, attacker_ip, ip_roles)
    for rec in packet_iter:
        wb.add(rec)
    return wb.flush(), wb.conflict_windows


def _count_proto_flags(a, rec, is_sender):
    """Per-packet protocol-mix and TCP-flag counts for one host (review M4: run on both the sender
    and the receiver). `contacted_ports` is a sent-only notion (ports this host reached out to)."""
    l3, l4 = rec["l3"], rec["l4"]
    if l3 == "arp":
        a.n_arp += 1
        op = rec.get("arp_op")
        if op == 1:
            a.arp_req += 1
        elif op == 2:
            a.arp_reply += 1
        # Gratuitous ARP = a host announcing its own IP (sender protocol addr == target protocol
        # addr), in EITHER a request (announcement/probe, op=1) or a reply (op=2). Counting only
        # replies missed the request form that arpspoof announcements use (review m-new-5).
        if op in (1, 2) and rec.get("arp_psrc") and rec.get("arp_psrc") == rec.get("arp_pdst"):
            a.arp_gratuitous += 1
    elif l4 == "tcp":
        a.n_tcp += 1
        if is_sender and rec.get("dport"):
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
    elif l4 == "udp":
        a.n_udp += 1
        if is_sender and rec.get("dport"):
            a.contacted_ports.add(rec["dport"])
    elif l4 == "icmp":
        a.n_icmp += 1
    else:
        a.n_other += 1


def _count_mqtt(a, rec):
    """Pure MQTT message-type counting for one host; run on both sender and receiver (review M4).
    No plausibility and no client-id ownership here — those are stateful and sender-only."""
    for m in rec.get("mqtt", []):
        a.mqtt_counts[m["type_name"]] = a.mqtt_counts.get(m["type_name"], 0) + 1
        t = m["type"]
        if t == mqtt.CONNECT:
            a.mqtt_connect += 1
        elif t == mqtt.CONNACK:
            if m.get("return_code"):  # non-zero = connection refused
                a.connack_fail += 1
        elif t == mqtt.PUBLISH:
            a.mqtt_publish += 1
            a.mqtt_payloads.append(m.get("payload_size", 0))
            topic = m.get("topic") or ""
            a.topic_depth_max = max(a.topic_depth_max, _topic_depth(topic))
            if topic.endswith("/cmd") or "/pump/cmd" in topic:
                a.is_command_pub += 1
        elif t == mqtt.SUBSCRIBE:
            for tf in m.get("topics", []):
                a.topic_depth_max = max(a.topic_depth_max, _topic_depth(tf))
                if "+" in tf or "#" in tf:
                    a.wildcard_subs += 1


def _stateful_mqtt(a, rec, active_cid, conn_cid, plaus):
    """Sender-only MQTT state: live client-id ownership/reuse (review M5) and payload plausibility
    (testbed-only). Kept off the receiver so a forwarded/received copy is not re-scored."""
    conn = _mqtt_conn_key(rec)
    for m in rec.get("mqtt", []):
        t = m["type"]
        if t == mqtt.CONNECT:
            cid = m.get("client_id")
            if cid and conn:
                held = active_cid.get(cid)
                if held is not None and held != conn:
                    a.clientid_reuse += 1   # a different live connection currently holds this id
                active_cid[cid] = conn
                conn_cid[conn] = cid
        elif t == mqtt.DISCONNECT:
            if conn is not None:
                cid0 = conn_cid.pop(conn, None)
                if cid0 is not None and active_cid.get(cid0) == conn:
                    del active_cid[cid0]
        elif t == mqtt.PUBLISH:
            jump, bad = plaus.update(m.get("topic") or "", m.get("app_payload", b""))
            if jump > a.vital_max_jump:
                a.vital_max_jump = jump
            if bad:
                a.vital_jump_count += 1
                a.implausible = 1


def _finalize(mac, win, a, session, ip_roles, conflict, attacker_mac):
    sizes = np.array(a.sizes, dtype=float) if a.sizes else np.array([0.0])
    tss = sorted(a.ts)
    iats = np.diff(tss) if len(tss) > 1 else np.array([0.0])
    pkts_total = a.pkts_sent + a.pkts_recv
    host_ip = max(a.ip_counts, key=a.ip_counts.get) if a.ip_counts else None
    # Role from the MAC, not from traffic (review M3): the attacker forwards packets carrying the
    # victims' IPs during ARP-MITM, so an IP-derived role would mislabel the attacker as a monitor/
    # broker. The MAC is the stable host identity; everything else keys off ip_roles by host_ip.
    host_role = "attacker" if mac == attacker_mac else ip_roles.get(host_ip, "unknown")
    mqtt_total = sum(a.mqtt_counts.values())
    payloads = np.array(a.mqtt_payloads, dtype=float) if a.mqtt_payloads else np.array([0.0])

    # Causal flow features (review B1 + m-new-1): computed from packets seen up to this window's end
    # only, using the current-life first-seen captured on the accumulator (reset on FIN/RST close or
    # >FLOW_IDLE_TIMEOUT idle), so a reopened 4-tuple is a new flow and age never grows unbounded.
    win_end = win + common.WINDOW_SECONDS
    nf_active_flows = len(a.flows)
    nf_new_flows = sum(1 for k in a.flows
                       if common.window_start(a.flow_fs[k]) == win)
    ages = [max(0.0, win_end - a.flow_fs[k]) for k in a.flows]
    nf_mean_flow_age = float(np.mean(ages)) if ages else 0.0

    return {
        # --- metadata (never model inputs) ---
        "session": session,
        "window_start_epoch": win,
        "window_start_iso": common.epoch_to_iso(win),
        "host_mac": mac,
        "host_ip": host_ip,
        "host_role": host_role,
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
        # --- features: protocol mix (review m-new-5: frac_udp dropped — the five fractions sum to
        # 1, so one is exactly collinear; udp share is recoverable from the rest) ---
        "frac_tcp": a.n_tcp / pkts_total if pkts_total else 0.0,
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
