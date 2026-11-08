"""Truthful MQTT-CONNECT detection for arp_mitm (review M8).

The old check ran `tcpdump -r … -A | grep -c <victim>` WITHOUT `-n`, so tcpdump resolved IPs to
container hostnames (which contain e.g. `bed1-monitor`) and grep matched header lines — the run
logged `connect_seen=True` with no CONNECT actually captured. This module instead parses the
captured pcap and reports only a genuine MQTT CONNECT whose client id or username is the victim's.

Standard-library only (the attacker image has no scapy): a minimal classic-pcap + Ethernet/IPv4/TCP
reader feeds payloads to an MQTT CONNECT parser. Testable on the host.
"""
import struct

ETH_HLEN = 14
ETYPE_IPV4 = 0x0800


def _varint(buf, i):
    """MQTT remaining-length varint. Returns (value, next_index)."""
    mult, val = 1, 0
    for _ in range(4):
        b = buf[i]
        i += 1
        val += (b & 0x7F) * mult
        if not (b & 0x80):
            return val, i
        mult *= 128
    raise ValueError("malformed varint")


def parse_connect(buf):
    """If buf begins with an MQTT CONNECT, return {'client_id', 'username'}; else None."""
    try:
        if len(buf) < 2 or (buf[0] >> 4) != 1:  # control type 1 = CONNECT
            return None
        i = 1
        _, i = _varint(buf, i)                  # remaining length (unused)
        pn_len = struct.unpack_from(">H", buf, i)[0]; i += 2 + pn_len   # protocol name
        level = buf[i]; i += 1
        flags = buf[i]; i += 1
        i += 2                                  # keepalive
        if level >= 5:                          # MQTT 5 properties block
            plen, i = _varint(buf, i); i += plen
        cid_len = struct.unpack_from(">H", buf, i)[0]; i += 2
        client_id = buf[i:i + cid_len].decode("utf-8", "replace"); i += cid_len
        if flags & 0x04:                        # will: skip will-props(v5) + topic + message
            if level >= 5:
                wpl, i = _varint(buf, i); i += wpl
            for _ in range(2):
                ln = struct.unpack_from(">H", buf, i)[0]; i += 2 + ln
        username = None
        if flags & 0x80:                        # username present
            ul = struct.unpack_from(">H", buf, i)[0]; i += 2
            username = buf[i:i + ul].decode("utf-8", "replace")
        return {"client_id": client_id, "username": username}
    except (IndexError, struct.error, ValueError):
        return None


def iter_tcp_payloads(path):
    """Yield (dst_port, payload_bytes) for each IPv4/TCP packet in a classic pcap (Ethernet link)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 24:
        return
    magic = data[:4]
    if magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    elif magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    else:
        return                                  # not a classic pcap we understand
    off = 24                                     # global header
    n = len(data)
    while off + 16 <= n:
        _, _, caplen, _ = struct.unpack_from(endian + "IIII", data, off)
        off += 16
        pkt = data[off:off + caplen]
        off += caplen
        if len(pkt) < ETH_HLEN + 20:
            continue
        if struct.unpack_from(">H", pkt, 12)[0] != ETYPE_IPV4:
            continue
        ip = pkt[ETH_HLEN:]
        ihl = (ip[0] & 0x0F) * 4
        if (ip[0] >> 4) != 4 or ip[9] != 6 or len(ip) < ihl + 20:   # IPv4, proto 6 = TCP
            continue
        tcp = ip[ihl:]
        dport = struct.unpack_from(">H", tcp, 2)[0]
        doff = (tcp[12] >> 4) * 4
        payload = tcp[doff:]
        if payload:
            yield dport, payload


def victim_connect_seen(path, victim, port=1883):
    """True iff the pcap holds a genuine MQTT CONNECT (on `port`) whose client id or username is
    `victim`. `victim` is the device's MQTT username (also its client id in this lab)."""
    for dport, payload in iter_tcp_payloads(path):
        if dport != port:
            continue
        c = parse_connect(payload)
        if c and victim in (c.get("client_id"), c.get("username")):
            return True
    return False
