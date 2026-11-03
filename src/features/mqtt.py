"""Minimal MQTT 3.1.1/5.0 control-packet parser over a TCP payload.

scapy does not decode MQTT reliably when several control packets are coalesced into one TCP
segment (common for our small, chatty traffic), so we walk the fixed headers ourselves. We only
need window-level signals, never a full decode: packet type, payload size, CONNECT client-id,
CONNACK return code, SUBSCRIBE topic filters, PUBLISH topic. Best-effort: a packet whose declared
remaining-length runs past the captured bytes (a PUBLISH split across TCP segments) is parsed as
far as the buffer allows and then we stop walking that segment.

Nothing here is a model input directly; these fields are aggregated into derived, non-identifying
features (counts, rates, wildcard/depth/is-command booleans) in windows.py. Raw topics and
client-ids stay out of the feature table per the leakage contract.
"""

# MQTT control packet type (high nibble of byte 0)
CONNECT, CONNACK, PUBLISH, PUBACK = 1, 2, 3, 4
SUBSCRIBE, SUBACK, UNSUBSCRIBE, UNSUBACK = 8, 9, 10, 11
PINGREQ, PINGRESP, DISCONNECT = 12, 13, 14

TYPE_NAMES = {
    CONNECT: "connect", CONNACK: "connack", PUBLISH: "publish", PUBACK: "puback",
    SUBSCRIBE: "subscribe", SUBACK: "suback", UNSUBSCRIBE: "unsubscribe",
    UNSUBACK: "unsuback", PINGREQ: "pingreq", PINGRESP: "pingresp", DISCONNECT: "disconnect",
}


def _varint(buf, i):
    """Decode an MQTT remaining-length varint starting at buf[i]. Returns (value, next_index) or
    (None, None) if the field is truncated/invalid."""
    mult, value, n = 1, 0, 0
    while True:
        if i >= len(buf) or n >= 4:
            return None, None
        b = buf[i]
        value += (b & 0x7F) * mult
        i += 1
        n += 1
        if not (b & 0x80):
            return value, i
        mult *= 128


def _str(buf, i):
    """Decode a 2-byte-length-prefixed UTF-8 string at buf[i]. Returns (text, next_index) or
    (None, None) on truncation."""
    if i + 2 > len(buf):
        return None, None
    ln = (buf[i] << 8) | buf[i + 1]
    j = i + 2
    if j + ln > len(buf):
        return None, None
    try:
        return buf[j:j + ln].decode("utf-8", "replace"), j + ln
    except Exception:
        return None, j + ln


def parse_segment(payload):
    """Parse one TCP payload (bytes) into a list of MQTT control-packet dicts.

    Each dict always has: type (int), type_name (str), length (fixed header remaining length).
    Type-specific extras: connect -> client_id; connack -> return_code (int); publish -> topic,
    qos, payload_size (bytes of the application message); subscribe -> topics (list of filters)."""
    out = []
    i, n = 0, len(payload)
    while i < n:
        b0 = payload[i]
        ptype = (b0 & 0xF0) >> 4
        if ptype not in TYPE_NAMES:
            break  # not (or no longer) aligned on an MQTT packet; stop walking this segment
        flags = b0 & 0x0F
        rem, j = _varint(payload, i + 1)
        if rem is None:
            break
        body_start = j
        body_end = min(j + rem, n)  # best-effort if the body is split across segments
        body = payload[body_start:body_end]
        rec = {"type": ptype, "type_name": TYPE_NAMES[ptype], "length": rem}

        if ptype == CONNECT:
            # variable header: protocol name (str), level (1), connect flags (1), keepalive (2);
            # MQTT 5 then inserts a properties block (varint length + that many bytes); then the
            # payload begins with the client id (str).
            name, k = _str(body, 0)
            if name is not None and k + 4 <= len(body):
                level = body[k]
                k += 4  # level(1) + connect flags(1) + keepalive(2)
                if level >= 5:  # MQTT 5: skip the CONNECT properties block
                    plen, k2 = _varint(body, k)
                    if plen is not None:
                        k = k2 + plen
                cid, _ = _str(body, k)
                rec["client_id"] = cid
        elif ptype == CONNACK:
            # [ack flags][reason/return code]
            if len(body) >= 2:
                rec["return_code"] = body[1]
        elif ptype == PUBLISH:
            qos = (flags & 0x06) >> 1
            topic, k = _str(body, 0)
            rec["topic"] = topic
            rec["qos"] = qos
            if k is not None:
                if qos > 0:
                    k += 2  # packet identifier
                rec["payload_size"] = max(0, len(body) - k)
                # Application bytes, kept only so windows.py can derive the payload-plausibility
                # feature (vital jumps). Never stored in the feature table.
                rec["app_payload"] = bytes(body[k:])
            else:
                rec["payload_size"] = 0
                rec["app_payload"] = b""
        elif ptype == SUBSCRIBE:
            # [packet id (2)] then repeated (topic filter str, options byte)
            topics, k = [], 2
            while k < len(body):
                tf, k2 = _str(body, k)
                if tf is None:
                    break
                topics.append(tf)
                k = k2 + 1  # skip the options/qos byte
            rec["topics"] = topics

        out.append(rec)
        if body_end < j + rem:
            break  # truncated body: the rest is in another segment; stop here
        i = body_end
    return out
