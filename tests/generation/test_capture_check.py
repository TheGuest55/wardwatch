"""Regression test for review M8: arp_mitm must report credential capture only when a genuine MQTT
CONNECT carrying the victim's identity was actually captured — not from a tcpdump hostname-resolution
artefact. Tests the paho-free CONNECT parser and the pcap reader on crafted bytes.
"""
import pathlib
import struct
import tempfile
import unittest

from orchestrator import capture_check  # services/attacker on PYTHONPATH


def _varint(n):
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def build_connect(client_id, username=None, level=4):
    name = b"MQTT"
    var = len(name).to_bytes(2, "big") + name + bytes([level])
    flags = 0x02 | (0x80 if username else 0)
    var += bytes([flags]) + (60).to_bytes(2, "big")
    if level >= 5:
        var += b"\x00"                              # 0-length properties
    body = var + len(client_id.encode()).to_bytes(2, "big") + client_id.encode()
    if username:
        body += len(username.encode()).to_bytes(2, "big") + username.encode()
    return bytes([0x10]) + _varint(len(body)) + body


def build_pcap(payload, dport=1883):
    eth = b"\xaa" * 6 + b"\xbb" * 6 + (0x0800).to_bytes(2, "big")
    tcp = (12345).to_bytes(2, "big") + dport.to_bytes(2, "big") + b"\x00" * 8 + bytes([5 << 4, 0]) \
        + b"\x00" * 6 + payload
    ip = bytes([0x45, 0]) + (20 + len(tcp)).to_bytes(2, "big") + b"\x00" * 4 + bytes([64, 6]) \
        + b"\x00\x00" + b"\x0a\x00\x00\x01" + b"\x0a\x00\x00\x02"
    frame = eth + ip + tcp
    gh = struct.pack("<IHHiIII", 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1)
    rec = struct.pack("<IIII", 0, 0, len(frame), len(frame))
    return gh + rec + frame


class TestParseConnect(unittest.TestCase):
    def test_v311_connect_with_username(self):
        c = capture_check.parse_connect(build_connect("bed1-monitor", "bed1-monitor", level=4))
        self.assertEqual(c["client_id"], "bed1-monitor")
        self.assertEqual(c["username"], "bed1-monitor")

    def test_v5_connect_parses_client_id(self):
        c = capture_check.parse_connect(build_connect("bed3-monitor", "bed3-monitor", level=5))
        self.assertEqual(c["client_id"], "bed3-monitor")

    def test_non_connect_returns_none(self):
        self.assertIsNone(capture_check.parse_connect(b"\x30\x05hello"))  # PUBLISH
        self.assertIsNone(capture_check.parse_connect(b""))


class TestVictimConnectSeen(unittest.TestCase):
    def _write(self, data):
        f = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
        f.write(data); f.close()
        return f.name

    def test_true_for_victim_connect(self):
        p = self._write(build_pcap(build_connect("bed2-monitor", "bed2-monitor")))
        self.assertTrue(capture_check.victim_connect_seen(p, "bed2-monitor"))
        pathlib.Path(p).unlink()

    def test_false_for_other_user(self):
        p = self._write(build_pcap(build_connect("bed2-monitor", "bed2-monitor")))
        self.assertFalse(capture_check.victim_connect_seen(p, "bed4-monitor"))
        pathlib.Path(p).unlink()

    def test_false_when_not_on_mqtt_port(self):
        p = self._write(build_pcap(build_connect("bed2-monitor", "bed2-monitor"), dport=4444))
        self.assertFalse(capture_check.victim_connect_seen(p, "bed2-monitor"))
        pathlib.Path(p).unlink()

    def test_false_for_non_connect_traffic(self):
        p = self._write(build_pcap(b"\x30\x0cward1/x payload"))  # a PUBLISH, no CONNECT
        self.assertFalse(capture_check.victim_connect_seen(p, "bed2-monitor"))
        pathlib.Path(p).unlink()


if __name__ == "__main__":
    unittest.main()
