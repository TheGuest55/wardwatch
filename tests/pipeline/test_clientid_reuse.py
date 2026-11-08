"""Pipeline test for review M5: client-id reuse = the id is held by a *different live* TCP
connection, not 'differs from the first MAC ever seen'. A benign device reconnecting after its
previous session closed must not be flagged; an overlapping second connection (eviction) is."""
import unittest

from src.features import windows
from src import common

DEV = "11:11:11:11:11:11"
ATT = "de:ad:be:ef:00:01"
BROKER = "00:00:00:00:00:10"
DEV_IP, ATT_IP, BROKER_IP = "172.28.10.11", "172.28.10.66", "172.28.10.10"
CID = "bed1-monitor"


def _connect(ts, smac, sip, sport):
    return {"ts": float(ts), "length": 90, "eth_src": smac, "eth_dst": BROKER,
            "ip_src": sip, "ip_dst": BROKER_IP, "sport": sport, "dport": 1883,
            "l3": "ip", "l4": "tcp", "syn": False, "ack": True, "rst": False, "fin": False,
            "mqtt": [{"type": 1, "type_name": "connect", "client_id": CID}]}


def _fin(ts, smac, sip, sport):
    return {"ts": float(ts), "length": 54, "eth_src": smac, "eth_dst": BROKER,
            "ip_src": sip, "ip_dst": BROKER_IP, "sport": sport, "dport": 1883,
            "l3": "ip", "l4": "tcp", "syn": False, "ack": True, "rst": False, "fin": True}


def _reuse_by_mac(rows):
    out = {}
    for r in rows:
        out[r["host_mac"]] = out.get(r["host_mac"], 0) + r["mqtt_clientid_reuse"]
    return out


class TestClientIdReuse(unittest.TestCase):
    def test_benign_reconnect_after_close_is_not_reuse(self):
        t = 5_000_000.0
        pkts = [
            _connect(t + 0, DEV, DEV_IP, 40001),
            _fin(t + 1, DEV, DEV_IP, 40001),         # clean close releases the id
            _connect(t + 2, DEV, DEV_IP, 40002),     # new connection, same id -> NOT reuse
        ]
        rows, _ = windows.build_rows(iter(pkts), "t", ATT, ATT_IP, {})
        self.assertEqual(_reuse_by_mac(rows).get(DEV, 0), 0)

    def test_overlapping_second_connection_is_reuse(self):
        t = 6_000_000.0
        pkts = [
            _connect(t + 0, DEV, DEV_IP, 40001),     # device holds the id (no close)
            _connect(t + 1, ATT, ATT_IP, 50001),     # attacker connects same id while live -> reuse
        ]
        rows, _ = windows.build_rows(iter(pkts), "t", ATT, ATT_IP, {})
        reuse = _reuse_by_mac(rows)
        self.assertEqual(reuse.get(ATT, 0), 1)       # the second (overlapping) connector is flagged
        self.assertEqual(reuse.get(DEV, 0), 0)       # the original owner is not

    def test_attacker_first_then_device_reconnect_not_flagged_on_device(self):
        # Review's s05 case: the first CONNECT captured for the id is the attacker's. After the
        # attacker's connection closes, the device reconnecting must not be flagged.
        t = 7_000_000.0
        pkts = [
            _connect(t + 0, ATT, ATT_IP, 50001),     # attacker claims id first (no prior owner)
            _fin(t + 1, ATT, ATT_IP, 50001),         # attacker connection closes -> id released
            _connect(t + 2, DEV, DEV_IP, 40001),     # device reconnects -> NOT reuse
        ]
        rows, _ = windows.build_rows(iter(pkts), "t", ATT, ATT_IP, {})
        self.assertEqual(_reuse_by_mac(rows).get(DEV, 0), 0)


if __name__ == "__main__":
    unittest.main()
