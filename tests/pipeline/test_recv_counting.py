"""Pipeline test for review M4: MQTT/flag/protocol features count packets a host RECEIVES as well
as sends, and frac_* sum to ~1 (sender-only numerators made them sum to ~0.5)."""
import unittest

from src.features import windows
from src import common

SUB = "52:54:00:55:55:55"  # locally-administered unicast (even first octet)
BROKER = "00:00:00:00:00:10"
BROKER_IP, SUB_IP = "172.28.10.10", "172.28.10.20"


def _publish_to(ts, dmac, dip, dport):
    return {"ts": float(ts), "length": 120, "eth_src": BROKER, "eth_dst": dmac,
            "ip_src": BROKER_IP, "ip_dst": dip, "sport": 1883, "dport": dport,
            "l3": "ip", "l4": "tcp", "syn": False, "ack": True, "rst": False, "fin": False,
            "mqtt": [{"type": 3, "type_name": "publish", "topic": "ward1/bed1/monitor/vitals",
                      "qos": 0, "payload_size": 80, "app_payload": b""}]}


def _connack_fail_to(ts, dmac, dip, dport):
    return {"ts": float(ts), "length": 60, "eth_src": BROKER, "eth_dst": dmac,
            "ip_src": BROKER_IP, "ip_dst": dip, "sport": 1883, "dport": dport,
            "l3": "ip", "l4": "tcp", "syn": False, "ack": True, "rst": False, "fin": False,
            "mqtt": [{"type": 2, "type_name": "connack", "return_code": 5}]}


class TestRecvCounting(unittest.TestCase):
    def test_subscriber_sees_received_publishes(self):
        t = 8_000_000.0
        pkts = [_publish_to(t + i, SUB, SUB_IP, 40000) for i in range(4)]
        rows, _ = windows.build_rows(iter(pkts), "t", "zz:zz", "9.9.9.9", {})
        sub = [r for r in rows if r["host_mac"] == SUB]
        self.assertTrue(sub)
        self.assertGreaterEqual(sum(r["mqtt_publish"] for r in sub), 4)  # received publishes visible
        self.assertGreater(sum(r["mqtt_pkts"] for r in sub), 0)

    def test_received_connack_failure_visible_on_client(self):
        t = 8_100_000.0
        rows, _ = windows.build_rows(iter([_connack_fail_to(t, SUB, SUB_IP, 40000)]),
                                     "t", "zz:zz", "9.9.9.9", {})
        sub = [r for r in rows if r["host_mac"] == SUB]
        self.assertEqual(sum(r["mqtt_connack_failures"] for r in sub), 1)

    def test_frac_sum_to_one(self):
        t = 8_200_000.0
        pkts = [_publish_to(t + i, SUB, SUB_IP, 40000) for i in range(3)]
        rows, _ = windows.build_rows(iter(pkts), "t", "zz:zz", "9.9.9.9", {})
        for r in rows:
            if r["pkts_total"]:
                s = (r["frac_tcp"] + r["frac_udp"] + r["frac_arp"]
                     + r["frac_icmp"] + r["frac_other"])
                self.assertAlmostEqual(s, 1.0, places=6, msg=r["host_mac"])


if __name__ == "__main__":
    unittest.main()
