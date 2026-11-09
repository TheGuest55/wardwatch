"""Pipeline tests for review m-new-1 (flow lifecycle: close on FIN/RST, expire after idle) and
m-new-5 (drop the collinear frac_udp; count gratuitous ARP requests, not only replies)."""
import unittest

from src.features import windows
from src import common

W = int(common.WINDOW_SECONDS)


def _tcp(ts, a_mac, b_mac, a_ip, b_ip, sport, dport, length=100, fin=False, rst=False, syn=False):
    return {"ts": float(ts), "length": length, "eth_src": a_mac, "eth_dst": b_mac,
            "ip_src": a_ip, "ip_dst": b_ip, "sport": sport, "dport": dport,
            "l3": "ip", "l4": "tcp", "syn": syn, "ack": True, "rst": rst, "fin": fin}


def _arp(ts, src_mac, op, psrc, pdst, length=42):
    return {"ts": float(ts), "length": length, "eth_src": src_mac, "eth_dst": windows.BROADCAST,
            "ip_src": None, "ip_dst": None, "sport": None, "dport": None,
            "l3": "arp", "l4": None, "arp_op": op, "arp_psrc": psrc, "arp_pdst": pdst,
            "arp_hwsrc": src_mac}


def _rows_by_win(rows, mac):
    return {r["window_start_epoch"]: r for r in rows if r["host_mac"] == mac}


class TestFlowLifecycle(unittest.TestCase):
    A, B = "aa:aa:aa:aa:aa:aa", "bb:bb:bb:bb:bb:bb"
    T = 1_000_000.0  # on a window boundary

    def test_fin_closes_flow_then_reopens_as_new(self):
        pkts = [
            _tcp(self.T + 0, self.A, self.B, "10.0.0.1", "10.0.0.2", 5000, 1883, syn=True),
            _tcp(self.T + 2, self.A, self.B, "10.0.0.1", "10.0.0.2", 5000, 1883, fin=True),
            _tcp(self.T + 7, self.A, self.B, "10.0.0.1", "10.0.0.2", 5000, 1883),  # next window
        ]
        rows, _ = windows.build_rows(iter(pkts), "t", "zz:zz", "9.9.9.9", {})
        byw = _rows_by_win(rows, self.A)
        w1 = common.window_start(self.T + 7)
        # The reopened 4-tuple is a NEW flow in w1, and its age is bounded by one window — not the
        # ~7 s that a never-resetting first-seen would give.
        self.assertEqual(byw[w1]["nf_new_flows"], 1)
        self.assertEqual(byw[w1]["nf_active_flows"], 1)
        self.assertLessEqual(byw[w1]["nf_mean_flow_age"], common.WINDOW_SECONDS)

    def test_idle_timeout_reopens_flow(self):
        pkts = [
            _tcp(self.T + 0, self.A, self.B, "10.0.0.1", "10.0.0.2", 5000, 1883),
            _tcp(self.T + 200, self.A, self.B, "10.0.0.1", "10.0.0.2", 5000, 1883),  # >120 s idle
        ]
        rows, _ = windows.build_rows(iter(pkts), "t", "zz:zz", "9.9.9.9", {})
        byw = _rows_by_win(rows, self.A)
        wlate = common.window_start(self.T + 200)
        self.assertEqual(byw[wlate]["nf_new_flows"], 1)          # reopened after idle
        self.assertLessEqual(byw[wlate]["nf_mean_flow_age"], common.WINDOW_SECONDS)

    def test_continuous_flow_not_reset_under_idle(self):
        # 1 pkt/s for 300 s: gaps < timeout, no FIN -> one flow, age grows with elapsed time.
        pkts = [_tcp(self.T + i, self.A, self.B, "10.0.0.1", "10.0.0.2", 5000, 1883)
                for i in range(300)]
        rows, _ = windows.build_rows(iter(pkts), "t", "zz:zz", "9.9.9.9", {})
        byw = _rows_by_win(rows, self.A)
        wlate = common.window_start(self.T + 250)
        self.assertEqual(byw[wlate]["nf_new_flows"], 0)
        self.assertGreaterEqual(byw[wlate]["nf_mean_flow_age"], 250.0)


class TestProtoFeatures(unittest.TestCase):
    A, B = "aa:aa:aa:aa:aa:aa", "bb:bb:bb:bb:bb:bb"
    T = 2_000_000.0

    def test_frac_udp_dropped(self):
        pkts = [_tcp(self.T, self.A, self.B, "10.0.0.1", "10.0.0.2", 5000, 1883)]
        rows, _ = windows.build_rows(iter(pkts), "t", "zz:zz", "9.9.9.9", {})
        row = rows[0]
        self.assertNotIn("frac_udp", row)
        self.assertIn("frac_tcp", row)

    def test_gratuitous_arp_request_counted(self):
        attacker = "4a:30:f3:96:5f:fc"
        pkts = [
            _arp(self.T + 0, attacker, 1, "172.28.10.11", "172.28.10.11"),  # gratuitous REQUEST
            _arp(self.T + 1, attacker, 2, "172.28.10.10", "172.28.10.10"),  # gratuitous REPLY
            _arp(self.T + 2, attacker, 1, "172.28.10.1", "172.28.10.9"),    # normal request
        ]
        rows, _ = windows.build_rows(iter(pkts), "t", attacker, "9.9.9.9", {})
        row = _rows_by_win(rows, attacker)[common.window_start(self.T)]
        self.assertEqual(row["arp_gratuitous"], 2)   # request + reply forms both counted
        self.assertEqual(row["arp_requests"], 2)
        self.assertEqual(row["arp_replies"], 1)


if __name__ == "__main__":
    unittest.main()
