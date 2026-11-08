"""Pipeline test for review B1: transport-flow features must be causal — computed from packets seen
up to each window's end only, never encoding a flow's whole (future) lifetime.

Runs in the analysis image (numpy present). Feeds synthetic per-packet dicts to windows.build_rows.
"""
import unittest

from src.features import windows
from src import common

W = int(common.WINDOW_SECONDS)


def _pkt(ts, a_mac, b_mac, a_ip, b_ip, sport, dport, length=100):
    return {
        "ts": float(ts), "length": length,
        "eth_src": a_mac, "eth_dst": b_mac,
        "ip_src": a_ip, "ip_dst": b_ip,
        "sport": sport, "dport": dport,
        "l3": "ip", "l4": "tcp",
        "syn": False, "ack": True, "rst": False, "fin": False,
    }


class TestCausalFlow(unittest.TestCase):
    def _rows_by_win(self, rows, mac):
        return {r["window_start_epoch"]: r for r in rows if r["host_mac"] == mac}

    def test_first_window_age_is_small_not_lifetime(self):
        # One long-lived flow A<->B, 1 pkt/s for 600 s (spans 120 windows).
        A, B = "aa:aa:aa:aa:aa:aa", "bb:bb:bb:bb:bb:bb"
        start = 1_000_000.0  # already on a window boundary
        pkts = [_pkt(start + i, A, B, "10.0.0.1", "10.0.0.2", 5000, 1883) for i in range(600)]
        rows, _ = windows.build_rows(iter(pkts), "t", "zz:zz", "9.9.9.9", {})
        byw = self._rows_by_win(rows, A)

        w0 = common.window_start(start)
        first = byw[w0]
        # First window: age <= one window, NOT the ~600 s lifetime; it is new and active here.
        self.assertLessEqual(first["nf_mean_flow_age"], common.WINDOW_SECONDS)
        self.assertEqual(first["nf_new_flows"], 1)
        self.assertEqual(first["nf_active_flows"], 1)

        # A much later window: still one active flow, no longer "new", age grows but never exceeds
        # elapsed-to-window-end (causality) and is far below the full lifetime at that point.
        wlate = common.window_start(start + 300)
        late = byw[wlate]
        self.assertEqual(late["nf_new_flows"], 0)
        self.assertEqual(late["nf_active_flows"], 1)
        self.assertLessEqual(late["nf_mean_flow_age"], (wlate + W) - start)
        self.assertGreaterEqual(late["nf_mean_flow_age"], 300.0)  # ~ elapsed so far

    def test_active_counts_distinct_flows(self):
        A, B, C = "aa:aa:aa:aa:aa:aa", "bb:bb:bb:bb:bb:bb", "cc:cc:cc:cc:cc:cc"
        t = 2_000_000.0
        pkts = [
            _pkt(t + 0, A, B, "10.0.0.1", "10.0.0.2", 5000, 1883),
            _pkt(t + 1, A, C, "10.0.0.1", "10.0.0.3", 5001, 1883),  # 2nd distinct flow, same window
            _pkt(t + 2, A, B, "10.0.0.1", "10.0.0.2", 5000, 1883),  # same flow as #1 (reverse ok)
        ]
        rows, _ = windows.build_rows(iter(pkts), "t", "zz:zz", "9.9.9.9", {})
        a_row = self._rows_by_win(rows, A)[common.window_start(t)]
        self.assertEqual(a_row["nf_active_flows"], 2)
        self.assertEqual(a_row["nf_new_flows"], 2)


if __name__ == "__main__":
    unittest.main()
