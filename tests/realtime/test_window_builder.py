"""WindowBuilder: incremental == batch, late packets, and the stored r05 session rows."""
import pickle
import random
import unittest

import numpy as np
import pandas as pd

from src import common
from src.features import windows
from src.realtime import check_session as cs
from src.realtime import stream
from tests.realtime.helpers import IP_ROLES, MACS, T0, synthetic_packets, tcp

ATT, ATT_IP = MACS["c"], "10.0.0.13"


def _builder(with_peers=False):
    return windows.WindowBuilder("t", ATT, ATT_IP, IP_ROLES, with_peers=with_peers)


def _strip(rows):
    return [{k: v for k, v in r.items() if k != "peer_pkts"} for r in rows]


class TestIncrementalEqualsBatch(unittest.TestCase):
    def test_build_rows_is_add_all_then_flush(self):
        pk = synthetic_packets()
        batch, conflicts = windows.build_rows(iter(pk), "t", ATT, ATT_IP, IP_ROLES)
        wb = _builder()
        for r in pk:
            wb.add(r)
        rows = wb.flush()
        self.assertEqual(pickle.dumps(batch), pickle.dumps(rows))
        self.assertEqual(conflicts, wb.conflict_windows)
        self.assertTrue(any(r["ip_mac_conflict"] for r in batch))           # the ARP conflict is in the data
        self.assertTrue(any(r["mqtt_clientid_reuse"] for r in batch))        # client-id reuse is in the data

    def test_incremental_closes_equal_batch_rows_for_any_close_schedule(self):
        pk = synthetic_packets()
        batch, _ = windows.build_rows(iter(pk), "t", ATT, ATT_IP, IP_ROLES)
        want = {(r["host_mac"], r["window_start_epoch"]): r for r in batch}
        for seed in range(4):
            rnd = random.Random(seed)
            wb = _builder()
            got = []
            for r in pk:
                wb.add(r)
                if rnd.random() < 0.15:                                      # close a window once it is >= 1 s old
                    got += wb.close_before(r["ts"] - 1.0)
            got += wb.flush()
            self.assertEqual(wb.late_packets, 0)
            self.assertEqual(len(got), len(want))
            for r in got:
                self.assertEqual(pickle.dumps(r), pickle.dumps(want[(r["host_mac"], r["window_start_epoch"])]))

    def test_flow_features_survive_reopen_idle_and_cross_window_age(self):
        pk = synthetic_packets()
        wb = _builder()
        rows = []
        for r in pk:
            wb.add(r)
            rows += wb.close_before(r["ts"] - 1.0)
        rows += wb.flush()
        df = pd.DataFrame(rows)
        self.assertGreater(df["nf_mean_flow_age"].max(), 60)         # a long-lived flow ages across windows
        self.assertGreater(df["nf_new_flows"].sum(), 5)              # reopen after FIN / idle counts as new flows
        self.assertLess(df["nf_mean_flow_age"].max(), 400)

    def test_peer_pkts_is_metadata_only(self):
        pk = synthetic_packets()
        wb = _builder(with_peers=True)
        for r in pk:
            wb.add(r)
        rows = wb.flush()
        self.assertTrue(all("peer_pkts" in r for r in rows))
        self.assertTrue(set(rows[0]) - {"peer_pkts"} <= set(common.META_COLUMNS) | set(common.FEATURES)
                        | set(common.METADATA_EXTRAS) | {"window_start_iso"})
        self.assertNotIn("peer_pkts", common.FEATURES)
        batch, _ = windows.build_rows(iter(pk), "t", ATT, ATT_IP, IP_ROLES)
        self.assertEqual(pickle.dumps(batch), pickle.dumps(_strip(rows)))   # same rows with or without peers

    def test_peer_counts_are_unicast_sent_and_received(self):
        wb = _builder(with_peers=True)
        wb.add(tcp(T0 + 1, "a", "br"))
        wb.add(tcp(T0 + 2, "a", "br"))
        wb.add(tcp(T0 + 3, "br", "a"))
        rows = {r["host_mac"]: r for r in wb.flush()}
        self.assertEqual(rows[MACS["a"]]["peer_pkts"], {MACS["br"]: [2, 1]})
        self.assertEqual(rows[MACS["br"]]["peer_pkts"], {MACS["a"]: [1, 2]})

    def test_late_packets_are_counted_and_never_dropped(self):
        wb = _builder()
        wb.add(tcp(T0 + 1, "a", "br"))
        first = wb.close_before(T0 + 20)                 # window [T0, T0+5) is long closed
        self.assertTrue(first)
        self.assertEqual(wb.late_packets, 0)
        wb.add(tcp(T0 + 2, "a", "br"))                   # belongs to a closed window
        self.assertEqual(wb.late_packets, 1)
        self.assertEqual(len(wb.late_windows), 1)
        rows = wb.flush()                                # processed anyway: visible as an extra row
        self.assertTrue(any(r["host_mac"] == MACS["a"] for r in rows))


class TestReplayTimer(unittest.TestCase):
    def test_windows_close_by_timer_not_by_next_packet(self):
        t = [0.0]
        slept = []

        def clock():
            return t[0]

        def sleep(d):
            slept.append(d)
            t[0] += d
        pk = [tcp(T0 + 1, "a", "br"), tcp(T0 + 2, "br", "a"), tcp(T0 + 100, "a", "br")]   # 98 s of silence
        wb = _builder()
        ticks = []
        stream.replay(iter(pk), wb, lambda rows, ctx: ticks.append((ctx["window_end"], len(rows))), speed=1.0,
                      clock=clock, sleep=sleep)
        self.assertEqual(wb.late_packets, 0)
        # the first window is delivered at its end + grace while the next packet is still 98 s away
        self.assertEqual(ticks[0][0], (int(T0) // 5) * 5 + 5)
        self.assertEqual(ticks[0][1], 2)
        self.assertGreaterEqual(sum(slept), 98.0 - 1e-6)     # real-time: slept to the packet timestamps

    def test_speed_zero_never_sleeps_and_gives_the_same_ticks(self):
        pk = synthetic_packets()
        out = {}
        for speed in (0, 50):
            wb = _builder()
            rows = []
            sl = []
            t = [0.0]
            stream.replay(iter(pk), wb, lambda r, ctx: rows.extend(r), speed=speed,
                          clock=lambda: t[0], sleep=lambda d: (sl.append(d), t.__setitem__(0, t[0] + d)))
            out[speed] = (pickle.dumps(rows), len(sl), wb.late_packets)
        self.assertEqual(out[0][0], out[50][0])
        self.assertEqual(out[0][1], 0)
        self.assertEqual((out[0][2], out[50][2]), (0, 0))


class TestAgainstStoredSession(unittest.TestCase):
    """Real r05 pcaps (mounted read-only): the streamed rows of fully closed windows equal the stored parquet."""

    def test_r05_first_15_minutes_equal_the_stored_rows(self):
        rows, wb, st = cs.build_session("r05", max_seconds=900)
        ref = pd.read_parquet(cs.V2 / "sessions" / "r05.parquet")
        w = np.array([r["window_start_epoch"] for r in rows])
        lo, hi = w.min() + 5, w.max()                          # drop the partial first and last windows
        sel = [r for r in rows if lo <= r["window_start_epoch"] < hi]
        ref = ref[(ref["window_start_epoch"] >= lo) & (ref["window_start_epoch"] < hi)]
        self.assertGreater(len(sel), 1500)
        self.assertEqual(cs.compare(sel, ref), [])
        self.assertEqual(wb.late_packets, 0)


if __name__ == "__main__":
    unittest.main()


class TestCheckSessionReport(unittest.TestCase):
    def test_report_carries_provenance_and_parquet_hashes(self):
        import json
        import os
        import tempfile
        from pathlib import Path
        env = {"WARDWATCH_GIT_COMMIT": "a" * 40, "WARDWATCH_GIT_DIRTY": "", "WARDWATCH_IMAGE_DIGEST": "sha256:" + "b" * 64}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "check_session.json"
                doc = cs.write_report([{"session": "x", "exact": True, "parquet_sha256": "c" * 64},
                                       {"session": "y", "exact": True, "parquet_sha256": "d" * 64}], out)
                got = json.loads(out.read_text())
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        p = got["provenance"]
        self.assertEqual(p["git_commit"], "a" * 40)
        self.assertIs(p["git_dirty"], False)
        self.assertEqual(p["image_digest"], "sha256:" + "b" * 64)
        self.assertRegex(p["utc"], r"^\d{4}-\d\d-\d\dT")
        self.assertEqual(p["parquet_sha256"], {"x": "c" * 64, "y": "d" * 64})
        self.assertTrue(got["all_exact"])

    def test_check_result_has_the_real_parquet_hash(self):
        import hashlib
        h = hashlib.sha256((cs.V2 / "sessions" / "r05.parquet").read_bytes()).hexdigest()
        from src.models import tracking
        self.assertEqual(tracking._sha256_file(cs.V2 / "sessions" / "r05.parquet"), h)
