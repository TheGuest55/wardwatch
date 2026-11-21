"""src/realtime/livetail.py: a growing capture read through the same decoder gives the same packets, rotation and partial writes included;
and (full data) the whole live-tail pipeline output equals the replay output on a short recorded capture."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
from scapy.utils import wrpcap

from src.features import packets
from src.realtime import livetail
from tests.needs import needs


def make_pcap(path, n, t0):
    pk = []
    for i in range(n):
        p = Ether(src="aa:aa:aa:aa:aa:01", dst="aa:aa:aa:aa:aa:02") / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40000 + i, dport=1883)
        p.time = t0 + i * 0.01
        pk.append(p)
    wrpcap(str(path), pk)


def feed(src, dst, chunk, pause, done):
    """Copy src into dst in chunks that deliberately split records, like a writer that is still going."""
    data = Path(src).read_bytes()
    with open(dst, "wb") as f:
        for i in range(0, len(data), chunk):
            f.write(data[i:i + chunk])
            f.flush()
            time.sleep(pause)
    done.set()


class TestTail(unittest.TestCase):
    def test_a_growing_file_gives_the_same_packets_as_reading_it_whole(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            make_pcap(d / "src.pcap", 300, 1_700_000_000.0)
            want = list(packets.iter_packets(d / "src.pcap"))
            out = d / "tail"
            out.mkdir()
            done = threading.Event()
            threading.Thread(target=feed, args=(d / "src.pcap", out / "ward_20261121T000000Z.pcap", 997, 0.002, done), daemon=True).start()
            got = list(livetail.follow_packets(out, idle_stop_s=1.0, stop=done.is_set, poll_s=0.01))
            self.assertEqual(got, want)

    def test_rotation_moves_on_to_the_next_file_in_name_order(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            make_pcap(d / "a.pcap", 50, 1_700_000_000.0)
            make_pcap(d / "b.pcap", 50, 1_700_000_100.0)
            out = d / "tail"
            out.mkdir()
            (out / "ward_1.pcap").write_bytes((d / "a.pcap").read_bytes())
            (out / "ward_2.pcap").write_bytes((d / "b.pcap").read_bytes())
            got = list(livetail.follow_packets(out, idle_stop_s=0.5, poll_s=0.01))
            self.assertEqual([p["ts"] for p in got][:3] + [got[-1]["ts"]],
                             [1_700_000_000.0, 1_700_000_000.01, 1_700_000_000.02, 1_700_000_100.49])
            self.assertEqual(len(got), 100)

    def test_no_capture_at_all_ends_after_the_wait(self):
        with tempfile.TemporaryDirectory() as d:
            t = time.time()
            self.assertEqual(list(livetail.follow_packets(d, wait_first_s=0.3, poll_s=0.01)), [])
            self.assertLess(time.time() - t, 3)


@needs("RAW", "MODELS", "DATASET")
class TestPipelineEquivalence(unittest.TestCase):
    def test_live_tail_alerts_rows_and_incidents_equal_the_replay_on_a_recorded_excerpt(self):
        from src.experiments import run_e
        from src.realtime import alerts as alerts_mod
        from src.realtime import stream
        from src.realtime.pipeline import Pair
        from src.triage import offline
        a = json.loads((offline.RUNS / "triage_A.json").read_text())
        key, gap = a["params"]["grouping_key"], a["params"]["grouping_gap_s"]
        pair = Pair()
        pcap = stream.pcap_paths("r05")[0]
        # the first 25 s of r05 (the start-up windows alert), copied to a growing file in odd-sized chunks
        raw = [rec for rec in stream.session_packets("r05", max_seconds=25)]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ra = run_e.replay_session("r05", pair, key, gap, speed=0, live_dir=d / "a", packet_iter=iter(raw), alerts=alerts_mod.AlertSink(d / "a.jsonl", {"mode": "x"}), with_post=False)
            tail = d / "tail"
            tail.mkdir()
            data = pcap.read_bytes()
            # find the byte length of the first 25 s: the packets' file offsets are not needed; feed the whole file and stop at the same packet count
            n = len(raw)
            done = threading.Event()
            threading.Thread(target=feed, args=(pcap, tail / "ward_20261109T000000Z.pcap", 65_537, 0.0005, done), daemon=True).start()

            def limited():
                for i, rec in enumerate(livetail.follow_packets(tail, idle_stop_s=2.0, stop=done.is_set, poll_s=0.01)):
                    if i >= n:
                        return
                    yield rec
            rb = run_e.replay_session("r05", pair, key, gap, speed=0, live_dir=d / "b", packet_iter=limited(), alerts=alerts_mod.AlertSink(d / "b.jsonl", {"mode": "x"}), with_post=False)
            self.assertEqual(ra["rows"], rb["rows"])
            la = [json.loads(l) for l in (d / "a.jsonl").read_text().splitlines()][1:]
            lb = [json.loads(l) for l in (d / "b.jsonl").read_text().splitlines()][1:]
            self.assertEqual(la, lb)
            self.assertEqual(ra["triager"].records, rb["triager"].records)
            self.assertGreater(len(la), 0)


if __name__ == "__main__":
    unittest.main()
