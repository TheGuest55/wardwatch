"""Live-tail: follow a capture that tcpdump is still writing, and hand its packets to the same replay loop.

    follow_packets(pcap_dir, idle_stop_s=..., stop=...)  ->  iterator of packet records (src/features/packets.py)

tcpdump (-U, packet-buffered) appends whole records to ward_<UTC>.pcap and starts a new file when it rotates. A TailFile
looks like a binary file to scapy's PcapReader, but a read that reaches the end of what has been written WAITS for more
instead of returning short; so packets are decoded by exactly the same code as in a replay (features/packets.iter_packets),
and `stream.replay(follow_packets(...), builder, on_tick, speed=1.0)` runs the same window builder, timers and triage as Experiment E.
A file ends when a newer file exists (rotation) and nothing more arrives, when the stop callback is true, or after `idle_stop_s`
seconds without any new byte. Standard library plus the repo's packet decoder.
"""
import time
from pathlib import Path

from ..features import packets


class TailFile:
    def __init__(self, path, poll_s=0.05, is_last=lambda: False, stop=lambda: False, idle_stop_s=None, clock=time.monotonic, sleep=time.sleep):
        self.name = str(path)
        self.f = open(path, "rb")
        self.poll_s, self.is_last, self.stop, self.idle_stop_s = poll_s, is_last, stop, idle_stop_s
        self.clock, self.sleep = clock, sleep
        self.last_data = clock()

    def read(self, n=-1):
        if n is None or n < 0:
            return self.f.read()
        buf = b""
        while len(buf) < n:
            chunk = self.f.read(n - len(buf))
            if chunk:
                buf += chunk
                self.last_data = self.clock()
                continue
            idle = bool(self.idle_stop_s) and self.clock() - self.last_data > self.idle_stop_s
            if not self.is_last() or self.stop() or idle:      # rotated away (all bytes read), told to stop, or idle: end of file
                break
            self.sleep(self.poll_s)
        return buf

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def follow_packets(pcap_dir, idle_stop_s=None, stop=lambda: False, poll_s=0.05, wait_first_s=None, clock=time.monotonic, sleep=time.sleep):
    """Packet records of every ward_*.pcap in `pcap_dir` as they are written, files in name (= UTC time) order."""
    d = Path(pcap_dir)
    done = set()
    t_wait = clock()
    while True:
        files = sorted(p for p in d.glob("*.pcap") if p.name not in done)
        if not files:
            if stop() or (wait_first_s is not None and not done and clock() - t_wait > wait_first_s) \
                    or (idle_stop_s and done and clock() - t_wait > idle_stop_s):
                return
            sleep(poll_s)
            continue
        cur = files[0]
        done.add(cur.name)

        def is_last(cur=cur):                            # True while no newer file exists: keep waiting for more bytes
            return not any(p.name > cur.name for p in d.glob("*.pcap"))

        try:
            tf = TailFile(cur, poll_s, is_last, stop, idle_stop_s, clock, sleep)
            # the global header may not be flushed yet: TailFile.read waits for it like any other bytes
            yield from packets.iter_packets(tf)
        except EOFError:
            pass
        finally:
            tf.close()
        t_wait = clock()
        if stop():
            return
