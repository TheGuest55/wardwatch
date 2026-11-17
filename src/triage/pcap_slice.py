"""Evidence slices: the packets of an incident, cut from the raw session pcaps (read-only) and hashed.

WARNING: a slice is raw traffic and contains the lab's plaintext MQTT credentials (the broker runs without
TLS). Slices are written only under the gitignored data/processed/v2/{triage,live}/slices/ and must never be
committed or published.

The slice keeps every packet whose Ethernet source or destination is one of the given MACs and whose timestamp lies
in [t_start, t_end). It is written as a classic little-endian microsecond pcap with a fixed global header and the
packets' original bytes and timestamps, so the same inputs always give the same bytes (and the same SHA-256).
The source pcaps are only ever opened for reading, and only the pcaps passed in.
"""
import hashlib
import struct
from pathlib import Path

GLOBAL_HDR = struct.Struct("<IHHiIII")
REC_HDR = struct.Struct("<IIII")
MAGIC_LE = 0xA1B2C3D4
MAGIC_LE_NS = 0xA1B23C4D


def _read_pcap(path, offset=None):
    """Yield (ts_sec, ts_usec, incl_len, orig_len, data) for a classic pcap (LE or BE, usec or nsec). `offset`
    (a record boundary from PcapIndex) starts reading there instead of at the first packet."""
    with open(path, "rb") as f:
        head = f.read(24)
        if len(head) < 24:
            return
        magic_le = struct.unpack("<I", head[:4])[0]
        if magic_le in (MAGIC_LE, MAGIC_LE_NS):
            end, nano = "<", magic_le == MAGIC_LE_NS
        elif struct.unpack(">I", head[:4])[0] in (MAGIC_LE, MAGIC_LE_NS):
            end, nano = ">", struct.unpack(">I", head[:4])[0] == MAGIC_LE_NS
        else:
            raise ValueError(f"{path}: not a classic pcap (magic {head[:4].hex()})")
        rec = struct.Struct(end + "IIII")
        linktype = struct.unpack(end + "I", head[20:24])[0]
        yield ("hdr", linktype)
        if offset:
            f.seek(offset)
        while True:
            h = f.read(16)
            if len(h) < 16:
                return
            sec, frac, incl, orig = rec.unpack(h)
            data = f.read(incl)
            yield (sec, frac // 1000 if nano else frac, incl, orig, data)


SLACK_S = 5.0          # capture timestamps are monotonic to within a microsecond; 5 s is a wide safety margin
INDEX_EVERY = 2048


class PcapIndex:
    """Record-boundary offsets every INDEX_EVERY packets with their timestamps, so a slice reads only the packets
    near its time range instead of the whole file. Built by one pass over the record headers (no payload read).
    The output of write_slice is identical with or without an index as long as timestamps are monotonic to
    within SLACK_S (checked by the tests on the real pcaps)."""

    def __init__(self, path):
        self.path, self.offsets, self.ts = Path(path), [], []
        with open(path, "rb") as f:
            head = f.read(24)
            endian = "<" if struct.unpack("<I", head[:4])[0] in (MAGIC_LE, MAGIC_LE_NS) else ">"
            nano = struct.unpack(endian + "I", head[:4])[0] == MAGIC_LE_NS
            rec = struct.Struct(endian + "IIII")
            pos, n = 24, 0
            while True:
                h = f.read(16)
                if len(h) < 16:
                    break
                sec, frac, incl, _ = rec.unpack(h)
                if n % INDEX_EVERY == 0:
                    self.offsets.append(pos)
                    self.ts.append(sec + (frac / 1e9 if nano else frac / 1e6))
                pos += 16 + incl
                f.seek(pos)
                n += 1

    def start_offset(self, t_start):
        import bisect
        i = bisect.bisect_left(self.ts, t_start - SLACK_S) - 1
        return self.offsets[max(i, 0)] if self.offsets else None


_INDEX = {}


def get_index(path):
    p = str(path)
    if p not in _INDEX:
        _INDEX[p] = PcapIndex(p)
    return _INDEX[p]


def _mac(b):
    return ":".join(f"{x:02x}" for x in b)


def write_slice(pcaps, macs, t_start, t_end, out_path, indexed=True):
    """Write the slice; return {"path", "sha256", "n_packets", "bytes"}. Packets keep file order. With indexed=True
    only the part of each pcap near [t_start, t_end) is read (same output; see PcapIndex)."""
    macs = {m.lower() for m in macs}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    h = hashlib.sha256()
    with open(out_path, "wb") as out:
        def put(b):
            out.write(b)
            h.update(b)
        put(GLOBAL_HDR.pack(MAGIC_LE, 2, 4, 0, 0, 262144, 1))
        for p in pcaps:
            off = get_index(p).start_offset(t_start) if indexed else None
            for item in _read_pcap(p, off):
                if item[0] == "hdr":
                    continue
                sec, usec, incl, orig, data = item
                t = sec + usec / 1e6
                if indexed and t >= t_end + SLACK_S:
                    break
                if not (t_start <= t < t_end) or len(data) < 12:
                    continue
                if _mac(data[0:6]) in macs or _mac(data[6:12]) in macs:
                    put(REC_HDR.pack(sec, usec, incl, orig))
                    put(data)
                    n += 1
    return {"path": str(out_path), "sha256": h.hexdigest(), "n_packets": n, "bytes": out_path.stat().st_size}
