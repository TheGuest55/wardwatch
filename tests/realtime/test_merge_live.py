"""merge_live: unique ids, session-vs-file check, slice verification (the dashboard rule), sorted output, manifest."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.realtime import merge_live as ml


def _inc(root, session, n, first, payload=b"x"):
    iid = f"{session}-n-{n}"
    p = Path(root) / "live" / "slices" / f"{iid}.pcap"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload + iid.encode())
    return {"incident_id": iid, "session": session, "first_seen": first, "pcap_slice": f"live/slices/{iid}.pcap",
            "pcap_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def _write(root, session, recs):
    (Path(root) / "live").mkdir(parents=True, exist_ok=True)
    (Path(root) / "live" / f"incidents_{session}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


class TestMergeLive(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def _merge(self):
        return ml.merge(self.root / "live", self.root)

    def test_merges_sorts_and_writes_manifest(self):
        _write(self.root, "r07", [_inc(self.root, "r07", 2, "2026-11-09T05:20:00"), _inc(self.root, "r07", 1, "2026-11-09T05:10:00")])
        _write(self.root, "r05", [_inc(self.root, "r05", 1, "2026-11-09T00:30:00")])
        m = self._merge()
        lines = [json.loads(l) for l in (self.root / "live" / "incidents.jsonl").read_text().splitlines()]
        self.assertEqual([r["incident_id"] for r in lines], ["r05-n-1", "r07-n-1", "r07-n-2"])
        self.assertEqual(m["output"]["n_incidents"], 3)
        self.assertEqual(m["slices_verified"], 3)
        man = json.loads((self.root / "live" / "incidents_manifest.json").read_text())
        self.assertEqual(man["output"]["sha256"], hashlib.sha256((self.root / "live" / "incidents.jsonl").read_bytes()).hexdigest())
        self.assertEqual({s["file"]: s["n_incidents"] for s in man["sources"]},
                         {"incidents_r05.jsonl": 1, "incidents_r07.jsonl": 2})
        for k in ("git_commit", "git_dirty", "image_digest", "utc"):
            self.assertIn(k, man["provenance"])

    def test_remerge_is_idempotent_and_ignores_its_own_output(self):
        _write(self.root, "r05", [_inc(self.root, "r05", 1, "2026-11-09T00:30:00")])
        self._merge()
        a = (self.root / "live" / "incidents.jsonl").read_bytes()
        self._merge()
        self.assertEqual(a, (self.root / "live" / "incidents.jsonl").read_bytes())

    def test_duplicate_id_stops(self):
        r = _inc(self.root, "r05", 1, "t")
        _write(self.root, "r05", [r, r])
        with self.assertRaises(SystemExit):
            self._merge()
        self.assertFalse((self.root / "live" / "incidents.jsonl").exists())

    def test_session_must_match_the_file(self):
        _write(self.root, "r05", [_inc(self.root, "r07", 1, "t")])
        with self.assertRaises(SystemExit):
            self._merge()

    def test_tampered_missing_and_escaping_slices_stop_without_writing(self):
        for kind in ("tampered", "missing", "outside", "raw"):
            with self.subTest(kind):
                r = _inc(self.root, "r05", 1, "t")
                p = self.root / r["pcap_slice"]
                if kind == "tampered":
                    p.write_bytes(b"changed")
                elif kind == "missing":
                    p.unlink()
                elif kind == "outside":
                    r["pcap_slice"] = "../escape.pcap"
                else:
                    (self.root / "raw").mkdir(exist_ok=True)
                    (self.root / "raw" / "a.pcap").write_bytes(b"z")
                    r["pcap_slice"] = "raw/a.pcap"
                    r["pcap_sha256"] = hashlib.sha256(b"z").hexdigest()
                _write(self.root, "r05", [r])
                with self.assertRaises(SystemExit):
                    self._merge()
                self.assertFalse((self.root / "live" / "incidents.jsonl").exists())

    def test_verify_hash_statuses(self):
        r = _inc(self.root, "r05", 1, "t")
        self.assertEqual(ml.verify_hash(r["pcap_slice"], r["pcap_sha256"], self.root)[0], "verified")
        self.assertEqual(ml.verify_hash(r["pcap_slice"], "0" * 64, self.root)[0], "MISMATCH")
        self.assertEqual(ml.verify_hash(None, None, self.root)[0], "unverified")

    def test_no_sources_stops(self):
        (self.root / "live").mkdir()
        with self.assertRaises(SystemExit):
            self._merge()


if __name__ == "__main__":
    unittest.main()
