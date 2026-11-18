import hashlib
import json

from src.dashboard import incidents, loaders
from helpers import run


def make(root, slice_bytes=b"pcap-bytes", recorded=None):
    live = root / "live"
    live.mkdir(parents=True, exist_ok=True)
    (live / "slice1.pcap").write_bytes(slice_bytes)
    good = hashlib.sha256(b"pcap-bytes").hexdigest()
    rows = [
        {"incident_id": "i1", "verdict": "Known attack (arp_mitm)", "target_host": "m1", "source": "attacker",
         "first_seen": "t0", "last_seen": "t1", "attack_class": "arp_mitm", "pcap_slice": "live/slice1.pcap",
         "pcap_sha256": recorded or good, "top_features": [{"feature": "ip_mac_conflict", "text": "IP seen with a new MAC"}],
         "timeline": [{"t": "t0", "event": "first alert"}]},
        {"incident_id": "i2", "verdict": "Detectors disagree → review", "attack_class": "pump_cmd"},
        {"incident_id": "i3", "verdict": "Something new"},
    ]
    (live / "incidents.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_verdict_counts_cover_the_five_named_cells_and_unknowns(tmp_path):
    make(tmp_path)
    counts, cross = incidents.verdict_counts(loaders.load_incidents(tmp_path).data)
    assert counts["Known attack"] == 1 and counts["Detectors disagree: review"] == 1
    assert counts["Possible novel attack"] == 0 and counts["Something new"] == 1
    assert cross.loc["pump_cmd", "Detectors disagree: review"] == 1


def test_hash_verified_mismatch_refused_and_missing(tmp_path):
    make(tmp_path)
    good = hashlib.sha256(b"pcap-bytes").hexdigest()
    assert incidents.verify_hash("live/slice1.pcap", good, tmp_path)[0] == "verified"
    assert incidents.verify_hash("live/slice1.pcap", "0" * 64, tmp_path)[0] == "MISMATCH"
    assert incidents.verify_hash("../../etc/passwd", good, tmp_path)[0] == "refused"
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "x.pcap").write_bytes(b"x")
    assert incidents.verify_hash("raw/x.pcap", good, tmp_path)[0] == "refused"
    assert incidents.verify_hash("live/nope.pcap", good, tmp_path)[0] == "unverified"
    assert incidents.verify_hash(None, good, tmp_path)[0] == "unverified"


def test_incidents_page_stub_and_populated(tmp_path, monkeypatch):
    at = run("src/dashboard/pages/6_Incidents.py", tmp_path, monkeypatch)
    assert not at.exception and any("not available" in i.value for i in at.info)
    make(tmp_path)
    at = run("src/dashboard/pages/6_Incidents.py", tmp_path, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert any("hash verified" in s.value for s in at.success)
