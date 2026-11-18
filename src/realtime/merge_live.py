"""Merge the per-session live incident files into the one file the dashboard reads.

    python -m src.realtime.merge_live

Reads data/processed/v2/live/incidents_<S>.jsonl (written by run_e, one per replayed session) and writes
  live/incidents.jsonl          every incident, sorted by (session, first_seen, incident_id)
  live/incidents_manifest.json  the source files and the output with their sha256 and line counts, provenance
                                (git commit, dirty flag, image digest from the WARDWATCH_* env), UTC time and the
                                result of the slice check.
Stops (SystemExit, nothing written) on: a duplicate incident id, a line whose `session` differs from its file's, a line
without a session, or any pcap slice that is missing, outside the data root, under raw/, or whose sha256 differs from
the recorded one. The slice check is the dashboard's verify_hash rule (src/dashboard/incidents.py on the dashboard branch):
the path is relative to the data root, must stay inside it and never under a `raw` directory, and its SHA-256 must equal
`pcap_sha256`.

Slices contain plaintext lab MQTT credentials; this only reads and hashes them, it never copies or moves them.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from .. import common
from ..models import tracking

V2 = common.PROCESSED_DIR / "v2"
LIVE = V2 / "live"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(slice_path, expected, root):
    """(status, detail): the dashboard's rule. status is verified | unverified | refused | MISMATCH."""
    if not slice_path or not expected:
        return "unverified", "no slice path or no recorded hash"
    root = Path(root).resolve()
    p = Path(slice_path)
    p = (p if p.is_absolute() else root / p).resolve()
    if root not in p.parents or "raw" in p.relative_to(root).parts:
        return "refused", "slice path is outside the data root (or under raw/)"
    if not p.is_file():
        return "unverified", f"slice file not found ({p.name})"
    got = _sha(p)
    return ("verified", got) if got == str(expected).lower() else ("MISMATCH", f"file hashes to {got}")


def read_sources(live_dir):
    files = sorted(Path(live_dir).glob("incidents_*.jsonl"))
    if not files:
        raise SystemExit(f"[merge_live] no incidents_<session>.jsonl under {live_dir}")
    out = []
    for f in files:
        session = f.stem[len("incidents_"):]
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        for r in recs:
            if r.get("session") != session:
                raise SystemExit(f"[merge_live] {f.name}: incident {r.get('incident_id')} has session "
                                 f"{r.get('session')!r}, expected {session!r}")
        out.append((f, session, recs))
    return out


def merge(live_dir=LIVE, root=V2):
    live_dir = Path(live_dir)
    sources = read_sources(live_dir)
    seen, allrecs = {}, []
    for f, session, recs in sources:
        for r in recs:
            iid = r.get("incident_id")
            if iid in seen:
                raise SystemExit(f"[merge_live] duplicate incident id {iid!r} ({seen[iid]} and {f.name})")
            seen[iid] = f.name
            allrecs.append(r)
    bad = []
    for r in allrecs:
        st, detail = verify_hash(r.get("pcap_slice"), r.get("pcap_sha256"), root)
        if st != "verified":
            bad.append(f"{r['incident_id']}: {st} ({detail})")
    if bad:
        raise SystemExit(f"[merge_live] {len(bad)} slice(s) failed verification, nothing written: {bad[:5]}")
    allrecs.sort(key=lambda r: (r["session"], r["first_seen"], r["incident_id"]))
    body = "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in allrecs)
    out_path = live_dir / "incidents.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.write_text(body)
    tmp.replace(out_path)
    prov = tracking.run_provenance(seed=None, split_path=None, dataset_path=None)
    manifest = {"provenance": {"git_commit": prov["git_commit"], "git_dirty": prov["git_dirty"],
                               "image_digest": prov["image_digest"],
                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                "sources": [{"file": f.name, "session": s, "sha256": _sha(f), "n_incidents": len(recs)}
                            for f, s, recs in sources],
                "output": {"file": out_path.name, "sha256": _sha(out_path), "n_incidents": len(allrecs)},
                "slices_verified": len(allrecs), "slice_rule": "dashboard verify_hash: inside data root, not under raw/, sha256 equal"}
    (live_dir / "incidents_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", default=str(LIVE))
    ap.add_argument("--root", default=str(V2))
    a = ap.parse_args(argv)
    m = merge(a.live_dir, a.root)
    print(f"[merge_live] {m['output']['n_incidents']} incidents from {len(m['sources'])} session file(s), "
          f"{m['slices_verified']} slices verified; wrote {m['output']['file']} (sha256 {m['output']['sha256'][:12]}) "
          f"and incidents_manifest.json")


if __name__ == "__main__":
    main()
