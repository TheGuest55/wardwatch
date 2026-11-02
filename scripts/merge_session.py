#!/usr/bin/env python3
"""Finalise a session directory: merge per-device benign-event files into one
benign_events.jsonl, then write manifest.json with the SHA-256 of every file. Host-side,
standard library only."""
import hashlib
import json
import sys
from pathlib import Path


def merge_benign(sdir):
    bdir = sdir / "benign_events"
    if not bdir.is_dir():
        (sdir / "benign_events.jsonl").touch()
        return 0
    events = []
    for f in sorted(bdir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    events.sort(key=lambda e: e.get("ts") or e.get("start_utc", ""))
    out = sdir / "benign_events.jsonl"
    out.write_text("".join(json.dumps(e) + "\n" for e in events))
    return len(events)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(sdir):
    files = {}
    for p in sorted(sdir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(sdir))] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    gt = sdir / "ground_truth.jsonl"
    runs = len(gt.read_text().splitlines()) if gt.exists() else 0
    manifest = {"session": sdir.name, "files": files, "n_files": len(files), "n_runs": runs}
    (sdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    sdir = Path(sys.argv[1])
    n = merge_benign(sdir)
    m = write_manifest(sdir)
    print(f"session {sdir.name}: {n} benign events, {m['n_runs']} attack runs, "
          f"{m['n_files']} files hashed")


if __name__ == "__main__":
    main()
