"""The results bundle: exactly the files the dashboard reads, nothing else, with hard safety checks.

    python scripts/make_results_bundle.py [--data-root PATH] [--out DIR] [--creds PATH]

WHAT IS INCLUDED is derived from the loaders (src/dashboard/loaders.py, metrics_view.py, triage_view.py), not from "all of
runs/": every run record load_records() would read (the same discovery rule: any *.json under runs/ with experiment, model and
a metrics dict), the prediction parquets and LSTM curve files those records point to (loaders._pred_candidates and
load_lstm_curves), runs/lstm/selected_config.json, runs/lstm/explanations.json, runs/calibration/<model>_details.json,
runs/expA/autoencoder_details.json, runs/bootstrap/*.json, runs/expA/predictions/<model>_multiclass_s0.parquet,
testbed_windows.parquet (the dataset summary), live/incidents.jsonl and live/recorded/*.jsonl (a recorded alert stream of one real
replay, which the "Live alerts" page plays back) and live/screenshots/*.png (screenshots of the page during the live run;
live/alerts_stream.jsonl, the file a running replay writes, live/demo/ and live_sessions/ are NOT bundled).

WHAT IS EXCLUDED: PCAP slices (live/slices, triage/slices: they hold plaintext lab MQTT credentials), models/, logs, the
peers sidecars, data/raw, and every other file.

HARD CHECKS (any failure stops the build and writes nothing):
  * no *.pcap / *.pcapng, and no file named like creds / accounts / password / secret;
  * no bundled file contains any username or password from data/raw/creds/accounts.json. That file is read in memory for the
    scan only and is never copied; error messages name files, never the strings.

OUTPUT: <out>/wardwatch_results_<sha>.zip (+ .zip.sha256), MANIFEST.sha256 and BUNDLE.json (also inside the zip, at the data
root, where the dashboard looks for BUNDLE.json). MANIFEST.sha256 lists "<sha256>  <path>" for every other file in the zip.
"""
import hashlib
import json
import re
import subprocess
import time
import zipfile
from pathlib import Path

from . import config, loaders

SECRET_NAME = re.compile(r"cred|account|passw|secret", re.I)
EXCLUDED_PARTS = ("slices", "models", "raw", "peers")
FIXED = ("runs/lstm/selected_config.json", "runs/lstm/explanations.json", "runs/expA/autoencoder_details.json",
         "testbed_windows.parquet", "live/incidents.jsonl")
ZIP_TIME = (2026, 11, 18, 0, 0, 0)


class BundleError(SystemExit):
    """A hard check failed: nothing is written."""


def _sha(path_or_bytes):
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def collect(root):
    """Sorted relative POSIX paths (under `root`) the dashboard reads."""
    root = Path(root).resolve()
    found = set()
    rec_df = _records(root)
    for _, r in rec_df.iterrows():
        found.add(Path(r["path"]).resolve())
        for p in loaders._pred_candidates(dict(r), root):
            found.add(Path(p).resolve())
        found.add((Path(r["path"]).parent / "curves" / f"{r['run_name']}.json").resolve())
    for rel in FIXED:
        found.add(root / rel)
    for m in loaders.MODELS:
        found.add(root / "runs" / "calibration" / f"{m}_details.json")
        found.add(root / "runs" / "expA" / "predictions" / f"{m}_multiclass_s0.parquet")
    for p in (root / "runs" / "bootstrap").glob("*.json"):
        found.add(p.resolve())
    for p in (root / "live" / "recorded").glob("*.jsonl"):
        found.add(p.resolve())
    for p in (root / "live" / "screenshots").glob("*.png"):
        found.add(p.resolve())
    out = []
    for p in found:
        if not p.is_file():
            continue
        try:
            out.append(p.relative_to(root).as_posix())
        except ValueError:                       # a results/runs directory outside the data root: not bundled
            continue
    return sorted(set(out))


def _records(root):
    """The run-record rows load_records() builds, restricted to this root's runs/ directory."""
    import os
    old = os.environ.get("WARDWATCH_RESULTS_ROOT")
    os.environ["WARDWATCH_RESULTS_ROOT"] = str(Path(root) / "runs")           # no extra result roots outside the data root
    try:
        r = loaders.load_records(root)
    finally:
        if old is None:
            os.environ.pop("WARDWATCH_RESULTS_ROOT", None)
        else:
            os.environ["WARDWATCH_RESULTS_ROOT"] = old
    if not r.ok:
        raise BundleError(f"[bundle] no run records under {root}: {r.reason}")
    return r.data


def load_secrets(creds_path):
    """Usernames and passwords of data/raw/creds/accounts.json, in memory only (never written anywhere)."""
    d = json.loads(Path(creds_path).read_text())
    vals = set()
    for k, v in (d.items() if isinstance(d, dict) else enumerate(d)):
        for x in ((k, v) if isinstance(d, dict) else (v.get("username") if isinstance(v, dict) else None,
                                                      v.get("password") if isinstance(v, dict) else None)):
            if isinstance(x, str) and len(x) >= 3:
                vals.add(x)
    if not vals:
        raise BundleError(f"[bundle] {creds_path}: no usernames/passwords found: the credential scan cannot run")
    return sorted(vals)


def check_paths(rels):
    bad = []
    for rel in rels:
        p = Path(rel)
        low = rel.lower()
        if rel == "live/alerts_stream.jsonl" or rel.startswith("live/demo/"):
            bad.append(rel)
        if low.endswith((".pcap", ".pcapng", ".log")) or set(p.parts) & set(EXCLUDED_PARTS) \
                or SECRET_NAME.search(p.name) or p.name.startswith("peers_"):
            bad.append(rel)
    if bad:
        raise BundleError(f"[bundle] refused: excluded kinds of file selected: {bad[:5]}")


def check_tree(root):
    """A PCAP or a credentials-named file anywhere under runs/, or directly in the data root or live/, stops the build even
    though it would not be selected: such a file has no business among the results, so its presence is reported, not ignored."""
    root = Path(root)
    bad = []
    for base, recursive in ((root / "runs", True), (root, False), (root / "live", False)):
        if not base.is_dir():
            continue
        for p in (base.rglob("*") if recursive else base.iterdir()):
            if p.is_file() and (p.name.lower().endswith((".pcap", ".pcapng")) or SECRET_NAME.search(p.name)):
                bad.append(p.relative_to(root).as_posix())
    if bad:
        raise BundleError(f"[bundle] STOP: pcap or credential-named file(s) among the results: {sorted(bad)[:5]}")


def scan_secrets(root, rels, secrets):
    needles = [s.encode() for s in secrets]
    hit = []
    for rel in rels:
        data = (Path(root) / rel).read_bytes()
        if any(n in data for n in needles):
            hit.append(rel)
    if hit:
        raise BundleError(f"[bundle] STOP: a username or password from the credentials file appears in {len(hit)} bundled "
                          f"file(s), e.g. {hit[:3]} (the string is not printed). Nothing written.")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        import os
        return os.environ.get("WARDWATCH_GIT_COMMIT") or "unknown"


def build(root, out_dir, creds_path, commit=None):
    root, out_dir = Path(root).resolve(), Path(out_dir)
    check_tree(root)
    rels = collect(root)
    if not rels:
        raise BundleError("[bundle] nothing to bundle")
    check_paths(rels)
    scan_secrets(root, rels, load_secrets(creds_path))
    commit = commit or git_commit()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = "".join(f"{_sha(root / r)}  {r}\n" for r in rels)
    info = {"commit": commit, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "file_count": len(rels) + 2,
            "source_data_root": str(root), "contents": "files the dashboard reads; no PCAP slices, models, logs or raw data",
            "manifest": "MANIFEST.sha256 (every file in this bundle except itself and BUNDLE.json)"}
    zpath = out_dir / f"wardwatch_results_{commit[:7]}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        def put(name, data):
            zi = zipfile.ZipInfo(name, ZIP_TIME)
            zi.compress_type, zi.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            z.writestr(zi, data)
        for r in rels:
            put(r, (root / r).read_bytes())
        put("MANIFEST.sha256", manifest)
        put("BUNDLE.json", json.dumps(info, indent=2))
    (out_dir / "MANIFEST.sha256").write_text(manifest)
    (out_dir / "BUNDLE.json").write_text(json.dumps({**info, "zip": zpath.name, "zip_bytes": zpath.stat().st_size,
                                                     "zip_sha256": _sha(zpath)}, indent=2))
    (out_dir / (zpath.name + ".sha256")).write_text(f"{_sha(zpath)}  {zpath.name}\n")
    return {"zip": str(zpath), "files": len(rels) + 2, "bytes": zpath.stat().st_size, "sha256": _sha(zpath), "commit": commit}


def verify_manifest(directory):
    """Re-hash an unpacked bundle against its MANIFEST.sha256. Returns a list of problems (empty = verified)."""
    d = Path(directory)
    m = d / "MANIFEST.sha256"
    if not m.exists():
        return ["MANIFEST.sha256 missing"]
    probs, listed = [], set()
    for line in m.read_text().splitlines():
        h, _, rel = line.partition("  ")
        listed.add(rel)
        p = d / rel
        if not p.is_file():
            probs.append(f"missing: {rel}")
        elif _sha(p) != h:
            probs.append(f"hash differs: {rel}")
    extra = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} - listed - {"MANIFEST.sha256", "BUNDLE.json"}
    probs += [f"not in the manifest: {e}" for e in sorted(extra)]
    return probs


def is_bundle(root=None):
    return (Path(root) if root is not None else config.data_root()).joinpath("BUNDLE.json").exists()
