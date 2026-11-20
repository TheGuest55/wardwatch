"""Write a sanitised, publishable copy of the sessions' ground truth: data/processed/v2/ground_truth/<S>/ground_truth.jsonl.

    python scripts/make_sanitised_gt.py [--raw data/raw/sessions] [--out data/processed/v2/ground_truth] [--creds data/raw/creds/accounts.json]

Keeps ONLY what the labelling/exclusion code needs: run_id, session, class, start_utc, end_utc, src_ip, src_mac and, if a
run carries them, the structured `victims` [{ip,mac}]. Drops params, notes, targets, stage, outcome and anything else (the
raw files name lab accounts in their notes/params). Before writing, every output line is scanned in memory for each
credential string (every key and string value of the accounts file and of each session's stolen_creds.json); a hit stops
the build and prints only the count, never the value. Sessions are those listed in data/raw/sessions/split_v2.json.
A session with no raw ground_truth.jsonl (a normal-traffic capture) gets an EMPTY file, but only if its processed parquet holds no
attack, boundary or victim row; otherwise the build stops.
"""
import argparse
import json
import sys
from pathlib import Path

KEEP = ("run_id", "session", "class", "start_utc", "end_utc", "src_ip", "src_mac", "victims")


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield str(k)
            if k != "via":                      # stolen_creds.json "via" = the attack class that stole it (public), not a credential
                yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)


def secrets_for(session_dir, creds_file):
    out = set()
    for p in (Path(creds_file), Path(session_dir) / "stolen_creds.json"):
        if p.exists():
            out |= {s for s in _strings(json.loads(p.read_text())) if len(s) >= 6}
    return out


def sanitise_line(rec):
    return {k: rec[k] for k in KEEP if k in rec and rec[k] is not None}


def scan(text, secrets):
    """Number of credential strings found in `text` (values are never returned)."""
    return sum(1 for s in secrets if s in text)


def _check_normal_session(s, sessions_dir):
    import pandas as pd
    f = Path(sessions_dir) / f"{s}.parquet"
    if not f.exists():
        sys.exit(f"[sanitised-gt] no ground truth for {s} and {f} is missing: cannot confirm it is a normal-traffic session")
    labels = set(pd.read_parquet(f, columns=["label"])["label"].astype(str))
    if labels - {"benign"}:
        sys.exit(f"[sanitised-gt] {s} has no ground truth but its windows carry labels {sorted(labels - {'benign'})}")


def build(raw, out, creds_file, sessions_dir="data/processed/v2/sessions"):
    raw, out = Path(raw), Path(out)
    split = json.loads((raw / "split_v2.json").read_text())
    sessions = sorted({s for names in split.values() for s in names})
    results = {}
    for s in sessions:
        src = raw / s / "ground_truth.jsonl"
        if src.exists():
            lines = [json.dumps(sanitise_line(json.loads(l)), sort_keys=True) for l in src.read_text().splitlines() if l.strip()]
        else:
            _check_normal_session(s, sessions_dir)
            lines = []
        text = "\n".join(lines) + "\n" if lines else ""
        hits = scan(text, secrets_for(raw / s, creds_file))
        if hits:
            sys.exit(f"[sanitised-gt] REFUSING: {hits} credential string(s) found in the sanitised output of {s} (values not printed)")
        results[s] = text
    for s, text in results.items():
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        (d / "ground_truth.jsonl").write_text(text)
    return sessions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default="data/raw/sessions")
    ap.add_argument("--out", default="data/processed/v2/ground_truth")
    ap.add_argument("--creds", default="data/raw/creds/accounts.json")
    ap.add_argument("--sessions-dir", default="data/processed/v2/sessions")
    a = ap.parse_args(argv)
    sessions = build(a.raw, a.out, a.creds, a.sessions_dir)
    print(f"[sanitised-gt] wrote {len(sessions)} files under {a.out}: {', '.join(sessions)} (credential scan: 0 hits)")


if __name__ == "__main__":
    main()
