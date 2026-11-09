"""Build the single labelled dataset from captured sessions.

    python -m src.dataset [--split data/raw/sessions/split.json]
                          [--out-dir data/processed] [--force] [--include-dry]

Extracts each session (reusing an existing per-session parquet unless --force), drops boundary
windows (counted in the report), stamps the train/val/test `split` from split.json, concatenates
and writes data/processed/testbed_windows.parquet plus feature_list.json (feature vs metadata vs
testbed-only vs nfstream). Dry-run sessions are excluded unless --include-dry (used only to smoke
the builder against the t0* fixtures before the real sessions exist).

Fitting discipline this enables downstream (./CLAUDE.md): the `split` column lets the modelling
phase fit scalers + SMOTE on train only, thresholds/calibration on validation only, nothing on
test. Splitting is by session (time-based), never a random shuffle.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import common
from .features import extract


def _metadata_columns(df):
    extra = [c for c in common.METADATA_EXTRAS if c in df.columns]
    return [c for c in common.META_COLUMNS if c in df.columns] + extra


def _feature_columns(df):
    """Features are exactly common.FEATURES present in df (review N6: an explicit allow-list, not
    'everything that isn't metadata'). Refuses drift: a missing allow-listed feature, or any emitted
    column that is neither a feature nor metadata, is a contract violation."""
    common.assert_feature_metadata_disjoint()
    feats = [f for f in common.FEATURES if f in df.columns]
    missing = [f for f in common.FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"extractor did not emit allow-listed features: {missing}")
    known = set(common.FEATURES) | set(_metadata_columns(df))
    unknown = [c for c in df.columns if c not in known]
    if unknown:
        raise SystemExit(
            f"columns are neither in the feature allow-list nor metadata: {unknown}. "
            "Add each to common.FEATURES (model input) or META_COLUMNS/METADATA_EXTRAS (metadata).")
    return feats


def discover_sessions(include_dry):
    """Sessions that have a config.json, newest first by name. Dry-run sessions excluded unless
    include_dry."""
    out = []
    for d in sorted(common.SESSIONS_DIR.glob("*")):
        cfg = d / "config.json"
        if not cfg.exists():
            continue
        dry = bool(json.loads(cfg.read_text()).get("dry_run", False))
        if dry and not include_dry:
            continue
        out.append(d.name)
    return out


def session_df(session, force, out_dir):
    cached = Path(out_dir) / "sessions" / f"{session}.parquet"
    if cached.exists() and not force:
        return pd.read_parquet(cached)
    df, _ = extract.extract_session(session)
    cached.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cached, index=False)
    return df


def load_split(split_path):
    p = Path(split_path)
    if not p.exists():
        raise SystemExit(
            f"split file not found: {p}. Create it once the real sessions are recorded, e.g. "
            '{"train": ["s01","s02","s03","n01"], "val": ["s04"], "test": ["s05"]}')
    spec = json.loads(p.read_text())
    sess_split = {}
    for name in ("train", "val", "test"):
        for s in spec.get(name, []):
            sess_split[s] = name
    return sess_split


def build(split_path, force, include_dry, out_dir=None):
    out_dir = Path(out_dir) if out_dir is not None else common.PROCESSED_DIR
    sess_split = load_split(split_path)
    sessions = discover_sessions(include_dry)
    if not sessions:
        raise SystemExit("no sessions found to build from")

    parts, report = [], {"sessions": {}, "boundary_dropped": 0}
    for s in sessions:
        if s not in sess_split:
            print(f"[WARN] session {s} not in split file; skipping", flush=True)
            continue
        df = session_df(s, force, out_dir)
        n_boundary = int(df["is_boundary"].sum())
        kept = df[~df["is_boundary"]].copy()
        kept["split"] = sess_split[s]
        parts.append(kept)
        report["boundary_dropped"] += n_boundary
        report["sessions"][s] = {
            "split": sess_split[s], "rows_kept": len(kept), "boundary_dropped": n_boundary,
            "label_counts": kept["label"].value_counts().to_dict(),
        }

    if not parts:
        raise SystemExit("no non-boundary rows collected (check split file names)")

    full = pd.concat(parts, ignore_index=True)
    full = full.sort_values(["session", "window_start_epoch", "host_mac"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "testbed_windows.parquet"
    full.to_parquet(out, index=False)

    features = _feature_columns(full)
    feature_list = {
        "n_features": len(features),
        "features": features,
        "metadata": _metadata_columns(full),
        "testbed_only": [c for c in common.TESTBED_ONLY_FEATURES if c in features],
        "flow_features": [c for c in common.FLOW_FEATURES if c in features],
    }
    (out_dir / "feature_list.json").write_text(json.dumps(feature_list, indent=2))

    report["out"] = str(out)
    report["total_rows"] = len(full)
    report["n_features"] = len(features)
    report["split_counts"] = full["split"].value_counts().to_dict()
    report["label_counts"] = full["label"].value_counts().to_dict()
    print(json.dumps(report, indent=2, default=str))
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(common.SPLIT_PATH))
    ap.add_argument("--out-dir", default=str(common.PROCESSED_DIR),
                    help="directory for testbed_windows.parquet, feature_list.json and the "
                         "per-session parquet cache (default data/processed; use a separate dir "
                         "per corpus so builds never overwrite each other)")
    ap.add_argument("--force", action="store_true", help="re-extract sessions even if cached")
    ap.add_argument("--include-dry", action="store_true",
                    help="include dry-run sessions (smoke-testing the builder only)")
    args = ap.parse_args()
    build(args.split, args.force, args.include_dry, args.out_dir)


if __name__ == "__main__":
    main()
