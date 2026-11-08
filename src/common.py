"""Shared constants and helpers for the analysis pipeline: paths, the 5-second window grid, lab
topology from config/ward.json, and the metadata-vs-feature column contract.

Leakage contract (./CLAUDE.md): IPs, MACs, ports, client-ID strings, raw topics and timestamps are
metadata only, never model inputs. This module is the single source of truth for which columns are
features; everything else the extractor emits is metadata and is excluded from feature_list.json.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

WINDOW_SECONDS = 5.0

# Repo-relative paths (the container mounts the repo at /work and runs with cwd=/work).
ROOT = Path(".")
CONFIG_PATH = ROOT / "config" / "ward.json"
SESSIONS_DIR = ROOT / "data" / "raw" / "sessions"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_PATH = SESSIONS_DIR / "split.json"

# Metadata columns: emitted for forensics/labelling/splitting, never fed to a model. Any column the
# extractor produces that is NOT in FEATURE order below is treated as metadata.
META_COLUMNS = [
    "session", "window_start_iso", "window_start_epoch",
    "host_mac", "host_ip", "host_role",
    "label", "attack_class", "run_id", "is_boundary", "split",
]

# Features that only the testbed can provide (payload plausibility needs decrypted JSON vitals).
# Cross-dataset experiments (Experiment C) must drop these; feature_list.json flags them.
TESTBED_ONLY_FEATURES = [
    "vital_max_jump", "vital_jump_count", "implausible_vital",
]

# Causal transport-flow features, computed per window from packets seen up to the window end only
# (src/features/windows.py). Review B1 replaced NFStream's whole-lifetime flow aggregates (which
# leaked the future and acted as a host-lifetime identifier) with these live-computable stats.
FLOW_FEATURES = [
    "nf_active_flows", "nf_new_flows", "nf_mean_flow_age",
]
# Back-compat alias (older report/datacheck code referenced this name).
NFSTREAM_FEATURES = FLOW_FEATURES


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def ip_role_map(cfg):
    """Return {ip: role_label} for the configured topology. Roles are generic (not per-host
    identifiers): 'broker', 'nurse', 'attacker', 'monitor', 'pump', 'wearable'."""
    prefix = cfg["network"]["subnet"].rsplit(".", 1)[0] + "."
    roles = {
        cfg["broker"]["ip"]: "broker",
        cfg["nurse"]["ip"]: "nurse",
        cfg["attacker"]["ip"]: "attacker",
    }
    for b in range(1, cfg["beds"] + 1):
        roles[f"{prefix}{10 * b + 1}"] = "monitor"
        roles[f"{prefix}{10 * b + 2}"] = "pump"
        roles[f"{prefix}{10 * b + 3}"] = "wearable"
    return roles


def window_start(epoch_seconds):
    """Floor a Unix timestamp to its 5-second window start (epoch seconds)."""
    return int(epoch_seconds // WINDOW_SECONDS) * int(WINDOW_SECONDS)


def iso_to_epoch(s):
    """Parse an ISO-8601 timestamp (trailing 'Z' or explicit offset) to epoch seconds (float)."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def epoch_to_iso(epoch_seconds):
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()
