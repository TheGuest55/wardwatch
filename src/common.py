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

# Explicit model-input allow-list (review N6): the dataset's features are exactly these columns, in
# this order — never "every column that isn't metadata". Any column the extractor emits that is not
# listed here and not in META_COLUMNS is a contract violation the builder refuses, so a new feature
# must be added here deliberately and a stray/renamed column can't silently leak in. Keep disjoint
# from META_COLUMNS (assert_feature_metadata_disjoint / datacheck enforce it).
FEATURES = [
    # volume / size / timing
    "pkts_total", "pkts_sent", "pkts_recv", "bytes_total", "bytes_sent", "bytes_recv",
    "mean_pkt_size", "std_pkt_size", "min_pkt_size", "max_pkt_size", "mean_iat", "std_iat",
    "unique_peers", "unique_peer_ips", "unique_contacted_ports",
    # protocol mix (frac_udp dropped as collinear, review m-new-5)
    "frac_tcp", "frac_arp", "frac_icmp", "frac_other",
    # TCP flags
    "tcp_syn", "tcp_synack", "tcp_rst", "tcp_fin", "tcp_syn_rate",
    # MQTT
    "mqtt_pkts", "mqtt_connect", "mqtt_publish", "mqtt_subscribe", "mqtt_connack", "mqtt_pingreq",
    "mqtt_disconnect", "mqtt_connect_rate", "mqtt_publish_rate", "mqtt_mean_payload",
    "mqtt_max_payload", "mqtt_connack_failures", "mqtt_clientid_reuse",
    "topic_wildcard_subs", "topic_max_depth", "topic_is_command",
    # L2 / ARP
    "arp_requests", "arp_replies", "arp_gratuitous", "arp_reply_rate", "ip_mac_conflict",
    # payload plausibility (testbed only)
    "vital_max_jump", "vital_jump_count", "implausible_vital",
    # causal transport-flow stats (review B1 + m-new-1)
    "nf_active_flows", "nf_new_flows", "nf_mean_flow_age",
]

# Features that only the testbed can provide (payload plausibility needs decrypted JSON vitals).
# Cross-dataset experiments (Experiment C) must drop these; feature_list.json flags them.
TESTBED_ONLY_FEATURES = [
    "vital_max_jump", "vital_jump_count", "implausible_vital",
]

# Metadata extras the extractor emits beyond META_COLUMNS (forensic/diagnostic, never model inputs).
METADATA_EXTRAS = ["touches_attacker", "dry_run"]


def assert_feature_metadata_disjoint():
    """Features and metadata must not overlap (review N6: a column is either a model input or
    metadata, never both). Raises ValueError naming any offender."""
    meta = set(META_COLUMNS) | set(METADATA_EXTRAS)
    overlap = sorted(set(FEATURES) & meta)
    if overlap:
        raise ValueError(f"features and metadata overlap: {overlap}")
    dupes = sorted({f for f in FEATURES if FEATURES.count(f) > 1})
    if dupes:
        raise ValueError(f"duplicate features in allow-list: {dupes}")

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
