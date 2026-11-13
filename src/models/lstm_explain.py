"""Post-alert explanation for the LSTM by OCCLUSION (outside the scoring path; no SHAP deep explainer).

For one input sequence [k, d] (d = n_feat + 1 with the is_present channel, or n_feat without), set one
feature (or one feature group) to SCALED ZERO across the last `last_n` PRESENT windows and record the
drop in the attack score (1 - P(benign)). Scaled zero is the training mean, i.e. "nothing unusual",
and it is the same fill value the model sees for gaps and padding. The is_present channel and the
filled/padded windows are never touched. A positive delta means that feature (group) raised the
attack score. Returns the top 5 as plain-language strings plus the raw deltas.
"""
import numpy as np
import torch

from .. import common
from . import lstm

# Explicit feature -> group mapping, built from common.FEATURES (checked below: every feature lands in
# exactly one group, nothing left over).
GROUP_MEMBERS = {
    "MQTT": [
        "mqtt_pkts", "mqtt_connect", "mqtt_publish", "mqtt_subscribe", "mqtt_connack", "mqtt_pingreq",
        "mqtt_disconnect", "mqtt_connect_rate", "mqtt_publish_rate", "mqtt_mean_payload",
        "mqtt_max_payload", "mqtt_connack_failures", "mqtt_clientid_reuse",
        "topic_wildcard_subs", "topic_max_depth", "topic_is_command",
        "vital_max_jump", "vital_jump_count", "implausible_vital"],      # payload plausibility rides on MQTT
    "TCP flags": ["frac_tcp", "tcp_syn", "tcp_synack", "tcp_rst", "tcp_fin", "tcp_syn_rate"],
    "volume": [
        "pkts_total", "pkts_sent", "pkts_recv", "bytes_total", "bytes_sent", "bytes_recv",
        "mean_pkt_size", "std_pkt_size", "min_pkt_size", "max_pkt_size", "mean_iat", "std_iat",
        "unique_peers", "unique_peer_ips", "unique_contacted_ports", "frac_icmp", "frac_other"],
    "ARP": ["frac_arp", "arp_requests", "arp_replies", "arp_gratuitous", "arp_reply_rate",
            "ip_mac_conflict"],
    "flow": ["nf_active_flows", "nf_new_flows", "nf_mean_flow_age"],
}


def build_groups(features=None):
    """{group: [features]} restricted to `features` (default common.FEATURES), asserting that every
    feature of common.FEATURES lands in exactly one group and no group member is unknown."""
    seen = {}
    for g, members in GROUP_MEMBERS.items():
        for f in members:
            if f in seen:
                raise AssertionError(f"feature {f!r} is in both {seen[f]!r} and {g!r}")
            seen[f] = g
    missing = [f for f in common.FEATURES if f not in seen]
    unknown = [f for f in seen if f not in common.FEATURES]
    if missing or unknown:
        raise AssertionError(f"group mapping mismatch: unmapped={missing} unknown={unknown}")
    wanted = list(common.FEATURES if features is None else features)
    stray = [f for f in wanted if f not in seen]
    if stray:
        raise AssertionError(f"features without a group: {stray}")
    return {g: [f for f in members if f in set(wanted)] for g, members in GROUP_MEMBERS.items()}


_WORDS = {"mqtt": "MQTT", "tcp": "TCP", "arp": "ARP", "ip": "IP", "mac": "MAC", "nf": "flow",
          "pkts": "packets", "pkt": "packet", "iat": "inter-arrival time", "syn": "SYN",
          "synack": "SYN-ACK", "rst": "RST", "fin": "FIN", "connack": "CONNACK", "pingreq": "PINGREQ",
          "clientid": "client-ID", "vital": "vital-sign", "frac": "fraction", "icmp": "ICMP",
          "recv": "received", "subs": "subscriptions", "std": "std-dev", "min": "min", "max": "max"}


def humanise(name):
    """mqtt_publish_rate -> 'MQTT publish rate' (group names pass through unchanged)."""
    if name in GROUP_MEMBERS:
        return f"{name} group"
    return " ".join(_WORDS.get(t, t) for t in name.split("_"))


def _scores(model, X):
    with torch.no_grad():
        p = torch.softmax(model.eval()(torch.as_tensor(X)), dim=1).numpy()
    return (1.0 - p[:, 0]).astype(np.float64)


def _sentence(name, delta, m):
    verb = "raised" if delta >= 0 else "lowered"
    return f"{humanise(name)} in the last {m} windows {verb} the attack score by {abs(delta):.2f}"


def explain(model, seq, last_n=3, groups=None, feature_names=None, use_presence=True, top=5):
    """Occlusion explanation of one sequence.

    seq: float32 [k, d] or [1, k, d]; the first len(feature_names) channels are the scaled features and,
    with use_presence, the next one is is_present. groups: None (feature level only) or {name: [feature
    names]} (adds a group level). Returns
      {"base_score", "last_n", "n_windows_occluded", "present_positions",
       "feature": {"delta": {name: d}, "top": [str...]}, "group": {...} (only when groups given)}
    delta = base score - score with that feature/group at scaled zero in the last present windows."""
    if feature_names is None:
        raise ValueError("feature_names is required")
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 3:
        if seq.shape[0] != 1:
            raise ValueError("explain takes one sequence")
        seq = seq[0]
    n_feat = len(feature_names)
    d_expected = n_feat + (1 if use_presence else 0)
    if seq.shape[1] != d_expected:
        raise ValueError(f"sequence has {seq.shape[1]} channels, expected {d_expected}")
    present = lstm.present_mask(torch.as_tensor(seq)[None], n_feat, use_presence)[0].numpy()
    pos = np.flatnonzero(present)[-int(last_n):]              # the last n PRESENT windows
    if len(pos) == 0:
        raise ValueError("sequence has no present window")
    idx = {f: i for i, f in enumerate(feature_names)}
    base = float(_scores(model, seq[None])[0])

    def occlude(cols):
        out = seq.copy()
        out[np.ix_(pos, cols)] = 0.0                           # only feature channels, only present windows
        return out

    def level(items):
        batch = np.stack([occlude([idx[f] for f in fs]) for _, fs in items])
        s = _scores(model, batch)
        delta = {name: float(base - s[i]) for i, (name, _) in enumerate(items)}
        order = sorted(delta, key=lambda n: (-delta[n], n))
        return {"delta": delta, "top": [_sentence(n, delta[n], len(pos)) for n in order[:top]],
                "top_names": order[:top]}

    out = {"base_score": base, "last_n": int(last_n), "n_windows_occluded": int(len(pos)),
           "present_positions": [int(p) for p in pos],
           "feature": level([(f, [f]) for f in feature_names])}
    if groups:
        items = [(g, fs) for g, fs in groups.items() if fs]
        out["group"] = level(items)
    return out
