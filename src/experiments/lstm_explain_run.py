"""Occlusion explanations for one test alert per attack class, from the selected seed-0 A model.

  python -m src.experiments.lstm_explain_run

An alert = a test row with label attack and score >= the model's validation threshold. Per class the
alert is the MEDIAN-in-time one among the candidates (deterministic, not cherry-picked), taken from a
non-attacker host where one exists (host_role is recorded either way). A class with no alert is
recorded explicitly as "no_alert"; nothing is substituted. Output: runs/lstm/explanations.json.

Delta = drop in attack LOG-ODDS (lstm_explain). Two views: "last3" (main: the last 3 present windows)
and "all12" (secondary: every present window of the k=12 sequence). --same-alerts-as asserts the
alerts equal those of an earlier output (the user's instruction: re-explain the SAME alerts).

Verdict rule, FIXED BEFORE THE RUN (committed with this code): a view is "useful" for a class iff
its top GROUP by log-odds delta is one of EXPECTED_GROUPS[class] (the attack's mechanism) AND that
delta >= MIN_LOGODDS (1 nat, i.e. occluding the group cuts the attack odds by at least e). The top-5
features and whether the top feature is in EXPECTED_FEATURES[class] are recorded for the human read.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import common
from ..models import lstm
from ..models import lstm_explain as ex
from ..models import sequence as seq

V2 = common.PROCESSED_DIR / "v2"
RUNS = V2 / "runs" / "lstm"
RUN_NAME = "A_k12_h32_s0"
SPLIT_PATH = "data/raw/sessions/split_v2.json"
KEYS = ["session", "host_mac", "window_start_epoch"]
VIEWS = {"last3": 3, "all12": None}          # None = k (every present window)
MIN_LOGODDS = 1.0
# What each attack physically changes (docs/proposal threat model), by group and by feature.
EXPECTED_GROUPS = {
    "arp_mitm": {"ARP"},
    "bruteforce": {"MQTT", "TCP flags", "flow"},       # repeated CONNECT/CONNACK-fail over new TCP conns
    "eavesdrop": {"MQTT", "volume"},                   # wildcard SUBSCRIBE + broker fan-out of publishes
    "mqtt_flood": {"MQTT", "volume", "flow"},
    "pump_cmd": {"MQTT"},
    "recon_scan": {"TCP flags", "volume", "flow"},     # SYN/RST to many ports and peers
    "spoof_vitals": {"MQTT"},                          # publishes as a device; implausible vitals
    "syn_flood": {"TCP flags", "volume", "flow"},
}
EXPECTED_FEATURES = {
    "arp_mitm": {"ip_mac_conflict", "arp_replies", "arp_reply_rate", "arp_gratuitous", "arp_requests",
                 "frac_arp"},
    "bruteforce": {"mqtt_connect", "mqtt_connect_rate", "mqtt_connack", "mqtt_connack_failures",
                   "mqtt_disconnect", "tcp_syn", "tcp_fin", "tcp_rst", "nf_new_flows"},
    "eavesdrop": {"mqtt_subscribe", "topic_wildcard_subs", "mqtt_publish", "mqtt_publish_rate",
                  "mqtt_pkts", "bytes_sent", "pkts_sent"},
    "mqtt_flood": {"mqtt_connect", "mqtt_connect_rate", "mqtt_publish", "mqtt_publish_rate",
                   "mqtt_pkts", "mqtt_connack", "pkts_total", "pkts_recv", "pkts_sent", "nf_new_flows",
                   "nf_active_flows"},
    "pump_cmd": {"topic_is_command", "mqtt_publish", "mqtt_publish_rate", "mqtt_connect",
                 "mqtt_clientid_reuse"},
    "recon_scan": {"tcp_syn", "tcp_syn_rate", "tcp_rst", "tcp_synack", "unique_contacted_ports",
                   "unique_peers", "unique_peer_ips", "nf_new_flows", "nf_active_flows"},
    "spoof_vitals": {"vital_max_jump", "vital_jump_count", "implausible_vital", "mqtt_clientid_reuse",
                     "mqtt_publish", "mqtt_publish_rate", "mqtt_connect"},
    "syn_flood": {"tcp_syn", "tcp_syn_rate", "tcp_synack", "tcp_rst", "pkts_total", "pkts_recv",
                  "pkts_sent", "nf_new_flows", "nf_active_flows"},
}


def verdict(cls, e):
    """Apply the fixed rule to one view's explain() output."""
    g_top = e["group"]["top_names"][0]
    g_delta = e["group"]["delta"][g_top]
    f_top = e["feature"]["top_names"][0]
    useful = g_top in EXPECTED_GROUPS[cls] and g_delta >= MIN_LOGODDS
    return {"useful": bool(useful), "top_group": g_top, "top_group_logodds": g_delta,
            "top_group_expected": g_top in EXPECTED_GROUPS[cls],
            "top_feature": f_top, "top_feature_logodds": e["feature"]["delta"][f_top],
            "top_feature_expected": f_top in EXPECTED_FEATURES[cls]}


def pick_alerts(preds, threshold, classes=seq.ATTACK_CLASSES):
    """{class: row or None}. Candidates are attack rows with score >= threshold; non-attacker hosts are
    preferred; the median by (window_start_epoch, session, host_mac) is taken."""
    t = preds[(preds["split"] == "test") & (preds["label"] == "attack") & (preds["score"] >= threshold)]
    out = {}
    for c in classes:
        cand = t[t["attack_class"] == c]
        non_atk = cand[cand["host_role"] != "attacker"]
        pool = non_atk if len(non_atk) else cand
        if len(pool) == 0:
            out[c] = None
            continue
        pool = pool.sort_values(["window_start_epoch", "session", "host_mac"]).reset_index(drop=True)
        out[c] = pool.iloc[(len(pool) - 1) // 2]
    return out


def run(artefact_path, df, preds, out_json, same_alerts_as=None):
    art = lstm.load_artefact(artefact_path)
    model, names, thr, k = art["model"], art["feature_names"], art["threshold"], art["k"]
    use_presence = art["cfg"]["use_presence"]
    test = seq.build_sequences(df, art["scaler"], names, k, use_presence=use_presence,
                               splits=("test",))["test"]
    where = {tuple(r): i for i, r in enumerate(test.meta[KEYS].itertuples(index=False, name=None))}
    groups = ex.build_groups(names)
    picks = pick_alerts(preds, thr)
    if same_alerts_as is not None:
        prev = json.loads(Path(same_alerts_as).read_text())["classes"]
        for c in seq.ATTACK_CLASSES:
            want = None if prev[c].get("status") != "ok" else tuple(prev[c][x] for x in KEYS)
            got = None if picks[c] is None else (picks[c]["session"], picks[c]["host_mac"],
                                                 int(picks[c]["window_start_epoch"]))
            if want != got:
                raise SystemExit(f"{c}: alert {got} differs from {same_alerts_as} {want}; stopping")
    views = {v: (n if n is not None else k) for v, n in VIEWS.items()}
    result = {"model": str(artefact_path), "k": k, "threshold": thr, "views": views,
              "main_view": "last3",
              "occlusion": "feature or group set to scaled zero (the training mean) over the last "
                           "n present windows; is_present and filled windows untouched; delta = "
                           "base attack log-odds - occluded attack log-odds (positive = raised it); "
                           "delta_prob = the same in attack-score units",
              "verdict_rule": {"min_logodds": MIN_LOGODDS,
                               "expected_groups": {c: sorted(v) for c, v in EXPECTED_GROUPS.items()},
                               "expected_features": {c: sorted(v) for c, v in EXPECTED_FEATURES.items()}},
              "same_alerts_as": None if same_alerts_as is None else str(same_alerts_as),
              "groups": groups, "classes": {}}
    for c in seq.ATTACK_CLASSES:
        row = picks[c]
        if row is None:
            result["classes"][c] = {"status": "no_alert"}
            print(f"{c}: no alert", flush=True)
            continue
        i = where[(row["session"], row["host_mac"], int(row["window_start_epoch"]))]
        entry = {"status": "ok", "session": row["session"], "host_mac": row["host_mac"],
                 "window_start_epoch": int(row["window_start_epoch"]), "run_id": row["run_id"],
                 "host_role": row["host_role"], "attack_class": c, "score": float(row["score"]),
                 "threshold": thr, "views": {}}
        for v, n in views.items():
            e = ex.explain(model, test.X[i], last_n=n, groups=groups, feature_names=names,
                           use_presence=use_presence)
            entry["recomputed_score"] = e["base_score"]
            entry["base_logodds"] = e["base_logodds"]
            entry["views"][v] = {
                "n_windows_occluded": e["n_windows_occluded"], "verdict": verdict(c, e),
                "feature_top5": e["feature"]["top"], "group_top5": e["group"]["top"],
                "feature_delta": e["feature"]["delta"], "group_delta": e["group"]["delta"],
                "feature_delta_prob": e["feature"]["delta_prob"],
                "group_delta_prob": e["group"]["delta_prob"]}
            vd = entry["views"][v]["verdict"]
            print(f"{c} [{v}] {row['host_role']} {row['session']} score={row['score']:.5f} "
                  f"logodds={e['base_logodds']:.2f} | top group {vd['top_group']} "
                  f"{vd['top_group_logodds']:+.2f} | top feature {vd['top_feature']} "
                  f"{vd['top_feature_logodds']:+.2f} | useful={vd['useful']}", flush=True)
        result["classes"][c] = entry
    out_json.write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--same-alerts-as", default=None,
                    help="earlier explanations JSON whose alerts must be reproduced exactly")
    args = ap.parse_args()
    os.chdir(common.ROOT)
    lstm.set_determinism(0, 1)
    df = seq.load_sessions(V2 / "sessions", SPLIT_PATH)
    preds = pd.read_parquet(RUNS / "predictions" / f"{RUN_NAME}.parquet")
    run(V2 / "models" / "lstm" / f"{RUN_NAME}.pt", df, preds, RUNS / "explanations.json",
        same_alerts_as=args.same_alerts_as)
    print(f"wrote {RUNS / 'explanations.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
