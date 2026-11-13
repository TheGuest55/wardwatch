"""Occlusion explanations for one test alert per attack class, from the selected seed-0 A model.

  python -m src.experiments.lstm_explain_run

An alert = a test row with label attack and score >= the model's validation threshold. Per class the
alert is the MEDIAN-in-time one among the candidates (deterministic, not cherry-picked), taken from a
non-attacker host where one exists (host_role is recorded either way). A class with no alert is
recorded explicitly as "no_alert"; nothing is substituted. Output: runs/lstm/explanations.json.
"""
import json
import os
import sys

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


def run(artefact_path, df, preds, out_json, last_n=3):
    art = lstm.load_artefact(artefact_path)
    model, names, thr, k = art["model"], art["feature_names"], art["threshold"], art["k"]
    use_presence = art["cfg"]["use_presence"]
    test = seq.build_sequences(df, art["scaler"], names, k, use_presence=use_presence,
                               splits=("test",))["test"]
    where = {tuple(r): i for i, r in enumerate(test.meta[KEYS].itertuples(index=False, name=None))}
    groups = ex.build_groups(names)
    picks = pick_alerts(preds, thr)
    result = {"model": str(artefact_path), "k": k, "threshold": thr, "last_n": last_n,
              "occlusion": "feature or group set to scaled zero (the training mean) over the last "
                           "n present windows; is_present and filled windows untouched; delta = "
                           "base attack score - occluded attack score (positive = raised the score)",
              "groups": groups, "classes": {}}
    for c in seq.ATTACK_CLASSES:
        row = picks[c]
        if row is None:
            result["classes"][c] = {"status": "no_alert"}
            print(f"{c}: no alert", flush=True)
            continue
        i = where[(row["session"], row["host_mac"], int(row["window_start_epoch"]))]
        e = ex.explain(model, test.X[i], last_n=last_n, groups=groups, feature_names=names,
                       use_presence=use_presence)
        result["classes"][c] = {
            "status": "ok", "session": row["session"], "host_mac": row["host_mac"],
            "window_start_epoch": int(row["window_start_epoch"]), "run_id": row["run_id"],
            "host_role": row["host_role"], "attack_class": c, "score": float(row["score"]),
            "recomputed_score": e["base_score"], "threshold": thr,
            "n_windows_occluded": e["n_windows_occluded"],
            "feature_top5": e["feature"]["top"], "group_top5": e["group"]["top"],
            "feature_delta": e["feature"]["delta"], "group_delta": e["group"]["delta"]}
        print(f"{c}: {row['host_role']} {row['session']} score={row['score']:.3f} | "
              f"{e['feature']['top'][0]} | {e['group']['top'][0]}", flush=True)
    out_json.write_text(json.dumps(result, indent=2))
    return result


def main():
    os.chdir(common.ROOT)
    lstm.set_determinism(0, 1)
    df = seq.load_sessions(V2 / "sessions", SPLIT_PATH)
    preds = pd.read_parquet(RUNS / "predictions" / f"{RUN_NAME}.parquet")
    run(V2 / "models" / "lstm" / f"{RUN_NAME}.pt", df, preds, RUNS / "explanations.json")
    print(f"wrote {RUNS / 'explanations.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
