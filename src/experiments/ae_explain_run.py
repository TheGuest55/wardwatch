"""Per-alert reconstruction-error explanations from the seed-0 Experiment-A autoencoder, on the SAME
windows as the LSTM's runs/lstm/explanations.json (one per attack class), so the triage view is
comparable. If the AE did not flag such a window (score < its threshold) that is recorded, and the AE's
highest-scoring true positive of the class is explained as well; a class with none is recorded as
"no_true_positive". Output: runs/expA/autoencoder_explanations.json.

  python -m src.experiments.ae_explain_run
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from .. import common
from ..models import ae_explain as ex
from ..models import autoencoder as ae
from ..models import data as mdata

V2 = common.PROCESSED_DIR / "v2"
KEYS = ["session", "host_mac", "window_start_epoch"]


def _entry(model, scaler, feats, raw, meta_row, thr):
    e = ex.explain(model, scaler, raw, feats)
    return {"session": meta_row["session"], "host_mac": meta_row["host_mac"],
            "window_start_epoch": int(meta_row["window_start_epoch"]), "host_role": meta_row["host_role"],
            "attack_class": meta_row["attack_class"], "score": e["score"], "threshold": thr,
            "flagged": bool(e["score"] >= thr), "top": e["top"],
            "top_features": [{k: d[k] for k in ("feature", "share", "raw_value", "expected_value")}
                             for d in e["features"][:5]],
            "features": e["features"]}


def run(artefact_path, ds, lstm_json, out_json):
    art = ex.load_artefact(artefact_path)
    model, scaler, thr, feats = art["model"], art["scaler"], art["threshold"], art["feature_names"]
    if feats != ds.feature_names:
        raise SystemExit("artefact feature list differs from the dataset's")
    meta, X = ds.meta("test").reset_index(drop=True), ds.X("test")
    score = ae.anomaly_scores(model, scaler.transform(X))
    where = {tuple(r): i for i, r in enumerate(meta[KEYS].itertuples(index=False, name=None))}
    lstm = json.loads(Path(lstm_json).read_text())["classes"]
    out = {"model": str(artefact_path), "threshold": thr, "lstm_explanations": str(lstm_json), "classes": {}}
    for c, ent in lstm.items():
        if ent.get("status") != "ok":
            out["classes"][c] = {"status": "no_lstm_alert"}
            continue
        i = where[tuple(ent[k] for k in KEYS)]
        res = {"status": "ok", "lstm_alert": _entry(model, scaler, feats, X[i], meta.iloc[i], thr)}
        if not res["lstm_alert"]["flagged"]:
            tp = np.flatnonzero((meta["attack_class"].to_numpy() == c) & (meta["label"].to_numpy() == "attack")
                                & (score >= thr))
            if len(tp):
                j = int(tp[np.argmax(score[tp])])
                res["best_true_positive"] = _entry(model, scaler, feats, X[j], meta.iloc[j], thr)
            else:
                res["best_true_positive"] = "no_true_positive"
        out["classes"][c] = res
        a = res["lstm_alert"]
        print(f"{c}: score={a['score']:.4f} thr={thr:.4f} flagged={a['flagged']} | " + " | ".join(a["top"][:2]))
    Path(out_json).write_text(json.dumps(out, indent=2))
    return out


def main():
    os.chdir(common.ROOT)
    df, fl = mdata.load_dataset(V2 / "testbed_windows.parquet", V2 / "feature_list.json")
    ds = mdata.Dataset(df, fl, scale=False)
    run(V2 / "models" / "expA" / "autoencoder_seed0.pt", ds, V2 / "runs" / "lstm" / "explanations.json",
        V2 / "runs" / "expA" / "autoencoder_explanations.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
