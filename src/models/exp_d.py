"""Experiment D — INDICATIVE benchmark on CICIoMT2024's own feature CSVs + official split (./CLAUDE.md).

    python -m src.models.exp_d [--models xgboost random_forest] [--drop iat] [--raw PATH] [--for-report]

Input : data/raw/ciciomt2024/WiFi_and_MQTT/attacks/csv/{train,test}/*.pcap.csv   (CIC's Wi-Fi/MQTT attack CSVs,
        45 features, one class per file; downloaded after CIC's registration form). --raw PATH points elsewhere
        (read-only: nothing under data/raw is ever written, moved or copied).
Output: data/processed/v2/runs/expD/{model}[_noiat].json (+ _details.json)

Why "indicative": these are CIC's features (not our extractor), so the numbers compare our MODEL
choice against the published baselines on the published protocol — they say nothing about our
testbed or our features. Experiment C is the like-for-like cross-dataset test.

Protocol:
  * Labels from the file name, as CIC defines them (README): 19 classes = file stem with the
    "_train"/"_test" suffix and any trailing part number removed (TCP_IP-DDoS-SYN1..4 -> one class);
    6 categories = Benign / Spoofing / Recon / MQTT / DoS / DDoS; 2 = benign vs attack.
  * Train on CIC's train files, test on CIC's test files (official split, untouched).
  * Validation for early stopping/threshold = the LAST 10% of rows of every train file (time order
    within the file; never a random shuffle), so test is touched once.
  * Same XGBoost / RF settings family as our testbed experiments; one fixed config per model (no
    grid on test). Seeds 0..N-1 -> mean +/- std.
  * Metrics: accuracy, macro precision/recall/F1 (the paper reports these), plus binary F1 for the
    2-class task; per-class recall in the details file.
  * Leakage check: feature importances are logged; a near-perfect score driven by one feature (IAT
    is the known suspect in CIC's sister dataset CICIoT2023, where it tracks capture time) is
    re-run with --drop iat and both are reported.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
import pandas as pd

from .. import common
from . import tracking
from .train_tree import N_JOBS, V2

RAW = common.ROOT / "data" / "raw" / "ciciomt2024" / "WiFi_and_MQTT" / "attacks" / "csv"
OUT = V2 / "runs" / "expD"
VAL_TAIL = 0.10

# Published baselines: Dadkhah et al., CICIoMT2024 (preprints.org 202402.0898 v1, Table 8; journal
# version: Internet of Things 28, 2024). Recall/precision/F1 are their (macro) values as printed.
PUBLISHED = {
    "2": {"LR": (0.99464682, 0.93505479, 0.94600144, 0.94045653),
          "AdaBoost": (0.99807704, 0.97788699, 0.9797801, 0.97883158),
          "DNN": (0.99617391, 0.95233191, 0.96276398, 0.95748541),
          "RF": (0.99837193, 0.97640269, 0.98754544, 0.98190642)},
    "6": {"LR": (0.74399479, 0.60901055, 0.73229673, 0.60443814),
          "AdaBoost": (0.90303572, 0.77537141, 0.8417807, 0.80562143),
          "DNN": (0.78052413, 0.76020495, 0.76809269, 0.73354862),
          "RF": (0.99388111, 0.9379039, 0.94855819, 0.94221191)},
    "19": {"LR": (0.74562534, 0.4495978, 0.57814073, 0.43098172),
           "AdaBoost": (0.23653652, 0.34171028, 0.50168939, 0.30109723),
           "DNN": (0.77656609, 0.60194873, 0.73835033, 0.57906237),
           "RF": (0.99553086, 0.89345693, 0.94831885, 0.90736136)},
}   # tuple order: accuracy, recall, precision, F1

CATEGORY = [(r"^Benign$", "Benign"), (r"^ARP_Spoofing$", "Spoofing"), (r"^Recon-", "Recon"),
            (r"^MQTT-", "MQTT"), (r"^TCP_IP-DDoS-", "DDoS"), (r"^TCP_IP-DoS-", "DoS")]
N_CLASSES = {"19": 19, "6": 6, "2": 2}


def class_of(filename):
    """CIC file name -> 19-class label. 'TCP_IP-DDoS-SYN3_train.pcap.csv' -> 'TCP_IP-DDoS-SYN'."""
    stem = Path(filename).name.split(".")[0]
    stem = re.sub(r"_(train|test)$", "", stem)
    return re.sub(r"\d+$", "", stem)


def category_of(cls):
    for pat, cat in CATEGORY:
        if re.search(pat, cls):
            return cat
    raise ValueError(f"no category for class {cls!r}")


def load_split(split, raw=RAW):
    files = sorted((raw / split).glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSVs in {raw / split}")
    parts = []
    for f in files:
        df = pd.read_csv(f, dtype=np.float32)
        df.columns = [c.strip() for c in df.columns]
        n = len(df)
        df["c19"] = class_of(f.name)
        df["source_file"] = f.name
        # position within the file (capture order) -> the train-file tail becomes validation
        df["is_val"] = (np.arange(n) >= int(np.floor(n * (1 - VAL_TAIL)))) if split == "train" else False
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    # labels as categoricals: millions of Python string objects otherwise (the first full run
    # was OOM-killed at 8 GB)
    for c in ("c19", "source_file"):
        df[c] = df[c].astype("category")
    df["c6"] = df["c19"].map(category_of).astype("category")
    df["c2"] = pd.Categorical(np.where(df["c19"] == "Benign", "Benign", "Attack"))
    return df


def to_arrays(tr, te, feats, tasks):
    """Build the float32 matrices ONCE (fit / val / test) and small label arrays per task, so no
    task re-copies the 7M-row table."""
    val = tr["is_val"].to_numpy(bool)
    A = {"Xf": tr.loc[~val, feats].to_numpy(np.float32), "Xv": tr.loc[val, feats].to_numpy(np.float32),
         "Xt": te[feats].to_numpy(np.float32), "y": {}}
    for t in tasks:
        c = f"c{t}"
        A["y"][t] = (tr[c].to_numpy()[~val], tr[c].to_numpy()[val], te[c].to_numpy())
    return A


def check_complete(tr, te):
    """Refuse to run on a partial download: every class must be present in both splits."""
    a, b = set(tr["c19"]), set(te["c19"])
    problems = []
    if len(a) != 19:
        problems.append(f"train has {len(a)} classes, expected 19")
    if a != b:
        problems.append(f"classes differ: only-train={sorted(a - b)} only-test={sorted(b - a)}")
    return problems


def make_model(name, n_classes, seed):
    if name == "xgboost":
        from xgboost import XGBClassifier
        kw = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", n_jobs=N_JOBS, random_state=seed,
                  early_stopping_rounds=30)
        if n_classes == 2:
            return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **kw)
        return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **kw)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        # sklearn defaults (100 trees, full depth) — the paper does not report its RF settings
        return RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1,
                                      n_jobs=N_JOBS, random_state=seed)
    raise ValueError(name)


def scores(y, p, labels):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro",
                                                    zero_division=0)
    return {"accuracy": float(accuracy_score(y, p)), "precision_macro": float(pr),
            "recall_macro": float(rc), "f1_macro": float(f1)}


def run(name, A, feats, task, seed):
    from sklearn.preprocessing import LabelEncoder
    lf, lv, lt = A["y"][task]
    enc = LabelEncoder().fit(np.concatenate([lf, lv]))
    Xf, yf = A["Xf"], enc.transform(lf)
    Xv, yv = A["Xv"], enc.transform(lv)
    Xt, yt = A["Xt"], enc.transform(lt)
    m = make_model(name, len(enc.classes_), seed)
    t0 = time.perf_counter()
    if name == "xgboost":
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    else:
        m.fit(Xf, yf)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pt = m.predict(Xt)
    pred_s = time.perf_counter() - t0
    out = scores(yt, pt, list(range(len(enc.classes_))))
    out.update({"train_s": train_s, "predict_us_per_row": pred_s / len(Xt) * 1e6,
                "val_" + "f1_macro": scores(yv, m.predict(Xv), list(range(len(enc.classes_))))["f1_macro"]})
    if task == "2":
        from sklearn.metrics import f1_score
        atk = int(np.flatnonzero(enc.classes_ == "Attack")[0])
        out["f1_attack"] = float(f1_score(yt == atk, pt == atk))
    per_class = {c: float((pt[yt == i] == i).mean()) for i, c in enumerate(enc.classes_)}
    if name == "xgboost":
        imp = m.get_booster().get_score(importance_type="gain")
        vals = [imp.get(f"f{i}", 0.0) for i in range(len(feats))]
    else:
        vals = list(m.feature_importances_)
    tot = sum(vals) or 1.0
    top = sorted(((f, v / tot) for f, v in zip(feats, vals)), key=lambda kv: -kv[1])[:10]
    return out, per_class, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["xgboost", "random_forest"])
    ap.add_argument("--tasks", nargs="+", default=["2", "6", "19"], choices=list(N_CLASSES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--drop", nargs="*", default=[], help="feature names to drop, e.g. IAT")
    ap.add_argument("--raw", type=Path, default=RAW,
                    help="folder holding train/ and test/ with CIC's *.pcap.csv files (default: %(default)s)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="run even if some classes are missing (smoke test only; never report)")
    ap.add_argument("--for-report", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    raw = args.raw
    tr, te = load_split("train", raw), load_split("test", raw)
    problems = check_complete(tr, te)
    if problems and not args.allow_partial:
        raise SystemExit("incomplete CIC download: " + "; ".join(problems))
    if problems and args.for_report:
        raise SystemExit("--for-report refused on a partial download")
    if problems:   # smoke test only: score test on the classes train has seen
        te = te[te["c19"].isin(set(tr["c19"]))].reset_index(drop=True)
        print("PARTIAL DOWNLOAD (smoke test, not reportable):", "; ".join(problems))
    drop = {d.lower() for d in args.drop}
    meta = {"c19", "c6", "c2", "source_file", "is_val"}
    feats = [c for c in tr.columns if c not in meta and c.lower() not in drop]
    print(f"train {len(tr):,} rows ({int(tr['is_val'].sum()):,} val tail) | test {len(te):,} | "
          f"{len(feats)} features | dropped {sorted(drop) or 'none'} | classes {tr['c19'].nunique()}")
    print("  test rows per 6-cat:", te["c6"].value_counts().to_dict())
    tag = "" if not drop else "_no" + "_".join(sorted(drop)).lower()
    rows = {"train": len(tr), "test": len(te)}
    files = {"train": sorted(map(str, tr["source_file"].unique())),
             "test": sorted(map(str, te["source_file"].unique()))}
    A = to_arrays(tr, te, feats, args.tasks)
    del tr, te
    import gc
    gc.collect()

    for name in args.models:
        results, details = {}, {}
        for task in args.tasks:
            per_seed, pcs, tops = [], [], []
            for seed in range(args.seeds):
                m, pc, top = run(name, A, feats, task, seed)
                per_seed.append(m)
                pcs.append(pc)
                tops.append(top)
                print(f"  [{name}] {task:>2}-class seed {seed}: acc {m['accuracy']:.4f} "
                      f"P {m['precision_macro']:.4f} R {m['recall_macro']:.4f} "
                      f"F1 {m['f1_macro']:.4f} | train {m['train_s']:.0f}s")
            for k in per_seed[0]:
                v = np.array([s[k] for s in per_seed], dtype=float)
                results[f"{task}c_{k}_mean"], results[f"{task}c_{k}_std"] = (float(v.mean()),
                                                                              float(v.std()))
            details[task] = {"per_seed": per_seed, "per_class_recall_seed0": pcs[0],
                             "top_features_seed0": tops[0]}
            print(f"  [{name}] {task}-class top features: {[(f, round(v, 3)) for f, v in tops[0][:5]]}")
        print(f"\n[{name}{tag}] vs published (Dadkhah et al. Table 8; best published = RF):")
        print(f"  {'task':5s} {'metric':10s} {'ours':>16s} {'pub RF':>8s} {'pub best':>9s}")
        for task in args.tasks:
            for i, (k, lbl) in enumerate((("accuracy", "accuracy"), ("recall_macro", "recall"),
                                          ("precision_macro", "precision"), ("f1_macro", "F1"))):
                pub = {m: v[i] for m, v in PUBLISHED[task].items()}
                best = max(pub, key=pub.get)
                print(f"  {task:>3}c  {lbl:10s} {results[f'{task}c_{k}_mean']:.4f} ± "
                      f"{results[f'{task}c_{k}_std']:.4f} {pub['RF']:8.4f} {pub[best]:6.4f} ({best})")
                results[f"{task}c_{k}_minus_published_rf"] = results[f"{task}c_{k}_mean"] - pub["RF"]
        OUT.mkdir(parents=True, exist_ok=True)
        det = OUT / f"{name}{tag}_details.json"
        det.write_text(json.dumps({"features": feats, "dropped": sorted(drop), "tasks": details,
                                   "rows": rows, "files": files},
                                  indent=2, default=str))
        rec = tracking.build_run_record(
            experiment="D" if not drop else f"D-drop-{'-'.join(sorted(drop))}", model=name,
            role="classifier", params={"seeds": args.seeds, "tasks": args.tasks, "dropped": sorted(drop),
                                       "val": f"last {VAL_TAIL:.0%} of each train file",
                                       "partial": bool(problems)},
            metrics=results, tags={"dataset": "CICIoMT2024 WiFi/MQTT CSV (official split)",
                                   "details": str(det), "indicative": True, "raw": str(raw),
                                   "attacker_excluded_view": "not applicable (no host column in CIC's CSVs)",
                                   "comparable_to_A_C": False},
            seed=0, split_path=None, dataset_path=None, for_report=False)
        # provenance for D: the CIC CSVs are the dataset; hash the file list, not one file
        rec["provenance"]["split_file"] = "CIC official train/test folders"
        rec["provenance"]["split_sha256"] = tracking.hashlib.sha256(
            "\n".join(files["train"] + files["test"]).encode()).hexdigest()
        rec["provenance"]["dataset_sha256"] = tracking.hashlib.sha256(
            "".join(tracking._sha256_file(p) for p in sorted(raw.rglob("*.csv"))).encode()).hexdigest()
        rec["for_report"] = bool(args.for_report)
        tracking.log_run(rec, json_path=str(OUT / f"{name}{tag}.json"), use_mlflow=not args.no_mlflow)
        print(f"[{name}] record -> {OUT / (name + tag + '.json')}")


if __name__ == "__main__":
    main()
