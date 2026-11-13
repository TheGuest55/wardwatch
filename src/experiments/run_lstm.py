"""Resumable job for the LSTM classifier: grid selection, Experiment A (+ k and presence ablations)
and Experiment B (leave-one-attack-out). See docs/phase3-lstm-buildplan.md.

Run order (a run is skipped if its record JSON already exists, so the job resumes):
  1. grid    4 runs  grid_k{6,12}_h{32,64}_s0   Experiment A, seed 0, VALIDATION metrics only,
                                                for_report=False -> selected_config.json
  2. a      15 runs  A_k{1,6,12}_h{sel}_s{0..4} selected config first; determinism check vs the grid
  3. presence 5 runs Anopres_k{sel}_h{sel}_s{0..4}
  4. b      40 runs  B_{class}_s{seed}          seed-major (all 8 classes at seed 0, then seed 1, ...)
Output (all under data/processed/v2/): runs/lstm/<run>.json (record), runs/lstm/predictions/<run>.parquet,
runs/lstm/curves/<run>.json, runs/lstm/failures/<run>.txt, models/lstm/<run>.pt, plus
selected_config.json, determinism_check.json, STOP.md, FLAG.md, summary.md, summary.csv.
Progress goes to stdout; the launch command redirects it to runs/lstm/job.log.

  python -m src.experiments.run_lstm --threads N [--only grid|a|presence|b|summary] [--smoke]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import common
from ..models import data as mdata
from ..models import lstm
from ..models import metrics as mm
from ..models import sequence as seq
from ..models import tracking

V2 = common.PROCESSED_DIR / "v2"
RUNS = V2 / "runs" / "lstm"
MODELS = V2 / "models" / "lstm"
SPLIT_PATH = "data/raw/sessions/split_v2.json"
DATASET_PATH = "data/processed/v2/testbed_windows.parquet"

GRID = [(6, 32), (6, 64), (12, 32), (12, 64)]
K_ABLATION = (1, 6, 12)
SEEDS = (0, 1, 2, 3, 4)
TIE_TOL = 0.005
F1_STOP = 0.80
FLAG_RECALL = 0.98
FLAG_CLASSES = ("eavesdrop", "pump_cmd", "spoof_vitals")
EPOCHS_ASSUMED = 15
N_RUNS = len(GRID) + len(K_ABLATION) * len(SEEDS) + len(SEEDS) + len(seq.ATTACK_CLASSES) * len(SEEDS)


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


@dataclass
class Spec:
    name: str
    experiment: str            # 'A' or 'B' (record field)
    stage: str                 # grid | A | presence | B
    k: int
    hidden: int
    seed: int
    use_presence: bool = True
    exclude: str = None
    for_report: bool = True
    val_only: bool = False
    ablation: str = None


@dataclass
class Ctx:
    df: pd.DataFrame
    feats: list
    scaler: object
    hashes: dict
    threads: int
    runs_dir: Path
    models_dir: Path
    smoke: bool = False
    epochs: int = None                      # None -> lstm default (30)
    cache: dict = field(default_factory=dict)

    def sets(self, k, use_presence, exclude):
        key = (k, use_presence)
        if exclude is None and key in self.cache:
            return self.cache[key]
        out = seq.build_sequences(self.df, self.scaler, self.feats, k, use_presence=use_presence,
                                  exclude_class=exclude)
        if exclude is None:
            while len(self.cache) >= 3:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = out
        return out


# ----------------------------------------------------------------------------------------- setup
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_context(threads, runs_dir=None, models_dir=None, smoke=False, epochs=None):
    split_map = json.loads(Path(SPLIT_PATH).read_text())
    df = seq.load_sessions(V2 / "sessions", SPLIT_PATH)
    _, fl = mdata.load_dataset()
    feats = mdata.feature_columns(fl, df)
    scaler = seq.fit_scaler(df, fl)                       # train-only, as the harness fits it
    hashes = {s: sha256_file(V2 / "sessions" / f"{s}.parquet")
              for sessions in split_map.values() for s in sessions}
    if runs_dir is not None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
    return Ctx(df, feats, scaler, hashes, threads, runs_dir, models_dir, smoke, epochs)


def preflight():
    """Fail fast (before any training) if a report-grade record would be refused for null provenance."""
    rec = tracking.build_run_record("A", "lstm", "classifier", seed=0, split_path=SPLIT_PATH,
                                    dataset_path=DATASET_PATH, for_report=True)
    tracking.assert_report_provenance(rec)
    return rec["provenance"]


# --------------------------------------------------------------------------------------- one run
def _blocks(prefix, meta, y_bin, score, thr, out):
    """Binary metrics on all rows and on rows with host_role != attacker, plus recall on attacker
    rows and on broker rows (attack rows only)."""
    role = meta["host_role"].to_numpy()
    for tag, mask in (("all", np.ones(len(y_bin), bool)), ("noatk", role != "attacker")):
        for key, v in mm.binary_metrics(y_bin[mask], score[mask], threshold=thr).items():
            out[f"{prefix}_{tag}_{key}"] = v
    pred = score >= thr
    for r in ("attacker", "broker"):
        m = (y_bin == 1) & (role == r)
        out[f"{prefix}_recall_{r}"] = float(pred[m].mean()) if m.any() else float("nan")
        out[f"{prefix}_n_attack_{r}"] = int(m.sum())


def _pred_frame(split, s, probs, classes):
    f = s.meta[["session", "host_mac", "host_role", "window_start_epoch", "run_id",
                "attack_class", "label"]].copy()
    f.insert(0, "split", split)
    f["y_bin"] = s.y_bin
    f["score"] = lstm.binary_score(probs).astype(np.float64)
    for i, c in enumerate(classes):
        f[f"p_{c}"] = probs[:, i].astype(np.float64)
    return f


def permutation_top(model, s, feats, seed, budget_s=60.0, top=10):
    """Cheap permutation importance on a B test set: shuffle one feature channel across samples
    (same permutation at every position of the window) and measure the drop in binary PR-AUC. Capped
    at ~budget_s seconds; the sample keeps every attack row and up to 3000 benign rows."""
    rng = np.random.RandomState(seed)
    pos = np.flatnonzero(s.y_bin == 1)
    neg = np.flatnonzero(s.y_bin == 0)
    neg = rng.choice(neg, size=min(3000, len(neg)), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    X, y = s.X[idx], s.y_bin[idx]
    base = lstm._pr_auc(y, lstm.binary_score(lstm.predict_proba(model, X)))
    t0, res = time.perf_counter(), []
    for j, name in enumerate(feats):
        if time.perf_counter() - t0 > budget_s:
            break
        Xp = X.copy()
        Xp[:, :, j] = X[rng.permutation(len(X))][:, :, j]
        ap = lstm._pr_auc(y, lstm.binary_score(lstm.predict_proba(model, Xp)))
        res.append((base - ap, name))
    res.sort(reverse=True)
    return [(n, round(d, 4)) for d, n in res[:top]], len(res), base


def run_one(ctx, spec):
    t_start = time.perf_counter()
    sets = ctx.sets(spec.k, spec.use_presence, spec.exclude)
    classes = seq.class_list(spec.exclude)
    tr, va, te = sets["train"], sets["val"], sets["test"]
    if len(tr.y_class) == 0 or len(va.y_class) == 0:
        raise RuntimeError(f"{spec.name}: empty train or val sequence set")
    if tr.y_class.min() < 0 or va.y_class.min() < 0:
        raise RuntimeError(f"{spec.name}: train/val contain a class without an output unit")
    n_feat = len(ctx.feats)

    cfg = lstm.make_cfg(hidden=spec.hidden, seed=spec.seed, threads=ctx.threads,
                        use_presence=spec.use_presence, n_feat=n_feat,
                        **({"epochs": ctx.epochs} if ctx.epochs else {}))
    lstm.set_determinism(spec.seed, ctx.threads)
    model = lstm.LSTMClassifier(tr.X.shape[-1], spec.hidden, cfg["layers"], cfg["dropout"], len(classes))
    t0 = time.perf_counter()
    res = lstm.train(model, {"X_train": tr.X, "y_train": tr.y_class,
                             "X_val": va.X, "y_val": va.y_class}, cfg)
    train_seconds = time.perf_counter() - t0

    p_val = lstm.predict_proba(model, va.X)
    s_val = lstm.binary_score(p_val)
    thr = lstm.pick_threshold(va.y_bin, s_val)             # validation only
    metrics = {"best_val_pr_auc": res["best_val_pr_auc"], "epochs_run": res["epochs_run"],
               "train_seconds": train_seconds,
               "mean_epoch_seconds": float(np.mean(res["epoch_seconds"])),
               "n_train_sequences": int(len(tr.y_class)), "n_val_sequences": int(len(va.y_class))}
    _blocks("val", va.meta, va.y_bin, s_val, thr, metrics)
    preds = [_pred_frame("val", va, p_val, classes)]

    p_test = None
    if not spec.val_only:                                  # grid runs never score test
        p_test = lstm.predict_proba(model, te.X)
        s_test = lstm.binary_score(p_test)
        _blocks("test", te.meta, te.y_bin, s_test, thr, metrics)
        if spec.exclude is None:                           # macro-F1: classifiers, Experiment A only
            metrics["test_macro_f1"] = lstm.macro_f1(te.y_class, p_test)
            m = (te.meta["host_role"] != "attacker").to_numpy()
            metrics["test_noatk_macro_f1"] = lstm.macro_f1(te.y_class[m], p_test[m])
        preds.append(_pred_frame("test", te, p_test, classes))

    params = {"k": spec.k, "hidden": spec.hidden, "layers": cfg["layers"], "dropout": cfg["dropout"],
              "lr": cfg["lr"], "weight_decay": cfg["weight_decay"], "noise_sigma": cfg["noise_sigma"],
              "threads": ctx.threads, "seed": spec.seed, "held_out_class": spec.exclude,
              "best_epoch": res["best_epoch"], "threshold": thr, "use_presence": spec.use_presence,
              "batch_size": cfg["batch_size"], "max_epochs": cfg["epochs"],
              "patience": cfg["patience"], "n_classes": len(classes)}
    tags = {"run_name": spec.name, "stage": spec.stage, "session_parquet_sha256": ctx.hashes}
    if spec.ablation:
        tags["ablation"] = spec.ablation
    record = tracking.build_run_record(
        spec.experiment, "lstm", "classifier", params=params, metrics=metrics, tags=tags,
        seed=spec.seed, split_path=SPLIT_PATH, dataset_path=DATASET_PATH, for_report=spec.for_report)

    for sub in ("predictions", "curves"):
        (ctx.runs_dir / sub).mkdir(parents=True, exist_ok=True)
    pd.concat(preds, ignore_index=True).to_parquet(ctx.runs_dir / "predictions" / f"{spec.name}.parquet")
    (ctx.runs_dir / "curves" / f"{spec.name}.json").write_text(json.dumps(
        {"history": res["history"], "best_epoch": res["best_epoch"], "classes": classes,
         "class_weights": res["class_weights"], "epoch_seconds": res["epoch_seconds"]}, indent=2))
    lstm.save_artefact(ctx.models_dir / f"{spec.name}.pt", model, ctx.scaler, spec.k, ctx.feats,
                       classes, thr, cfg)
    # the record goes last: its existence marks the run as done (resume)
    tracking.log_run(record, json_path=str(ctx.runs_dir / f"{spec.name}.json"))
    log(f"done {spec.name}: best_epoch={res['best_epoch']} epochs={res['epochs_run']} "
        f"val_f1={metrics['val_all_f1']:.4f} val_pr_auc={metrics['val_all_pr_auc']:.4f}"
        + ("" if spec.val_only else f" test_f1={metrics['test_all_f1']:.4f}")
        + f" ({time.perf_counter() - t_start:.0f}s)")

    if spec.stage == "B" and spec.exclude in FLAG_CLASSES:
        r = metrics["test_noatk_recall"]
        if r >= FLAG_RECALL:
            top, n_done, base = permutation_top(model, te, ctx.feats, spec.seed)
            with open(ctx.runs_dir / "FLAG.md", "a") as f:
                f.write(f"- class={spec.exclude} seed={spec.seed} attacker-excluded test recall={r:.4f}; "
                        f"recall on attacker rows={metrics['test_recall_attacker']:.4f}, "
                        f"broker rows={metrics['test_recall_broker']:.4f}; "
                        f"top permutation features (PR-AUC drop, {n_done}/{len(ctx.feats)} features, "
                        f"base PR-AUC {base:.4f}): {top}\n")
            log(f"FLAG {spec.name}: attacker-excluded recall {r:.4f} >= {FLAG_RECALL}")
    return record


def run_safe(ctx, spec):
    """Run unless the record exists. A failure is written to failures/<run>.txt and the job goes on;
    in smoke mode every error is fatal."""
    if (ctx.runs_dir / f"{spec.name}.json").exists():
        log(f"skip {spec.name} (record exists)")
        return "skipped"
    log(f"start {spec.name}")
    try:
        run_one(ctx, spec)
        return "ok"
    except Exception:
        if ctx.smoke:
            raise
        (ctx.runs_dir / "failures").mkdir(parents=True, exist_ok=True)
        (ctx.runs_dir / "failures" / f"{spec.name}.txt").write_text(traceback.format_exc())
        log(f"FAILED {spec.name}: {traceback.format_exc().splitlines()[-1]}")
        return "failed"


# --------------------------------------------------------------------------------------- stages
def _read(ctx, name):
    p = ctx.runs_dir / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def stop(ctx, why):
    (ctx.runs_dir / "STOP.md").write_text(f"# STOP\n\n{why}\n")
    log(f"STOP: {why}")


def stage_grid(ctx):
    for k, h in GRID:
        run_safe(ctx, Spec(f"grid_k{k}_h{h}_s0", "A", "grid", k, h, 0, for_report=False, val_only=True))


def select_config(ctx):
    """Highest validation PR-AUC (all rows); anything within TIE_TOL of the best is a tie, and a tie
    goes to the smaller model (k first, then hidden). Written before any later run starts."""
    path = ctx.runs_dir / "selected_config.json"
    rows = []
    for k, h in GRID:
        r = _read(ctx, f"grid_k{k}_h{h}_s0")
        if r is not None:
            rows.append({"k": k, "hidden": h, "val_pr_auc": r["metrics"]["val_all_pr_auc"],
                         "val_f1": r["metrics"]["val_all_f1"]})
    if len(rows) < len(GRID):
        stop(ctx, f"grid incomplete: {len(rows)}/{len(GRID)} runs have records (see failures/).")
        return None
    best = max(r["val_pr_auc"] for r in rows)
    tied = [r for r in rows if r["val_pr_auc"] >= best - TIE_TOL]
    sel = min(tied, key=lambda r: (r["k"], r["hidden"]))
    out = {"k": sel["k"], "hidden": sel["hidden"], "val_pr_auc": sel["val_pr_auc"],
           "val_f1": sel["val_f1"], "rule": f"highest validation PR-AUC; within {TIE_TOL} is a tie, "
           "smaller model wins (k first, then hidden)", "grid": rows}
    if path.exists():
        old = json.loads(path.read_text())
        if (old["k"], old["hidden"]) != (out["k"], out["hidden"]):
            stop(ctx, f"selected_config.json says {old['k']}/{old['hidden']} but the grid now selects "
                      f"{out['k']}/{out['hidden']}: not overwriting.")
            return None
        return old
    path.write_text(json.dumps(out, indent=2))
    log(f"selected config: k={out['k']} hidden={out['hidden']} (val PR-AUC {out['val_pr_auc']:.4f})")
    if sel["val_f1"] < F1_STOP:
        stop(ctx, f"grid-best validation binary F1 {sel['val_f1']:.4f} < {F1_STOP} "
                  f"(selected config k={sel['k']}, hidden={sel['hidden']}): pipeline bug suspected.")
        return None
    return out


def determinism_check(ctx, sel):
    """The selected config at seed 0 (stage A) must reproduce the grid run's validation metrics
    exactly. Returns True/False, or None if the A run does not exist yet."""
    g = _read(ctx, f"grid_k{sel['k']}_h{sel['hidden']}_s0")
    a = _read(ctx, f"A_k{sel['k']}_h{sel['hidden']}_s0")
    if g is None or a is None:
        return None
    keys = sorted(k for k in g["metrics"] if k.startswith("val_") or k == "best_val_pr_auc")
    bad = []
    for k in keys:
        x, y = g["metrics"][k], a["metrics"].get(k)
        if not (x == y or (isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y))):
            bad.append({"metric": k, "grid": x, "A": y})
    for k in ("best_epoch", "threshold"):
        if g["params"][k] != a["params"][k]:
            bad.append({"metric": k, "grid": g["params"][k], "A": a["params"][k]})
    out = {"ok": not bad, "n_compared": len(keys) + 2, "mismatches": bad,
           "grid_run": f"grid_k{sel['k']}_h{sel['hidden']}_s0",
           "A_run": f"A_k{sel['k']}_h{sel['hidden']}_s0"}
    (ctx.runs_dir / "determinism_check.json").write_text(json.dumps(out, indent=2, default=str))
    log(f"determinism check: {'OK' if out['ok'] else 'FAILED'} ({out['n_compared']} values compared)")
    if bad:
        stop(ctx, f"determinism check failed, mismatches: {bad}")
    return out["ok"]


def stage_a(ctx, sel):
    k0, h = sel["k"], sel["hidden"]
    order = [k0] + [k for k in K_ABLATION if k != k0]
    for k in order:
        for seed in SEEDS:
            ab = None if k == k0 else f"k_{k}"
            run_safe(ctx, Spec(f"A_k{k}_h{h}_s{seed}", "A", "A", k, h, seed, ablation=ab))
            if k == k0 and seed == 0 and determinism_check(ctx, sel) is False:
                return False
    return True


def stage_presence(ctx, sel):
    for seed in SEEDS:
        run_safe(ctx, Spec(f"Anopres_k{sel['k']}_h{sel['hidden']}_s{seed}", "A", "presence",
                           sel["k"], sel["hidden"], seed, use_presence=False, ablation="no_presence"))


def stage_b(ctx, sel):
    for seed in SEEDS:
        for cls in seq.ATTACK_CLASSES:
            run_safe(ctx, Spec(f"B_{cls}_s{seed}", "B", "B", sel["k"], sel["hidden"], seed, exclude=cls))


# -------------------------------------------------------------------------------------- summary
def load_records(runs_dir):
    rows = []
    for p in sorted(runs_dir.glob("*.json")):
        r = json.loads(p.read_text())
        if "params" not in r or "experiment" not in r:
            continue
        row = {"run": p.stem, "stage": r["tags"].get("stage"), "experiment": r["experiment"],
               "ablation": r["tags"].get("ablation"), "for_report": r["for_report"],
               "k": r["params"]["k"], "hidden": r["params"]["hidden"], "seed": r["params"]["seed"],
               "held_out_class": r["params"]["held_out_class"], "best_epoch": r["params"]["best_epoch"],
               "threshold": r["params"]["threshold"], "use_presence": r["params"]["use_presence"]}
        row.update({k: v for k, v in r["metrics"].items() if isinstance(v, (int, float))})
        rows.append(row)
    return pd.DataFrame(rows)


def _ms(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "n/a"
    return f"{s.mean():.3f} ± {s.std(ddof=1) if len(s) > 1 else float('nan'):.3f}"


def _md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_summary(ctx):
    df = load_records(ctx.runs_dir)
    L = ["# LSTM summary", ""]
    sel_p = ctx.runs_dir / "selected_config.json"
    sel = json.loads(sel_p.read_text()) if sel_p.exists() else None
    L += ["Experiment C: not applicable to the LSTM (CIC feature CSVs have no hosts or time order).", ""]
    if (ctx.runs_dir / "STOP.md").exists():
        L += ["**STOP.md present:**", "", (ctx.runs_dir / "STOP.md").read_text(), ""]
    if (ctx.runs_dir / "FLAG.md").exists():
        L += ["**FLAG.md present (Experiment B near-perfect recall, check for leakage first):**", "",
              (ctx.runs_dir / "FLAG.md").read_text(), ""]
    if sel:
        L += [f"## Selected config\n\nk={sel['k']}, hidden={sel['hidden']} (validation PR-AUC "
              f"{sel['val_pr_auc']:.4f}; rule: {sel['rule']})", "",
              _md(["k", "hidden", "val PR-AUC", "val F1"],
                  [[g["k"], g["hidden"], f"{g['val_pr_auc']:.4f}", f"{g['val_f1']:.4f}"] for g in sel["grid"]]), ""]
    d = ctx.runs_dir / "determinism_check.json"
    if d.exists():
        c = json.loads(d.read_text())
        L += [f"## Determinism check\n\n{'OK' if c['ok'] else 'FAILED'}: {c['A_run']} vs {c['grid_run']}, "
              f"{c['n_compared']} validation values compared, mismatches: {c['mismatches']}", ""]
    counts = df.groupby("stage").size().to_dict() if len(df) else {}
    L += [f"## Runs completed\n\ngrid {counts.get('grid', 0)}/4, A {counts.get('A', 0)}/15, "
          f"presence {counts.get('presence', 0)}/5, B {counts.get('B', 0)}/40", ""]
    fails = sorted(p.stem for p in (ctx.runs_dir / "failures").glob("*.txt")) \
        if (ctx.runs_dir / "failures").exists() else []
    L += ["## Failures", "", ", ".join(fails) if fails else "none", ""]

    if sel and len(df):
        cols = [("precision", "P"), ("recall", "R"), ("f1", "F1"), ("pr_auc", "PR-AUC"),
                ("roc_auc", "ROC-AUC"), ("fpr_at_recall_90", "FPR@R90")]
        groups = [(f"A selected (k={sel['k']}, presence)", df[(df.stage == "A") & df.ablation.isna()])]
        for k in K_ABLATION:
            if k != sel["k"]:
                groups.append((f"k-ablation k={k}", df[(df.stage == "A") & (df.ablation == f"k_{k}")]))
        groups.append(("no is_present", df[df.stage == "presence"]))
        for tag, title in (("all", "all rows"), ("noatk", "attacker host excluded")):
            rows = []
            for name, g in groups:
                row = [name, len(g)] + [_ms(g.get(f"test_{tag}_{c}", pd.Series(dtype=float))) for c, _ in cols]
                row += [_ms(g.get(f"test_{'macro_f1' if tag == 'all' else 'noatk_macro_f1'}",
                                  pd.Series(dtype=float)))]
                row += [_ms(g.get("test_recall_attacker", pd.Series(dtype=float))),
                        _ms(g.get("test_recall_broker", pd.Series(dtype=float)))]
                rows.append(row)
            L += [f"## Experiment A, test, {title} (mean ± std over seeds)", "",
                  _md(["config", "n"] + [n for _, n in cols] + ["macro-F1", "recall attacker rows",
                                                                 "recall broker rows"], rows), ""]
        b = df[df.stage == "B"]
        if len(b):
            rows = []
            for cls in seq.ATTACK_CLASSES:
                g = b[b.held_out_class == cls]
                rows.append([cls, len(g)] + [_ms(g.get(f"test_all_{c}", pd.Series(dtype=float)))
                                             for c in ("precision", "recall", "f1", "pr_auc", "fpr_at_recall_90")]
                            + [_ms(g.get("test_noatk_recall", pd.Series(dtype=float))),
                               _ms(g.get("test_noatk_f1", pd.Series(dtype=float))),
                               _ms(g.get("test_recall_attacker", pd.Series(dtype=float))),
                               _ms(g.get("test_recall_broker", pd.Series(dtype=float)))])
            L += ["## Experiment B (leave-one-attack-out), test = benign + withheld class", "",
                  _md(["withheld", "n", "P", "R", "F1", "PR-AUC", "FPR@R90", "R (no attacker)",
                       "F1 (no attacker)", "R attacker rows", "R broker rows"], rows), ""]
        # per-run recall on test for the selected A config (one run of a class at 1.0 and another at
        # 0.0 would mean the model learned the run, not the attack)
        pr = []
        for r in df[(df.stage == "A") & df.ablation.isna()].itertuples():
            p = ctx.runs_dir / "predictions" / f"{r.run}.parquet"
            if not p.exists():
                continue
            t = pd.read_parquet(p)
            t = t[(t.split == "test") & (t.y_bin == 1)]
            t = t.assign(hit=(t.score >= r.threshold).astype(float))
            for (c, run_id), g in t.groupby(["attack_class", "run_id"]):
                pr.append({"attack_class": c, "run_id": run_id, "seed": r.seed, "recall": g.hit.mean(),
                           "n": len(g)})
        if pr:
            pr = pd.DataFrame(pr)
            rows = [[c, rid, int(g.n.iloc[0]), f"{g.recall.mean():.3f}", f"{g.recall.min():.3f}",
                     f"{g.recall.max():.3f}"]
                    for (c, rid), g in pr.groupby(["attack_class", "run_id"])]
            L += ["## Per-run recall on test (selected A config, over seeds)", "",
                  _md(["class", "run_id", "attack rows", "mean", "min", "max"], rows), ""]
    (ctx.runs_dir / "summary.md").write_text("\n".join(L) + "\n")
    if len(df):
        df.to_csv(ctx.runs_dir / "summary.csv", index=False)
    log(f"summary written to {ctx.runs_dir}")


# ----------------------------------------------------------------------------------------- smoke
def smoke_subset(df):
    """A handful of hosts PER SESSION (MACs differ between sessions): one each of attacker, broker,
    nurse, monitor and pump."""
    role = df.groupby(["session", "host_mac"])["host_role"].agg(lambda s: s.mode().iloc[0])
    keep = []
    for sess, r in role.groupby(level="session"):
        r = r.droplevel("session")
        for want in ("attacker", "broker", "nurse", "monitor", "pump"):
            keep += [(sess, m) for m in sorted(r[r == want].index)[:1]]
    idx = pd.MultiIndex.from_frame(df[["session", "host_mac"]])
    return df[idx.isin(keep)].reset_index(drop=True), keep


def run_smoke(threads):
    log("SMOKE: subset of hosts, one epoch, seed 0, every record fatal on error")
    full = load_context(threads)
    sub_df, keep = smoke_subset(full.df)
    log(f"smoke hosts: {len(keep)} (session, host) pairs of "
        f"{full.df.groupby(['session', 'host_mac']).ngroups}; {len(sub_df)} of {len(full.df)} rows")
    sctx = Ctx(sub_df, full.feats, full.scaler, full.hashes, threads, RUNS / "smoke", MODELS / "smoke",
               smoke=True, epochs=1)
    sctx.runs_dir.mkdir(parents=True, exist_ok=True)
    sctx.models_dir.mkdir(parents=True, exist_ok=True)
    specs = [Spec("smoke_grid_k6_h32_s0", "A", "grid", 6, 32, 0, for_report=False, val_only=True),
             Spec("smoke_A_k6_h32_s0", "A", "A", 6, 32, 0),
             Spec("smoke_Anopres_k6_h32_s0", "A", "presence", 6, 32, 0, use_presence=False,
                  ablation="no_presence"),
             Spec("smoke_B_eavesdrop_s0", "B", "B", 6, 32, 0, exclude="eavesdrop")]
    recs = {}
    for sp in specs:
        p = sctx.runs_dir / f"{sp.name}.json"
        if p.exists():
            p.unlink()                                        # a smoke run always executes
        recs[sp.name] = run_one(sctx, sp)
    # the record of the smoke must be complete and re-readable
    for name, rec in recs.items():
        if not rec["for_report"]:
            continue
        stored = json.loads((sctx.runs_dir / f"{name}.json").read_text())
        missing = [k for k in tracking.PROVENANCE_REQUIRED if stored["provenance"].get(k) is None]
        log(f"smoke record {name}: for_report=True, provenance complete={not missing} {missing or ''}")
        if missing:
            raise SystemExit(f"smoke record {name} has null provenance: {missing}")

    log("SMOKE: timing one full-size epoch (k=12, hidden=64, full train/val sets)")
    t0 = time.perf_counter()
    sets = full.sets(12, True, None)
    t_build = time.perf_counter() - t0
    cfg = lstm.make_cfg(hidden=64, seed=0, threads=threads, epochs=1, n_feat=len(full.feats))
    lstm.set_determinism(0, threads)
    tr, va = sets["train"], sets["val"]
    model = lstm.LSTMClassifier(tr.X.shape[-1], 64, 1, cfg["dropout"], 9)
    res = lstm.train(model, {"X_train": tr.X, "y_train": tr.y_class, "X_val": va.X, "y_val": va.y_class}, cfg)
    ep = res["epoch_seconds"][0]
    total = N_RUNS * EPOCHS_ASSUMED * ep
    timing = {"threads": threads, "k": 12, "hidden": 64, "n_train_sequences": int(len(tr.y_class)),
              "n_val_sequences": int(len(va.y_class)), "sequence_build_seconds": t_build,
              "epoch_seconds": ep, "n_runs": N_RUNS, "epochs_assumed": EPOCHS_ASSUMED,
              "projected_total_seconds": total, "projected_total_hours": total / 3600,
              "note": "upper bound: every run is priced at k=12, hidden=64; k=1/6 and hidden=32 runs are cheaper"}
    (sctx.runs_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    log(f"SMOKE timing: one full-size epoch (k=12, h=64, {threads} threads) = {ep:.1f}s "
        f"(sequence build {t_build:.1f}s)")
    log(f"SMOKE projection: {N_RUNS} runs x {EPOCHS_ASSUMED} epochs x {ep:.1f}s = "
        f"{total / 3600:.1f} h (upper bound)")
    write_summary(sctx)
    log("SMOKE passed")


# ------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=1, help="torch threads; keep identical across runs")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", choices=["grid", "a", "presence", "b", "summary"])
    args = ap.parse_args(argv)
    os.chdir(common.ROOT)
    log(f"threads={args.threads} smoke={args.smoke} only={args.only}")
    prov = preflight()
    log(f"provenance ok: git_commit={prov['git_commit']} git_dirty={prov['git_dirty']} "
        f"image_digest={str(prov['image_digest'])[:19]}")
    if args.smoke:
        run_smoke(args.threads)
        return 0

    ctx = load_context(args.threads, RUNS, MODELS)
    if (RUNS / "STOP.md").exists():
        log("STOP.md exists from an earlier run: resolve it (delete the file) before relaunching")
        return 3
    if args.only in (None, "grid"):
        stage_grid(ctx)
    if args.only == "summary":
        write_summary(ctx)
        return 0
    sel = select_config(ctx)
    if args.only == "grid":
        write_summary(ctx)
        return 0
    if sel is None:
        write_summary(ctx)
        return 3
    ok = True
    if args.only in (None, "a"):
        ok = stage_a(ctx, sel)
    if ok and args.only in (None, "presence"):
        stage_presence(ctx, sel)
    if ok and args.only in (None, "b"):
        stage_b(ctx, sel)
    if ok and determinism_check(ctx, sel) is False:
        ok = False
    write_summary(ctx)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
