"""LSTM Experiment C (cross-dataset, testbed <-> CICIoMT2024), frozen config, no tuning.

  python -m src.experiments.run_lstm_c --check      # no training: rows/sequences per domain, split, class
  python -m src.experiments.run_lstm_c --smoke      # 1 seed, 1 epoch, both directions -> runs/lstm_c/smoke/
  python -m src.experiments.run_lstm_c --threads 1  # 10 runs: seeds 0..4 x {tb2cic, cic2tb}, resumable
  python -m src.experiments.run_lstm_c --threads 1 --size-variant nosize   # frame-size sensitivity

Config: k and hidden come from runs/lstm/selected_config.json and must be k=12, hidden=32 (is_present
on); everything else is lstm.make_cfg's default. Each run: fit the SOURCE train-only scaler, build the
source train/val sequences, early-stop on SOURCE val PR-AUC, pick the threshold on SOURCE val (as
trained), then score source test (in-domain reference) and target test (cross-domain), both with the
source scaler, which is never refit on the target. There is no threshold-refit variant.

Inputs are the shared 48 features + is_present; outputs are benign + the 4 shared classes. CIC is the
single-attacker (DoS) extraction; a DDoS file stops the run. Metrics are reported on all rows and with
host_role == attacker excluded (the CIC attacker is role "attacker" in EVERY file, Benign included).
Rule 3: cross-domain F1 >= 0.98 on either view appends the top-10 permutation features to FLAG.md.
Outputs: data/processed/v2/runs/lstm_c/{<run>.json, predictions/, curves/, summary.md/.csv}, artefacts in
data/processed/v2/models/lstm_c/ (each holds the SOURCE scaler).
--size-variant nosize drops train_tree.SIZE_FEATURES (7 frame-size features) in both domains and writes to
runs/lstm_c_nosize/ and models/lstm_c_nosize/; records are tagged sensitivity=nosize and are NOT inputs to the
selection rule.
"""
import argparse
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
from ..models import lstm
from ..models import sequence as seq
from ..models import sequence_c as sc
from ..models import tracking
from ..models import train_tree as tt
from . import run_lstm as rl

V2 = common.PROCESSED_DIR / "v2"
RUNS = V2 / "runs" / "lstm_c"
MODELS = V2 / "models" / "lstm_c"
SELECTED = V2 / "runs" / "lstm" / "selected_config.json"
SEEDS = (0, 1, 2, 3, 4)
FROZEN_K, FROZEN_H = 12, 32
DIRECTIONS = (("testbed", "cic", "tb2cic"), ("cic", "testbed", "cic2tb"))
FLAG_F1 = 0.98
log = rl.log


class StopRun(Exception):
    """A condition the plan says to stop and ask about (STOP.md is written)."""


def run_name(tag, seed):
    return f"C_{tag}_s{seed}"


@dataclass
class Ctx:
    paths: dict
    doms: dict
    k: int
    hidden: int
    threads: int
    runs_dir: Path
    models_dir: Path
    hashes: dict
    smoke: bool = False
    epochs: int = None
    size_variant: str = "none"
    scalers: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)

    def scaler(self, src):
        if src not in self.scalers:
            self.scalers[src] = sc.fit_source_scaler(self.doms[src])       # source train rows only
        return self.scalers[src]

    def sets(self, dom, src, splits):
        key = (dom, src, splits)
        if key not in self.cache:
            self.cache[key] = sc.domain_sequences(self.doms[dom], self.scaler(src), self.k, splits=splits)
        return self.cache[key]


def load_selected(path=SELECTED):
    sel = json.loads(Path(path).read_text())
    if (sel["k"], sel["hidden"]) != (FROZEN_K, FROZEN_H):
        raise SystemExit(f"selected_config.json says k={sel['k']} hidden={sel['hidden']}; Experiment C is "
                         f"frozen at k={FROZEN_K} hidden={FROZEN_H}. Stopping.")
    return sel["k"], sel["hidden"]


def make_ctx(paths, threads, runs_dir, models_dir, smoke=False, epochs=None, selected=SELECTED,
             size_variant="none"):
    k, h = load_selected(selected)
    doms = sc.load_pair(paths, size_variant)
    split_map = json.loads(Path(paths["split_path"]).read_text())
    hashes = {"session_parquet_sha256": {s: rl.sha256_file(Path(paths["sessions_dir"]) / f"{s}.parquet")
                                         for ss in split_map.values() for s in ss},
              "cic_windows_sha256": rl.sha256_file(paths["cic_windows"]),
              "cic_feature_list_sha256": rl.sha256_file(paths["cic_fl"])}
    if not Path(paths["cic_report"]).exists():
        raise SystemExit(f"{paths['cic_report']} is missing: extract_cic must have written it")
    hashes["cic_extract_report_sha256"] = rl.sha256_file(paths["cic_report"])
    hashes["cic_files"] = sc.check_cic_files(doms["cic"].df)
    runs_dir, models_dir = Path(runs_dir), Path(models_dir)
    for d in (runs_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)
    if size_variant != "none":
        hashes.update({"sensitivity": size_variant, "size_variant": size_variant,
                       "dropped_features": [f for f in tt.SIZE_FEATURES if f not in doms["testbed"].feats]})
    return Ctx(paths, doms, k, h, threads, runs_dir, models_dir, hashes, smoke, epochs, size_variant=size_variant)


def preflight(paths):
    rec = tracking.build_run_record("C", "lstm", "classifier", seed=0, split_path=paths["split_path"],
                                    dataset_path=paths["tb_windows"], for_report=True)
    tracking.assert_report_provenance(rec)
    return rec["provenance"]


# ------------------------------------------------------------------------------------ metrics
def _blocks_c(prefix, meta, y_bin, score, thr, out):
    """run_lstm's blocks (all rows / attacker-excluded, attacker and broker recall) + FPR + recall per
    shared class, on both views."""
    rl._blocks(prefix, meta, y_bin, score, thr, out)
    role = meta["host_role"].to_numpy()
    pred = score >= thr
    cls = meta["attack_class"].astype(str).to_numpy()
    for tag, mask in (("all", np.ones(len(y_bin), bool)), ("noatk", role != "attacker")):
        neg = mask & (y_bin == 0)
        out[f"{prefix}_{tag}_fpr"] = float(pred[neg].mean()) if neg.any() else float("nan")
        for c in sc.SHARED:
            m = mask & (y_bin == 1) & (cls == c)
            out[f"{prefix}_{tag}_recall_{c}"] = float(pred[m].mean()) if m.any() else float("nan")
            out[f"{prefix}_{tag}_n_{c}"] = int(m.sum())


def _frame(domain_role, dataset, split, s, probs):
    f = rl._pred_frame(split, s, probs, sc.C_CLASSES)
    f.insert(1, "domain", domain_role)
    f.insert(2, "dataset", dataset)
    return f


# ------------------------------------------------------------------------------------- one run
def run_one(ctx, src, tgt, tag, seed):
    t_start = time.perf_counter()
    name = run_name(tag, seed)
    src_sets = ctx.sets(src, src, seq.SPLITS)
    tgt_sets = ctx.sets(tgt, src, ("test",))
    tr, va, te_s, te_t = src_sets["train"], src_sets["val"], src_sets["test"], tgt_sets["test"]
    if len(tr.y_class) == 0 or len(va.y_class) == 0:
        raise StopRun(f"{name}: empty source train or val sequence set")
    if int(va.y_bin.sum()) == 0 or int(tr.y_bin.sum()) == 0:
        raise StopRun(f"{name}: source ({src}) {'val' if va.y_bin.sum() == 0 else 'train'} has no attack "
                      f"rows after the context rule; early stopping and the threshold need them")
    for s in (tr, va, te_s, te_t):
        if len(s.y_class) and s.y_class.min() < 0:
            raise RuntimeError(f"{name}: a target without an output unit survived the context rule")
    n_feat = len(ctx.doms[src].feats)
    cfg = lstm.make_cfg(hidden=ctx.hidden, seed=seed, threads=ctx.threads, use_presence=True,
                        n_feat=n_feat, **({"epochs": ctx.epochs} if ctx.epochs else {}))
    lstm.set_determinism(seed, ctx.threads)
    model = lstm.LSTMClassifier(tr.X.shape[-1], ctx.hidden, cfg["layers"], cfg["dropout"], len(sc.C_CLASSES))
    t0 = time.perf_counter()
    res = lstm.train(model, {"X_train": tr.X, "y_train": tr.y_class,
                             "X_val": va.X, "y_val": va.y_class}, cfg)      # early stop: SOURCE val
    train_seconds = time.perf_counter() - t0

    p_val = lstm.predict_proba(model, va.X)
    s_val = lstm.binary_score(p_val)
    thr = lstm.pick_threshold(va.y_bin, s_val)                               # SOURCE val only
    p_src, p_tgt = lstm.predict_proba(model, te_s.X), lstm.predict_proba(model, te_t.X)
    s_src, s_tgt = lstm.binary_score(p_src), lstm.binary_score(p_tgt)
    metrics = {"best_val_pr_auc": res["best_val_pr_auc"], "epochs_run": res["epochs_run"],
               "train_seconds": train_seconds, "mean_epoch_seconds": float(np.mean(res["epoch_seconds"])),
               "n_train_sequences": int(len(tr.y_class)), "n_val_sequences": int(len(va.y_class)),
               "n_src_test_sequences": int(len(te_s.y_class)), "n_tgt_test_sequences": int(len(te_t.y_class))}
    _blocks_c("val", va.meta, va.y_bin, s_val, thr, metrics)
    _blocks_c("src_test", te_s.meta, te_s.y_bin, s_src, thr, metrics)
    _blocks_c("tgt_test", te_t.meta, te_t.y_bin, s_tgt, thr, metrics)

    padded = {"train": sc.padded_share(tr), "val": sc.padded_share(va), "src_test": sc.padded_share(te_s),
              "tgt_test": sc.padded_share(te_t)}
    rows = sc.row_table(ctx.doms[src], src_sets)
    params = {"k": ctx.k, "hidden": ctx.hidden, "seed": seed, "threshold": thr,
              "best_epoch": res["best_epoch"], "direction": tag, "source": src, "target": tgt,
              "n_features": n_feat, "shared_classes": list(sc.SHARED), "use_presence": True,
              "threshold_rule": "source-val-max-F1 (as trained)", "layers": cfg["layers"],
              "dropout": cfg["dropout"], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"],
              "noise_sigma": cfg["noise_sigma"], "batch_size": cfg["batch_size"],
              "max_epochs": cfg["epochs"], "patience": cfg["patience"], "threads": ctx.threads,
              "n_classes": len(sc.C_CLASSES), "classes": list(sc.C_CLASSES), "size_variant": ctx.size_variant}
    counts = {src: {sp: int(len(src_sets[sp].y_class)) for sp in seq.SPLITS}, tgt: {"test": int(len(te_t.y_class))}}
    tags = {"run_name": name, "stage": "C", **ctx.hashes, "sequence_counts": counts,
            "padded_context_share": padded, "row_table_source": rows,
            "tgt_test_rows": {"tree_rows": int(len(ctx.doms[tgt].tree_ds.meta("test"))),
                              "lstm_targets": int(len(te_t.y_class))}}
    record = tracking.build_run_record("C", "lstm", "classifier", params=params, metrics=metrics,
                                       tags=tags, seed=seed, split_path=ctx.paths["split_path"],
                                       dataset_path=ctx.paths["tb_windows"], for_report=True)

    for sub in ("predictions", "curves"):
        (ctx.runs_dir / sub).mkdir(parents=True, exist_ok=True)
    pd.concat([_frame("val", src, "val", va, p_val), _frame("src_test", src, "test", te_s, p_src),
               _frame("tgt_test", tgt, "test", te_t, p_tgt)], ignore_index=True
              ).to_parquet(ctx.runs_dir / "predictions" / f"{name}.parquet")
    (ctx.runs_dir / "curves" / f"{name}.json").write_text(json.dumps(
        {"history": res["history"], "best_epoch": res["best_epoch"], "classes": sc.C_CLASSES,
         "class_weights": res["class_weights"], "epoch_seconds": res["epoch_seconds"]}, indent=2))
    lstm.save_artefact(ctx.models_dir / f"{name}.pt", model, ctx.scaler(src), ctx.k,
                       ctx.doms[src].feats, sc.C_CLASSES, thr, cfg)
    tracking.log_run(record, json_path=str(ctx.runs_dir / f"{name}.json"))   # last: marks the run done
    log(f"done {name}: best_epoch={res['best_epoch']} src_f1={metrics['src_test_all_f1']:.4f} "
        f"tgt_f1={metrics['tgt_test_all_f1']:.4f} tgt_noatk_f1={metrics['tgt_test_noatk_f1']:.4f} "
        f"({time.perf_counter() - t_start:.0f}s)")

    f1s = (metrics["tgt_test_all_f1"], metrics["tgt_test_noatk_f1"])
    if max(f1s) >= FLAG_F1:                                                  # rule 3
        top, n_done, base = rl.permutation_top(model, te_t, ctx.doms[src].feats, seed)
        with open(ctx.runs_dir / "FLAG.md", "a") as f:
            f.write(f"- direction={tag} seed={seed} cross-domain F1 all rows={f1s[0]:.4f}, "
                    f"attacker-excluded={f1s[1]:.4f} (>= {FLAG_F1}); top permutation features (PR-AUC drop, "
                    f"{n_done}/{n_feat} features, base PR-AUC {base:.4f}): {top}\n")
        log(f"FLAG {name}: cross-domain F1 {max(f1s):.4f} >= {FLAG_F1}")
    return record


def run_safe(ctx, src, tgt, tag, seed):
    name = run_name(tag, seed)
    if (ctx.runs_dir / f"{name}.json").exists():
        log(f"skip {name} (record exists)")
        return "skipped"
    log(f"start {name}")
    try:
        run_one(ctx, src, tgt, tag, seed)
        return "ok"
    except StopRun:
        raise
    except Exception:
        if ctx.smoke:
            raise
        (ctx.runs_dir / "failures").mkdir(parents=True, exist_ok=True)
        (ctx.runs_dir / "failures" / f"{name}.txt").write_text(traceback.format_exc())
        log(f"FAILED {name}: {traceback.format_exc().splitlines()[-1]}")
        return "failed"


def run_all(ctx, seeds=SEEDS):
    out = {}
    for seed in seeds:                                    # seed-major: a cut-short job has whole seeds
        for src, tgt, tag in DIRECTIONS:
            out[run_name(tag, seed)] = run_safe(ctx, src, tgt, tag, seed)
    return out


# ---------------------------------------------------------------------------------------- check
def check(paths, k=FROZEN_K):
    """No training: load both domains, assert the features match, print rows and sequences."""
    doms = sc.load_pair(paths)
    print(f"features: {len(doms['testbed'].feats)} shared (+ is_present) = identical in both domains")
    stems = sc.check_cic_files(doms["cic"].df)
    print(f"CIC files (no DDoS): train/val {len(stems['train'])}: {stems['train']}")
    print(f"                     test      {len(stems['test'])}: {stems['test']}")
    problems = []
    for src, tgt, tag in DIRECTIONS:
        print(f"[{tag}] source scaler is fit on {len(doms[src].tree_ds.X('train'))} {src} train rows")
    for name, dom in doms.items():
        scaler = sc.fit_source_scaler(doms[name])
        sets = sc.domain_sequences(dom, scaler, k)
        print(f"\n== {name}: targets per split/class (k={k}), share with padded context")
        for sp in seq.SPLITS:
            s = sets[sp]
            names = np.array(sc.C_CLASSES)[s.y_class] if len(s.y_class) else np.array([])
            print(f"  {sp:5s} n={len(s.y_class):7d} " + " ".join(f"{c}={int((names == c).sum())}"
                  for c in sc.C_CLASSES) + f" | padded {sc.padded_share(s):.1%}")
            if sp in ("train", "val") and int(s.y_bin.sum()) == 0:
                problems.append(f"{name} {sp} has no attack sequences")
            missing = [c for c in sc.C_CLASSES[1:] if int((names == c).sum()) == 0]
            if missing:
                print(f"    classes with no {sp} targets: {missing}")
        print(f"  rows (trees' count next to the LSTM target count; difference by cause):")
        for r in sc.row_table(dom, sets):
            print(f"    {r['split']:5s} raw={r['raw_rows']} victim={r['victim']} boundary={r['boundary']} "
                  f"non-shared attack={r['nonshared_attack']} | trees={r['tree_rows']} "
                  f"- dropped by context rule={r['dropped_by_context_rule']} = LSTM={r['lstm_targets']}")
    for p in problems:
        print("PROBLEM:", p)
    return 3 if problems else 0


# -------------------------------------------------------------------------------------- summary
def _ms(v):
    v = pd.to_numeric(pd.Series(v), errors="coerce").dropna()
    return "n/a" if v.empty else f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else float('nan'):.3f}"


def load_c_records(runs_dir):
    rows = []
    for p in sorted(Path(runs_dir).glob("C_*.json")):
        r = json.loads(p.read_text())
        row = {"run": p.stem, "direction": r["params"]["direction"], "seed": r["params"]["seed"],
               "source": r["params"]["source"], "threshold": r["params"]["threshold"],
               "best_epoch": r["params"]["best_epoch"]}
        row.update({k: v for k, v in r["metrics"].items() if isinstance(v, (int, float))})
        rows.append(row)
    recs = {p.stem: json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("C_*.json"))}
    return pd.DataFrame(rows), recs


def selection_scores(df, key="tgt_test_all_f1"):
    """As-trained cross-domain F1 averaged over both directions, per seed (needs both directions)."""
    piv = df.pivot_table(index="seed", columns="direction", values=key)
    both = piv.dropna()
    return both.mean(axis=1) if len(both) else pd.Series(dtype=float)


def write_summary(runs_dir):
    runs_dir = Path(runs_dir)
    df, recs = load_c_records(runs_dir)
    n_feat = sorted({r["params"]["n_features"] for r in recs.values()}) or [48]
    sens = sorted({(r.get("tags") or {}).get("sensitivity") for r in recs.values()} - {None, "none"})
    L = ["# LSTM Experiment C summary", "",
         f"Frozen config k={FROZEN_K} hidden={FROZEN_H}, {'/'.join(map(str, n_feat))} shared features + is_present, "
         "5 output classes. "
         "Threshold from SOURCE validation (as trained); the target is scored with the SOURCE scaler.", ""]
    if sens:
        L += [f"SENSITIVITY records ({', '.join(sens)}): excluded from the selection rule.", ""]
    if (runs_dir / "FLAG.md").exists():
        L += ["**FLAG.md present (cross-domain F1 >= 0.98):**", "", (runs_dir / "FLAG.md").read_text(), ""]
    if df.empty:
        (runs_dir / "summary.md").write_text("\n".join(L + ["No records yet."]))
        return df
    csv = []
    for _, _, tag in DIRECTIONS:
        d = df[df.direction == tag]
        if d.empty:
            continue
        L += [f"## {tag} ({len(d)} seeds)", ""]
        hdr = ["view", "domain", "F1", "precision", "recall", "FPR", "PR-AUC"]
        rows = []
        for view, vname in (("all", "all rows"), ("noatk", "attacker excluded")):
            for dom, dname in (("tgt_test", "cross-domain (target test)"), ("src_test", "in-domain (source test)")):
                cells = [_ms(d[f"{dom}_{view}_{m}"]) for m in ("f1", "precision", "recall", "fpr", "pr_auc")]
                rows.append([vname, dname] + cells)
                csv.append({"direction": tag, "view": view, "domain": dom,
                            **{m: pd.to_numeric(d[f"{dom}_{view}_{m}"]).mean() for m in
                               ("f1", "precision", "recall", "fpr", "pr_auc")}})
        L += [rl._md(hdr, rows), "", "Recall per shared class (target test / source test):", ""]
        rows = []
        for c in sc.SHARED:
            rows.append([c] + [_ms(d[f"{dom}_{view}_recall_{c}"]) for view in ("all", "noatk")
                               for dom in ("tgt_test", "src_test")])
        L += [rl._md(["class", "target all", "source all", "target noatk", "source noatk"], rows), "",
              "Recall on attacker / broker attack rows (target test): "
              f"{_ms(d['tgt_test_recall_attacker'])} / {_ms(d['tgt_test_recall_broker'])}", ""]
    L += ["## Experiment-C selection score (as-trained F1, mean of both directions per seed)", ""]
    sel_rows = []
    for view, key in (("all rows", "tgt_test_all_f1"), ("attacker excluded", "tgt_test_noatk_f1")):
        s = selection_scores(df, key)
        sel_rows.append([view, len(s), _ms(s)])
    L += [rl._md(["view", "seeds with both directions", "F1 mean ± std"], sel_rows), ""]
    L += ["## Rows: trees' count vs LSTM targets (difference by cause)", "",
          "XGBoost/RF rows = benign + shared-class rows (victim/boundary and non-shared attack rows are not "
          "in them). LSTM targets = those rows minus the targets whose k-window context holds a window of a "
          "class with no output unit.", ""]
    by_src = {}
    for r in recs.values():
        by_src.setdefault(r["params"]["source"], r["tags"]["row_table_source"])
    tbl = []
    for dom in ("testbed", "cic"):
        for r in by_src.get(dom, []):
            tbl.append([dom, r["split"], r["raw_rows"], r["victim"], r["boundary"], r["nonshared_attack"],
                        r["tree_rows"], r["dropped_by_context_rule"], r["lstm_targets"]])
    L += [rl._md(["domain", "split", "raw rows", "victim", "boundary", "non-shared attack", "trees' rows",
                  "dropped by context rule", "LSTM targets"], tbl), ""]
    first = next(iter(recs.values()))
    L += ["## Counts", "", f"Sequence counts (run {first['tags']['run_name']}): "
          f"{json.dumps(first['tags']['sequence_counts'])}", "",
          f"Padded-context share (run {first['tags']['run_name']}): "
          f"{json.dumps({k: round(v, 3) for k, v in first['tags']['padded_context_share'].items()})}", "",
          f"CIC files: {json.dumps(first['tags']['cic_files'])}", ""]
    (runs_dir / "summary.md").write_text("\n".join(L))
    pd.DataFrame(csv).to_csv(runs_dir / "summary.csv", index=False)
    return df


# ------------------------------------------------------------------------------------------ main
def run_smoke(paths, threads, runs_dir=None, models_dir=None, selected=SELECTED):
    log("SMOKE: seed 0, one epoch, both directions; a provenance refusal is fatal")
    runs_dir = Path(runs_dir) if runs_dir else RUNS / "smoke"
    ctx = make_ctx(paths, threads, runs_dir, Path(models_dir) if models_dir else MODELS / "smoke",
                   smoke=True, epochs=1, selected=selected)
    for src, tgt, tag in DIRECTIONS:
        p = ctx.runs_dir / f"{run_name(tag, 0)}.json"
        if p.exists():
            p.unlink()
        run_one(ctx, src, tgt, tag, 0)
        stored = json.loads(p.read_text())
        missing = [k for k in tracking.PROVENANCE_REQUIRED if stored["provenance"].get(k) is None]
        log(f"smoke record {p.stem}: for_report={stored['for_report']} provenance complete={not missing}")
        if missing or not stored["for_report"]:
            raise SystemExit(f"smoke record {p.stem} incomplete: {missing}")
    write_summary(ctx.runs_dir)
    log("SMOKE passed")
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cic-dir", default=str(sc.CIC_DIR))
    ap.add_argument("--size-variant", choices=["none", "nosize"], default="none",
                    help="nosize drops train_tree.SIZE_FEATURES; writes runs/lstm_c_nosize/ (sensitivity, not selection)")
    args = ap.parse_args(argv)
    runs = RUNS if args.size_variant == "none" else RUNS.parent / f"lstm_c_{args.size_variant}"
    models = MODELS if args.size_variant == "none" else MODELS.parent / f"lstm_c_{args.size_variant}"
    os.chdir(common.ROOT)
    paths = sc.default_paths(args.cic_dir)
    if args.check:
        return check(paths)
    prov = preflight(paths)
    log(f"provenance ok: git_commit={prov['git_commit']} git_dirty={prov['git_dirty']} "
        f"image_digest={str(prov['image_digest'])[:19]}")
    if args.smoke:
        run_smoke(paths, args.threads)
        return 0
    if (runs / "STOP.md").exists():
        log("STOP.md exists from an earlier run: resolve it (delete the file) before relaunching")
        return 3
    ctx = make_ctx(paths, args.threads, runs, models, size_variant=args.size_variant)
    log(f"size_variant={args.size_variant}: {len(ctx.doms['testbed'].feats)} shared features -> {runs}")
    try:
        status = run_all(ctx)
    except StopRun as e:
        (runs / "STOP.md").write_text(f"# STOP\n\n{e}\n")
        log(f"STOP: {e}")
        write_summary(runs)
        return 3
    write_summary(runs)
    return 0 if "failed" not in status.values() else 3


if __name__ == "__main__":
    sys.exit(main())
