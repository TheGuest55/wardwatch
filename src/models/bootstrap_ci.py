"""Block-bootstrap confidence intervals for headline metrics and model comparisons (./CLAUDE.md).

    python -m src.models.bootstrap_ci                       # default: XGBoost vs RF (Exp. A) and
                                                            #   time vs random split, if exported
    python -m src.models.bootstrap_ci --pred A_xgboost --pred A_autoencoder [--for-report]
    python -m src.models.bootstrap_ci --export-a xgboost    # write A_xgboost predictions first

Why blocks: consecutive 5-second windows are near-identical, so resampling single windows would
treat ~50 copies of one attack as 50 independent pieces of evidence and give falsely narrow
intervals. Seeds only vary the model, not the data, so they cannot give these intervals either.

Blocks (test split):
  * each attack RUN (run_id) is one block; runs are resampled with replacement WITHIN their attack
    class (stratified), so every replicate keeps the designed attack mix;
  * benign windows are cut into fixed time chunks per session (default 300 s, all hosts together),
    resampled with replacement.
Percentile intervals (default 95%, 2,000 replicates, fixed seed).

Comparisons:
  * PAIRED (same test rows, e.g. two models on Experiment A): the same resampled blocks are applied
    to both models; reports the CI of (A - B) and the share of replicates with A <= B.
  * UNPAIRED (different test rows, e.g. time vs random split): independent bootstraps; CI of the
    difference of the draws.
Per-class intervals are NOT reported: the test split holds only 2 runs per attack class.
"""
import argparse
import itertools
import json
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np

from . import tracking
from .predictions import PRED_DIR, export_experiment_a, load_predictions
from .train_tree import DATASET, SPLIT, V2

OUT_DIR = V2 / "runs" / "bootstrap"
METRICS = ("f1", "precision", "recall", "fpr", "pr_auc")


def make_blocks(df, benign_block_s=300):
    """(block_id, stratum) per row. Attack rows: block = run, stratum = attack class. Benign rows:
    block = (session, time chunk), stratum = 'benign'."""
    is_att = (df["label"] == "attack").to_numpy()
    t = df["window_start_epoch"].to_numpy(dtype=float)
    sess = df["session"].astype(str).to_numpy()
    t0 = df.groupby("session")["window_start_epoch"].transform("min").to_numpy(dtype=float)
    chunk = ((t - t0) // benign_block_s).astype(int)
    run = df["run_id"].to_numpy()
    cls = df["attack_class"].astype(str).to_numpy()
    if is_att.any() and df.loc[is_att, "run_id"].isna().any():
        raise ValueError("attack rows without run_id: cannot form run blocks")
    blocks = np.array([f"run:{run[i]}" if is_att[i] else f"benign:{sess[i]}:{chunk[i]}"
                       for i in range(len(df))])
    strata = np.array([f"attack:{cls[i]}" if is_att[i] else "benign" for i in range(len(df))])
    return blocks, strata


class BlockSampler:
    """Draws row-index replicates by resampling blocks with replacement within each stratum."""

    def __init__(self, blocks, strata, seed=0):
        self.rng = np.random.default_rng(seed)
        blocks, strata = np.asarray(blocks), np.asarray(strata)
        self.groups = {}
        for s in sorted(set(strata.tolist())):
            ids = blocks[strata == s]
            uniq = sorted(set(ids.tolist()))
            self.groups[s] = [np.flatnonzero((blocks == b) & (strata == s)) for b in uniq]

    def draw(self):
        parts = []
        for rows in self.groups.values():
            pick = self.rng.integers(0, len(rows), size=len(rows))
            parts.extend(rows[i] for i in pick)
        return np.concatenate(parts)

    def n_blocks(self):
        return {s: len(r) for s, r in self.groups.items()}


def metrics_at(y, pred, score=None):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    out = {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r,
           "fpr": fp / (fp + tn) if fp + tn else 0.0}
    if score is not None and 0 < y.sum() < len(y):
        from sklearn.metrics import average_precision_score
        out["pr_auc"] = float(average_precision_score(y, score))
    return out


def _ci(draws, alpha):
    a = np.asarray(draws, dtype=float)
    return float(np.nanpercentile(a, 100 * alpha / 2)), float(np.nanpercentile(a, 100 * (1 - alpha / 2)))


def bootstrap(df, n_boot=2000, seed=0, alpha=0.05, benign_block_s=300):
    """CI per metric for one prediction table. Returns ({metric: {point, lo, hi}}, draws, n_blocks)."""
    blocks, strata = make_blocks(df, benign_block_s)
    s = BlockSampler(blocks, strata, seed)
    y, pred, score = df["y"].to_numpy(), df["pred"].to_numpy(), df["score"].to_numpy()
    point = metrics_at(y, pred, score)
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        i = s.draw()
        m = metrics_at(y[i], pred[i], score[i])
        for k in draws:
            draws[k].append(m.get(k, np.nan))
    out = {k: {"point": point[k], "lo": _ci(draws[k], alpha)[0], "hi": _ci(draws[k], alpha)[1]}
           for k in point}
    return out, {k: np.asarray(v) for k, v in draws.items()}, s.n_blocks()


def same_rows(a, b):
    key = ["session", "host_mac", "window_start_epoch"]
    return len(a) == len(b) and a[key].reset_index(drop=True).equals(b[key].reset_index(drop=True))


def paired_diff(a, b, n_boot=2000, seed=0, alpha=0.05, benign_block_s=300):
    """CI of metric(a) - metric(b) with the SAME resampled blocks for both (rows must match)."""
    if not same_rows(a, b):
        raise ValueError("paired comparison needs identical test rows in the same order")
    if not np.array_equal(a["y"].to_numpy(), b["y"].to_numpy()):
        raise ValueError("labels differ between the two prediction tables")
    blocks, strata = make_blocks(a, benign_block_s)
    s = BlockSampler(blocks, strata, seed)
    y = a["y"].to_numpy()
    pa, sa, pb, sb = (a["pred"].to_numpy(), a["score"].to_numpy(),
                      b["pred"].to_numpy(), b["score"].to_numpy())
    ma, mb = metrics_at(y, pa, sa), metrics_at(y, pb, sb)
    draws = {k: [] for k in ma if k in mb}
    for _ in range(n_boot):
        i = s.draw()
        x, z = metrics_at(y[i], pa[i], sa[i]), metrics_at(y[i], pb[i], sb[i])
        for k in draws:
            draws[k].append(x.get(k, np.nan) - z.get(k, np.nan))
    return {k: {"diff": ma[k] - mb[k], "lo": _ci(v, alpha)[0], "hi": _ci(v, alpha)[1],
                "share_a_le_b": float(np.mean(np.asarray(v) <= 0))} for k, v in draws.items()}


def unpaired_diff(draws_a, draws_b, point_a, point_b, alpha=0.05):
    out = {}
    for k in draws_a:
        if k in draws_b:
            d = draws_a[k] - draws_b[k]
            out[k] = {"diff": point_a[k]["point"] - point_b[k]["point"], "lo": _ci(d, alpha)[0],
                      "hi": _ci(d, alpha)[1], "share_a_le_b": float(np.mean(d <= 0))}
    return out


def _fmt(r):
    return f"{r['point']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]"


def _fmt_d(r):
    sig = "" if r["lo"] <= 0 <= r["hi"] else "  *CI excludes 0*"
    return f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{sig}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[],
                    help="prediction table name (e.g. A_xgboost) or path; repeatable")
    ap.add_argument("--export-a", action="append", default=[], choices=["xgboost", "random_forest"],
                    help="export Experiment A seed-0 test predictions for a tree model first")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--benign-block-s", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="default", help="name of this comparison's output record")
    ap.add_argument("--for-report", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    for m in args.export_a:
        print(f"exported {export_experiment_a(m)}")
    names = args.pred
    if not names:   # default: whatever headline tables exist
        names = [p.stem for p in sorted(PRED_DIR.glob("*.parquet"))] if PRED_DIR.exists() else []
    if not names:
        raise SystemExit("no prediction tables; use --export-a xgboost or save_predictions(...)")
    tables = dict(load_predictions(n) for n in names)
    kw = dict(n_boot=args.n_boot, alpha=args.alpha, benign_block_s=args.benign_block_s)

    single, draws, nblocks = {}, {}, {}
    for n, df in tables.items():
        single[n], draws[n], nblocks[n] = bootstrap(df, seed=args.seed, **kw)
        att = sum(v for k, v in nblocks[n].items() if k != "benign")
        print(f"\n{n}: {len(df)} windows, {int(df['y'].sum())} attack | blocks: {att} attack runs, "
              f"{nblocks[n].get('benign', 0)} benign chunks")
        for k in METRICS:
            if k in single[n]:
                print(f"  {k:9s} {_fmt(single[n][k])}")

    comps = {}
    for a, b in itertools.combinations(tables, 2):
        paired = same_rows(tables[a], tables[b])
        res = (paired_diff(tables[a], tables[b], seed=args.seed, **kw) if paired
               else unpaired_diff(draws[a], draws[b], single[a], single[b], args.alpha))
        comps[f"{a} - {b}"] = {"paired": paired, **res}
        print(f"\n{a} - {b} ({'paired' if paired else 'unpaired'}):")
        for k in METRICS:
            if k in res:
                print(f"  {k:9s} {_fmt_d(res[k])}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    det = OUT_DIR / f"{args.tag}_details.json"
    det.write_text(json.dumps({"intervals": single, "comparisons": comps, "n_blocks": nblocks,
                               "settings": vars(args)}, indent=2, default=str))
    flat = {}
    for n, r in single.items():
        for k, v in r.items():
            flat.update({f"{n}_{k}": v["point"], f"{n}_{k}_lo": v["lo"], f"{n}_{k}_hi": v["hi"]})
    for c, r in comps.items():
        tag = c.replace(" - ", "_minus_")
        for k, v in r.items():
            if isinstance(v, dict):
                flat.update({f"{tag}_{k}_diff": v["diff"], f"{tag}_{k}_lo": v["lo"],
                             f"{tag}_{k}_hi": v["hi"]})
    rec = tracking.build_run_record(
        experiment="bootstrap-ci", model="+".join(tables), role="comparison",
        params={"n_boot": args.n_boot, "alpha": args.alpha, "benign_block_s": args.benign_block_s,
                "blocks": "attack runs stratified by class + benign time chunks", "tables": names},
        metrics=flat, tags={"dataset": "v2", "details": str(det)}, seed=args.seed,
        split_path=str(SPLIT), dataset_path=str(DATASET), for_report=args.for_report)
    tracking.log_run(rec, json_path=str(OUT_DIR / f"{args.tag}.json"), use_mlflow=not args.no_mlflow)
    print(f"\nrecord -> {OUT_DIR / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
