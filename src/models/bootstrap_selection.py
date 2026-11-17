"""Block-bootstrap CIs for the selection-rule scores (Experiment B, Experiment C as trained, and their mean)
and for paired model differences.

    python -m src.models.bootstrap_selection --for-report     # -> runs/bootstrap/selection.{json,md}

Statistics (binary F1, all rows, exactly as the per-seed records define them):
  B(model)   = mean over seeds of the mean over the 8 withheld classes of test F1 (benign test + class X);
  C(model)   = mean over seeds of the mean over both directions of the as-trained TARGET-test F1;
  sel(model) = (B + C) / 2, the fixed selection rule (./CLAUDE.md).

Resampling (bootstrap_ci.make_blocks + BlockSampler, unchanged): every attack RUN is one block, resampled
within its attack class; benign windows are cut into 300 s chunks per session. The block universe is the union
of all rows used anywhere (testbed test + CIC test), so ONE replicate assigns one weight to each block and every
model, class, seed and direction is evaluated under that same replicate. A row's weight = how many times its
block was drawn; F1 is computed from weighted tp/fp/fn. Different row sets (the LSTM's context rule drops ~4% of
testbed rows in C) are therefore still paired: same blocks, each model on its own rows. Seeds vary the model only,
so they are averaged inside each replicate, not resampled.

Point check (hard stop): with every weight = 1 each table's F1 must equal the recorded per-seed F1 within 1e-9.
Sensitivity records (tags.sensitivity, e.g. nosize) are never selection inputs: they are read only from their own
directories, reported as a separate block of C intervals, and the selection numbers assert none are used.

Caveat: CIC test holds ONE file each for arp_mitm and syn_flood, so those strata cannot vary and the C intervals
understate the data uncertainty on the CIC side.
"""
import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
import pandas as pd

from . import tracking
from .bootstrap_ci import BlockSampler, _ci, make_blocks
from .train_tree import DATASET, SPLIT, V2

B_CLASSES = ("arp_mitm", "bruteforce", "eavesdrop", "mqtt_flood", "pump_cmd", "recon_scan", "spoof_vitals",
             "syn_flood")
SEEDS = (0, 1, 2, 3, 4)
DIRECTIONS = ("tb2cic", "cic2tb")
MODELS = ("xgboost", "random_forest", "lstm", "autoencoder", "isolation_forest")
DIFFS = (("xgboost", "random_forest"), ("xgboost", "lstm"), ("isolation_forest", "autoencoder"))
KEYS = ["dataset", "session", "host_mac", "window_start_epoch"]
POINT_TOL = 1e-9


# ------------------------------------------------------------------------------------ table discovery
def b_paths(runs, model, X, seed):
    if model == "lstm":
        return runs / "lstm" / f"B_{X}_s{seed}.json", runs / "lstm" / "predictions" / f"B_{X}_s{seed}.parquet"
    return runs / "expB" / f"{model}_{X}_s{seed}.json", runs / "expB" / "predictions" / f"{model}_{X}_s{seed}.parquet"


def c_paths(runs, model, tag, seed, variant="none"):
    if model == "lstm":
        d = runs / ("lstm_c" if variant == "none" else f"lstm_c_{variant}")
        return d / f"C_{tag}_s{seed}.json", d / "predictions" / f"C_{tag}_s{seed}.parquet"
    d = runs / ("expC" if variant == "none" else f"expC_{variant}")
    return d / f"{model}_{tag}_astrained_s{seed}.json", d / "predictions" / f"{model}_{tag}_astrained_s{seed}.parquet"


def _load(rec_path, pred_path, exp, variant):
    rec = json.loads(Path(rec_path).read_text())
    sens = (rec.get("tags") or {}).get("sensitivity")
    if variant == "none" and sens not in (None, "none"):
        raise SystemExit(f"{rec_path} is a sensitivity record ({sens}); it must not feed the selection rule")
    if variant != "none" and sens != variant:
        raise SystemExit(f"{rec_path} is not tagged sensitivity={variant}")
    p = pd.read_parquet(pred_path)
    if exp == "B":
        p = p[p["split"] == "test"].copy()
        p["dataset"] = "testbed"
        f1_key = "test_all_f1"
    else:
        p = p[p["domain"] == "tgt_test"].copy()
        f1_key = "tgt_test_all_f1"
    p = p[p["label"].isin(["attack", "benign"])]
    p["session"] = p["session"].astype(str).str.replace(r"@(train|val|test)$", "", regex=True)
    p["window_start_epoch"] = p["window_start_epoch"].astype(float)
    p["host_mac"] = p["host_mac"].astype(str)
    pred = p["pred"].to_numpy().astype(int) if "pred" in p else \
        (p["score"].to_numpy() >= float(rec["params"]["threshold"])).astype(int)
    return {"df": p.reset_index(drop=True), "y": (p["label"] == "attack").to_numpy().astype(int), "pred": pred,
            "recorded_f1": float(rec["metrics"][f1_key]), "record": str(rec_path),
            "git_commit": rec["provenance"]["git_commit"], "for_report": bool(rec.get("for_report"))}


def collect(runs, models=MODELS, seeds=None, classes=None, variants=("none",)):
    """{(exp, variant, model, unit, seed): table}; unit = class (B) or direction (C). Missing nosize tables are
    skipped (a variant is reported only where its records exist); missing main tables are an error."""
    runs, T = Path(runs), {}
    seeds, classes = seeds or SEEDS, classes or B_CLASSES
    for m in models:
        for X in classes:
            for s in seeds:
                T[("B", "none", m, X, s)] = _load(*b_paths(runs, m, X, s), "B", "none")
        for v in variants:
            for tag in DIRECTIONS:
                for s in seeds:
                    rp, pp = c_paths(runs, m, tag, s, v)
                    if v != "none" and not rp.exists():
                        continue
                    T[("C", v, m, tag, s)] = _load(rp, pp, "C", v)
    return T


# ------------------------------------------------------------------------------------ blocks
def universe(tables):
    """Union of all rows (by KEYS) with labels; blocks from bootstrap_ci.make_blocks, prefixed by dataset."""
    cols = KEYS + ["label", "run_id", "attack_class"]
    U = pd.concat([t["df"][cols] for t in tables.values()], ignore_index=True)
    U = U.drop_duplicates(KEYS).reset_index(drop=True)
    lab = U.groupby(KEYS)["label"].nunique()
    if (lab > 1).any():
        raise SystemExit("a row has different labels in different prediction tables")
    blocks, strata = [], []
    for ds, g in U.groupby("dataset", sort=True):
        b, s = make_blocks(g.assign(session=g["session"].astype(str)))
        blocks.append(pd.Series([f"{ds}|{x}" for x in b], index=g.index))
        strata.append(pd.Series([f"{ds}|{x}" for x in s], index=g.index))
    U["block"] = pd.concat(blocks).sort_index().to_numpy()
    U["stratum"] = pd.concat(strata).sort_index().to_numpy()
    codes, uniq = pd.factorize(U["block"], sort=True)
    U["block_id"] = codes
    return U, list(uniq)


def attach(tables, U):
    """Per table: per-block tp/fp/fn counts (weights act on blocks)."""
    idx = U.set_index(KEYS)["block_id"]
    nb = int(U["block_id"].max()) + 1
    for t in tables.values():
        bid = idx.reindex(pd.MultiIndex.from_frame(t["df"][KEYS])).to_numpy()
        if np.isnan(bid.astype(float)).any():
            raise AssertionError("a table row is missing from the block universe")
        bid = bid.astype(int)
        y, p = t["y"], t["pred"]
        t["tp"] = np.bincount(bid[(y == 1) & (p == 1)], minlength=nb).astype(float)
        t["fp"] = np.bincount(bid[(y == 0) & (p == 1)], minlength=nb).astype(float)
        t["fn"] = np.bincount(bid[(y == 1) & (p == 0)], minlength=nb).astype(float)


def f1_w(t, w):
    tp, fp, fn = t["tp"] @ w, t["fp"] @ w, t["fn"] @ w
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


def representatives(U):
    """(row index, block id) of one representative universe row per block."""
    first = U.groupby("block_id").head(1)
    return first.index.to_numpy(), first["block_id"].to_numpy(), len(U), int(U["block_id"].max()) + 1


def block_weights(sampler, reps):
    """One replicate -> weight per block (times drawn), via BlockSampler.draw() on universe rows: every row of
    a drawn block is returned once per draw, so the representative row's count is the block's weight."""
    rows_idx, bids, n_rows, nb = reps
    counts = np.bincount(sampler.draw(), minlength=n_rows)
    w = np.zeros(nb)
    w[bids] = counts[rows_idx]
    return w


# ------------------------------------------------------------------------------------ statistics
def scores(tables, w, models, variants=("none",)):
    out = {}
    for m in models:
        b = [np.mean([f1_w(tables[("B", "none", m, X, s)], w) for X in B_CLASSES]) for s in SEEDS]
        out[("B", m)] = float(np.mean(b))
        for v in variants:
            keys = [(s, tag) for s in SEEDS for tag in DIRECTIONS if ("C", v, m, tag, s) in tables]
            if not keys:
                continue
            c = [np.mean([f1_w(tables[("C", v, m, tag, s)], w) for tag in DIRECTIONS]) for s in SEEDS]
            name = "C" if v == "none" else f"C_{v}"
            out[(name, m)] = float(np.mean(c))
            if v == "none":
                out[("sel", m)] = (out[("B", m)] + out[("C", m)]) / 2
            else:
                out[(f"sel_{v}", m)] = (out[("B", m)] + out[(name, m)]) / 2
    return out


def check_points(tables, nb):
    w = np.ones(nb)
    bad = []
    for k, t in tables.items():
        got = f1_w(t, w)
        if abs(got - t["recorded_f1"]) > POINT_TOL:
            bad.append((k, got, t["recorded_f1"]))
    if bad:
        raise SystemExit(f"point estimates differ from the records (first 5): {bad[:5]}")


def run(runs, n_boot=2000, seed=0, alpha=0.05, models=MODELS, variants=("none", "nosize")):
    tables = collect(runs, models, variants=variants)
    present = tuple(v for v in variants if v == "none" or any(k[1] == v for k in tables))
    U, blocks = universe(tables)
    attach(tables, U)
    check_points(tables, len(blocks))
    point = scores(tables, np.ones(len(blocks)), models, present)
    sampler = BlockSampler(U["block"].to_numpy(), U["stratum"].to_numpy(), seed)
    reps = representatives(U)
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        s = scores(tables, block_weights(sampler, reps), models, present)
        for k in draws:
            draws[k].append(s[k])
    draws = {k: np.asarray(v) for k, v in draws.items()}
    intervals = {f"{st}/{m}": {"point": point[(st, m)], "lo": _ci(draws[(st, m)], alpha)[0],
                               "hi": _ci(draws[(st, m)], alpha)[1]} for st, m in point}
    diffs = {}
    for a, b in DIFFS:
        if a not in models or b not in models:
            continue
        for st in sorted({k[0] for k in point}):
            if (st, a) in point and (st, b) in point:
                d = draws[(st, a)] - draws[(st, b)]
                diffs[f"{st}/{a}-{b}"] = {"diff": point[(st, a)] - point[(st, b)], "lo": _ci(d, alpha)[0],
                                          "hi": _ci(d, alpha)[1], "share_a_le_b": float(np.mean(d <= 0))}
    nblocks = U.groupby("stratum")["block"].nunique().to_dict()
    inputs = {"/".join(map(str, k)): {"record": t["record"], "git_commit": t["git_commit"],
                                      "for_report": t["for_report"]} for k, t in tables.items()}
    return {"intervals": intervals, "differences": diffs, "n_blocks": nblocks, "inputs": inputs,
            "variants": list(present), "settings": {"n_boot": n_boot, "seed": seed, "alpha": alpha,
                                                    "benign_block_s": 300}}


def to_markdown(res):
    L = ["# Selection-rule scores: block-bootstrap 95% CIs", "",
         f"{res['settings']['n_boot']} replicates; blocks = attack runs (stratified by class) + 300 s benign chunks per "
         "session; one replicate is shared by every model, so differences are paired. Binary F1, all rows, as trained.",
         "", "| statistic | model | point | 95% CI |", "|---|---|---|---|"]
    for k, v in res["intervals"].items():
        st, m = k.split("/")
        L.append(f"| {st} | {m} | {v['point']:.4f} | [{v['lo']:.4f}, {v['hi']:.4f}] |")
    L += ["", "| difference | point | 95% CI | share of replicates a <= b |", "|---|---|---|---|"]
    for k, v in res["differences"].items():
        star = "" if v["lo"] <= 0 <= v["hi"] else " *"
        L.append(f"| {k} | {v['diff']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}]{star} | {v['share_a_le_b']:.3f} |")
    L += ["", "`*` = the CI excludes 0. Sensitivity rows (C_nosize, sel_nosize) are NOT the selection rule.",
          "", "Blocks per stratum: " + ", ".join(f"{k} {v}" for k, v in sorted(res["n_blocks"].items())),
          "", "Caveat: CIC test has one file each for arp_mitm and syn_flood; those strata cannot vary, so the C "
              "intervals understate CIC-side uncertainty."]
    commits = sorted({v["git_commit"] for v in res["inputs"].values()})
    L += ["", f"Input records come from commits: {', '.join(c[:7] for c in commits)}."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--for-report", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)
    runs = V2 / "runs"
    res = run(runs, args.n_boot, args.seed, args.alpha)
    out = runs / "bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    det = out / "selection_details.json"
    det.write_text(json.dumps(res, indent=2, default=str))
    (out / "selection.md").write_text(to_markdown(res))
    flat = {}
    for k, v in res["intervals"].items():
        n = re.sub(r"[^A-Za-z0-9_]", "_", k)
        flat.update({n: v["point"], f"{n}_lo": v["lo"], f"{n}_hi": v["hi"]})
    for k, v in res["differences"].items():
        n = re.sub(r"[^A-Za-z0-9_]", "_", k)
        flat.update({f"{n}_diff": v["diff"], f"{n}_lo": v["lo"], f"{n}_hi": v["hi"]})
    rec = tracking.build_run_record(
        experiment="bootstrap-ci", model="+".join(MODELS), role="comparison",
        params={"n_boot": args.n_boot, "alpha": args.alpha, "benign_block_s": 300, "statistics": "B, C, (B+C)/2",
                "blocks": "attack runs stratified by class + benign 300 s chunks; testbed and CIC test, one universe"},
        metrics=flat, tags={"dataset": "v2+cic", "details": str(det), "variants": res["variants"]}, seed=args.seed,
        split_path=str(SPLIT), dataset_path=str(DATASET), for_report=args.for_report)
    tracking.log_run(rec, json_path=str(out / "selection.json"), use_mlflow=not args.no_mlflow)
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
