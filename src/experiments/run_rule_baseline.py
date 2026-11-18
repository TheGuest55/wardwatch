"""Report-grade runs of the non-ML rule baseline for Experiments A, B and C.

    python -m src.experiments.run_rule_baseline --experiment A [--for-report]      (then B, C)

The rules and the validation-only threshold search are in src/models/rule_baseline.py. Everything else reuses
train_autoencoder's machinery so the baseline is scored exactly like the ML models: the Env paths, the metric blocks
(_blocks: all rows and attacker-host-excluded, plus per-class recall and recall on attacker / broker rows), the per-
experiment prediction layout, the B protocol (class X withheld from the validation rows, test = benign + X) and the
shared-feature domains of C (train_tree._domain).

  A  fit on validation, score val + test. 1 record (seed 0). Predictions: runs/expA/predictions/rule_baseline_s0.parquet
     and runs/predictions/A_rule_baseline.parquet. The record's tags carry the nurse-station wildcard statistics.
  B  per class X: thresholds fitted on validation WITHOUT X, test on benign + X. 8 records (params.held_out_class).
     A FLAG.md line (working rule 3) names the rule(s) that fired if the attacker-excluded F1 reaches 0.98.
  C  thresholds fitted on the SOURCE validation rows and applied unchanged to the target test (nothing refit).
     2 records (tb2cic, cic2tb), variant "astrained".
Every record: role "rule", model "rule_baseline", tags.deterministic = true, seed 0 (the rules have no randomness),
for_report with provenance. The score is a rule-strength score, not a probability; pred = score >= 1.

Output: runs/exp{A,B,C}/rule_baseline_*.json, rule_baseline_summary.{json,md} per experiment, and (after any run)
runs/rule_baseline_summary.{json,md} combining what exists, including the B and C mean (comparison only: the rule
baseline is never a selection candidate).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..models import metrics as mmetrics
from ..models import predictions, tracking
from ..models import rule_baseline as rb
from ..models import train_autoencoder as ta

MODEL = "rule_baseline"
FLAG_F1 = 0.98
SEED = 0
RULE_DOC = ("rule-strength score = max(x_f / t_f over rate rules, 1 + topic_wildcard_subs for the wildcard rule); "
            "pred = score >= 1; not a probability")


def _record(env, exp, params, metrics, tags):
    return tracking.build_run_record(
        experiment=exp, model=MODEL, role="rule", params=params, metrics=metrics,
        tags={"deterministic": True, "score": RULE_DOC, **tags}, seed=SEED, split_path=str(env.split),
        dataset_path=str(env.dataset), for_report=env.for_report)


def _params(rule, extra):
    return {"thresholds": rule.describe(), "rule_features": list(rb.RULE_FEATURES), "combine": "OR",
            "threshold_rule": "validation-only coordinate ascent maximising the OR's val F1; ties to the higher threshold",
            "pred_rule": "score >= 1", "seed": SEED, **extra}


def _fit(ds_val_X, y, feats, meta):
    return rb.RuleBaseline().fit(ds_val_X, y, feats, meta["split"].to_numpy())


def _flag(env_dir, line):
    with open(env_dir / "FLAG.md", "a") as f:
        f.write(line + "\n")
    print("FLAG:", line, flush=True)


def _top_rules(rule, X, feats, mask):
    cnt = rule.fired(X[mask], feats).sum(axis=0)
    return {rb.RULE_FEATURES[j]: int(c) for j, c in enumerate(cnt) if c}


# ------------------------------------------------------------------------------------ nurse wildcard
def wildcard_stats(ds, feats, rule, test_pred, y_test):
    """Benign nurse-station windows with topic_wildcard_subs >= 1: count and share per split and per session
    (nurse role from host_role, evaluation metadata); the wildcard rule's firing on all benign windows; and its
    share of the test false positives. Nothing here is used to fit anything."""
    j = feats.index(rb.WILDCARD)
    out = {"per_split": {}, "per_session": {}, "benign_all_windows": {}}
    for sp in ("train", "val", "test"):
        m, X, y = ds.meta(sp), ds.X(sp), ds.y(sp)
        w = X[:, j] >= 1
        ben = (m["label"].to_numpy() == "benign")
        nurse = ben & (m["host_role"].to_numpy() == "nurse")
        out["per_split"][sp] = {"nurse_benign_windows": int(nurse.sum()), "with_wildcard": int((nurse & w).sum()),
                                "share": float((nurse & w).sum() / nurse.sum()) if nurse.any() else None}
        for s in sorted(set(m["session"].astype(str))):
            k = nurse & (m["session"].astype(str).to_numpy() == s)
            out["per_session"][s] = {"split": sp, "nurse_benign_windows": int(k.sum()), "with_wildcard": int((k & w).sum()),
                                     "share": float((k & w).sum() / k.sum()) if k.any() else None}
        out["benign_all_windows"][sp] = {"benign_windows": int(ben.sum()), "wildcard_rule_fires": int((ben & w).sum()),
                                         "share": float((ben & w).sum() / ben.sum()) if ben.any() else None}
    mt = ds.meta("test")
    w = ds.X("test")[:, j] >= 1
    fp = (test_pred == 1) & (y_test == 0)
    fired = rule.fired(ds.X("test"), feats)
    only_wild = fired[:, -1] & ~fired[:, :-1].any(axis=1)
    out["test_false_positives"] = {
        "n": int(fp.sum()), "wildcard_rule_fired": int((fp & w).sum()),
        "share_wildcard_fired": float((fp & w).sum() / fp.sum()) if fp.any() else None,
        "only_wildcard_fired": int((fp & only_wild).sum()),
        "from_nurse_station": int((fp & (mt["host_role"].to_numpy() == "nurse")).sum())}
    return out


# ------------------------------------------------------------------------------------ Experiment A
def run_a(env):
    ta.preflight(env)
    ds = ta._load(env, scale=False)
    feats = ds.feature_names
    mva, mte, yva, yte = ds.meta("val"), ds.meta("test"), ds.y("val"), ds.y("test")
    rule = _fit(ds.X("val"), yva, feats, mva)
    s_va, s_te = rule.score(ds.X("val")), rule.score(ds.X("test"))
    pred_te = rule.predict(ds.X("test"))
    metrics = {}
    ta._blocks("val", mva, yva, s_va, 1.0, metrics)
    ta._blocks("test", mte, yte, s_te, 1.0, metrics)
    wc = wildcard_stats(ds, feats, rule, pred_te, yte)
    d = env.exp_dir("A")
    rec = _record(env, "A", _params(rule, {"threshold": 1.0}), metrics,
                  {"dataset": "v2", "stage": "A", "wildcard": wc,
                   "rules_fired_on_test_attacks": _top_rules(rule, ds.X("test"), feats, yte == 1),
                   "rules_fired_on_test_false_positives": _top_rules(rule, ds.X("test"), feats, (pred_te == 1) & (yte == 0))})
    ta._write_preds(d / "predictions" / f"{MODEL}_s{SEED}.parquet",
                    [ta._pred_frame("val", mva, yva, s_va, 1.0), ta._pred_frame("test", mte, yte, s_te, 1.0)])
    predictions.save_predictions(mte, yte, s_te, pred_te, "A", MODEL, out_dir=env.pred_dir)
    ta._log(env, rec, d / f"{MODEL}_s{SEED}.json")
    print(f"[{MODEL}] A thresholds {rule.describe()} (val F1 {rule.val_f1_:.4f}); test F1 {metrics['test_all_f1']:.4f} "
          f"(attacker-excluded {metrics['test_noatk_f1']:.4f}) FPR {metrics['test_all_fpr']:.5f}", flush=True)
    print(f"  nurse benign windows with wildcard: {wc['per_split']}; test FPs {wc['test_false_positives']}", flush=True)
    S = ta.write_summary(env, "A", MODEL)
    write_combined(env)
    return S


# ------------------------------------------------------------------------------------ Experiment B
def run_b(env):
    ta.preflight(env)
    ds = ta._load(env, scale=False)
    feats = ds.feature_names
    mva, mte = ds.meta("val"), ds.meta("test")

    def cls_of(m):
        return np.where(m["label"].to_numpy() == "attack", m["attack_class"].astype(str), "benign")

    c_va, c_te = cls_of(mva), cls_of(mte)
    Xva, yva, Xte, yte = ds.X("val"), ds.y("val"), ds.X("test"), ds.y("test")
    classes = sorted(set(cls_of(ds.meta("train"))) - {"benign"})
    d = env.exp_dir("B")
    for X in classes:
        if int(yte[(c_te == "benign") | (c_te == X)].sum()) == 0:
            raise SystemExit(f"class {X} has no test rows: Experiment B undefined for it")
        if int((c_va == X).sum()) == int(yva.sum()):
            raise SystemExit(f"class {X} is the only attack class in validation: nothing left to fit on")
    for X in classes:
        kva = c_va != X                                   # withhold X from the threshold-fitting validation rows
        kte = (c_te == "benign") | (c_te == X)            # test: benign + X only
        yva_x, mva_x = yva[kva], mva[kva]
        rule = _fit(Xva[kva], yva_x, feats, mva_x)
        s_va, s_te = rule.score(Xva[kva]), rule.score(Xte[kte])
        yte_x, mte_x = yte[kte], mte[kte]
        metrics = {"n_test_attack": int(yte_x.sum())}
        ta._blocks("val", mva_x, yva_x, s_va, 1.0, metrics)
        ta._blocks("test", mte_x, yte_x, s_te, 1.0, metrics)
        rec = _record(env, "B", _params(rule, {"threshold": 1.0, "held_out_class": X,
                                               "withheld_from": "val+test (the rules have no training)"}), metrics,
                      {"dataset": "v2", "stage": "B",
                       "rules_fired_on_withheld_attacks": _top_rules(rule, Xte[kte], feats, yte_x == 1)})
        name = f"{MODEL}_{X}_s{SEED}"
        ta._write_preds(d / "predictions" / f"{name}.parquet",
                        [ta._pred_frame("val", mva_x, yva_x, s_va, 1.0), ta._pred_frame("test", mte_x, yte_x, s_te, 1.0)])
        ta._log(env, rec, d / f"{name}.json")
        print(f"  withhold {X:13s} F1 {metrics['test_all_f1']:.4f} (attacker-excluded {metrics['test_noatk_f1']:.4f}) "
              f"recall {metrics['test_all_recall']:.4f} FPR {metrics['test_all_fpr']:.5f} thresholds "
              f"{ {k: v for k, v in rule.describe().items() if v is not None and k != rb.WILDCARD} }", flush=True)
        if metrics["test_noatk_f1"] >= FLAG_F1:                                  # working rule 3
            _flag(d, f"- {MODEL} class={X} attacker-excluded F1={metrics['test_noatk_f1']:.4f}; rules fired on the "
                     f"withheld attacks: {_top_rules(rule, Xte[kte], feats, yte_x == 1)}")
    S = ta.write_summary(env, "B", MODEL)
    write_combined(env)
    return S


# ------------------------------------------------------------------------------------ Experiment C
def run_c(env):
    ta.preflight(env)
    doms, tags = ta.load_c_domains(env)
    d = env.exp_dir("C")
    for src, tgt in (("testbed", "cic"), ("cic", "testbed")):
        ds_s, ds_t = doms[src], doms[tgt]
        feats = ds_s.feature_names
        mv, ms, mt = ds_s.meta("val"), ds_s.meta("test"), ds_t.meta("test")
        yva = ds_s.y("val")
        rule = _fit(ds_s.X("val"), yva, feats, mv)               # SOURCE validation only; nothing is refit on the target
        s_val, s_src, s_tgt = rule.score(ds_s.X("val")), rule.score(ds_s.X("test")), rule.score(ds_t.X("test"))
        metrics = {}
        ta._blocks("val", mv, yva, s_val, 1.0, metrics)
        ta._blocks("src_test", ms, ds_s.y("test"), s_src, 1.0, metrics)
        ta._blocks("tgt_test", mt, ds_t.y("test"), s_tgt, 1.0, metrics)
        tag = {"testbed": "tb2cic", "cic": "cic2tb"}[src]
        rec = _record(env, "C", _params(rule, {"threshold": 1.0, "direction": tag, "variant": "astrained", "source": src,
                                               "target": tgt, "n_features": len(feats),
                                               "scaler": "none (rules act on raw feature values)",
                                               "threshold_rule_C": "source-val thresholds applied unchanged to the target"}),
                      metrics, {"dataset": "v2+cic", "stage": "C", **tags,
                                "rules_fired_on_target_attacks": _top_rules(rule, ds_t.X("test"), feats, ds_t.y("test") == 1)})
        name = f"{MODEL}_{tag}_astrained_s{SEED}"
        ta._write_preds(d / "predictions" / f"{name}.parquet",
                        [ta._pred_frame("val", mv, yva, s_val, 1.0, "val", src),
                         ta._pred_frame("test", ms, ds_s.y("test"), s_src, 1.0, "src_test", src),
                         ta._pred_frame("test", mt, ds_t.y("test"), s_tgt, 1.0, "tgt_test", tgt)])
        ta._log(env, rec, d / f"{name}.json")
        print(f"  {tag}: target F1 {metrics['tgt_test_all_f1']:.4f} (attacker-excluded {metrics['tgt_test_noatk_f1']:.4f}) | "
              f"source F1 {metrics['src_test_all_f1']:.4f}; thresholds "
              f"{ {k: v for k, v in rule.describe().items() if v is not None and k != rb.WILDCARD} }", flush=True)
        if metrics["tgt_test_noatk_f1"] >= FLAG_F1:
            _flag(d, f"- {MODEL} {tag} attacker-excluded target F1={metrics['tgt_test_noatk_f1']:.4f}; rules fired on target "
                     f"attacks: {_top_rules(rule, ds_t.X('test'), feats, ds_t.y('test') == 1)}")
    S = ta.write_summary(env, "C", MODEL)
    write_combined(env)
    return S


# ------------------------------------------------------------------------------------ combined summary
def _rec(env, exp, pattern):
    return ta._records(env, exp, pattern)


def write_combined(env):
    """runs/rule_baseline_summary.{json,md}: what exists of A, B, C (both views) and the B and C mean, which is
    reported for comparison only (the rule baseline is not a selection candidate)."""
    A = [r for r in _rec(env, "A", rf"{MODEL}_s\d+\.json") if r["model"] == MODEL]
    B = [r for r in _rec(env, "B", rf"{MODEL}_.+_s\d+\.json") if r["model"] == MODEL]
    C = [r for r in _rec(env, "C", rf"{MODEL}_(tb2cic|cic2tb)_astrained_s\d+\.json") if r["model"] == MODEL]
    S = {"model": MODEL, "role": "rule", "note": "a rule-strength score, not a probability; never a selection winner",
         "n_records": {"A": len(A), "B": len(B), "C": len(C)}}
    L = ["# Rule baseline (rate rules + wildcard rule, OR)", "", RULE_DOC, ""]
    for view in ("all", "noatk"):
        S[view] = {}
        if A:
            S[view]["A"] = {m: A[0]["metrics"].get(f"test_{view}_{m}") for m in ta._METRICS}
        if B:
            S[view]["B_per_class"] = {r["params"]["held_out_class"]: r["metrics"].get(f"test_{view}_f1") for r in B}
            S[view]["B_mean_f1"] = float(np.nanmean(list(S[view]["B_per_class"].values())))
        if C:
            S[view]["C_f1"] = {r["params"]["direction"]: r["metrics"].get(f"tgt_test_{view}_f1") for r in C}
            if len(S[view]["C_f1"]) == 2:
                S[view]["C_mean_f1"] = float(np.mean(list(S[view]["C_f1"].values())))
        if "B_mean_f1" in S[view] and "C_mean_f1" in S[view]:
            S[view]["B_C_mean_f1"] = (S[view]["B_mean_f1"] + S[view]["C_mean_f1"]) / 2
    if A:
        S["thresholds_A"] = A[0]["params"]["thresholds"]
        S["wildcard"] = A[0]["tags"].get("wildcard")
    for view, label in (("all", "all rows"), ("noatk", "attacker excluded")):
        L += [f"## {label}", ""]
        s = S[view]
        if "A" in s:
            L.append("A: " + ", ".join(f"{m} {v:.4f}" for m, v in s["A"].items() if isinstance(v, float)))
        if "B_mean_f1" in s:
            L.append(f"B: mean F1 {s['B_mean_f1']:.4f}; per class " + ", ".join(f"{c} {v:.3f}" for c, v in s["B_per_class"].items()))
        if "C_mean_f1" in s:
            L.append(f"C (as trained): mean F1 {s['C_mean_f1']:.4f}; " + ", ".join(f"{k} {v:.3f}" for k, v in s["C_f1"].items()))
        if "B_C_mean_f1" in s:
            L.append(f"B and C mean (comparison only): {s['B_C_mean_f1']:.4f}")
        L.append("")
    if A:
        wc = S["wildcard"]
        L += ["## Wildcard rule and the nurse station (benign windows)", "",
              f"Thresholds fitted on validation: {S['thresholds_A']}", "",
              "| split | nurse benign windows | with wildcard sub | share |", "|---|---|---|---|"]
        for sp, v in wc["per_split"].items():
            share = "n/a" if v["share"] is None else format(v["share"], ".3f")
            L.append(f"| {sp} | {v['nurse_benign_windows']} | {v['with_wildcard']} | {share} |")
        fp = wc["test_false_positives"]
        L += ["", f"Wildcard rule fires on all benign windows (train/val/test): " +
              ", ".join(f"{sp} {v['wildcard_rule_fires']}/{v['benign_windows']}" for sp, v in wc["benign_all_windows"].items()),
              f"Test false positives: {fp['n']}; wildcard rule fired on {fp['wildcard_rule_fired']} "
              f"(share {fp['share_wildcard_fired']}), alone on {fp['only_wildcard_fired']}; from the nurse station {fp['from_nurse_station']}.", ""]
    Path(env.runs).mkdir(parents=True, exist_ok=True)
    (Path(env.runs) / f"{MODEL}_summary.json").write_text(json.dumps(S, indent=2, default=str))
    (Path(env.runs) / f"{MODEL}_summary.md").write_text("\n".join(L) + "\n")
    return S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["A", "B", "C"], required=True)
    ap.add_argument("--for-report", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true", default=True)
    args = ap.parse_args(argv)
    env = ta.Env(for_report=args.for_report, no_mlflow=True, seeds=(SEED,))
    {"A": run_a, "B": run_b, "C": run_c}[args.experiment](env)


if __name__ == "__main__":
    main()
