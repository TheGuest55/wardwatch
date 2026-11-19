"""Reproduce recorded results from the v2 dataset: retrain with the EXISTING entry points and compare with the recorded run.

    python -m src.reproduce A        # Experiment A, seed 0: XGBoost, Random Forest, Isolation Forest, autoencoder, rule baseline
    python -m src.reproduce B        # Experiment B, seed 0, two held-out classes (scan = easy, spoof_vitals = hard):
                                     #   XGBoost, Isolation Forest, rule baseline
    python -m src.reproduce lstm     # LSTM, seed 0, the selected configuration (k=12, hidden=32)

No new training code: this calls run_tree_report.run_a/run_b, train_autoencoder.run_a/run_b, run_rule_baseline.run_a/run_b and
run_lstm.run_one, with every output redirected to --out (default data/processed/v2/repro/). It never writes runs/, never marks a
record for_report, and refuses an --out that is, or is inside, the recorded runs/ directory. MLflow is pointed inside --out.

The recorded run is read from <data root>/runs/ (the results bundle provides it): expA/<model>_s0.json, expB/<model>_<class>_s0.json,
lstm/A_k12_h32_s0.json. Per model it compares test F1 (all rows) and test F1 (attacker host excluded) and prints PASS or FAIL
with the tolerance. Exit code 1 on any FAIL. A failing comparison is reported, never loosened.

Tolerances (absolute F1 difference) and what was measured (19 Nov 2026, one machine, wardwatch-analysis:0.2.0, 5 CPUs, two runs of each target
plus the recorded run): every compared test_* and val_* metric (424 for A, 492 for B, 46-50 for the LSTM) was IDENTICAL across the two reproductions
and equal to the recorded value (max |diff| = 0). The tree models, Isolation Forest, autoencoder and rule baseline are seeded; the LSTM uses
fixed CPU torch threads. The allowed bounds are therefore pure safety margins for other hardware or library builds, not fitted to make a result
pass: 0.005 (about the seed-to-seed std of those models over 5 seeds, 0.002-0.014, and 0.4% of their F1) and 0.02 for the LSTM (CPU torch
repeatability is not guaranteed across CPUs; its std over 5 seeds is 0.011). A comparison that fails is reported, never loosened.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import common

V2 = common.PROCESSED_DIR / "v2"
RECORDED = V2 / "runs"
OUT_DEFAULT = V2 / "repro"
TOLERANCES = {"tree": 0.005, "if": 0.005, "ae": 0.005, "rule": 0.005, "lstm": 0.02}
MODEL_TOL = {"xgboost": "tree", "random_forest": "tree", "isolation_forest": "if", "autoencoder": "ae",
             "rule_baseline": "rule", "lstm": "lstm"}
B_CLASSES = ("recon_scan", "spoof_vitals")
A_MODELS = ("xgboost", "random_forest", "isolation_forest", "autoencoder", "rule_baseline")
B_MODELS = ("xgboost", "isolation_forest", "rule_baseline")
METRICS = ("test_all_f1", "test_noatk_f1")


def check_out(out, recorded=RECORDED):
    """The reproduction may not write where recorded results live."""
    out, rec = Path(out).resolve(), Path(recorded).resolve()
    if out == rec or rec in out.parents or out in rec.parents or out == V2.resolve():
        raise SystemExit(f"[reproduce] refusing --out {out}: it is, contains or lies inside the recorded results ({rec}). "
                         f"Use a separate folder (default {OUT_DEFAULT}).")
    return out


def _env_setup(out):
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{out / 'mlruns'}")      # nothing goes to the repo's mlruns/
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


def _make_env(out, tree=False):
    from .experiments import run_tree_report as rtr
    from .models import train_autoencoder as ta
    kw = dict(runs=out / "runs", model_dir=out / "models", pred_dir=out / "predictions", for_report=False, no_mlflow=True,
              seeds=(0,))
    return (rtr.Env if tree else ta.Env)(**kw)


def read_metrics(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())["metrics"]


def compare(name, got, want, tol):
    """Rows (metric, recorded, reproduced, |diff|, PASS/FAIL). A missing recorded or reproduced value is a FAIL."""
    rows = []
    for m in METRICS:
        a, b = (want or {}).get(m), (got or {}).get(m)
        if a is None or b is None:
            rows.append({"run": name, "metric": m, "recorded": a, "reproduced": b, "diff": None, "tolerance": tol, "status": "FAIL"})
            continue
        d = abs(float(a) - float(b))
        rows.append({"run": name, "metric": m, "recorded": float(a), "reproduced": float(b), "diff": d, "tolerance": tol,
                     "status": "PASS" if d <= tol else "FAIL"})
    return rows


def report(rows, out, title, seconds):
    ok = all(r["status"] == "PASS" for r in rows)
    print(f"\n== reproduce {title}: recorded vs reproduced (tolerance = absolute F1 difference) ==")
    print(f"{'run':34s} {'metric':14s} {'recorded':>9s} {'reproduced':>11s} {'|diff|':>8s} {'tol':>6s}  status")
    for r in rows:
        f = lambda v: "n/a" if v is None else f"{v:.4f}"
        print(f"{r['run']:34s} {r['metric']:14s} {f(r['recorded']):>9s} {f(r['reproduced']):>11s} {f(r['diff']):>8s} "
              f"{r['tolerance']:>6.3f}  {r['status']}")
    print(f"RESULT {title}: {'PASS' if ok else 'FAIL'} ({len(rows)} comparisons, {seconds:.0f} s)")
    (Path(out) / f"reproduce_{title}.json").write_text(json.dumps({"title": title, "pass": ok, "rows": rows, "seconds": seconds}, indent=2))
    return ok


def run_A(out, models=A_MODELS, recorded=RECORDED):
    from .experiments import run_rule_baseline as rr
    from .experiments import run_tree_report as rtr
    from .models import train_autoencoder as ta
    rows = []
    for m in models:
        print(f"[reproduce A] {m}", flush=True)
        if m in ("xgboost", "random_forest"):
            rtr.run_a(m, _make_env(out, tree=True))
        elif m in ("isolation_forest", "autoencoder"):
            ta.run_a(m, _make_env(out))
        else:
            rr.run_a(_make_env(out))
        rows += compare(f"A {m} seed 0", read_metrics(out / "runs" / "expA" / f"{m}_s0.json"),
                        read_metrics(Path(recorded) / "expA" / f"{m}_s0.json"), TOLERANCES[MODEL_TOL[m]])
    return rows


def run_B(out, models=B_MODELS, classes=B_CLASSES, recorded=RECORDED):
    from .experiments import run_rule_baseline as rr
    from .experiments import run_tree_report as rtr
    from .models import train_autoencoder as ta
    rows = []
    for m in models:
        print(f"[reproduce B] {m} withholding {list(classes)}", flush=True)
        if m == "xgboost":
            rtr.run_b(m, _make_env(out, tree=True), only=classes)
        elif m == "isolation_forest":
            env = _make_env(out)
            ta.run_a(m, env)                                   # B reads the architecture A selected (grid, fast)
            ta.run_b(m, ta._load(env, scale=True), None, env=env, only=classes)
        else:
            rr.run_b(_make_env(out), only=classes)
        for c in classes:
            rows += compare(f"B {m} withhold {c}", read_metrics(out / "runs" / "expB" / f"{m}_{c}_s0.json"),
                            read_metrics(Path(recorded) / "expB" / f"{m}_{c}_s0.json"), TOLERANCES[MODEL_TOL[m]])
    return rows


def run_lstm(out, threads, recorded=RECORDED):
    from .experiments import run_lstm as rl
    sel = json.loads((Path(recorded) / "lstm" / "selected_config.json").read_text())
    k, h = sel["k"], sel["hidden"]
    name = f"A_k{k}_h{h}_s0"
    ctx = rl.load_context(threads, out / "runs" / "lstm", out / "models" / "lstm")
    print(f"[reproduce lstm] {name}, {threads} torch threads (the recorded run used "
          f"{json.loads((Path(recorded) / 'lstm' / (name + '.json')).read_text())['params']['threads']})", flush=True)
    spec = rl.Spec(name, "A", "A", k, h, 0, for_report=False)
    rl.run_safe(ctx, spec)
    return compare(f"LSTM {name}", read_metrics(out / "runs" / "lstm" / f"{name}.json"),
                   read_metrics(Path(recorded) / "lstm" / f"{name}.json"), TOLERANCES["lstm"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["A", "B", "lstm"])
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="output folder (never runs/); default %(default)s")
    ap.add_argument("--recorded", default=str(RECORDED), help="folder with the recorded runs (default %(default)s)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("ANALYSIS_CPUS", "5")),
                    help="torch threads for the LSTM; the recorded run used 5")
    a = ap.parse_args(argv)
    out = check_out(a.out, a.recorded)
    _env_setup(out)
    t0 = time.perf_counter()
    rows = {"A": lambda: run_A(out, recorded=a.recorded), "B": lambda: run_B(out, recorded=a.recorded),
            "lstm": lambda: run_lstm(out, a.threads, a.recorded)}[a.what]()
    ok = report(rows, out, a.what, time.perf_counter() - t0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
