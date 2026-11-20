"""Reproduce recorded results from the v2 dataset: retrain with the EXISTING entry points and compare with reproduce/expected.json.

    python -m src.reproduce A        # Experiment A, seed 0: XGBoost, Random Forest, Isolation Forest, autoencoder, rule baseline
    python -m src.reproduce B        # Experiment B, seed 0, two held-out classes (scan = easy, spoof_vitals = hard):
                                     #   XGBoost, Isolation Forest, rule baseline
    python -m src.reproduce lstm     # LSTM, seed 0, the configuration in expected.json (k=12, hidden=32)
    python -m src.reproduce --write-expected     # AUTHORS ONLY, on full data: (re)generate reproduce/expected.json from the run records

No new training code: this calls run_tree_report.run_a/run_b, train_autoencoder.run_a/run_b, run_rule_baseline.run_a/run_b and
run_lstm.run_one, with every output redirected to --out (default data/processed/v2/repro/). It never writes runs/, never marks a
record for_report, and refuses an --out that is, or is inside, the recorded runs/ directory. MLflow is pointed inside --out.

WHAT IT COMPARES WITH: the committed file reproduce/expected.json (--expected), NOT the run records and NOT the results bundle: reproducing
needs only the code and the v2 dataset. Before any training it checks that the local data/processed/v2/testbed_windows.parquet has the
SHA-256 recorded in expected.json (dataset_sha256) and stops with FAIL "wrong dataset" otherwise. Each entry of expected.json holds the
headline metrics of one recorded run (test F1, PR-AUC and FPR at 90% recall, for all rows and attacker host excluded), the run record's
path, commit and SHA-256 (the records carry no MLflow run id, so the record file's SHA-256 identifies it exactly), and the tolerance.
Every metric of the entry is compared with the entry's tolerance and the run prints PASS or FAIL; exit code 1 on any FAIL. A failing
comparison is reported, never loosened.

Tolerances (absolute difference), MEASURED (19 Nov 2026, one machine, wardwatch-analysis:0.2.0, 5 CPUs, two runs of each target plus the
recorded run): every compared test_* and val_* metric (424 for A, 492 for B, 46-50 for the LSTM) was IDENTICAL across the two
reproductions and equal to the recorded value (max |diff| = 0). The bounds are therefore pure safety margins for other hardware or library
builds, not fitted to make a result pass: 0.005 for the tree models, Isolation Forest, autoencoder and rule baseline (about the
seed-to-seed std of those models over 5 seeds, 0.002-0.014) and 0.02 for the LSTM (CPU torch repeatability is not guaranteed across CPUs;
its std over 5 seeds is 0.011).
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
EXPECTED_DEFAULT = common.ROOT / "reproduce" / "expected.json"
DATASET_FILE = V2 / "testbed_windows.parquet"
SPLIT_FILE = common.SESSIONS_DIR / "split_v2.json"
# the dashboard's HEADLINE_COLS (src/dashboard/metrics_view.py) for both views
HEADLINE = ("f1", "pr_auc", "fpr_at_recall_90")
METRICS = tuple(f"test_{v}_{m}" for v in ("all", "noatk") for m in HEADLINE)
GENERATOR = "python -m src.reproduce --write-expected"
NOTES = ("run_id is the SHA-256 of the run record file: the records carry no MLflow run id, so the file hash identifies the record exactly. "
         "metrics are the dashboard's headline columns (test F1, PR-AUC, FPR at 90% recall) for all rows (test_all_*) and attacker host excluded "
         "(test_noatk_*). Regenerate with: " + GENERATOR + " (authors only, on full data).")


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


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------- expected.json
def entry_key(experiment, model, cls=None):
    return f"{experiment}/{model}" + (f"/{cls}" if cls else "")


def lstm_key(k, hidden):
    return f"LSTM/A_k{k}_h{hidden}_s0"


def load_expected(path=EXPECTED_DEFAULT):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[reproduce] {p} not found: it ships with the repository (git checkout); regenerate with `{GENERATOR}` (authors, full data).")
    d = json.loads(p.read_text())
    for k in ("dataset_sha256", "split_sha256", "runs"):
        if k not in d:
            raise SystemExit(f"[reproduce] {p} lacks '{k}'")
    return d


def check_dataset(expected, dataset_file=DATASET_FILE):
    """FAIL 'wrong dataset' (SystemExit, before any training) unless the local testbed_windows.parquet is the one the results were made on."""
    p = Path(dataset_file)
    if not p.exists():
        raise SystemExit(f"[reproduce] FAIL: {p} not found. Download the v2 dataset release first (USER_MANUAL Level 2).")
    got = sha256_file(p)
    if got != expected["dataset_sha256"]:
        raise SystemExit(f"[reproduce] FAIL: wrong dataset. {p.name} has sha256 {got}, expected {expected['dataset_sha256']} "
                         f"(the file the recorded results were produced on). Re-download the v2 dataset release.")
    return got


def runs_for(expected, experiment, models=None, classes=None):
    """The expected entries of a target, in a fixed order; a missing entry is an error (the targets compare exactly these)."""
    out = []
    for key, e in expected["runs"].items():
        if e["experiment"] != experiment:
            continue
        if models is not None and e["model"] not in models:
            continue
        if classes is not None and e.get("class") not in classes:
            continue
        out.append((key, e))
    return out


def compare(name, got, entry, tol=None):
    """One row per metric of the expected entry: (recorded, reproduced, |diff|, tolerance, PASS/FAIL). A missing value is a FAIL."""
    tol = entry["tolerance"] if tol is None else tol
    want = entry["metrics"]
    rows = []
    for m in sorted(want):
        a, b = want.get(m), (got or {}).get(m)
        if a is None or b is None:
            rows.append({"run": name, "metric": m, "recorded": a, "reproduced": b, "diff": None, "tolerance": tol, "status": "FAIL"})
            continue
        d = abs(float(a) - float(b))
        rows.append({"run": name, "metric": m, "recorded": float(a), "reproduced": float(b), "diff": d, "tolerance": tol,
                     "status": "PASS" if d <= tol else "FAIL"})
    return rows


def read_metrics(path):
    """Metrics of a record this run just wrote (under --out)."""
    p = Path(path)
    return json.loads(p.read_text())["metrics"] if p.exists() else None


def report(rows, out, title, seconds):
    ok = all(r["status"] == "PASS" for r in rows)
    print(f"\n== reproduce {title}: expected (reproduce/expected.json) vs reproduced (tolerance = absolute difference) ==")
    print(f"{'run':34s} {'metric':26s} {'expected':>9s} {'reproduced':>11s} {'|diff|':>8s} {'tol':>6s}  status")
    for r in rows:
        f = lambda v: "n/a" if v is None else f"{v:.4f}"
        print(f"{r['run']:34s} {r['metric']:26s} {f(r['recorded']):>9s} {f(r['reproduced']):>11s} {f(r['diff']):>8s} "
              f"{r['tolerance']:>6.3f}  {r['status']}")
    print(f"RESULT {title}: {'PASS' if ok else 'FAIL'} ({len(rows)} comparisons, {seconds:.0f} s)")
    (Path(out) / f"reproduce_{title}.json").write_text(json.dumps({"title": title, "pass": ok, "rows": rows, "seconds": seconds}, indent=2))
    return ok


# ----------------------------------------------------------------------------------------------- targets
def run_A(out, expected, models=A_MODELS):
    from .experiments import run_rule_baseline as rr
    from .experiments import run_tree_report as rtr
    from .models import train_autoencoder as ta
    rows = []
    for key, e in runs_for(expected, "A", models):
        m = e["model"]
        print(f"[reproduce A] {m}", flush=True)
        if m in ("xgboost", "random_forest"):
            rtr.run_a(m, _make_env(out, tree=True))
        elif m in ("isolation_forest", "autoencoder"):
            ta.run_a(m, _make_env(out))
        else:
            rr.run_a(_make_env(out))
        rows += compare(f"A {m} seed 0", read_metrics(out / "runs" / "expA" / f"{m}_s0.json"), e)
    if not rows:
        raise SystemExit("[reproduce] expected.json has no Experiment A entries")
    return rows


def run_B(out, expected, models=B_MODELS, classes=B_CLASSES):
    from .experiments import run_rule_baseline as rr
    from .experiments import run_tree_report as rtr
    from .models import train_autoencoder as ta
    rows, done = [], set()
    entries = runs_for(expected, "B", models, classes)
    if not entries:
        raise SystemExit("[reproduce] expected.json has no Experiment B entries")
    for m in models:
        mine = [(k, e) for k, e in entries if e["model"] == m]
        if not mine:
            continue
        cls = sorted({e["class"] for _, e in mine})
        print(f"[reproduce B] {m} withholding {cls}", flush=True)
        if m == "xgboost":
            rtr.run_b(m, _make_env(out, tree=True), only=cls)
        elif m == "isolation_forest":
            env = _make_env(out)
            ta.run_a(m, env)                                   # B reads the architecture A selected (grid, fast)
            ta.run_b(m, ta._load(env, scale=True), None, env=env, only=cls)
        else:
            rr.run_b(_make_env(out), only=cls)
        for _, e in mine:
            rows += compare(f"B {m} withhold {e['class']}", read_metrics(out / "runs" / "expB" / f"{m}_{e['class']}_s0.json"), e)
    return rows


def check_ground_truth(split_file=SPLIT_FILE, sessions_dir=None):
    """The LSTM's sequence builder reads each session's ground_truth.jsonl (data/raw/sessions/<S>/): the dataset release does not contain them
    (they name lab accounts, so they are not published). Say so before any work instead of failing deep inside the builder."""
    sessions_dir = Path(sessions_dir) if sessions_dir else Path(split_file).parent
    split = json.loads(Path(split_file).read_text())
    missing = sorted(f"{s}" for names in split.values() for s in names if not (sessions_dir / s / "ground_truth.jsonl").exists())
    if missing:
        raise SystemExit(f"[reproduce] cannot run the LSTM target: data/raw/sessions/<session>/ground_truth.jsonl is missing for {missing}. "
                         f"The v2 dataset release does not contain the raw ground truth (it names lab accounts); the LSTM needs it to label "
                         f"boundary windows (USER_MANUAL Level 3). Experiments A and B (reproduce A, reproduce B) do not need it.")


def run_lstm(out, threads, expected):
    from .experiments import run_lstm as rl
    check_ground_truth()
    entries = runs_for(expected, "LSTM")
    if len(entries) != 1:
        raise SystemExit("[reproduce] expected.json must hold exactly one LSTM entry")
    key, e = entries[0]
    k, h = e["k"], e["hidden"]                                   # from expected.json, not from any run folder
    name = f"A_k{k}_h{h}_s0"
    ctx = rl.load_context(threads, out / "runs" / "lstm", out / "models" / "lstm")
    print(f"[reproduce lstm] {name}, {threads} torch threads (the recorded run used {e.get('threads', 'n/a')})", flush=True)
    rl.run_safe(ctx, rl.Spec(name, "A", "A", k, h, 0, for_report=False))
    return compare(f"LSTM {name}", read_metrics(out / "runs" / "lstm" / f"{name}.json"), e)


# ----------------------------------------------------------------------------------- --write-expected
def _need(record, what):
    if record is None:
        raise SystemExit(f"[reproduce] --write-expected: {what} is missing")


def _entry_from_record(path, repo_root, experiment, model, tol, cls=None, extra=None):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[reproduce] --write-expected: run record {p} is missing (authors run this on full data)")
    rec = json.loads(p.read_text())
    if not rec.get("for_report"):
        raise SystemExit(f"[reproduce] --write-expected: {p} is not a report-grade record (for_report is false)")
    metrics = {}
    for m in METRICS:
        v = rec["metrics"].get(m)
        if v is None or v != v:
            raise SystemExit(f"[reproduce] --write-expected: {p} lacks a usable {m}")
        metrics[m] = float(v)
    prov = rec["provenance"]
    e = {"experiment": experiment, "model": model, "record_path": Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix(),
         "run_id": sha256_file(p), "git_commit": prov["git_commit"], "tolerance": tol, "metrics": metrics}
    if cls:
        e["class"] = cls
    e.update(extra or {})
    return e, prov


def write_expected(records=RECORDED, out=EXPECTED_DEFAULT, repo_root=None, dataset_file=DATASET_FILE):
    """Build reproduce/expected.json from the run records (authors only, full data). Refuses if a record is missing or not for_report,
    and if the records disagree on the dataset or split hash, or the local dataset file is not the one they name."""
    repo_root = Path(repo_root or common.ROOT)
    records = Path(records)
    runs, provs = {}, []
    for m in A_MODELS:
        e, pv = _entry_from_record(records / "expA" / f"{m}_s0.json", repo_root, "A", m, TOLERANCES[MODEL_TOL[m]])
        runs[entry_key("A", m)] = e
        provs.append(pv)
    for m in B_MODELS:
        for c in B_CLASSES:
            e, pv = _entry_from_record(records / "expB" / f"{m}_{c}_s0.json", repo_root, "B", m, TOLERANCES[MODEL_TOL[m]], cls=c)
            runs[entry_key("B", m, c)] = e
            provs.append(pv)
    sel = json.loads((records / "lstm" / "selected_config.json").read_text())
    k, h = sel["k"], sel["hidden"]
    lp = records / "lstm" / f"A_k{k}_h{h}_s0.json"
    e, pv = _entry_from_record(lp, repo_root, "LSTM", "lstm", TOLERANCES["lstm"], extra={"k": k, "hidden": h, "run_name": f"A_k{k}_h{h}_s0"})
    e["threads"] = json.loads(lp.read_text())["params"].get("threads")
    runs[lstm_key(k, h)] = e
    provs.append(pv)
    ds, sp = {p["dataset_sha256"] for p in provs}, {p["split_sha256"] for p in provs}
    if len(ds) != 1 or len(sp) != 1:
        raise SystemExit(f"[reproduce] --write-expected: the records disagree on the dataset or split hash ({len(ds)} / {len(sp)} distinct)")
    ds, sp = ds.pop(), sp.pop()
    if Path(dataset_file).exists() and sha256_file(dataset_file) != ds:
        raise SystemExit(f"[reproduce] --write-expected: {dataset_file} does not have the dataset_sha256 the records name")
    doc = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generator": GENERATOR, "notes": NOTES,
           "dataset_sha256": ds, "split_sha256": sp, "runs": runs}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", choices=["A", "B", "lstm"])
    ap.add_argument("--write-expected", action="store_true", help="AUTHORS ONLY (full data): regenerate reproduce/expected.json from the run records")
    ap.add_argument("--out", default=None, help="output folder (never runs/); default %s; with --write-expected the expected.json path" % OUT_DEFAULT)
    ap.add_argument("--expected", default=str(EXPECTED_DEFAULT), help="the expected results (default %(default)s)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("ANALYSIS_CPUS", "5")),
                    help="torch threads for the LSTM; the recorded run used 5")
    a = ap.parse_args(argv)
    if a.write_expected:
        doc = write_expected(out=a.out or EXPECTED_DEFAULT)
        print(f"[reproduce] wrote {a.out or EXPECTED_DEFAULT}: {len(doc['runs'])} runs, dataset_sha256 {doc['dataset_sha256'][:12]}...")
        return 0
    if not a.what:
        ap.error("give A, B or lstm (or --write-expected)")
    out = check_out(a.out or OUT_DEFAULT)
    expected = load_expected(a.expected)
    check_dataset(expected)                                   # before any training: FAIL "wrong dataset"
    _env_setup(out)
    t0 = time.perf_counter()
    rows = {"A": lambda: run_A(out, expected), "B": lambda: run_B(out, expected),
            "lstm": lambda: run_lstm(out, a.threads, expected)}[a.what]()
    ok = report(rows, out, a.what, time.perf_counter() - t0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
