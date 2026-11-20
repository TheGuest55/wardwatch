import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROV = {"split_file": "s.json", "split_sha256": "a", "dataset_sha256": "b", "git_commit": "c0ffee",
        "git_dirty": False, "image_digest": "sha256:img"}


def record(model, experiment, seed, f1, params=None, tags=None, for_report=True, metrics=None):
    m = {"test_all_f1": f1, "test_all_pr_auc": f1 + 0.01, "test_all_fpr_at_recall_90": 0.001,
         "test_noatk_f1": f1 - 0.01, "test_noatk_pr_auc": f1, "test_noatk_fpr_at_recall_90": 0.002,
         "test_all_macro_f1": f1 - 0.02, "best_val_pr_auc": 0.9}
    m.update(metrics or {})
    return {"experiment": experiment, "model": model, "role": "classifier", "params": params or {},
            "metrics": m, "tags": tags or {}, "for_report": for_report,
            "provenance": {**PROV, "seed": seed}, "utc": "2026-11-13T00:00:00Z"}


def write(root, rel, obj):
    p = Path(root) / "runs" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    return p


@pytest.fixture
def root(tmp_path):
    """A small corpus: lstm complete (A, ablation, B, C, latency) plus every kind of record that must be
    kept out of headlines; xgboost has nothing; autoencoder has only a non-report record."""
    sel = {"k": 12, "hidden": 32}
    write(tmp_path, "lstm/selected_config.json", sel)
    for s in range(5):
        p = {"k": 12, "hidden": 32, "use_presence": True, "seed": s, "threshold": 0.5}
        write(tmp_path, f"lstm/A_k12_h32_s{s}.json",
              record("lstm", "A", s, 0.90 + 0.01 * s, p, {"run_name": f"A_k12_h32_s{s}", "stage": "A"}))
        write(tmp_path, f"lstm/A_k1_h32_s{s}.json",
              record("lstm", "A", s, 0.50, {**p, "k": 1}, {"run_name": f"A_k1_h32_s{s}"}))
    write(tmp_path, "lstm/grid_k6_h64_s0.json",                       # for_report False -> excluded
          record("lstm", "A", 0, 0.99, {"k": 6, "hidden": 64}, {"run_name": "grid_k6_h64_s0"}, False))
    write(tmp_path, "lstm/smoke/smoke_A_k6_h32_s0.json",              # smoke but for_report True
          record("lstm", "A", 0, 0.01, {"k": 12, "hidden": 32}, {"run_name": "smoke_A_k6_h32_s0"}))
    lat = record("lstm", "A", 0, 0.0, {"k": 12, "hidden": 32}, {"latency": True},
                 metrics={"median_ms": 0.3, "p95_ms": 0.4})
    write(tmp_path, "lstm/latency_A_k12_h32_s0.json", lat)
    for cls in ("arp_mitm", "pump_cmd"):
        for s in range(2):
            write(tmp_path, f"lstm/B_{cls}_s{s}.json",
                  record("lstm", "B", s, 0.8 if cls == "arp_mitm" else 0.4,
                         {"k": 12, "hidden": 32, "held_out_class": cls}, {"run_name": f"B_{cls}_s{s}"}))
    write(tmp_path, "lstm_c/C_tb2cic_s0.json", record(
        "lstm", "C", 0, 0.9, {"direction": "tb2cic", "threshold_rule": "source-val-max-F1 (as trained)"},
        {"run_name": "C_tb2cic_s0"}, metrics={"tgt_test_all_f1": 0.30, "tgt_test_noatk_f1": 0.2}))
    write(tmp_path, "expA/autoencoder_s0.json",                       # AE exists but is not report-grade
          record("autoencoder", "A", 0, 0.7, {}, {}, False))
    write(tmp_path, "lstm/not_a_record.json", {"hello": 1})
    (tmp_path / "runs" / "lstm" / "broken.json").write_text("{nope")
    return tmp_path


@pytest.fixture
def preds_root(root):
    """Predictions for lstm A_k12_h32_s0: 6 test rows incl. an attacker-host row, 2 val rows."""
    d = root / "runs" / "lstm" / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "split": ["test"] * 6 + ["val"] * 2,
        "session": "r05", "host_mac": "m", "window_start_epoch": range(8), "run_id": None,
        "host_role": ["monitor", "monitor", "pump", "pump", "attacker", "attacker", "monitor", "pump"],
        "attack_class": [None, None, "arp_mitm", "arp_mitm", "arp_mitm", None, None, None],
        "label": ["benign", "benign", "attack", "attack", "attack", "benign", "benign", "benign"],
        "y_bin": [0, 0, 1, 1, 1, 0, 0, 0],
        "score": [0.1, 0.6, 0.9, 0.4, 0.95, 0.2, 0.1, 0.1],
        "p_benign": 0.5, "p_arp_mitm": 0.5})
    df.to_parquet(d / "A_k12_h32_s0.parquet")
    return root


@pytest.fixture(autouse=True)
def _isolate_results_root(monkeypatch, tmp_path_factory):
    """The loaders also search <repo>/results/runs (the unzipped results bundle of USER_MANUAL Level 1). Point that
    extra directory at an empty one so no test sees real run records, whatever has been unzipped next to the code."""
    monkeypatch.setenv("WARDWATCH_RESULTS_ROOT", str(tmp_path_factory.mktemp("no_results")))
