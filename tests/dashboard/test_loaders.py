import numpy as np
import pandas as pd
import pytest

from src.dashboard import loaders, metrics_view as mv


@pytest.fixture
def df(root):
    r = loaders.load_records(root)
    assert r.ok
    return r.data


def test_non_records_are_skipped_and_broken_files_reported(df):
    assert not any("not_a_record" in p for p in df["path"])
    assert any("broken.json" in p for p in df.attrs["problems"])


def test_headline_membership_is_for_report_plus_explicit_exclusions(df):
    reasons = df.set_index("path")["exclude_reason"]
    by = {p.split("/runs/")[1]: r for p, r in reasons.items()}
    assert by["lstm/grid_k6_h64_s0.json"] == "not for_report"
    assert by["lstm/smoke/smoke_A_k6_h32_s0.json"] == "smoke run"
    assert by["lstm/latency_A_k12_h32_s0.json"] == "latency measurement"
    assert by["expA/autoencoder_s0.json"] == "not for_report"
    assert by["lstm/A_k12_h32_s0.json"] is None
    a = loaders.eligible(df, model="lstm", experiment="A")
    assert len(a) == 10 and set(a["config"]) == {"k=12 h=32", "k=1 h=32"}     # no grid, smoke or latency


def test_experiment_a_uses_selected_config_and_seed_stats(df, root):
    a = mv.experiment_a(df, "lstm", "all", root)
    assert a["config"] == "k=12 h=32" and a["n_seeds"] == 5
    mean, std, n = a["f1"]
    assert n == 5 and mean == pytest.approx(0.92) and std == pytest.approx(np.std([.90, .91, .92, .93, .94], ddof=1))
    assert mv.experiment_a(df, "lstm", "noatk", root)["f1"][0] == pytest.approx(0.91)   # attacker-excluded view


def test_experiment_b_mean_over_classes_and_missing_classes(df):
    b = mv.experiment_b(df, "lstm")
    assert b["mean_f1"] == pytest.approx(0.6) and b["n_classes"] == 2
    assert "syn_flood" in b["missing_classes"] and "arp_mitm" not in b["missing_classes"]


def test_experiment_c_needs_both_directions(df):
    c = mv.experiment_c(df, "lstm")
    assert c["both_directions"] is False and np.isnan(c["score"])


def test_headline_table_marks_missing_models_not_available(df, root):
    h = mv.headline(df, "all", root).set_index("model")
    assert h.loc["lstm", "F1"].startswith("0.920 ±")
    assert h.loc["lstm", "p95 latency (ms)"] == "0.40"
    for m in ("xgboost", "random_forest", "isolation_forest", "rule_baseline", "autoencoder"):
        assert h.loc[m, "F1"] == "not available" and h.loc[m, "B mean F1"] == "not available"
    assert h.loc["lstm", "C as-trained F1"].startswith("not available")
    # the non-report AE record never reaches the headline, and the reason is findable
    st = loaders.model_status(df)
    assert "not for_report" in st["autoencoder"] and st["xgboost"] == "no run records on disk" and st["lstm"] is None


def test_no_winner_declared_while_inputs_are_missing(df, root):
    sel = mv.selection(mv.headline(df, "all", root))
    assert sel["classifier"]["winner"] is None and sel["anomaly"]["winner"] is None
    assert any("lstm: Experiment C" in m for m in sel["classifier"]["missing"])
    assert any("autoencoder: Experiment B" in m for m in sel["anomaly"]["missing"])


def test_winner_declared_only_when_every_candidate_has_b_and_c():
    rows = [{"model": m, "_B": b, "_C": c} for m, b, c in
            [("autoencoder", .6, .4), ("isolation_forest", .5, .3),
             ("xgboost", .8, .5), ("random_forest", .7, .5), ("lstm", .9, .6)]]
    sel = mv.selection(pd.DataFrame(rows))
    assert sel["anomaly"]["winner"] == "autoencoder" and sel["classifier"]["winner"] == "lstm"
    rows[-1]["_C"] = np.nan                                           # LSTM cannot do C
    assert mv.selection(pd.DataFrame(rows))["classifier"]["winner"] is None


def test_predictions_normalised_and_attacker_excluded(preds_root):
    df = loaders.load_records(preds_root).data
    rec = df[(df["run_name"] == "A_k12_h32_s0") & df["eligible"]].iloc[0]
    p = loaders.load_predictions(rec, preds_root)
    assert p.ok and len(p.data) == 6 and set(p.data["split"]) == {"test"}
    assert list(p.data["pred"]) == [0, 1, 1, 0, 1, 0]               # score >= recorded threshold 0.5
    na = loaders.drop_attacker(p.data)
    assert na.ok and len(na.data) == 4 and "attacker" not in set(na.data["host_role"])
    assert len(loaders.load_predictions(rec, preds_root, split="val").data) == 2


def test_missing_predictions_and_missing_host_role_give_reasons(root):
    df = loaders.load_records(root).data
    rec = df[(df["run_name"] == "A_k1_h32_s0")].iloc[0]
    assert loaders.load_predictions(rec, root).reason == "no predictions file for this run"
    assert not loaders.drop_attacker(pd.DataFrame({"host_role": [None, None]})).ok


def test_empty_root_never_crashes(tmp_path):
    r = loaders.load_records(tmp_path)
    assert not r.ok and "no run records" in r.reason
    h = mv.headline(None, "all", tmp_path)
    assert (h["F1"] == "not available").all()
    assert not loaders.load_dataset_summary(tmp_path).ok
    assert not loaders.load_lstm_explanations(tmp_path).ok
    assert not loaders.load_calibration_details("xgboost", tmp_path).ok
    assert not loaders.load_bootstrap(tmp_path).ok
    assert not loaders.load_latency(None).ok


def test_incidents_stub_then_parse(tmp_path):
    assert "not available yet" in loaders.load_incidents(tmp_path).reason
    d = tmp_path / "live"
    d.mkdir()
    (d / "incidents.jsonl").write_text('{"incident_id": "i1", "verdict": "Known attack"}\nnot json\n'
                                       '{"incident_id": "i2"}\n')
    r = loaders.load_incidents(tmp_path)
    assert r.ok and list(r.data["incident_id"]) == ["i1", "i2"] and r.data.attrs["bad_lines"] == 1


def test_dataset_summary_rows_per_class_and_split(tmp_path):
    pd.DataFrame({"split": ["train", "train", "test", "test"], "label": ["benign", "attack", "benign", "attack"],
                  "attack_class": [None, "arp_mitm", None, "arp_mitm"]}).to_parquet(tmp_path / "testbed_windows.parquet")
    t = loaders.load_dataset_summary(tmp_path).data
    assert t.loc["arp_mitm", "train"] == 1 and t.loc["benign", "test"] == 1


def test_derived_curves_and_confusions(preds_root):
    from src.dashboard import derived
    df = loaders.load_records(preds_root).data
    rec = df[(df["run_name"] == "A_k12_h32_s0") & df["eligible"]].iloc[0]
    pred = loaders.load_predictions(rec, preds_root).data
    cm = derived.confusion_binary(pred)
    assert cm.values.tolist() == [[2, 1], [1, 2]]
    assert cm.loc["attack", "predicted attack"] == 2 and cm.loc["benign", "predicted attack"] == 1
    assert derived.pr(pred).ok and derived.roc(pred).ok
    mc = derived.confusion_multiclass(pred)
    assert mc.ok and mc.data.values.sum() == 6
    assert not derived.pr(pred.assign(y=0)).ok                      # one class only -> reason, not a crash
    assert not derived.confusion_multiclass(pred.drop(columns=["p_benign", "p_arp_mitm"])).ok
