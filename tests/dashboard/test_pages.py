"""AppTest smoke: every page renders without an exception, with data and with an empty data root."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = sorted(str(p) for p in [Path("src/dashboard/app.py"), *Path("src/dashboard/pages").glob("*.py")])


def run(page, root, monkeypatch):
    monkeypatch.setenv("WARDWATCH_DATA_ROOT", str(root))
    at = AppTest.from_file(page, default_timeout=60).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_data(page, preds_root, monkeypatch):
    at = run(page, preds_root, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_empty_root(page, tmp_path, monkeypatch):
    at = run(page, tmp_path, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]


def test_overview_headline_values(preds_root, monkeypatch):
    at = run("src/dashboard/app.py", preds_root, monkeypatch)
    heads = [d.value for d in at.dataframe]
    flat = " ".join(str(h.to_dict()) for h in heads)
    assert "0.920" in flat and "xgboost" in flat and "not available" in flat


def test_calibration_page_shows_ece_when_details_exist(preds_root, monkeypatch):
    import json
    d = preds_root / "runs" / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    bins = [{"lo": 0.0, "hi": 0.5, "n": 10, "mean_predicted": 0.2, "observed_attack_rate": 0.1},
            {"lo": 0.5, "hi": 1.0, "n": 0, "mean_predicted": None, "observed_attack_rate": None}]
    (d / "xgboost_details.json").write_text(json.dumps({
        "calibration_method": "isotonic", "reliability_test_raw": {"ece": 0.0123, "bins": bins},
        "reliability_test_calibrated": {"ece": 0.0045, "bins": bins}}))
    at = run("src/dashboard/pages/4_Calibration.py", preds_root, monkeypatch)
    assert not at.exception
    vals = {m.label: m.value for m in at.metric}
    assert vals["ECE raw"] == "0.0123" and vals["ECE calibrated"] == "0.0045" and vals["Method"] == "isotonic"


def test_explanations_page_renders_all_three_sources(preds_root, monkeypatch):
    import json
    (preds_root / "runs" / "lstm" / "explanations.json").write_text(json.dumps({
        "verdict_rule": {"expected_groups": {"arp_mitm": ["ARP"]}},
        "classes": {"arp_mitm": {"status": "ok", "run_id": "r07-arp_mitm-05", "host_role": "monitor",
                                 "score": 0.99, "threshold": 0.77, "views": {"last3": {
                                     "verdict": {"top_group": "ARP", "top_group_expected": True,
                                                 "top_feature": "ip_mac_conflict", "top_feature_expected": True},
                                     "feature_top5": ["IP MAC conflict raised the attack log-odds by 9.45"],
                                     "group_top5": ["ARP group raised the attack log-odds by 11.51"]}}},
                    "pump_cmd": {"status": "no alert"}}}))
    cal = preds_root / "runs" / "calibration"
    cal.mkdir(parents=True, exist_ok=True)
    (cal / "xgboost_details.json").write_text(json.dumps({
        "shap_global": [["tcp_syn", 0.5], ["arp_replies", 0.2]],
        "shap_per_class_top5": {"syn_flood": [["tcp_syn", 0.9]]},
        "example_explanations": {"syn_flood": {"reasons": [{"text": "TCP connection attempts (SYN) = 400"}]}},
        "shap_sanity": {"additivity_max_abs_error": 1e-7}}))
    (preds_root / "runs" / "expA").mkdir(exist_ok=True)
    (preds_root / "runs" / "expA" / "autoencoder_details.json").write_text(json.dumps(
        {"per_feature_error_seed0": [["pkts_total", 0.4], ["tcp_syn", 0.1]]}))
    at = run("src/dashboard/pages/5_Explanations.py", preds_root, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "IP MAC conflict raised the attack log-odds by 9.45" in text
    assert "r07-arp_mitm-05" in text
    tables = [d.value.to_dict("records") for d in at.dataframe]
    assert any(r.get("feature") == "tcp_syn" and r.get("meaning", "").startswith("TCP connection attempts")
               for t in tables for r in t)
