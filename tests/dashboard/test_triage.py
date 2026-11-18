"""Headline whitelist, the frame-size sensitivity table, the Triage / E page and the incident session filter."""
import json

import pytest
from conftest import PROV, record, write
from helpers import run

from src.dashboard import loaders, metrics_view as mv, triage_view as tv


def _df(root):
    r = loaders.load_records(root)
    assert r.ok
    return r.data


def _reasons(df):
    return {p.split("/runs/")[1]: r for p, r in df.set_index("path")["exclude_reason"].items()}


def _extra_records(root):
    """Records that used to leak into headlines, plus nosize C records, written next to the base corpus."""
    for exp, name, rel in (("triage", "xgboost+isolation_forest", "triage/triage_A.json"),
                           ("triage-calibration", "xgboost", "calibration/xgboost.json"),
                           ("bootstrap-ci", "xgboost", "bootstrap/ci.json")):
        write(root, rel, record(name, exp, 0, 0.9, {}, {}))
    write(root, "expE/E_r05.json",                       # E: tags.latency is a DICT of stats, for_report True
          record("xgboost+isolation_forest", "E", 0, 0.9, {}, {"latency": {"tick_ms": {"median": 1}}},
                 metrics={"tick_p95_ms": 22.0}))
    write(root, "expD/D_xgb_s0.json", record("xgboost", "D", 0, 0.95, {}, {}))
    for direction, f1 in (("tb2cic", 0.9), ("cic2tb", 0.8)):          # with size features: C headline for the lstm
        for s in range(2):
            write(root, f"lstm_c/C_{direction}_s{s}.json", record(
                "lstm", "C", s, 0.9, {"direction": direction, "threshold_rule": "source-val-max-F1 (as trained)"},
                {"run_name": f"C_{direction}_s{s}"}, metrics={"tgt_test_all_f1": f1 - 0.5, "tgt_test_noatk_f1": f1 - 0.6}))
            write(root, f"lstm_c_nosize/C_{direction}_s{s}.json", record(
                "lstm", "C", s, 0.9, {"direction": direction, "threshold_rule": "source-val-max-F1 (as trained)"},
                {"run_name": f"C_{direction}_s{s}", "sensitivity": "nosize"},
                metrics={"tgt_test_all_f1": 0.95, "tgt_test_noatk_f1": 0.95}))


class TestWhitelist:
    def test_non_model_experiments_are_excluded_with_the_reason(self, root):
        _extra_records(root)
        by = _reasons(_df(root))
        for rel in ("triage/triage_A.json", "calibration/xgboost.json", "bootstrap/ci.json", "expE/E_r05.json"):
            assert by[rel] == loaders.R_NOT_MODEL == "not a model experiment (shown on Triage / E)", rel

    def test_e_is_not_a_latency_row_even_though_its_tags_latency_is_a_dict(self, root):
        _extra_records(root)
        df = _df(root)
        e = df[df["path"].str.endswith("expE/E_r05.json")].iloc[0]
        assert e["kind"] == "metrics" and not e["eligible"]
        lat = loaders.load_latency(df)
        assert set(lat.data["model"]) == {"lstm"}          # only the real latency record

    def test_sensitivity_records_are_excluded_with_their_reason(self, root):
        _extra_records(root)
        by = _reasons(_df(root))
        assert by["lstm_c_nosize/C_tb2cic_s0.json"] == "sensitivity analysis (not a selection input)"
        assert by["lstm_c/C_tb2cic_s0.json"] is None

    def test_existing_rules_still_apply_and_d_is_allowed(self, root):
        _extra_records(root)
        by = _reasons(_df(root))
        assert by["lstm/grid_k6_h64_s0.json"] == "not for_report"
        assert by["lstm/smoke/smoke_A_k6_h32_s0.json"] == "smoke run"
        assert by["lstm/latency_A_k12_h32_s0.json"] == "latency measurement"
        assert by["expD/D_xgb_s0.json"] is None

    def test_c_headline_is_unaffected_by_nosize_records(self, root):
        base = mv.experiment_c(_df(root), "lstm")
        _extra_records(root)
        c = mv.experiment_c(_df(root), "lstm")
        assert c["both_directions"] and c["score"] == pytest.approx((0.4 + 0.3) / 2)
        assert base["both_directions"] is False or base["score"] != c["score"]      # base corpus had one direction only
        assert c["score"] != pytest.approx(0.95)                                    # nosize values never reach the headline

    def test_headline_and_selection_ignore_non_model_records(self, root):
        _extra_records(root)
        h = mv.headline(_df(root), "all", root).set_index("model")
        assert h.loc["lstm", "C as-trained F1"] == "0.350"
        assert h.loc["xgboost", "F1"] == "not available"       # the triage/calibration/bootstrap/D(xgb) records did not make it an A row


class TestSensitivityTable:
    def test_with_and_without_size_features_side_by_side(self, root):
        _extra_records(root)
        t = mv.c_sensitivity(_df(root), "all").set_index("model")
        assert t.loc["lstm", "C F1 with size features (headline)"] == "0.350"
        assert t.loc["lstm", "C F1 without size features"] == "0.950"
        assert t.loc["lstm", "difference"] == "+0.600"

    def test_empty_when_no_sensitivity_records(self, root):
        assert mv.c_sensitivity(_df(root), "all").empty

    def test_model_comparison_page_labels_the_table(self, root, monkeypatch):
        _extra_records(root)
        at = run("src/dashboard/pages/2_Model_comparison.py", root, monkeypatch)
        assert not at.exception, [e.value for e in at.exception]
        text = " ".join(c.value for c in at.caption)
        assert "NOT a selection input" in text
        tables = [d.value.to_dict("records") for d in at.dataframe]
        assert any(r.get("C F1 without size features") == "0.950" for t in tables for r in t)


# ---------------------------------------------------------------------------------------- Triage / E page
def _triage_a():
    cell = {"known": 5, "novel": 0, "review_anomaly_only": 1, "review_classifier_only": 0, "low_priority": 0, "no_alert": 2}
    det = {"pair_alerting": {"all": {"precision": .85, "recall": .86, "f1": .858, "tp": 6, "fp": 1, "fn": 1},
                             "noatk": {"precision": .9, "recall": .9, "f1": .9, "tp": 6, "fp": 1, "fn": 1}}}
    sess = {"detection": det, "cell_table": {"syn_flood": cell, "benign": cell},
            "grouping": {"accuracy": 0.25, "n_detected": 8, "correct": ["a", "b"], "purity": 1.0, "n_incidents": 90,
                         "n_false_incidents": 43, "incidents_per_run": 3.0, "undetected": []},
            "ttd": {"per_run": {"r05-syn_flood-02": 5.0, "r05-pump_cmd-08": None}, "median_s": 5.0},
            "false_novel": {"reboot": {"n_events": 30, "n_windows": 64, "novel": 0.0, "review": 0.047, "alert": 0.047}}}
    grid = [{"key": k, "gap_s": g, "accuracy": 0.25, "correct": 4, "detected": 16}
            for k in ("source", "source+target") for g in (10, 30)]
    return {"experiment": "triage", "model": "xgboost+isolation_forest", "role": "pair", "params": {}, "metrics": {},
            "tags": {"per_session": {"r05": sess, "r07": sess}, "grouping_grid": grid}, "for_report": True,
            "provenance": {**PROV, "seed": 0}, "utc": "2026-11-17T00:00:00Z"}


def _triage_b():
    agg = lambda m: {"mean": m, "std": 0.01}
    w = {"syn_flood": {"n_windows": 34, "n_runs": 2, "agg": {k: agg(1.0 if k == "known" else 0.0) for k in
                       ("novel_or_review", "novel", "review", "known", "low_priority", "no_alert", "runs_reaching_novel_or_review")}},
         "arp_mitm": {"n_windows": 195, "n_runs": 2, "agg": {k: agg(0.4 if k == "novel_or_review" else 0.0) for k in
                      ("novel_or_review", "novel", "review", "known", "low_priority", "no_alert", "runs_reaching_novel_or_review")}}}
    return {"experiment": "triage", "model": "xgboost+isolation_forest", "role": "pair", "params": {}, "metrics": {},
            "tags": {"withheld": w}, "for_report": True, "provenance": {**PROV, "seed": 0}, "utc": "u"}


def _diagnosis():
    run = {"class": "recon_scan", "n_alerted_windows": 96, "n_incidents": 17, "main_alerts": 75,
           "extra": [{"cause": "attribution"}] * 16, "mixes_other_runs": []}
    return {"experiment": "triage", "model": "xgboost+isolation_forest", "role": "pair", "params": {}, "metrics": {},
            "tags": {"per_session": {"r05": {"runs": {"r05-recon_scan-01": run}, "false_alarm": {"n": 43}}},
                     "totals": {"by_cause": {"attribution": 83, "gap": 0, "other": 0}, "by_transition": {"attacker->nurse": 17}}},
            "for_report": True, "provenance": {**PROV, "seed": 0}, "utc": "u"}


def _e(over=0):
    m = {"tick_median_ms": 13.6, "tick_p95_ms": 22.2, "tick_max_ms": 48.2, "row_median_ms": 1.1, "row_p95_ms": 1.9,
         "row_max_ms": 3.7, "ticks_over_budget": over, "budget_ms": 50.0, "n_ticks": 1020, "lag_s_max": 0.25, "lag_s_p95": 0.0076,
         "ingest_us_mean": 89.0, "shap_p95_s": 0.01, "slice_p95_s": 0.23, "wall_s": 5096.0, "n_incidents": 90,
         "pair_precision": .858, "pair_recall": .86, "pair_f1": .859, "xgboost_f1": .91, "isolation_forest_f1": .77}
    eq = {"late_packets": 0, "rows_vs_offline": {"n_rows": 12171, "max_rel_score_diff": 0.0, "band_flips": 0, "verdict_flips": 0},
          "vs_saved_A_s0": {"xgboost_rows": 12018, "xgboost_max_abs_diff": 0.0},
          "incidents_vs_offline": {"n_incidents": 90, "equal": True}}
    return {"experiment": "E", "model": "xgboost+isolation_forest", "role": "pair", "params": {}, "metrics": m,
            "tags": {"equivalence": eq, "ttd_offline": {"per_run": {"r05-syn_flood-02": 5.0}},
                     "ttd_live": {"r05-syn_flood-02": 6.0}}, "for_report": True, "provenance": {**PROV, "seed": 0}, "utc": "u"}


PAGE = "src/dashboard/pages/7_Triage_and_E.py"


class TestTriagePage:
    def test_renders_with_all_records(self, root, monkeypatch):
        write(root, "triage/triage_A.json", _triage_a())
        write(root, "triage/triage_B_withheld.json", _triage_b())
        write(root, "triage/grouping_diagnosis.json", _diagnosis())
        write(root, "expE/E_r05.json", _e(0))
        write(root, "expE/E_r07.json", _e(2))
        at = run(PAGE, root, monkeypatch)
        assert not at.exception, [e.value for e in at.exception]
        warn = " ".join(w.value for w in at.warning)
        assert "selection had no signal" in warn                         # the validation grid tied
        assert "syn_flood land in" in warn and "limitation of the verdict table" in warn
        assert "2 tick(s) over the 50.0 ms budget" in warn               # r07 had ticks over budget
        assert any("0 tick(s) over" in s.value for s in at.success)
        assert any("not a live capture" in c.value for c in at.caption)
        tables = [d.value.to_dict("records") for d in at.dataframe]
        assert any(r.get("detector") == "pair (alerting verdicts)" for t in tables for r in t)
        assert any(r.get("class") == "recon_scan" and r.get("extra by cause") == "attribution 16" for t in tables for r in t)

    def test_missing_files_show_not_available_per_section(self, root, monkeypatch):
        write(root, "triage/triage_A.json", _triage_a())
        at = run(PAGE, root, monkeypatch)
        assert not at.exception, [e.value for e in at.exception]
        info = " ".join(i.value for i in at.info)
        for what in ("withheld-attack analysis", "grouping diagnosis", "Experiment E r05", "Experiment E r07"):
            assert what + ": not available" in info, what

    def test_renders_with_no_triage_files_at_all(self, root, monkeypatch):
        at = run(PAGE, root, monkeypatch)
        assert not at.exception
        assert sum("not available" in i.value for i in at.info) >= 5

    def test_view_helpers_only_arrange_logged_values(self):
        a = _triage_a()
        assert tv.validation_tied(a) is True
        assert tv.grouping_summary(a, "r05")["accuracy"] == 0.25
        t = tv.ttd_table(a, "r05").set_index("run")
        assert t.loc["r05-pump_cmd-08", "time to detect (s)"] == "undetected"
        assert tv.loud_known(_triage_b()) == ["syn_flood"]
        assert tv.e_equivalence(_e())["incidents equal to offline"] is True
        a["tags"]["grouping_grid"][0]["accuracy"] = 0.5
        assert tv.validation_tied(a) is False


# ---------------------------------------------------------------------------------------- Incidents page
def _incidents(root):
    d = root / "live"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"incident_id": f"{s}-n-{i}", "session": s, "target_host": "10.0.0.10 (broker)", "source": "10.0.0.66 (attacker)",
             "first_seen": f"2026-11-09T0{i}:00:00", "last_seen": "x", "verdict": "Detectors disagree → review",
             "cell": "review_anomaly_only" if i else "review_classifier_only", "n_alerts": 2, "top_features": [],
             "pcap_slice": None, "pcap_sha256": None, "attack_class": None, "timeline": []}
            for s in ("r05", "r07") for i in range(2)]
    (d / "incidents.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class TestIncidentsSessionFilter:
    def _list(self, at):
        for d in at.dataframe:
            v = d.value
            if "incident_id" in v.columns:
                return v
        return None

    def test_all_then_one_session_and_cell_column(self, root, monkeypatch):
        _incidents(root)
        at = run("src/dashboard/pages/6_Incidents.py", root, monkeypatch)
        assert not at.exception, [e.value for e in at.exception]
        assert at.selectbox[0].options == ["All", "r05", "r07"]
        lst = self._list(at)
        assert len(lst) == 4 and "cell" in lst.columns and "session" in lst.columns
        at.selectbox[0].select("r05").run()
        assert not at.exception, [e.value for e in at.exception]
        lst = self._list(at)
        assert len(lst) == 2 and set(lst["session"]) == {"r05"}
        assert set(lst["cell"]) == {"review_anomaly_only", "review_classifier_only"}      # the two review cells stay apart

    def test_old_incident_files_without_session_still_render(self, root, monkeypatch):
        d = root / "live"
        d.mkdir(parents=True)
        (d / "incidents.jsonl").write_text(json.dumps({"incident_id": "a", "verdict": "Known attack (type named)"}) + "\n")
        at = run("src/dashboard/pages/6_Incidents.py", root, monkeypatch)
        assert not at.exception, [e.value for e in at.exception]
        assert at.selectbox[0].options == ["All"]


class TestBootstrapTab:
    def _page(self, root, monkeypatch):
        at = run("src/dashboard/pages/2_Model_comparison.py", root, monkeypatch)
        assert not at.exception, [e.value for e in at.exception]
        return [d.value.to_dict("records") for d in at.dataframe]

    def test_flat_selection_intervals_and_run_record_file_do_not_break_the_page(self, root, monkeypatch):
        d = root / "runs" / "bootstrap"
        d.mkdir(parents=True)
        (d / "selection.json").write_text(json.dumps(record("xgboost", "bootstrap-ci", 0, 0.9)))        # a run record, no intervals
        (d / "selection_details.json").write_text(json.dumps({
            "intervals": {"B/xgboost": {"point": 0.44, "lo": 0.39, "hi": 0.49}},
            "differences": {"sel/xgboost-random_forest": {"point": 0.01, "lo": -0.01, "hi": 0.03}}}))
        tables = self._page(root, monkeypatch)
        assert any(r.get("statistic / model") == "B/xgboost" and r.get("point") == 0.44 for t in tables for r in t)

    def test_nested_intervals_still_render(self, root, monkeypatch):
        d = root / "runs" / "bootstrap"
        d.mkdir(parents=True)
        (d / "ci.json").write_text(json.dumps({"intervals": {"xgb": {"f1": {"point": 0.9, "lo": 0.8, "hi": 0.95}}}}))
        tables = self._page(root, monkeypatch)
        assert any(r.get("metric") == "f1" and r.get("point") == 0.9 for t in tables for r in t)
