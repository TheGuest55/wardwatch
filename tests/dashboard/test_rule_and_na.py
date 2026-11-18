"""Macro-F1 keys, "not applicable" vs "not available", and the rule baseline in the views."""
import pandas as pd
import pytest
from conftest import record, write
from helpers import run

from src.dashboard import loaders, metrics_view as mv


def _df(root):
    r = loaders.load_records(root)
    assert r.ok
    return r.data


def _tree(model, seed, f1, macro):
    m = {"test_macro_f1": macro, "test_macro_f1_traintree": macro}          # real tree records: no per-view macro-F1
    rec = record(model, "A", seed, f1, {"max_depth": 4}, {}, metrics=m)
    rec["metrics"].pop("test_all_macro_f1")
    return rec


def _lstm_real(seed, macro, macro_noatk):
    rec = record("lstm", "A", seed, 0.9, {"k": 3, "hidden": 32, "use_presence": True}, {"run_name": f"A_k3_h32_s{seed}"},
                 metrics={"test_macro_f1": macro, "test_noatk_macro_f1": macro_noatk})
    rec["metrics"].pop("test_all_macro_f1")                                    # the real LSTM keys: plain + noatk
    return rec


def _rows(root, view="all"):
    t = mv.a_by_config(_df(root), view).set_index("model")
    return t


class TestMacroF1:
    def test_trees_show_the_logged_value_for_all_rows_and_say_so_for_attacker_excluded(self, root):
        for s in range(2):
            write(root, f"expA/xgboost_s{s}.json", _tree("xgboost", s, 0.89, 0.80 + 0.02 * s))
            write(root, f"expA/random_forest_s{s}.json", _tree("random_forest", s, 0.92, 0.70))
        t = _rows(root, "all")
        assert t.loc["xgboost", "macro-F1"].startswith("0.810")
        assert t.loc["random_forest", "macro-F1"].startswith("0.700")
        t = _rows(root, "noatk")
        assert t.loc["xgboost", "macro-F1"] == mv.NA_NOATK_MACRO == "not available (no attacker-excluded macro-F1 logged)"
        assert t.loc["random_forest", "macro-F1"] == mv.NA_NOATK_MACRO

    def test_lstm_real_keys_are_read_for_both_views_and_every_config(self, root):
        for s in range(2):
            write(root, f"lstm/A_k3_s{s}.json", _lstm_real(s, 0.60 + 0.1 * s, 0.50))
        t = mv.a_by_config(_df(root), "all").set_index(["model", "config"])
        assert t.loc[("lstm", "k=3 h=32"), "macro-F1"].startswith("0.650")        # real shape: plain test_macro_f1
        assert t.loc[("lstm", "k=12 h=32"), "macro-F1"].startswith("0.90")         # fixture shape: test_all_macro_f1, unchanged
        n = mv.a_by_config(_df(root), "noatk").set_index(["model", "config"])
        assert n.loc[("lstm", "k=3 h=32"), "macro-F1"].startswith("0.500")
        assert n.loc[("lstm", "k=12 h=32"), "macro-F1"] == mv.NA_NOATK_MACRO or n.loc[("lstm", "k=12 h=32"), "macro-F1"][0].isdigit()
        assert not t["macro-F1"].str.contains("not applicable").any()             # a classifier is never "not applicable"

    def test_macro_series_per_record_fallback_and_noatk_never_reuses_the_plain_key(self):
        d = pd.DataFrame
        assert list(mv.macro_series(d({"test_macro_f1": [0.5]}), "all")) == [0.5]
        assert list(mv.macro_series(d({"test_macro_f1": [0.5], "test_all_macro_f1": [0.6]}), "all")) == [0.6]
        # mixed records: the per-view key where present, the plain key where not (per record, not per frame)
        assert list(mv.macro_series(d({"test_macro_f1": [0.5, 0.7], "test_all_macro_f1": [None, 0.6]}), "all")) == [0.5, 0.6]
        assert mv.macro_series(d({"test_macro_f1": [0.5]}), "noatk") is None
        assert list(mv.macro_series(d({"test_noatk_macro_f1": [0.4]}), "noatk")) == [0.4]
        assert mv.macro_series(d({"x": [1]}), "all") is None


def _anomaly_and_rule(root):
    write(root, "expA/autoencoder_s0.json", record("autoencoder", "A", 0, 0.79, {}, {}))
    write(root, "expA/isolation_forest_s0.json", record("isolation_forest", "A", 0, 0.73, {}, {}))
    r = record("rule_baseline", "A", 0, 0.78, {}, {"deterministic": True})
    r["role"] = "rule"
    write(root, "expA/rule_baseline_s0.json", r)


class TestNotApplicable:
    def test_helper_by_role(self):
        for m in ("autoencoder", "isolation_forest", "rule_baseline"):
            for what in ("macro_f1", "multiclass", "calibration"):
                assert loaders.not_applicable(m, what)
        for m in ("xgboost", "random_forest", "lstm"):
            for what in ("macro_f1", "multiclass", "calibration"):
                assert loaders.not_applicable(m, what) is None

    def test_macro_cell_says_not_applicable_for_anomaly_and_rule(self, root):
        _anomaly_and_rule(root)
        for view in ("all", "noatk"):
            t = _rows(root, view)
            for m in ("autoencoder", "isolation_forest", "rule_baseline"):
                assert t.loc[m, "macro-F1"] == "not applicable", (m, view)

    def test_calibration_page_not_applicable_vs_not_available(self, root, monkeypatch):
        at = run("src/dashboard/pages/4_Calibration.py", root, monkeypatch)
        assert not at.exception, [e.value for e in at.exception]
        info = [i.value for i in at.info]
        for m in ("autoencoder", "isolation_forest", "rule_baseline"):
            pass
        assert sum("calibration: not applicable" in i for i in info) == 3
        assert sum(i.startswith("calibration: not available") for i in info) == 3     # xgboost, random_forest, lstm: just no data

    def _curves(self, root, monkeypatch, model):
        n = 6
        for sub, stem in (("expA", f"{model}_s0"),):
            d = root / "runs" / sub / "predictions"
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"split": "test", "session": "r05", "host_mac": "m", "window_start_epoch": range(n),
                          "host_role": ["monitor"] * n, "run_id": None, "attack_class": [None, None, "syn_flood"] * 2,
                          "label": ["benign", "benign", "attack"] * 2, "y_bin": [0, 0, 1] * 2,
                          "score": [0.1, 0.2, 0.9] * 2, "pred": [0, 0, 1] * 2}).to_parquet(d / f"{stem}.parquet")
        at = run("src/dashboard/pages/3_Curves_and_confusion.py", root, monkeypatch)
        assert not at.exception, [e.value for e in at.exception]
        at.selectbox[1].select(model).run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    @pytest.mark.parametrize("model", ["autoencoder", "isolation_forest", "rule_baseline"])
    def test_multiclass_confusion_not_applicable_for_anomaly_and_rule(self, root, monkeypatch, model):
        _anomaly_and_rule(root)
        at = self._curves(root, monkeypatch, model)
        info = " ".join(i.value for i in at.info)
        assert "multiclass confusion: not applicable" in info
        assert "no named attack type" in info

    def test_multiclass_confusion_for_a_classifier_is_still_the_old_path(self, root, monkeypatch):
        d = root / "runs" / "lstm" / "predictions"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"split": "test", "session": "r05", "host_mac": "m", "window_start_epoch": range(4),
                      "host_role": "monitor", "run_id": None, "attack_class": [None, None, "syn_flood", "syn_flood"],
                      "label": ["benign", "benign", "attack", "attack"], "y_bin": [0, 0, 1, 1], "score": [.1, .2, .9, .8],
                      "p_benign": [.9, .8, .1, .2], "p_syn_flood": [.1, .2, .9, .8]}).to_parquet(d / "A_k12_h32_s0.parquet")
        at = run("src/dashboard/pages/3_Curves_and_confusion.py", root, monkeypatch)
        assert not at.exception
        assert "not applicable" not in " ".join(i.value for i in at.info)


class TestRuleBaselineInViews:
    def _full(self, root):
        _anomaly_and_rule(root)
        for s in range(3):
            write(root, f"expA/isolation_forest_s{s}.json", record("isolation_forest", "A", s, 0.73, {}, {}))
        for cls in ("arp_mitm", "pump_cmd"):
            r = record("rule_baseline", "B", 0, 0.99, {"held_out_class": cls}, {"deterministic": True})
            r["role"] = "rule"
            write(root, f"expB/rule_baseline_{cls}_s0.json", r)
        for d in ("tb2cic", "cic2tb"):
            r = record("rule_baseline", "C", 0, 0.9, {"direction": d, "threshold_rule": "validation-only coordinate ascent"},
                       {"deterministic": True}, metrics={"tgt_test_all_f1": 0.97, "tgt_test_noatk_f1": 0.97})
            r["role"] = "rule"
            write(root, f"expC/rule_baseline_{d}_astrained_s0.json", r)

    def test_rule_baseline_row_has_role_rule_and_real_numbers(self, root):
        self._full(root)
        h = mv.headline(_df(root), "all", root).set_index("model")
        assert h.loc["rule_baseline", "role"] == "rule"
        assert h.loc["rule_baseline", "F1"].startswith("0.780")
        assert h.loc["rule_baseline", "B mean F1"].startswith("0.990")
        assert h.loc["rule_baseline", "C as-trained F1"] == "0.970"

    def test_rule_baseline_is_never_a_selection_winner_even_when_it_scores_highest(self, root):
        self._full(root)
        h = mv.headline(_df(root), "all", root)
        sel = mv.selection(h)
        assert set(sel) == {"anomaly", "classifier"}
        for role, s in sel.items():
            assert "rule_baseline" not in s["scores"] and s["winner"] != "rule_baseline"
            assert "rule_baseline" not in " ".join(s["missing"])
        assert (h.set_index("model").loc["rule_baseline", "_B"] or 0) > 0.9          # it does outscore them: still no winner

    def test_overview_and_comparison_pages_render_with_the_rule_baseline(self, root, monkeypatch):
        self._full(root)
        for page in ("src/dashboard/app.py", "src/dashboard/pages/2_Model_comparison.py"):
            at = run(page, root, monkeypatch)
            assert not at.exception, (page, [e.value for e in at.exception])
        text = " ".join(str(d.value.to_dict()) for d in at.dataframe)
        assert "rule_baseline" in text and "not applicable" in text
