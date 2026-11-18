"""Rule baseline: validation-only threshold fit, determinism, score/verdict agreement, B withholding, C source
thresholds, record contract and the nurse wildcard statistics, on synthetic frames."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import run_rule_baseline as rr
from src.models import rule_baseline as rb
from src.models import train_autoencoder as ta
from tests.pipeline.synth_c import make_synthetic

ENV = {"WARDWATCH_GIT_COMMIT": "0" * 40, "WARDWATCH_GIT_DIRTY": "false",
       "WARDWATCH_IMAGE_DIGEST": "sha256:" + "a" * 64}
FEATS = list(rb.RULE_FEATURES) + ["pkts_total"]


def arr(**cols):
    n = len(next(iter(cols.values())))
    X = np.zeros((n, len(FEATS)))
    for k, v in cols.items():
        X[:, FEATS.index(k)] = v
    return X


class TestFit(unittest.TestCase):
    def test_fit_refuses_test_and_mixed_rows(self):
        X, y = arr(mqtt_publish_rate=[1.0, 9.0]), np.array([0, 1])
        for bad in ("test", "train", ["val", "test"], np.array(["val", "test"], dtype=object)):
            with self.assertRaises(ValueError, msg=str(bad)):
                rb.RuleBaseline().fit(X, y, FEATS, bad)
        rb.RuleBaseline().fit(X, y, FEATS, "val")
        rb.RuleBaseline().fit(X, y, FEATS, np.array(["val", "val"], dtype=object))

    def test_fit_needs_attack_rows_and_all_rule_features(self):
        with self.assertRaises(ValueError):
            rb.RuleBaseline().fit(arr(tcp_syn_rate=[1.0, 2.0]), np.array([0, 0]), FEATS, "val")
        with self.assertRaises(ValueError):
            rb.RuleBaseline().fit(np.zeros((2, 2)), np.array([0, 1]), ["a", "b"], "val")

    def test_threshold_separates_and_ties_go_to_the_higher_value(self):
        rate = [0.1, 0.2, 0.3, 5.0, 6.0, 7.0]
        r = rb.RuleBaseline().fit(arr(mqtt_publish_rate=rate), np.array([0, 0, 0, 1, 1, 1]), FEATS, "val")
        self.assertEqual(r.thresholds["mqtt_publish_rate"], 5.0)      # any t in (0.3, 5.0] is perfect; the highest candidate wins
        self.assertEqual(r.val_f1_, 1.0)
        for f in rb.RATE_FEATURES:
            if f != "mqtt_publish_rate":
                self.assertEqual(r.thresholds[f], rb.INF)             # irrelevant rules stay off (no gain)

    def test_deterministic_and_row_order_invariant(self):
        rng = np.random.default_rng(0)
        n = 300
        X = arr(mqtt_connect_rate=rng.exponential(1, n), mqtt_publish_rate=rng.exponential(2, n),
                tcp_syn_rate=rng.exponential(1, n), arp_reply_rate=rng.exponential(0.1, n),
                mqtt_connack_failures=rng.poisson(0.2, n), topic_wildcard_subs=rng.poisson(0.03, n))
        y = ((X[:, 2] > 3) | (X[:, 3] > 0.4) | (rng.random(n) < 0.05)).astype(int)
        a = rb.RuleBaseline().fit(X, y, FEATS, "val")
        b = rb.RuleBaseline().fit(X.copy(), y.copy(), FEATS, "val")
        self.assertEqual(a.thresholds, b.thresholds)
        p = rng.permutation(n)
        c = rb.RuleBaseline().fit(X[p], y[p], FEATS, "val")
        self.assertEqual(a.thresholds, c.thresholds)

    def test_candidates_are_positive_values_that_occur_and_are_capped(self):
        v = np.concatenate([np.zeros(10), np.arange(1, 1001, dtype=float)])
        c = rb.candidates(v)
        self.assertLessEqual(len(c), rb.MAX_CANDIDATES)
        self.assertTrue((c > 0).all() and set(c) <= set(v))
        self.assertEqual(c[0], 1.0)
        self.assertEqual(c[-1], 1000.0)


class TestScoreAndVerdict(unittest.TestCase):
    def setUp(self):
        self.r = rb.RuleBaseline({"mqtt_connect_rate": 2.0, "mqtt_publish_rate": rb.INF, "tcp_syn_rate": 10.0,
                                  "arp_reply_rate": rb.INF, "mqtt_connack_failures": 3.0}, FEATS)
        self.X = arr(mqtt_connect_rate=[0.0, 1.0, 2.0, 4.0, 0.0, 0.0, 0.0],
                     tcp_syn_rate=[0, 0, 0, 0, 25.0, 0, 0],
                     mqtt_connack_failures=[0, 0, 0, 0, 0, 3, 0],
                     topic_wildcard_subs=[0, 0, 0, 0, 0, 0, 2])

    def test_pred_equals_score_ge_one_equals_the_or(self):
        s, p, f = self.r.score(self.X), self.r.predict(self.X), self.r.fired(self.X)
        np.testing.assert_array_equal(p, (s >= 1.0).astype(int))
        np.testing.assert_array_equal(p, f.any(axis=1).astype(int))
        self.assertEqual(list(p), [0, 0, 1, 1, 1, 1, 1])             # ">=": a value equal to its threshold fires

    def test_score_definition(self):
        s = self.r.score(self.X)
        self.assertAlmostEqual(s[1], 0.5)                             # connect 1.0 / 2.0
        self.assertAlmostEqual(s[3], 2.0)                             # connect 4.0 / 2.0
        self.assertAlmostEqual(s[4], 2.5)                             # syn 25 / 10
        self.assertAlmostEqual(s[6], 3.0)                             # wildcard: 1 + subs
        self.assertEqual(s[0], 0.0)

    def test_which_rules_fired(self):
        names = self.r.rules_that_fired(self.X)
        self.assertEqual(names[0], [])
        self.assertEqual(names[3], ["mqtt_connect_rate"])
        self.assertEqual(names[6], ["topic_wildcard_subs"])
        self.assertEqual(self.r.describe()["topic_wildcard_subs"], ">= 1 (fixed)")
        self.assertIsNone(self.r.describe()["mqtt_publish_rate"])


# ----------------------------------------------------------------------------------- the runner on synthetic frames
def _prepare(paths, cic_shift=0.0):
    """Make the synthetic attack rows carry rule signal: syn_flood -> tcp_syn_rate, mqtt_flood -> mqtt_publish_rate,
    arp_mitm -> arp_reply_rate, recon_scan -> mqtt_connect_rate, bruteforce -> mqtt_connack_failures; benign stays ~0."""
    sig = {"syn_flood": "tcp_syn_rate", "mqtt_flood": "mqtt_publish_rate", "arp_mitm": "arp_reply_rate",
           "recon_scan": "mqtt_connect_rate", "bruteforce": "mqtt_connack_failures"}
    for key in ("tb_windows", "cic_windows"):
        df = pd.read_parquet(paths[key])
        for f in rb.RULE_FEATURES + (rb.WILDCARD,):
            df[f] = 0.0
        for cls, f in sig.items():
            m = (df["label"] == "attack") & (df["attack_class"] == cls)
            df.loc[m, f] = 20.0 + (cic_shift if key == "cic_windows" else 0.0)
        df.to_parquet(paths[key], index=False)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old = {k: os.environ.get(k) for k in ENV}
        os.environ.update(ENV)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.paths = make_synthetic(cls.root / "data")
        _prepare(cls.paths)
        cls.env = ta.Env(dataset=cls.paths["tb_windows"], fl=cls.paths["tb_fl"], split=cls.paths["split_path"],
                         runs=cls.root / "out" / "runs", model_dir=cls.root / "out" / "models",
                         pred_dir=cls.root / "out" / "preds", cic_dir=Path(cls.paths["cic_windows"]).parent,
                         for_report=True, seeds=(0,))
        cls.fit_calls = []
        orig = rr._fit

        def spy(X, y, feats, meta):
            cls.fit_calls.append(meta[["split", "attack_class", "label"]].copy())
            return orig(X, y, feats, meta)
        rr._fit = spy
        try:
            rr.run_a(cls.env)
            rr.run_b(cls.env)
            rr.run_c(cls.env)
        finally:
            rr._fit = orig

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        cls._tmp.cleanup()

    def recs(self, exp, pattern):
        return [json.loads(p.read_text()) for p in sorted((self.env.runs / f"exp{exp}").glob(pattern))]


class TestRunnerRecords(Base):
    def test_counts_fields_and_provenance(self):
        a, b, c = (self.recs("A", "rule_baseline_s[0-9].json"), self.recs("B", "rule_baseline_*_s[0-9].json"),
                   self.recs("C", "rule_baseline_*_s[0-9].json"))
        self.assertEqual((len(a), len(b), len(c)), (1, 5, 2))
        self.assertEqual({r["params"]["held_out_class"] for r in b},
                         {"syn_flood", "arp_mitm", "bruteforce", "recon_scan", "mqtt_flood"})
        self.assertEqual({r["params"]["direction"] for r in c}, {"tb2cic", "cic2tb"})
        for r in a + b + c:
            self.assertEqual((r["role"], r["model"], r["for_report"], r["provenance"]["seed"]), ("rule", "rule_baseline", True, 0))
            self.assertIs(r["tags"]["deterministic"], True)
            self.assertEqual([k for k in ("seed", "split_sha256", "dataset_sha256", "git_commit", "git_dirty",
                                          "image_digest") if r["provenance"][k] is None], [])
            self.assertEqual(r["params"]["pred_rule"], "score >= 1")
        for r in a + b:
            for view in ("all", "noatk"):
                for m in ("f1", "precision", "recall", "fpr", "pr_auc", "roc_auc"):
                    self.assertIn(f"test_{view}_{m}", r["metrics"])
            self.assertIn("test_recall_attacker", r["metrics"])
        for r in c:
            self.assertIn("tgt_test_noatk_f1", r["metrics"])
            self.assertIn("src_test_all_f1", r["metrics"])
            self.assertEqual(r["params"]["variant"], "astrained")

    def test_summaries_exist_and_never_claim_a_probability(self):
        S = json.loads((self.env.runs / "rule_baseline_summary.json").read_text())
        self.assertEqual(S["n_records"], {"A": 1, "B": 5, "C": 2})
        self.assertIn("B_C_mean_f1", S["all"])
        self.assertIn("not a probability", S["note"])
        for exp in "ABC":
            self.assertTrue((self.env.runs / f"exp{exp}" / "rule_baseline_summary.json").exists())

    def test_a_predictions_in_both_formats_with_pred_equal_score_ge_one(self):
        per = pd.read_parquet(self.env.runs / "expA" / "predictions" / "rule_baseline_s0.parquet")
        shared = pd.read_parquet(self.env.pred_dir / "A_rule_baseline.parquet")
        self.assertEqual(set(per["split"]), {"val", "test"})
        np.testing.assert_array_equal(per["pred"], (per["score"] >= 1.0).astype(int))
        np.testing.assert_array_equal(shared["pred"], (shared["score"] >= 1.0).astype(int))
        self.assertEqual(len(shared), int((per["split"] == "test").sum()))

    def test_a_record_carries_wildcard_statistics(self):
        w = self.recs("A", "rule_baseline_s[0-9].json")[0]["tags"]["wildcard"]
        self.assertEqual(set(w["per_split"]), {"train", "val", "test"})
        self.assertIn("test_false_positives", w)

    def test_distinct_signals_do_not_generalise_in_b_so_no_flag(self):
        # each class has its OWN rule feature; withholding X removes the only evidence for its rule: F1 collapses (the
        # intended B behaviour for a threshold rule), so nothing reaches the 0.98 rule-3 flag.
        self.assertFalse((self.env.runs / "expB" / "FLAG.md").exists())
        for r in self.recs("B", "rule_baseline_*_s[0-9].json"):
            self.assertLess(r["metrics"]["test_noatk_f1"], rr.FLAG_F1, r["params"]["held_out_class"])

    def test_a_shared_signal_gives_b_f1_near_one_and_the_flag_names_the_rule(self):
        with tempfile.TemporaryDirectory() as t:
            paths = make_synthetic(Path(t) / "data")
            for key in ("tb_windows", "cic_windows"):
                df = pd.read_parquet(paths[key])
                for f in rb.RULE_FEATURES + (rb.WILDCARD,):
                    df[f] = 0.0
                df.loc[df["label"] == "attack", "mqtt_publish_rate"] = 30.0     # every class trips the same rule
                df.to_parquet(paths[key], index=False)
            env = ta.Env(dataset=paths["tb_windows"], fl=paths["tb_fl"], split=paths["split_path"], runs=Path(t) / "runs",
                         pred_dir=Path(t) / "p", cic_dir=Path(paths["cic_windows"]).parent, for_report=True, seeds=(0,))
            rr.run_b(env)
            txt = (env.runs / "expB" / "FLAG.md").read_text()
            self.assertIn("syn_flood", txt)
            self.assertIn("mqtt_publish_rate", txt)                              # the rule that fired is named


class TestValidationOnlyAndWithholding(Base):
    def test_every_fit_saw_validation_rows_only(self):
        self.assertGreaterEqual(len(self.fit_calls), 1 + 5 + 2)
        for m in self.fit_calls:
            self.assertEqual(set(m["split"]), {"val"})

    def test_b_removes_the_withheld_class_from_the_fitting_rows_and_from_test(self):
        b_calls = self.fit_calls[1:6]
        order = sorted(set(self.fit_calls[0]["attack_class"].dropna()) | {"syn_flood", "arp_mitm", "bruteforce",
                                                                        "recon_scan", "mqtt_flood"})
        for X, m in zip(order, b_calls):
            self.assertNotIn(X, set(m["attack_class"].dropna()), X)
        for r in self.recs("B", "rule_baseline_*_s[0-9].json"):
            X = r["params"]["held_out_class"]
            pr = pd.read_parquet(self.env.runs / "expB" / "predictions" / f"rule_baseline_{X}_s0.parquet")
            te = pr[pr["split"] == "test"]
            self.assertEqual(set(te.loc[te["label"] == "attack", "attack_class"]), {X})
            self.assertNotIn(X, set(pr.loc[pr["split"] == "val", "attack_class"].dropna()))

    def test_c_applies_the_source_thresholds_unchanged_to_the_target(self):
        doms, _ = ta.load_c_domains(self.env)
        for src, tgt, tag in (("testbed", "cic", "tb2cic"), ("cic", "testbed", "cic2tb")):
            ds_s, ds_t = doms[src], doms[tgt]
            want = rb.RuleBaseline().fit(ds_s.X("val"), ds_s.y("val"), ds_s.feature_names,
                                         ds_s.meta("val")["split"].to_numpy())
            rec = json.loads((self.env.runs / "expC" / f"rule_baseline_{tag}_astrained_s0.json").read_text())
            self.assertEqual(rec["params"]["thresholds"], want.describe())
            pr = pd.read_parquet(self.env.runs / "expC" / "predictions" / f"rule_baseline_{tag}_astrained_s0.parquet")
            tg = pr[pr["domain"] == "tgt_test"]
            np.testing.assert_allclose(tg["score"].to_numpy(), want.score(ds_t.X("test"), ds_s.feature_names))

    def test_c_thresholds_do_not_depend_on_the_target(self):
        with tempfile.TemporaryDirectory() as t:
            paths = make_synthetic(Path(t) / "data")
            _prepare(paths, cic_shift=500.0)                           # a very different target domain
            env = ta.Env(dataset=paths["tb_windows"], fl=paths["tb_fl"], split=paths["split_path"], runs=Path(t) / "runs",
                         pred_dir=Path(t) / "p", cic_dir=Path(paths["cic_windows"]).parent, for_report=True, seeds=(0,))
            rr.run_c(env)
            a = json.loads((env.runs / "expC" / "rule_baseline_tb2cic_astrained_s0.json").read_text())["params"]["thresholds"]
            b = self.recs("C", "rule_baseline_tb2cic_astrained_s0.json")[0]["params"]["thresholds"]
            self.assertEqual(a, b)


class _FakeDS:
    def __init__(self, parts):
        self._p = parts

    def meta(self, s):
        return self._p[s][0]

    def X(self, s):
        return self._p[s][1]

    def y(self, s):
        return self._p[s][2]


class TestWildcardStats(unittest.TestCase):
    def test_nurse_benign_wildcard_counts_per_split_and_session(self):
        feats = FEATS
        def part(rows):
            m = pd.DataFrame([r[:3] for r in rows], columns=["session", "host_role", "label"])
            subs = [r[3] for r in rows]
            X = arr(topic_wildcard_subs=subs)
            y = (m["label"] == "attack").astype(int).to_numpy()
            return m, X, y
        train = part([("s1", "nurse", "benign", 1), ("s1", "nurse", "benign", 0), ("s1", "monitor", "benign", 3)])
        val = part([("v1", "nurse", "benign", 2), ("v1", "nurse", "benign", 0), ("v1", "nurse", "benign", 0), ("v1", "nurse", "benign", 4)])
        test = part([("t1", "nurse", "benign", 1), ("t1", "nurse", "benign", 0), ("t1", "attacker", "attack", 5),
                     ("t2", "nurse", "benign", 0), ("t2", "pump", "benign", 1)])
        ds = _FakeDS({"train": train, "val": val, "test": test})
        rule = rb.RuleBaseline({f: rb.INF for f in rb.RATE_FEATURES}, feats)
        pred = rule.predict(test[1], feats)
        w = rr.wildcard_stats(ds, feats, rule, pred, test[2])
        self.assertEqual(w["per_split"]["train"], {"nurse_benign_windows": 2, "with_wildcard": 1, "share": 0.5})
        self.assertEqual(w["per_split"]["val"], {"nurse_benign_windows": 4, "with_wildcard": 2, "share": 0.5})
        self.assertEqual(w["per_split"]["test"]["nurse_benign_windows"], 3)
        self.assertEqual(w["per_split"]["test"]["with_wildcard"], 1)
        self.assertEqual(w["per_session"]["t1"]["nurse_benign_windows"], 2)
        self.assertEqual(w["per_session"]["t2"]["with_wildcard"], 0)
        self.assertEqual(w["benign_all_windows"]["test"], {"benign_windows": 4, "wildcard_rule_fires": 2, "share": 0.5})
        fp = w["test_false_positives"]                                  # wildcard-only rule: FPs = benign wildcard windows
        self.assertEqual((fp["n"], fp["wildcard_rule_fired"], fp["only_wildcard_fired"], fp["from_nurse_station"]), (2, 2, 2, 1))
        self.assertEqual(fp["share_wildcard_fired"], 1.0)


if __name__ == "__main__":
    unittest.main()
