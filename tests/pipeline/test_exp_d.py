"""Experiment D loader: the default raw path, labels from file names, the validation tail, and a tiny end-to-end run
on a synthetic CSV tree (--raw)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import exp_d


def write_tree(root, n=40, classes=("Benign", "ARP_Spoofing", "TCP_IP-DDoS-SYN1", "TCP_IP-DDoS-SYN2", "MQTT-DoS-Connect_Flood")):
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        d = Path(root) / split
        d.mkdir(parents=True)
        for c in classes:
            df = pd.DataFrame(rng.normal(size=(n, 3)) + (0 if c == "Benign" else 3), columns=[" Header_Length", "Rate ", "IAT"])
            df.to_csv(d / f"{c}_{split}.pcap.csv", index=False)


class TestPathAndLabels(unittest.TestCase):
    def test_default_raw_points_at_the_real_csv_folder(self):
        self.assertTrue(str(exp_d.RAW).endswith("ciciomt2024/WiFi_and_MQTT/attacks/csv"))
        self.assertTrue((exp_d.RAW / "train").is_dir() and (exp_d.RAW / "test").is_dir())
        self.assertEqual(len(list((exp_d.RAW / "train").glob("*.csv"))), 51)
        self.assertEqual(len(list((exp_d.RAW / "test").glob("*.csv"))), 21)

    def test_class_and_category(self):
        self.assertEqual(exp_d.class_of("TCP_IP-DDoS-SYN3_train.pcap.csv"), "TCP_IP-DDoS-SYN")
        self.assertEqual(exp_d.class_of("Benign_test.pcap.csv"), "Benign")
        self.assertEqual([exp_d.category_of(c) for c in ("Benign", "ARP_Spoofing", "Recon-OS_Scan", "MQTT-DoS-Connect_Flood",
                                                         "TCP_IP-DDoS-SYN", "TCP_IP-DoS-SYN")],
                         ["Benign", "Spoofing", "Recon", "MQTT", "DDoS", "DoS"])


class TestSubsample(unittest.TestCase):
    def test_seeded_stratified_train_only_keeps_validation_and_never_touches_test(self):
        with tempfile.TemporaryDirectory() as t:
            write_tree(t, n=100)
            tr = exp_d.load_split("train", Path(t))
            te = exp_d.load_split("test", Path(t))
        sub, info = exp_d.subsample_train(tr, 0.25, seed=0)
        sub2, _ = exp_d.subsample_train(tr, 0.25, seed=0)
        self.assertTrue(sub.equals(sub2))                                           # seeded: reproducible
        self.assertEqual(int(sub["is_val"].sum()), int(tr["is_val"].sum()))         # the validation tail is kept whole
        fit_before = tr[~tr["is_val"]].groupby("c19", observed=True).size()
        fit_after = sub[~sub["is_val"]].groupby("c19", observed=True).size()
        for c in fit_before.index:
            self.assertAlmostEqual(fit_after[c] / fit_before[c], 0.25, delta=0.02)   # stratified by class
        self.assertEqual(info["train_rows_after"], len(sub))
        self.assertLess(len(sub), len(tr))
        same, i1 = exp_d.subsample_train(tr, 1.0)
        self.assertIs(same, tr)
        self.assertEqual(i1, {"train_subsample": 1.0})
        with self.assertRaises(ValueError):
            exp_d.subsample_train(tr, 0)
        self.assertEqual(len(te), 5 * 100)                                           # the test frame is untouched by design

    def test_main_records_the_subsample_and_suffixes_the_file_name(self):
        import sys
        with tempfile.TemporaryDirectory() as t:
            write_tree(Path(t) / "csv", n=100)
            out = Path(t) / "out"
            orig, argv = exp_d.OUT, sys.argv
            exp_d.OUT = out
            sys.argv = ["exp_d", "--raw", str(Path(t) / "csv"), "--models", "random_forest", "--tasks", "2", "--seeds", "1",
                        "--allow-partial", "--no-mlflow", "--train-subsample", "0.25"]
            try:
                exp_d.main()
            finally:
                exp_d.OUT, sys.argv = orig, argv
            rec = json.loads((out / "random_forest_sub25.json").read_text())
            self.assertEqual(rec["params"]["train_subsample"], 0.25)
            self.assertEqual(rec["tags"]["train_subsample"], 0.25)
            self.assertLess(rec["params"]["train_rows_after"], rec["params"]["train_rows_before"])


class TestLoadSplit(unittest.TestCase):
    def test_reads_a_tiny_tree_with_stripped_columns_labels_and_val_tail(self):
        with tempfile.TemporaryDirectory() as t:
            write_tree(t)
            tr = exp_d.load_split("train", Path(t))
            te = exp_d.load_split("test", Path(t))
        self.assertEqual(list(tr.columns[:3]), ["Header_Length", "Rate", "IAT"])
        self.assertEqual(len(tr), 5 * 40)
        self.assertEqual(set(tr["c19"]), {"Benign", "ARP_Spoofing", "TCP_IP-DDoS-SYN", "MQTT-DoS-Connect_Flood"})
        self.assertEqual(set(tr["c2"]), {"Benign", "Attack"})
        self.assertEqual(int(tr["is_val"].sum()), 5 * 4)                     # the LAST 10% of every train file, in time order
        for _, g in tr.groupby("source_file", observed=True):
            v = g["is_val"].to_numpy()
            self.assertTrue(not v[:36].any() and v[36:].all())
        self.assertFalse(te["is_val"].any())                                  # test is never a validation set
        self.assertEqual(exp_d.check_complete(tr, te), ["train has 4 classes, expected 19"])

    def test_missing_split_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(SystemExit):
                exp_d.load_split("train", Path(t))

    def test_main_with_raw_runs_end_to_end_and_tags_the_record(self):
        with tempfile.TemporaryDirectory() as t:
            write_tree(Path(t) / "csv")
            out = Path(t) / "out"
            orig = exp_d.OUT
            exp_d.OUT = out
            try:
                import sys
                argv = sys.argv
                sys.argv = ["exp_d", "--raw", str(Path(t) / "csv"), "--models", "random_forest", "--tasks", "2", "--seeds", "1",
                            "--allow-partial", "--no-mlflow"]
                try:
                    exp_d.main()
                finally:
                    sys.argv = argv
            finally:
                exp_d.OUT = orig
            rec = json.loads((out / "random_forest.json").read_text())
            self.assertEqual(rec["experiment"], "D")
            self.assertIs(rec["tags"]["indicative"], True)
            self.assertIn("not applicable", rec["tags"]["attacker_excluded_view"])
            self.assertEqual(rec["tags"]["raw"], str(Path(t) / "csv"))
            self.assertFalse(rec["for_report"])
            self.assertIn("2c_f1_attack_mean", rec["metrics"])


if __name__ == "__main__":
    unittest.main()
