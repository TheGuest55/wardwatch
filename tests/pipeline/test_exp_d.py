"""Experiment D label mapping (src/models/exp_d.py): CIC file names -> 19 / 6 / 2 classes."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.exp_d import category_of, check_complete, class_of, load_split

TRAIN_FILES = """ARP_Spoofing Benign MQTT-DDoS-Connect_Flood MQTT-DDoS-Publish_Flood MQTT-DoS-Connect_Flood
MQTT-DoS-Publish_Flood MQTT-Malformed_Data Recon-OS_Scan Recon-Ping_Sweep Recon-Port_Scan
Recon-VulScan TCP_IP-DDoS-ICMP1 TCP_IP-DDoS-ICMP8 TCP_IP-DDoS-SYN1 TCP_IP-DDoS-SYN4 TCP_IP-DDoS-TCP1
TCP_IP-DDoS-UDP1 TCP_IP-DDoS-UDP8 TCP_IP-DoS-ICMP1 TCP_IP-DoS-SYN1 TCP_IP-DoS-TCP1
TCP_IP-DoS-UDP4""".split()
TEST_FILES = """ARP_Spoofing Benign MQTT-DDoS-Connect_Flood MQTT-DDoS-Publish_Flood MQTT-DoS-Connect_Flood
MQTT-DoS-Publish_Flood MQTT-Malformed_Data Recon-OS_Scan Recon-Ping_Sweep Recon-Port_Scan
Recon-VulScan TCP_IP-DDoS-ICMP1 TCP_IP-DDoS-ICMP2 TCP_IP-DDoS-SYN TCP_IP-DDoS-TCP TCP_IP-DDoS-UDP1
TCP_IP-DDoS-UDP2 TCP_IP-DoS-ICMP TCP_IP-DoS-SYN TCP_IP-DoS-TCP TCP_IP-DoS-UDP""".split()


class TestLabels(unittest.TestCase):
    def test_part_numbers_and_suffix_stripped(self):
        self.assertEqual(class_of("TCP_IP-DDoS-SYN3_train.pcap.csv"), "TCP_IP-DDoS-SYN")
        self.assertEqual(class_of("TCP_IP-DDoS-SYN_test.pcap.csv"), "TCP_IP-DDoS-SYN")
        self.assertEqual(class_of("ARP_Spoofing_train.pcap.csv"), "ARP_Spoofing")
        self.assertEqual(class_of("MQTT-DDoS-Connect_Flood_test.pcap.csv"), "MQTT-DDoS-Connect_Flood")

    def test_official_file_lists_give_19_classes_both_sides(self):
        tr = {class_of(f + "_train.pcap.csv") for f in TRAIN_FILES}
        te = {class_of(f + "_test.pcap.csv") for f in TEST_FILES}
        self.assertEqual(len(tr), 19)
        self.assertEqual(tr, te)

    def test_six_categories(self):
        cats = {category_of(class_of(f + "_test.pcap.csv")) for f in TEST_FILES}
        self.assertEqual(cats, {"Benign", "Spoofing", "Recon", "MQTT", "DoS", "DDoS"})
        self.assertEqual(category_of("MQTT-DDoS-Connect_Flood"), "MQTT")   # MQTT wins over DDoS
        self.assertEqual(category_of("TCP_IP-DDoS-UDP"), "DDoS")
        self.assertEqual(category_of("TCP_IP-DoS-UDP"), "DoS")

    def test_unknown_class_rejected(self):
        with self.assertRaises(ValueError):
            category_of("Bluetooth-DoS")


class TestLoadAndCompleteness(unittest.TestCase):
    def test_val_tail_and_partial_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split in ("train", "test"):
                (root / split).mkdir()
                for cls in ("Benign", "ARP_Spoofing"):
                    pd.DataFrame({"Rate": range(20), " IAT": range(20)}).to_csv(
                        root / split / f"{cls}_{split}.pcap.csv", index=False)
            tr, te = load_split("train", root), load_split("test", root)
        self.assertEqual(int(tr["is_val"].sum()), 4)                  # last 10% of each 20-row file
        self.assertTrue(tr.groupby("source_file")["is_val"].apply(lambda s: s.iloc[-2:].all()).all())
        self.assertFalse(te["is_val"].any())
        self.assertIn("IAT", tr.columns)                              # header whitespace stripped
        self.assertEqual(set(tr["c2"]), {"Benign", "Attack"})
        self.assertTrue(any("expected 19" in p for p in check_complete(tr, te)))


if __name__ == "__main__":
    unittest.main()
