"""Pipeline tests for review N2: the `victim` label for stolen-credential victim devices, victim-IP
derivation (pump_cmd from commands, spoof_vitals from notes, structured `victims` precedence,
eavesdrop none), and the spoof-victim PCAP confirmation (stop on disagreement)."""
import unittest

import pandas as pd

from src import common
from src.labels import label as L
from src.features import extract, mqtt

CFG = common.load_config()
PREFIX = CFG["network"]["subnet"].rsplit(".", 1)[0] + "."
MON3 = f"{PREFIX}31"   # bed3 monitor
PUMP2 = f"{PREFIX}22"  # bed2 pump
PUMP4 = f"{PREFIX}42"  # bed4 pump


class TestVictimIPs(unittest.TestCase):
    def test_spoof_from_notes(self):
        iv = {"class": "spoof_vitals", "notes": "published 60 spoofed vitals to bed3 as bed3-monitor",
              "params": {}, "victims": None}
        self.assertEqual(L.victim_ips_for_interval(iv, CFG), {MON3})

    def test_pump_from_commands(self):
        iv = {"class": "pump_cmd", "notes": "", "victims": None,
              "params": {"commands": [{"bed": 2}, {"bed": 4}, {"bed": 2}]}}
        self.assertEqual(L.victim_ips_for_interval(iv, CFG), {PUMP2, PUMP4})

    def test_structured_victims_take_precedence(self):
        iv = {"class": "spoof_vitals", "notes": "to bed3", "params": {},
              "victims": [{"ip": "10.9.9.9", "mac": "aa:bb"}]}
        self.assertEqual(L.victim_ips_for_interval(iv, CFG), {"10.9.9.9"})

    def test_eavesdrop_has_no_victim(self):
        iv = {"class": "eavesdrop", "notes": "subscribed ward1/#", "params": {}, "victims": None}
        self.assertEqual(L.victim_ips_for_interval(iv, CFG), set())


class TestVictimLabelling(unittest.TestCase):
    ATK = "172.28.10.66"

    def _iv(self, cls, start, end, **kw):
        return {"class": cls, "run_id": f"r-{cls}-01", "start": float(start), "end": float(end),
                "params": kw.get("params", {}), "notes": kw.get("notes", ""), "victims": None}

    def test_victim_attack_benign_boundary(self):
        spoof = self._iv("spoof_vitals", 1000, 1100,
                         notes="published 60 spoofed vitals to bed3 as bed3-monitor")
        df = pd.DataFrame([
            {"window_start_epoch": 1000, "touches_attacker": False, "host_ip": MON3},   # victim
            {"window_start_epoch": 1000, "touches_attacker": True, "host_ip": self.ATK},  # attack
            {"window_start_epoch": 1200, "touches_attacker": False, "host_ip": MON3},    # benign (outside)
            {"window_start_epoch": 1005, "touches_attacker": True, "host_ip": MON3},     # attack (victim host but touches)
        ])
        out = L.apply_labels(df, [spoof], CFG)
        self.assertEqual(list(out["label"]), ["victim", "attack", "benign", "attack"])
        # victim keeps class + run_id as metadata
        self.assertEqual(out.loc[0, "attack_class"], "spoof_vitals")
        self.assertEqual(out.loc[0, "run_id"], "r-spoof_vitals-01")

    def test_boundary_precedence(self):
        spoof = self._iv("spoof_vitals", 1002, 1100,
                         notes="to bed3 as bed3-monitor")   # starts mid-window 1000..1005
        df = pd.DataFrame([{"window_start_epoch": 1000, "touches_attacker": False, "host_ip": MON3}])
        out = L.apply_labels(df, [spoof], CFG)
        self.assertEqual(out.loc[0, "label"], "boundary")   # straddle beats victim


class TestSpoofConfirm(unittest.TestCase):
    ATK_IP, ATK_MAC = "172.28.10.66", "aa:bb:cc:dd:ee:ff"

    def _pub(self, ts, topic):
        return {"ts": float(ts), "ip_src": self.ATK_IP, "eth_src": self.ATK_MAC,
                "mqtt": [{"type": mqtt.PUBLISH, "topic": topic}]}

    def _run(self, iv, recs):
        obs = {}
        list(extract._spoof_tap(iter(recs), [iv], self.ATK_IP, self.ATK_MAC, obs))
        return obs

    def test_confirms_matching_bed(self):
        iv = {"class": "spoof_vitals", "run_id": "r-spoof-01", "start": 100.0, "end": 200.0,
              "notes": "to bed3 as bed3-monitor"}
        obs = self._run(iv, [self._pub(150, "ward1/bed3/monitor/vitals")])
        res = extract._confirm_spoof_victims([iv], obs)
        self.assertTrue(res["r-spoof-01"]["confirmed"])

    def test_disagreement_raises(self):
        iv = {"class": "spoof_vitals", "run_id": "r-spoof-01", "start": 100.0, "end": 200.0,
              "notes": "to bed3 as bed3-monitor"}
        obs = self._run(iv, [self._pub(150, "ward1/bed4/monitor/vitals")])  # PCAP says bed4!
        with self.assertRaises(SystemExit):
            extract._confirm_spoof_victims([iv], obs)


if __name__ == "__main__":
    unittest.main()
