"""Pipeline tests for review M3 (role from MAC) and M2 (ip_mac_conflict per involved host)."""
import unittest

from src.features import windows
from src import common

ATT = "de:ad:be:ef:00:01"
VIC = "11:11:11:11:11:11"
BYS = "22:22:22:22:22:22"
BROKER = "00:00:00:00:00:10"
BCAST = "ff:ff:ff:ff:ff:ff"
VIC_IP = "172.28.10.11"
IP_ROLES = {VIC_IP: "monitor", "172.28.10.10": "broker"}


def _tcp(ts, smac, dmac, sip, dip):
    return {"ts": float(ts), "length": 80, "eth_src": smac, "eth_dst": dmac,
            "ip_src": sip, "ip_dst": dip, "sport": 40000, "dport": 1883,
            "l3": "ip", "l4": "tcp", "syn": False, "ack": True, "rst": False, "fin": False}


def _arp_reply(ts, smac, psrc, pdst, hwsrc):
    return {"ts": float(ts), "length": 60, "eth_src": smac, "eth_dst": BCAST,
            "ip_src": psrc, "ip_dst": pdst, "sport": None, "dport": None,
            "l3": "arp", "l4": None, "arp_op": 2, "arp_psrc": psrc, "arp_pdst": pdst,
            "arp_hwsrc": hwsrc}


class TestRoleAndConflict(unittest.TestCase):
    def test_m3_attacker_role_from_mac_even_when_forwarding_victim_ip(self):
        t = 3_000_000.0
        # Attacker forwards packets carrying the victim's IP (MITM): its most-frequent IP is VIC_IP.
        pkts = [_tcp(t + i, ATT, BROKER, VIC_IP, "172.28.10.10") for i in range(3)]
        rows, _ = windows.build_rows(iter(pkts), "t", ATT, "9.9.9.9", IP_ROLES)
        att_rows = [r for r in rows if r["host_mac"] == ATT]
        self.assertTrue(att_rows)
        for r in att_rows:
            self.assertEqual(r["host_ip"], VIC_IP)        # IP-derived role would say "monitor"
            self.assertEqual(r["host_role"], "attacker")  # but MAC-derived role is correct

    def test_m2_conflict_flag_only_on_involved_hosts(self):
        t = 4_000_000.0
        pkts = [
            _tcp(t + 0, VIC, BROKER, VIC_IP, "172.28.10.10"),      # victim legitimately owns VIC_IP
            _tcp(t + 1, BYS, BROKER, "172.28.10.33", "172.28.10.10"),  # unrelated bystander
            _arp_reply(t + 2, ATT, VIC_IP, VIC_IP, ATT),           # attacker claims VIC_IP -> conflict
        ]
        rows, conflict_windows = windows.build_rows(iter(pkts), "t", ATT, "9.9.9.9", IP_ROLES)
        w = common.window_start(t)
        self.assertIn(w, conflict_windows)
        flag = {r["host_mac"]: r["ip_mac_conflict"] for r in rows if r["window_start_epoch"] == w}
        self.assertEqual(flag.get(ATT), 1)   # new owner: involved
        self.assertEqual(flag.get(VIC), 1)   # old owner: involved
        self.assertEqual(flag.get(BYS), 0)   # bystander: not involved


if __name__ == "__main__":
    unittest.main()
