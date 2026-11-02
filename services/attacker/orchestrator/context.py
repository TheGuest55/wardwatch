"""Run context shared by all attack handlers: lab topology from config/ward.json, the
attacker's own identity, the session output directory, and the stolen-credential store.

Credential handling (testbed artifice, documented in REPORT_NOTES): the attacker has no
credentials in the threat model. Credential-access attacks (bruteforce, arp_mitm) write what
they recover to stolen_creds.json; credential-abuse attacks read it. So the dataset always has
usable abuse traffic, the session may provide the real passwords in a mounted creds file; the
bruteforce wordlist seeds from it and it is the fallback when live recovery fails (flagged
creds_source=fallback in ground truth)."""
import json
import os
from pathlib import Path

from . import timing


class Context:
    def __init__(self, session_id, out_dir, seed):
        self.session_id = session_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Per-session RNG: reproducible, but different per session so attack timing varies.
        self.rng = timing.session_rng(seed, session_id)
        self.cfg = json.loads(Path("/config/ward.json").read_text())

        self.beds = self.cfg["beds"]
        self.broker_ip = self.cfg["broker"]["ip"]
        self.nurse_ip = self.cfg["nurse"]["ip"]
        self.nurse_user = "nurse-station"
        prefix = self.cfg["network"]["subnet"].rsplit(".", 1)[0] + "."
        self.monitor_ips = [f"{prefix}{10 * b + 1}" for b in range(1, self.beds + 1)]
        self.monitor_users = [f"bed{b}-monitor" for b in range(1, self.beds + 1)]
        # username -> its IP, for attacks that need to target a specific device
        self.user_ip = {f"bed{b}-monitor": f"{prefix}{10 * b + 1}" for b in range(1, self.beds + 1)}
        self.src_ip = self.cfg["attacker"]["ip"]
        self.src_mac = Path("/sys/class/net/eth0/address").read_text().strip()

        self.stolen_path = self.out_dir / "stolen_creds.json"
        self._provisioned = self._load_provisioned()

    def _load_provisioned(self):
        """Real lab passwords for seeding/fallback, if the session mounted them. Keyed by
        MQTT username. Absent in a pure black-box run."""
        p = Path("/run/creds/accounts.json")
        if p.exists():
            try:
                return json.loads(p.read_text())
            except ValueError:
                return {}
        return {}

    def provisioned(self, username):
        return self._provisioned.get(username)

    def save_stolen(self, creds):
        """Merge recovered credentials into stolen_creds.json."""
        existing = {}
        if self.stolen_path.exists():
            existing = json.loads(self.stolen_path.read_text())
        existing.update(creds)
        self.stolen_path.write_text(json.dumps(existing, indent=2))

    def load_stolen(self):
        if self.stolen_path.exists():
            return json.loads(self.stolen_path.read_text())
        return {}
