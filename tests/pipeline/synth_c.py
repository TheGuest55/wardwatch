"""Synthetic testbed + CIC-like frames for the Experiment-C tests (not a test module).

make_synthetic(root) writes, under `root`, a testbed (per-session parquets, split json, windows
parquet, feature list) and a CIC-like dataset (cic_windows.parquet, feature_list.json,
extract_report.json), all with the real 51 feature names, and returns the dict of paths that
src.models.sequence_c.default_paths() would give. CIC is shifted by +SHIFT sigma, has testbed-only
columns at zero, the attacker MAC with role "attacker" in EVERY file (Benign included), one
"unmapped" attack file, and train files that the extractor's 75/25 time cut splits into train/val.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import common

SHARED = ["arp_mitm", "mqtt_flood", "recon_scan", "syn_flood"]
NONSHARED_TB = ["bruteforce", "pump_cmd"]
N_WIN = 60
T0 = 1_700_000_000
MACS = {"mon": "aa:00:00:00:00:01", "brk": "aa:00:00:00:00:02", "atk": "aa:00:00:00:00:66"}
TB_CODE = ["syn_flood", "arp_mitm", "bruteforce", "recon_scan", "mqtt_flood"]
SHIFT = 10.0


def _frame(rng, session, run_id, blocks, shift, zero_tb_only, floor_t0=T0, victim_blocks=()):
    """blocks: {(lo, hi): attack_class}; the attacker and broker hosts are attack rows inside a block,
    the monitor stays benign (victim rows inside victim_blocks)."""
    rows = []
    feats = list(common.FEATURES)
    for mac_role, mac in MACS.items():
        role = {"mon": "monitor", "brk": "broker", "atk": "attacker"}[mac_role]
        for i in range(N_WIN):
            r = {"session": session, "host_mac": mac, "host_ip": "10.0.0." + mac[-2:],
                 "host_role": role, "run_id": None, "attack_class": None, "label": "benign",
                 "window_start_epoch": floor_t0 + 5 * i}
            for (lo, hi), cls in blocks.items():
                if lo <= i < hi and mac_role in ("atk", "brk"):
                    r.update(label="attack", attack_class=cls, run_id=run_id)
            for (lo, hi), cls in victim_blocks:
                if lo <= i < hi and mac_role == "mon":
                    r.update(label="victim", attack_class=cls, run_id=run_id)
            base = rng.normal(0, 1, len(feats)) + shift
            if r["label"] == "attack":
                base[: 6 + TB_CODE.index(r["attack_class"]) % 3 if r["attack_class"] in TB_CODE else 3] += 4.0
            for f, v in zip(feats, base):
                r[f] = float(v)
            if zero_tb_only:
                for f in common.TESTBED_ONLY_FEATURES:
                    r[f] = 0.0
            rows.append(r)
    return pd.DataFrame(rows)


def make_synthetic(root, seed=0, shift=SHIFT):
    root = Path(root)
    rng = np.random.default_rng(seed)
    tb_dir = root / "tb"
    (tb_dir / "sessions").mkdir(parents=True)
    split_map = {"train": ["tr1"], "val": ["va1"], "test": ["te1"]}
    (tb_dir / "split_v2.json").write_text(json.dumps(split_map))
    blocks = {(10, 16): "syn_flood", (20, 26): "arp_mitm", (30, 36): "bruteforce",
              (40, 46): "recon_scan", (50, 56): "mqtt_flood"}
    parts = []
    for sp, sess in (("train", "tr1"), ("val", "va1"), ("test", "te1")):
        df = _frame(rng, sess, f"{sess}-run", blocks, 0.0, False, victim_blocks=[((36, 40), "pump_cmd")])
        df.to_parquet(tb_dir / "sessions" / f"{sess}.parquet", index=False)
        df = df.copy()
        df["split"] = sp
        parts.append(df)
    pd.concat(parts, ignore_index=True).to_parquet(tb_dir / "testbed_windows.parquet", index=False)
    tb_only = list(common.TESTBED_ONLY_FEATURES)
    fl = {"n_features": len(common.FEATURES), "features": list(common.FEATURES), "testbed_only": tb_only}
    (tb_dir / "feature_list.json").write_text(json.dumps(fl))

    cic_dir = root / "cic"
    cic_dir.mkdir()
    files = [("Benign_train", "train", None), ("DoS-Unmapped_train", "train", "unmapped")]
    files += [(f"{c}_train", "train", c) for c in SHARED]
    files += [("Benign_test", "test", None), ("DoS-Unmapped_test", "test", "unmapped")]
    files += [(f"{c}_test", "test", c) for c in SHARED]
    frames = []
    for stem, cic_split, cls in files:
        blk = {} if cls is None else {(10, 16): cls, (46, 54): cls}
        d = _frame(rng, stem, stem, blk, shift, True)
        d["run_id"] = stem
        d["cic_class"] = cls or "Benign"
        d["is_boundary"] = False
        d["dry_run"] = False
        d["touches_attacker"] = d["label"] == "attack"
        if cic_split == "test":
            d["split"] = "test"
        else:
            wins = sorted(d["window_start_epoch"].unique())
            cut = wins[int(len(wins) * 0.75)]
            d["split"] = np.where(d["window_start_epoch"] >= cut, "val", "train")
        frames.append(d)
    pd.concat(frames, ignore_index=True).to_parquet(cic_dir / "cic_windows.parquet", index=False)
    (cic_dir / "feature_list.json").write_text(json.dumps(fl))
    (cic_dir / "extract_report.json").write_text(json.dumps({s + ".pcap": {"file": s + ".pcap"} for s, _, _ in files}))
    return {"sessions_dir": tb_dir / "sessions", "split_path": tb_dir / "split_v2.json",
            "tb_windows": tb_dir / "testbed_windows.parquet", "tb_fl": tb_dir / "feature_list.json",
            "cic_windows": cic_dir / "cic_windows.parquet", "cic_fl": cic_dir / "feature_list.json",
            "cic_report": cic_dir / "extract_report.json"}
