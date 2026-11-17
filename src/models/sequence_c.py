"""Domain loader for LSTM Experiment C (cross-dataset, testbed <-> CICIoMT2024).

The shared-feature contract is the tree models': `train_tree._domain` (benign rows + the shared attack
classes, testbed-only features dropped). The LSTM sees the same 48 shared features + is_present, and
the 5 output units benign + the 4 shared classes.

  * Each run uses ONE scaler: the SOURCE domain's, fit on its train rows only (after the tree's
    benign + shared filter). The target is always transformed with it and never refit.
  * A CIC file is cut 75/25 in time into train/val, so the CIC session key is `file@split`: no
    context window crosses the cut and the first k-1 windows of a val segment are left-padded.
  * Windows of classes with no output unit (non-shared testbed classes; CIC "unmapped") may not
    appear in any target's context (build_sequences drop_classes), in any split.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .. import common
from . import data as mdata
from . import sequence as seq
from . import train_tree as tt

V2 = common.PROCESSED_DIR / "v2"
CIC_DIR = common.PROCESSED_DIR / "cic"
SHARED = tt.SHARED_CLASSES
C_CLASSES = ["benign", "arp_mitm", "mqtt_flood", "recon_scan", "syn_flood"]
NONSHARED = {"testbed": set(seq.ATTACK_CLASSES) - set(SHARED), "cic": {"unmapped"}}
assert sorted(C_CLASSES[1:]) == sorted(SHARED)


def default_paths(cic_dir=CIC_DIR):
    cic_dir = Path(cic_dir)
    return {"sessions_dir": V2 / "sessions", "split_path": common.ROOT / "data/raw/sessions/split_v2.json",
            "tb_windows": V2 / "testbed_windows.parquet", "tb_fl": V2 / "feature_list.json",
            "cic_windows": cic_dir / "cic_windows.parquet", "cic_fl": cic_dir / "feature_list.json",
            "cic_report": cic_dir / "extract_report.json"}


@dataclass
class Domain:
    name: str
    df: pd.DataFrame        # ALL rows (every label), split + exclusion_class + session keys for sequences
    feats: list             # the shared features, in train_tree._domain order
    tree_ds: object         # the trees' Dataset for this domain (rows the XGBoost/RF run uses)
    path: Path
    fl_path: Path


def load_domain(name, paths=None, size_variant="none"):
    p = paths or default_paths()
    if name == "testbed":
        df = seq.load_sessions(p["sessions_dir"], p["split_path"])
        windows, fl = p["tb_windows"], p["tb_fl"]
    elif name == "cic":
        windows, fl = p["cic_windows"], p["cic_fl"]
        df = pd.read_parquet(windows)
        df["exclusion_class"] = df["attack_class"].astype(object).where(df["attack_class"].notna(), None)
        # the CIC cut sits inside a file: make each segment its own session
        df["session"] = df["session"].astype(str) + "@" + df["split"].astype(str)
        check_cic_files(df)
    else:
        raise ValueError(f"unknown domain {name!r}")
    ds = tt._domain(windows, fl, size_variant=size_variant)          # nosize: SIZE_FEATURES dropped
    return Domain(name, df.reset_index(drop=True), list(ds.feature_names), ds, Path(windows), Path(fl))


def load_pair(paths=None, size_variant="none"):
    """Both domains, with the shared-feature lists asserted equal (same names, same order)."""
    tb, cic = load_domain("testbed", paths, size_variant), load_domain("cic", paths, size_variant)
    if tb.feats != cic.feats:
        raise SystemExit(f"feature mismatch: {set(tb.feats) ^ set(cic.feats)} "
                         f"(or a different order)")
    return {"testbed": tb, "cic": cic}


def file_stem(session):
    return str(session).split("@")[0]


def check_cic_files(df):
    """Single-attacker captures only: any DDoS file (several attackers, one attacker address cannot
    label it) stops the run. Stems come from the run_id column. Returns {"train": [...], "test": [...]}."""
    stems = {"train": set(), "test": set()}
    for rid, sp in df[["run_id", "split"]].drop_duplicates().itertuples(index=False):
        stem = file_stem(rid)
        if "ddos" in stem.lower():
            raise SystemExit(f"[cic] {stem}: DDoS captures have several attackers and cannot be labelled "
                             f"from one attacker address; extract the single-attacker (DoS) files only")
        stems["test" if sp == "test" else "train"].add(stem)
    return {k: sorted(v) for k, v in stems.items()}


def fit_source_scaler(domain):
    """StandardScaler on the source's TRAIN rows only, after the tree's benign + shared filter (and
    victim/boundary removal): identical to Dataset(scale=True).scaler on those rows."""
    return StandardScaler().fit(domain.tree_ds.X("train"))


def domain_sequences(domain, scaler, k, use_presence=True, splits=seq.SPLITS):
    return seq.build_sequences(domain.df, scaler, domain.feats, k, use_presence=use_presence,
                               classes=C_CLASSES, drop_classes=NONSHARED[domain.name], splits=splits)


# ------------------------------------------------------------------------------------ diagnostics
def padded_share(s):
    """Share of targets whose context contains a padded or filled window (is_present = 0)."""
    if len(s.X) == 0:
        return float("nan")
    return float((s.X[:, :, -1] == 0).any(axis=1).mean())


def row_table(domain, sets):
    """Rows per split: the trees' count next to the LSTM target count, with the difference by cause.
    raw = every row; victim/boundary and non-shared attack rows are not in the trees' set; the trees'
    rows minus the targets the context rule dropped = the LSTM targets (asserted)."""
    out = []
    for sp in seq.SPLITS:
        d = domain.df[domain.df["split"] == sp]
        lab = d["label"]
        non_shared = (lab == "attack") & ~d["attack_class"].isin(SHARED)
        tree_n = len(domain.tree_ds.meta(sp))
        s = sets[sp]
        row = {"domain": domain.name, "split": sp, "raw_rows": int(len(d)),
               "victim": int((lab == "victim").sum()), "boundary": int((lab == "boundary").sum()),
               "nonshared_attack": int(non_shared.sum()), "tree_rows": int(tree_n),
               "lstm_targets": int(len(s.y_class))}
        row["dropped_by_context_rule"] = row["tree_rows"] - row["lstm_targets"]
        if row["raw_rows"] - row["victim"] - row["boundary"] - row["nonshared_attack"] != tree_n:
            raise AssertionError(f"{domain.name}/{sp}: tree rows do not reconcile with the row causes")
        if row["dropped_by_context_rule"] < 0:
            raise AssertionError(f"{domain.name}/{sp}: more LSTM targets than tree rows")
        t = d[(lab == "benign") | ((lab == "attack") & d["attack_class"].isin(SHARED))]
        tc = np.where(t["label"] == "attack", t["attack_class"].astype(str), "benign")
        row["per_class"] = {c: {"tree": int((tc == c).sum()),
                                "lstm": int((np.array(C_CLASSES)[s.y_class] == c).sum()) if len(s.y_class) else 0}
                            for c in C_CLASSES}
        out.append(row)
    return out
