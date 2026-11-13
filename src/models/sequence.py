"""Causal sequence builder for the LSTM classifier (and the RollingBuffer the live pipeline uses).

One example = the k windows of one host that END at the target window (never later). Each host is
put on its session's full 5-second grid; gaps and the left padding of the first k-1 windows are
**scaled zeros with is_present = 0**, and present windows carry is_present = 1. The scaler is the
train-only scaler of the shared harness (`data.Dataset(scale=True).scaler`).

Leakage control: the tensors hold only the model features (+ is_present). Session, MAC, IP, role,
run id, attack class, timestamps and labels travel alongside as a metadata DataFrame and are never
inputs. Targets are windows labelled `attack` or `benign` only; `victim` and `boundary` windows (and
filled gaps) may appear as context but never as targets.
"""
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as mdata
from ..labels import label as labelling

SPLITS = mdata.SPLITS
STEP = 5                                            # seconds per window
# Fixed multiclass order: benign = 0, then these (sorted) attack classes.
ATTACK_CLASSES = ("arp_mitm", "bruteforce", "eavesdrop", "mqtt_flood",
                  "pump_cmd", "recon_scan", "spoof_vitals", "syn_flood")
CLASS_ORDER = ("benign",) + ATTACK_CLASSES
TARGET_LABELS = ("attack", "benign")
SEP = "|"                                           # separator inside exclusion_class
META_KEYS = ["session", "host_mac", "host_ip", "host_role", "run_id", "attack_class",
             "window_start_epoch", "label"]


def class_list(exclude_class=None):
    """Output classes: benign + the attack classes, with the withheld class removed (Experiment B)."""
    if exclude_class is not None and exclude_class not in ATTACK_CLASSES:
        raise ValueError(f"unknown attack class {exclude_class!r}")
    return ["benign"] + [c for c in ATTACK_CLASSES if c != exclude_class]


@dataclass
class SeqSet:
    X: np.ndarray          # [n, k, d(+1)] float32
    y_class: np.ndarray    # int64; -1 = a class with no output unit (withheld class X in B test)
    y_bin: np.ndarray      # int64; 1 = attack
    meta: pd.DataFrame     # one row per target, META_KEYS columns


def exclusion_classes(df, ground_truth_dir):
    """Metadata column `exclusion_class` (never an input): the attack class(es) a window belongs to for
    the purpose of Experiment B exclusion, joined with "|". Normal rows carry their attack_class
    (victim rows keep theirs; attack rows have it). BOUNDARY rows have attack_class = None (the
    labeller blanks it), so each gets the class of every ground-truth run it straddles, by exactly the
    label.py rule: window [w0, w0 + WINDOW) straddles a run if w0 < start < w1 or w0 < end < w1. A
    boundary window that straddles no run is a labelling bug and fails loudly. The label, attack_class
    and run_id columns are left untouched."""
    out = df["attack_class"].astype(object).where(df["attack_class"].notna(), None).copy()
    bmask = (df["label"] == "boundary").to_numpy()
    if not bmask.any():
        return out
    W = labelling.WINDOW
    ivs = {}
    for sess in df.loc[bmask, "session"].unique():
        ivs[sess], _, _ = labelling.load_ground_truth(Path(ground_truth_dir) / str(sess))
    for i in np.flatnonzero(bmask):
        w0 = int(df["window_start_epoch"].iat[i])
        w1 = w0 + W
        sess = df["session"].iat[i]
        classes = sorted({iv["class"] for iv in ivs[sess]
                          if (w0 < iv["start"] < w1) or (w0 < iv["end"] < w1)})
        if not classes:
            raise ValueError(f"boundary window {sess}/{df['host_mac'].iat[i]}/{w0} straddles no "
                             f"ground-truth run: labelling bug")
        out.iat[i] = SEP.join(classes)
    return out


def load_sessions(sessions_dir, split_path, ground_truth_dir=None):
    """Read every per-session parquet listed in split_v2.json and assign its `split` from that file
    (the per-session parquets have no split column). Asserts the splits do not overlap. Adds the
    metadata column `exclusion_class` from the read-only ground truth in `ground_truth_dir`
    (default: the folder holding split_path, i.e. data/raw/sessions/<session>/ground_truth.jsonl)."""
    split_map = json.loads(Path(split_path).read_text())
    gt_dir = Path(ground_truth_dir) if ground_truth_dir else Path(split_path).parent
    seen = {}
    for split, sessions in split_map.items():
        for s in sessions:
            if s in seen:
                raise ValueError(f"session {s!r} is in both {seen[s]!r} and {split!r}")
            seen[s] = split
    unknown = set(split_map) - set(SPLITS)
    if unknown:
        raise ValueError(f"unexpected split names in {split_path}: {sorted(unknown)}")
    frames = []
    for split in SPLITS:
        for s in split_map.get(split, []):
            f = Path(sessions_dir) / f"{s}.parquet"
            part = pd.read_parquet(f)
            part["split"] = split
            frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    if "label" in df.columns and "attack_class" in df.columns:
        df["exclusion_class"] = exclusion_classes(df, gt_dir)
    return df


def fit_scaler(df_v2, feature_list):
    """Train-only StandardScaler, exactly as the harness fits it (victim/boundary rows excluded)."""
    return mdata.Dataset(df_v2, feature_list, scale=True).scaler


def _coerce(values):
    """One window's raw values -> float vector, same NaN/inf -> 0 coercion as data._matrix."""
    out = np.empty(len(values), dtype=float)
    for i, v in enumerate(values):
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = 0.0
        out[i] = x if np.isfinite(x) else 0.0
    return out


def build_sequences(df, scaler, feature_names, k, use_presence=True, exclude_class=None,
                    splits=SPLITS):
    """Return {split: SeqSet}. `df` must carry the session, host_mac, window_start_epoch, label,
    attack_class, split and feature columns (all rows, including victim/boundary).

    exclude_class=X (Experiment B): in train and val, every sequence containing ANY window with
    attack_class == X is dropped (whatever its label); in test, context is unchanged and the targets
    are benign + X only. Withheld-class test targets get y_class = -1."""
    feature_names = list(feature_names)
    nf = len(feature_names)
    classes = class_list(exclude_class)
    cls_idx = {c: i for i, c in enumerate(classes)}

    key = ["session", "host_mac", "window_start_epoch"]
    if df.duplicated(key).any():
        raise ValueError("duplicate (session, host_mac, window_start_epoch) keys")
    df = df.reset_index(drop=True)
    sess_split = df.groupby("session")["split"].nunique()
    if (sess_split > 1).any():
        raise ValueError(f"sessions spanning several splits: {list(sess_split[sess_split > 1].index)}")

    Xs = scaler.transform(mdata._matrix(df, feature_names)).astype(np.float32)
    epoch = df["window_start_epoch"].to_numpy(dtype=np.int64)
    label = df["label"].to_numpy()
    aclass = df["attack_class"].astype(object).where(df["attack_class"].notna(), None).to_numpy()
    if exclude_class is not None:
        if "exclusion_class" not in df.columns:
            raise ValueError("exclude_class needs the exclusion_class column (use load_sessions): "
                             "boundary windows carry no attack_class")
        xcls = df["exclusion_class"].astype(object).where(df["exclusion_class"].notna(), "").to_numpy()
        in_x = np.array([exclude_class in str(v).split(SEP) for v in xcls], dtype=bool)
    bounds = df.groupby("session")["window_start_epoch"].agg(["min", "max"])
    off = np.arange(k)

    out = {}
    for split in splits:
        xs, yc, yb, metas = [], [], [], []
        sub = df[df["split"] == split]
        for (sess, mac), g in sub.groupby(["session", "host_mac"], sort=True):
            t0, t1 = int(bounds.loc[sess, "min"]), int(bounds.loc[sess, "max"])
            pos = g.index.to_numpy()
            rel = epoch[pos] - t0
            if (rel % STEP).any():
                raise ValueError(f"session {sess}: windows off the {STEP}s grid")
            gi = rel // STEP                                   # grid index of each present window
            T = (t1 - t0) // STEP + 1
            P = np.zeros((T + k - 1, nf + 1), dtype=np.float32)    # left pad k-1 zero rows
            P[k - 1 + gi, :nf] = Xs[pos]
            P[k - 1 + gi, nf] = 1.0
            hasx = np.zeros(T + k - 1, dtype=bool)
            if exclude_class is not None:
                hasx[k - 1 + gi] = in_x[pos]

            lab = label[pos]
            ac = aclass[pos]
            is_t = np.isin(lab, TARGET_LABELS)
            if exclude_class is not None and split == "test":
                is_t &= (lab == "benign") | (ac == exclude_class)
            sel = np.flatnonzero(is_t)
            if sel.size == 0:
                continue
            ti = gi[sel][:, None] + off[None, :]                # [n, k] indices into P, ending at target
            if exclude_class is not None and split != "test":
                keep = ~hasx[ti].any(axis=1)
                sel, ti = sel[keep], ti[keep]
                if sel.size == 0:
                    continue
            seq = P[ti]
            xs.append(seq if use_presence else seq[:, :, :nf])
            lab_s, ac_s = lab[sel], ac[sel]
            yc.append(np.array([0 if l == "benign" else cls_idx.get(a, -1)
                                for l, a in zip(lab_s, ac_s)], dtype=np.int64))
            yb.append((lab_s == "attack").astype(np.int64))
            metas.append(g.iloc[sel][META_KEYS].reset_index(drop=True))
        d = nf + (1 if use_presence else 0)
        if xs:
            out[split] = SeqSet(np.concatenate(xs), np.concatenate(yc), np.concatenate(yb),
                                pd.concat(metas, ignore_index=True))
        else:
            out[split] = SeqSet(np.zeros((0, k, d), np.float32), np.zeros(0, np.int64),
                                np.zeros(0, np.int64), pd.DataFrame(columns=META_KEYS))
    return out


class RollingBuffer:
    """Per-host rolling window for the live pipeline. push() places a window by TIMESTAMP, fills
    skipped grid steps itself (scaled zeros, is_present = 0), keeps the last k windows and returns the
    [1, k, d] tensor for the newest window, identical to the offline builder's tensor for it."""

    def __init__(self, k, scaler, feature_names, use_presence=True):
        self.k = int(k)
        self.scaler = scaler
        self.feature_names = list(feature_names)
        self.use_presence = use_presence
        self._nf = len(self.feature_names)
        self._d = self._nf + (1 if use_presence else 0)
        self._rows = deque(maxlen=self.k)
        self._last = None

    def _fill_row(self):
        return np.zeros(self._d, dtype=np.float32)

    def push(self, raw_row, window_start_epoch):
        e = int(window_start_epoch)
        if self._last is not None:
            delta = e - self._last
            if delta <= 0:
                raise ValueError(f"window {e} is not after the previous window {self._last}")
            if delta % STEP:
                raise ValueError(f"window {e} is off the {STEP}s grid (previous {self._last})")
            for _ in range(delta // STEP - 1):
                self._rows.append(self._fill_row())
        vals = [raw_row[f] for f in self.feature_names]
        scaled = self.scaler.transform(_coerce(vals)[None, :])[0].astype(np.float32)
        row = self._fill_row()
        row[:self._nf] = scaled
        if self.use_presence:
            row[self._nf] = 1.0
        self._rows.append(row)
        self._last = e
        out = np.zeros((1, self.k, self._d), dtype=np.float32)
        n = len(self._rows)
        out[0, self.k - n:] = np.stack(list(self._rows))
        return out
