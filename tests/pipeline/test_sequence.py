"""Tests for the causal sequence builder (src/models/sequence.py): no crossing of session/host/split,
the target is the last element and nothing later appears, gaps and padding are scaled zeros with
is_present = 0, victim/boundary rows are context only, Experiment B exclusion, RollingBuffer equals
the offline tensor, and (real v2 data) target rows equal the harness Dataset rows."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import common
from src.models import data as mdata
from src.models import sequence as seq

V2 = common.PROCESSED_DIR / "v2"
SPLIT_PATH = common.ROOT / "data" / "raw" / "sessions" / "split_v2.json"
FEATS = ["f1", "f2"]
K = 4
T0 = {"a": 1000, "b": 5000, "c": 9000}
SESSION_CODE = {"a": 1, "b": 2, "c": 3}
HOST_CODE = {"H1": 1, "H2": 2, "H3": 3}


class _Ident:
    """Identity 'scaler' so the encoded ids in f1 can be read back from the tensor."""
    def transform(self, x):
        return np.asarray(x, dtype=float).copy()


def code(session, host, i):
    return SESSION_CODE[session] * 1000 + HOST_CODE[host] * 100 + i + 1   # never 0


def _rows(session, split, host, steps, labels=None, classes=None, excl=None):
    out = []
    for i in steps:
        out.append({
            "session": session, "split": split, "host_mac": host, "host_ip": "ip-" + host,
            "host_role": "monitor", "run_id": None,
            "window_start_epoch": T0[session] + 5 * i,
            "f1": float(code(session, host, i)), "f2": float(i),
            "label": (labels or {}).get(i, "benign"),
            "attack_class": (classes or {}).get(i),
            # boundary windows have attack_class None; exclusion_class carries the straddled run's class
            "exclusion_class": (excl or {}).get(i, (classes or {}).get(i)),
        })
    return out


def synth():
    """Session a (train): H1 present 0..9 except 4,5 (gap); victim at 3 and boundary at 7 (both class
    syn_flood), attack syn_flood at 8; H2 only 0..2 (< K windows); H3 only 6..9 (late start).
    Session b (val): H1 0..5 with an eavesdrop attack at 5. Session c (test): H1 0..9, attack
    syn_flood at 4, attack eavesdrop at 7."""
    rows = []
    rows += _rows("a", "train", "H1", [0, 1, 2, 3, 6, 7, 8, 9],
                  labels={3: "victim", 7: "boundary", 8: "attack"},
                  classes={3: "syn_flood", 8: "syn_flood"}, excl={7: "syn_flood"})
    rows += _rows("a", "train", "H2", [0, 1, 2])
    # H3: its first window is a boundary row of a syn_flood run (attack_class None, as the labeller leaves it)
    rows += _rows("a", "train", "H3", [6, 7, 8, 9], labels={6: "boundary"}, excl={6: "syn_flood"})
    rows += _rows("b", "val", "H1", range(6), labels={5: "attack"}, classes={5: "eavesdrop"})
    rows += _rows("c", "test", "H1", range(10), labels={4: "attack", 7: "attack"},
                  classes={4: "syn_flood", 7: "eavesdrop"})
    return pd.DataFrame(rows)


def reference_seq(df, session, host, i, k, d_presence=True):
    """Independent, loop-based expected tensor for the target at grid step i."""
    g = df[(df.session == session) & (df.host_mac == host)]
    have = {int((e - T0[session]) // 5): float(f) for e, f in zip(g.window_start_epoch, g.f1)}
    out = np.zeros((k, 3), dtype=np.float32)
    for p in range(k):
        step = i - (k - 1 - p)
        if step >= 0 and step in have:
            out[p, 0] = have[step]
            out[p, 1] = float(step)
            out[p, 2] = 1.0
    return out if d_presence else out[:, :2]


class TestSynthetic(unittest.TestCase):
    def setUp(self):
        self.df = synth()
        self.out = seq.build_sequences(self.df, _Ident(), FEATS, K)

    def test_targets_are_attack_or_benign_only(self):
        tgt = pd.concat([s.meta for s in self.out.values()])
        self.assertTrue(set(tgt["label"]) <= {"attack", "benign"})
        want = self.df[self.df.label.isin(["attack", "benign"])]
        self.assertEqual(len(tgt), len(want))

    def test_every_sequence_matches_independent_reference(self):
        """Covers: no crossing of session/host/split, last element = target, nothing later,
        gap fill = zeros + is_present 0, left padding of the first k-1 windows."""
        n = 0
        for split, s in self.out.items():
            for j in range(len(s.meta)):
                m = s.meta.iloc[j]
                i = int((m["window_start_epoch"] - T0[m["session"]]) // 5)
                np.testing.assert_array_equal(
                    s.X[j], reference_seq(self.df, m["session"], m["host_mac"], i, K))
                self.assertEqual(s.X[j, -1, 0], code(m["session"], m["host_mac"], i))  # target is last
                n += 1
        self.assertGreater(n, 20)

    def test_no_sequence_crosses_a_session_or_split(self):
        for split, s in self.out.items():
            self.assertTrue((s.meta["session"].map({"a": "train", "b": "val", "c": "test"}) == split).all())
            ids = s.X[:, :, 0][s.X[:, :, 2] == 1]
            self.assertTrue(set((ids // 1000).astype(int)) <= set(SESSION_CODE.values()))
            for j in range(len(s.meta)):
                sc = SESSION_CODE[s.meta.iloc[j]["session"]]
                hc = HOST_CODE[s.meta.iloc[j]["host_mac"]]
                present = s.X[j, :, 0][s.X[j, :, 2] == 1].astype(int)
                self.assertTrue(((present // 1000) == sc).all())
                self.assertTrue((((present % 1000) // 100) == hc).all())

    def test_victim_and_boundary_are_context_only(self):
        tr = self.out["train"]
        epochs = set(tr.meta[tr.meta.host_mac == "H1"]["window_start_epoch"])
        self.assertNotIn(T0["a"] + 15, epochs)       # victim at step 3
        self.assertNotIn(T0["a"] + 35, epochs)       # boundary at step 7
        j = int(np.flatnonzero((tr.meta.host_mac == "H1") &
                               (tr.meta.window_start_epoch == T0["a"] + 40))[0])   # attack at step 8
        ids = tr.X[j, :, 0]
        self.assertIn(float(code("a", "H1", 7)), ids)    # boundary row visible as context

    def test_gap_fill_is_scaled_zero_not_raw_zero(self):
        raw = synth()
        sc = StandardScaler().fit(raw[FEATS].to_numpy())
        out = seq.build_sequences(raw, sc, FEATS, K)
        s = out["train"]
        j = int(np.flatnonzero((s.meta.host_mac == "H1") &
                               (s.meta.window_start_epoch == T0["a"] + 30))[0])   # step 6, gap at 4,5
        self.assertEqual(s.X[j, 1, 2], 0.0)              # step 4 missing
        self.assertTrue((s.X[j, 1, :2] == 0.0).all())
        self.assertTrue((s.X[j, 2, :2] == 0.0).all())    # step 5 missing
        self.assertEqual(s.X[j, 3, 2], 1.0)
        self.assertTrue((s.X[j, 3, :2] != 0.0).any())

    def test_left_padding_and_short_host(self):
        s = self.out["train"]
        j = int(np.flatnonzero((s.meta.host_mac == "H2") &
                               (s.meta.window_start_epoch == T0["a"]))[0])   # first window of H2
        self.assertTrue((s.X[j, :K - 1] == 0).all())
        self.assertEqual(s.X[j, K - 1, 2], 1.0)

    def test_use_presence_false_drops_channel_keeps_zero_fill(self):
        nop = seq.build_sequences(self.df, _Ident(), FEATS, K, use_presence=False)
        for split in seq.SPLITS:
            self.assertEqual(nop[split].X.shape[-1], len(FEATS))
            np.testing.assert_array_equal(nop[split].X, self.out[split].X[:, :, :len(FEATS)])
            absent = self.out[split].X[:, :, 2] == 0
            self.assertTrue((nop[split].X[absent] == 0).all())

    def test_class_encoding_and_order(self):
        self.assertEqual(seq.CLASS_ORDER[0], "benign")
        self.assertEqual(list(seq.ATTACK_CLASSES), sorted(seq.ATTACK_CLASSES))
        self.assertEqual(len(seq.ATTACK_CLASSES), 8)
        s = self.out["test"]
        eav = seq.CLASS_ORDER.index("eavesdrop")
        self.assertEqual(set(s.y_class[s.meta.attack_class == "eavesdrop"]), {eav})
        self.assertEqual(set(s.y_class[s.y_bin == 0]), {0})

    def test_exclude_class_train_val_has_no_class_x_window_of_any_label(self):
        out = seq.build_sequences(self.df, _Ident(), FEATS, K, exclude_class="syn_flood")
        x_codes = {float(code("a", "H1", i)) for i in (3, 7, 8)} | {float(code("c", "H1", 4))}
        for split in ("train", "val"):
            ids = out[split].X[:, :, 0]
            self.assertFalse(np.isin(ids, list(x_codes)).any())
        # target at step 9 of session a/H1 has the boundary (7) and attack (8) in context -> dropped
        tr = out["train"].meta
        self.assertFalse(((tr.host_mac == "H1") & (tr.window_start_epoch == T0["a"] + 45)).any())
        # step 0 of the same host is untouched
        self.assertTrue(((tr.host_mac == "H1") & (tr.window_start_epoch == T0["a"])).any())

    def test_boundary_window_straddling_class_x_is_excluded_from_train_val(self):
        """H3/a step 6 is a boundary window (attack_class None) of a syn_flood run. Without the
        exclusion_class column it would never match; every sequence that contains it must go."""
        full = seq.build_sequences(self.df, _Ident(), FEATS, K)["train"].meta
        out = seq.build_sequences(self.df, _Ident(), FEATS, K, exclude_class="syn_flood")["train"]
        b = float(code("a", "H3", 6))
        self.assertFalse((out.X[:, :, 0] == b).any())
        h3 = lambda m: m[(m.host_mac == "H3")]
        self.assertEqual(len(h3(full)), 3)              # steps 7,8,9 are targets; 6 is context only
        self.assertEqual(len(h3(out.meta)), 0)          # all three contain the boundary window (k=4)
        # exclusion by another class leaves them
        other = seq.build_sequences(self.df, _Ident(), FEATS, K, exclude_class="eavesdrop")["train"]
        self.assertEqual(len(h3(other.meta)), 3)

    def test_exclude_class_without_exclusion_column_fails_loudly(self):
        with self.assertRaises(ValueError):
            seq.build_sequences(self.df.drop(columns=["exclusion_class"]), _Ident(), FEATS, K,
                                exclude_class="syn_flood")

    def test_exclude_class_test_targets_are_benign_plus_x(self):
        out = seq.build_sequences(self.df, _Ident(), FEATS, K, exclude_class="syn_flood")
        t = out["test"]
        self.assertEqual(set(t.meta.attack_class.dropna()), {"syn_flood"})   # eavesdrop attack gone
        self.assertEqual(int(t.y_bin.sum()), 1)
        self.assertEqual(set(t.y_class[t.y_bin == 1]), {-1})                 # no output unit for X
        # context unchanged: the later windows still show the unmodified stream
        ref = seq.build_sequences(self.df, _Ident(), FEATS, K)["test"]
        j = int(np.flatnonzero(t.meta.window_start_epoch == T0["c"] + 20)[0])
        jr = int(np.flatnonzero(ref.meta.window_start_epoch == T0["c"] + 20)[0])
        np.testing.assert_array_equal(t.X[j], ref.X[jr])
        self.assertEqual(len(seq.class_list("syn_flood")), 8)               # benign + 7 outputs
        self.assertNotIn("syn_flood", seq.class_list("syn_flood"))
        self.assertTrue(t.y_class.max() <= 7)

    def test_no_metadata_in_the_tensor(self):
        self.assertEqual(self.out["train"].X.shape[-1], len(FEATS) + 1)
        for c in seq.META_KEYS:
            self.assertNotIn(c, FEATS)

    def test_duplicate_keys_rejected(self):
        dup = pd.concat([self.df, self.df.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            seq.build_sequences(dup, _Ident(), FEATS, K)


class TestRollingBuffer(unittest.TestCase):
    def _check(self, scaler, use_presence):
        df = synth()
        off = seq.build_sequences(df, scaler, FEATS, K, use_presence=use_presence)
        checked = 0
        for split, s in off.items():
            for (sess, host), g in df[df.split == split].groupby(["session", "host_mac"]):
                buf = seq.RollingBuffer(K, scaler, FEATS, use_presence=use_presence)
                for _, r in g.sort_values("window_start_epoch").iterrows():
                    got = buf.push(r[FEATS].to_dict(), r["window_start_epoch"])
                    m = ((s.meta.session == sess) & (s.meta.host_mac == host) &
                         (s.meta.window_start_epoch == r["window_start_epoch"]))
                    if m.any():   # targets only; victim/boundary rows just advance the buffer
                        j = int(np.flatnonzero(m)[0])
                        self.assertEqual(got.shape, (1, K, s.X.shape[-1]))
                        np.testing.assert_array_equal(got[0], s.X[j])
                        checked += 1
        self.assertGreater(checked, 25)

    def test_equals_offline_with_gaps_and_short_host(self):
        sc = StandardScaler().fit(synth()[FEATS].to_numpy())
        self._check(sc, True)

    def test_equals_offline_without_presence(self):
        sc = StandardScaler().fit(synth()[FEATS].to_numpy())
        self._check(sc, False)

    def test_out_of_order_or_off_grid_push_rejected(self):
        buf = seq.RollingBuffer(K, _Ident(), FEATS)
        buf.push({"f1": 1, "f2": 1}, 100)
        with self.assertRaises(ValueError):
            buf.push({"f1": 1, "f2": 1}, 100)
        with self.assertRaises(ValueError):
            buf.push({"f1": 1, "f2": 1}, 103)

    def test_nan_and_inf_coerced_to_zero_like_the_harness(self):
        sc = _Ident()
        buf = seq.RollingBuffer(2, sc, FEATS)
        got = buf.push({"f1": float("nan"), "f2": float("inf")}, 0)
        self.assertTrue((got[0, -1, :2] == 0).all())


class TestLoadSessions(unittest.TestCase):
    def test_assigns_split_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for s in ("x", "y"):
                pd.DataFrame({"session": [s], "window_start_epoch": [0]}).to_parquet(d / f"{s}.parquet")
            ok = d / "split.json"
            ok.write_text(json.dumps({"train": ["x"], "val": ["y"], "test": []}))
            df = seq.load_sessions(d, ok)
            self.assertEqual(dict(zip(df.session, df.split)), {"x": "train", "y": "val"})
            bad = d / "bad.json"
            bad.write_text(json.dumps({"train": ["x"], "val": ["x"], "test": []}))
            with self.assertRaises(ValueError):
                seq.load_sessions(d, bad)


@unittest.skipUnless((V2 / "sessions" / "r05.parquet").exists() and SPLIT_PATH.exists(),
                     "real v2 data not available")
class TestRealV2(unittest.TestCase):
    K = 6

    @classmethod
    def setUpClass(cls):
        cls.df = seq.load_sessions(V2 / "sessions", SPLIT_PATH)
        cls.tw, cls.fl = mdata.load_dataset()
        cls.feats = mdata.feature_columns(cls.fl, cls.df)
        cls.scaler = seq.fit_scaler(cls.df, cls.fl)
        cls.out = seq.build_sequences(cls.df, cls.scaler, cls.feats, cls.K)

    def test_scaler_equals_the_harness_scaler(self):
        ds = mdata.Dataset(self.tw, self.fl, scale=True)
        np.testing.assert_allclose(self.scaler.mean_, ds.scaler.mean_, rtol=0, atol=1e-9)
        np.testing.assert_allclose(self.scaler.scale_, ds.scaler.scale_, rtol=0, atol=1e-9)

    def test_input_dimension_is_52(self):
        self.assertEqual(len(self.feats), 51)
        self.assertEqual(self.out["test"].X.shape[1:], (self.K, 52))
        nop = seq.build_sequences(self.df, self.scaler, self.feats, 2, use_presence=False)
        self.assertEqual(nop["test"].X.shape[-1], 51)

    def test_targets_equal_dataset_rows_per_split(self):
        ds = mdata.Dataset(self.tw, self.fl, scale=True)
        key = ["session", "host_mac", "window_start_epoch"]
        self.assertEqual(len(self.out["test"].meta), 24840)
        for split in seq.SPLITS:
            want = ds.meta(split)
            got = self.out[split].meta
            self.assertEqual(len(got), len(want), split)
            m = got.merge(want[key + ["label", "split"]].rename(columns={"split": "split_ds"}),
                          on=key, how="outer", indicator=True, suffixes=("", "_ds"))
            self.assertTrue((m["_merge"] == "both").all(), split)
            self.assertTrue((m["label"] == m["label_ds"]).all(), split)
            self.assertTrue((m["split_ds"] == split).all(), split)
            self.assertEqual(int(self.out[split].y_bin.sum()), int(ds.y(split).sum()))

    def test_split_assignment_matches_testbed_windows_for_every_row(self):
        key = ["session", "host_mac", "window_start_epoch"]
        m = self.df.merge(self.tw[key + ["split", "label"]], on=key, how="inner", suffixes=("", "_tw"))
        self.assertEqual(len(m), len(self.tw))
        self.assertTrue((m["split"] == m["split_tw"]).all())
        self.assertTrue((m["label"] == m["label_tw"]).all())

    def test_context_has_victim_and_boundary_but_targets_do_not(self):
        for s in self.out.values():
            self.assertTrue(set(s.meta["label"]) <= {"attack", "benign"})
        self.assertTrue(self.df["label"].isin(["victim", "boundary"]).any())

    @staticmethod
    def _independent_straddles(df):
        """{(session, host_mac, epoch): set(classes)} for boundary rows, straight from ground_truth.jsonl
        (own parser, own time conversion) with the label.py rule w0 < start < w0+5 or w0 < end < w0+5."""
        from datetime import datetime, timezone
        def ts(z):
            return datetime.strptime(z, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
        gt = {}
        for sess in df["session"].unique():
            f = SPLIT_PATH.parent / sess / "ground_truth.jsonl"      # normal sessions (n02) have none
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
            gt[sess] = [(r["class"], ts(r["start_utc"]), ts(r["end_utc"])) for r in rows]
        b = df[df["label"] == "boundary"]
        out = {}
        for r in b.itertuples():
            w0 = r.window_start_epoch
            out[(r.session, r.host_mac, w0)] = {c for c, s0, e0 in gt[r.session]
                                                if (w0 < s0 < w0 + 5) or (w0 < e0 < w0 + 5)}
        return out

    def test_every_boundary_row_gets_exactly_the_straddled_classes(self):
        ind = self._independent_straddles(self.df)
        self.assertEqual(len(ind), int((self.df["label"] == "boundary").sum()))
        self.assertGreater(len(ind), 0)
        self.assertTrue(all(len(v) >= 1 for v in ind.values()))
        b = self.df[self.df["label"] == "boundary"]
        self.assertTrue(b["attack_class"].isna().all())            # the labeller blanks it
        for r in b.itertuples():
            self.assertEqual(set(r.exclusion_class.split(seq.SEP)),
                             ind[(r.session, r.host_mac, r.window_start_epoch)])
        nb = self.df[self.df["label"] != "boundary"]
        same = (nb["exclusion_class"].fillna("") == nb["attack_class"].fillna(""))
        self.assertTrue(same.all())
        # label / attack_class / run_id are left as the labeller wrote them
        orig = pd.concat([pd.read_parquet(V2 / "sessions" / f"{s}.parquet") for s in
                          json.loads(SPLIT_PATH.read_text())["train"] + json.loads(SPLIT_PATH.read_text())["val"]
                          + json.loads(SPLIT_PATH.read_text())["test"]], ignore_index=True)
        pd.testing.assert_series_equal(orig["label"].reset_index(drop=True),
                                       self.df["label"].reset_index(drop=True), check_names=False)
        pd.testing.assert_series_equal(orig["attack_class"].reset_index(drop=True),
                                       self.df["attack_class"].reset_index(drop=True), check_names=False)

    def test_exclude_class_removes_every_class_x_window_from_train_and_val(self):
        key = ["session", "host_mac", "window_start_epoch"]
        ind = self._independent_straddles(self.df)
        for x in seq.ATTACK_CLASSES:
            out = seq.build_sequences(self.df, self.scaler, self.feats, self.K, exclude_class=x)
            xrows = self.df[self.df["attack_class"] == x][key]
            bx = pd.DataFrame([k for k, v in ind.items() if x in v], columns=key)
            xrows = pd.concat([xrows, bx], ignore_index=True)
            self.assertGreater(len(xrows), 0)
            self.assertGreater(len(bx), 0, x)           # every class has boundary windows
            for split in ("train", "val"):
                base = self.out[split].meta[key].copy()
                kept = out[split].meta[key].copy()
                # independent expectation: drop every target within k-1 steps after a class-X window
                bad = set()
                xs = {(r.session, r.host_mac): [] for r in xrows.itertuples()}
                for r in xrows.itertuples():
                    xs[(r.session, r.host_mac)].append(r.window_start_epoch)
                for r in base.itertuples():
                    for xe in xs.get((r.session, r.host_mac), ()):
                        if 0 <= r.window_start_epoch - xe <= 5 * (self.K - 1):
                            bad.add((r.session, r.host_mac, r.window_start_epoch))
                            break
                want = set(map(tuple, base.to_numpy())) - bad
                self.assertEqual(set(map(tuple, kept.to_numpy())), want, (x, split))
                self.assertFalse((out[split].meta["attack_class"] == x).any())
            t = out["test"]
            self.assertEqual(set(t.meta.loc[t.y_bin == 1, "attack_class"]), {x})
            self.assertEqual(len(t.meta), int((self.out["test"].meta.label == "benign").sum()
                                              + ((self.out["test"].meta.attack_class == x)).sum()))
            self.assertEqual(len(seq.class_list(x)), 8)


if __name__ == "__main__":
    unittest.main()
