"""Calibration, confidence bands and SHAP explanations for the tree classifiers (triage inputs).

    python -m src.models.calibrate_explain --model xgboost [--for-report]
    python -m src.models.calibrate_explain --model random_forest [--for-report]

Needs the Experiment A seed-0 binary model (`python -m src.models.train_tree --model <m>`).

Why (./CLAUDE.md, forensic triage): raw classifier scores are not probabilities (Guo et al. [10]), so
a 0.9 does not mean "90% likely an attack". Triage combines the classifier with the anomaly detector
and shows a confidence on every incident card, so the score must be calibrated first; and every alert
needs "why" in plain language (TreeSHAP, Lundberg & Lee [9]).

Protocol (no test data is used to fit or choose anything):
  1. Raw scores from the frozen Experiment A seed-0 model on validation (r04, r06) and test.
  2. Calibration method (Platt vs isotonic) chosen by CROSS-FITTING over the two validation sessions:
     fit on r04 -> ECE on r06, fit on r06 -> ECE on r04, lower mean ECE wins (tie -> Platt). The
     winner is then refit on all of validation.
  3. Confidence cut-offs on CALIBRATED validation probabilities:
       hi = lowest cut-off whose validation precision >= --target-precision (default 0.95)
            -> p >= hi is "confident attack";
       lo = highest cut-off that still keeps >= --target-recall (default 0.95) of validation attacks
            at or above it -> p < lo is "confident benign"; anything between is "unsure".
  4. Test is scored once: ECE/Brier before vs after, how many windows/attacks land in each band
     (overall and per attack class), and FPR inside "confident attack".
  5. TreeSHAP on test: global mean |SHAP|, top features per attack class, and three sanity checks —
     additivity (XGBoost: base + sum(SHAP) == raw margin), rank agreement with the model's own
     importances (Spearman), and that no identifier/metadata column is among the explained inputs.
  6. Single-window latency of calibrated scoring and of scoring + explanation.

Artifacts for the triage layer (data/processed/v2/models/expA/):
  {model}_calibrator.joblib, {model}_triage_config.json; use `CalibratedDetector.load(model)`.
"""
import argparse
import json
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np

from .. import common
from . import data as mdata
from . import metrics as mmetrics
from . import tracking
from .calibrator import Calibrator
from .train_tree import DATASET, GRIDS, MODEL_DIR, SPLIT, V2, importances, multiclass_targets

OUT_DIR = V2 / "runs" / "calibration"
EPS = 1e-6
BANDS = ("confident_benign", "unsure", "confident_attack")

# Plain-language meaning of every model input, for incident cards. Keys must equal feature_list.json.
FEATURE_TEXT = {
    "pkts_total": "packets sent + received in the 5 s window",
    "pkts_sent": "packets sent by the host",
    "pkts_recv": "packets received by the host",
    "bytes_total": "bytes sent + received",
    "bytes_sent": "bytes sent by the host",
    "bytes_recv": "bytes received by the host",
    "mean_pkt_size": "average packet size",
    "std_pkt_size": "spread of packet sizes",
    "min_pkt_size": "smallest packet size",
    "max_pkt_size": "largest packet size",
    "mean_iat": "average time between packets",
    "std_iat": "spread of time between packets",
    "unique_peers": "number of distinct machines (MAC) talked to",
    "unique_peer_ips": "number of distinct IP addresses talked to",
    "unique_contacted_ports": "number of distinct destination ports contacted",
    "frac_tcp": "share of packets that are TCP",
    "frac_arp": "share of packets that are ARP",
    "frac_icmp": "share of packets that are ICMP",
    "frac_other": "share of packets of other protocols",
    "tcp_syn": "TCP connection attempts (SYN)",
    "tcp_synack": "TCP connection accepts (SYN-ACK)",
    "tcp_rst": "TCP resets (refused/aborted connections)",
    "tcp_fin": "TCP connection closes (FIN)",
    "tcp_syn_rate": "TCP connection attempts per second",
    "mqtt_pkts": "MQTT packets",
    "mqtt_connect": "MQTT CONNECT (log-in) attempts",
    "mqtt_publish": "MQTT PUBLISH messages",
    "mqtt_subscribe": "MQTT SUBSCRIBE requests",
    "mqtt_connack": "MQTT CONNACK (log-in answers)",
    "mqtt_pingreq": "MQTT keep-alive pings",
    "mqtt_disconnect": "MQTT DISCONNECT messages",
    "mqtt_connect_rate": "MQTT log-in attempts per second",
    "mqtt_publish_rate": "MQTT messages published per second",
    "mqtt_mean_payload": "average MQTT message size",
    "mqtt_max_payload": "largest MQTT message",
    "mqtt_connack_failures": "failed MQTT log-ins (wrong password / not allowed)",
    "mqtt_clientid_reuse": "connections reusing a client ID that is already connected",
    "topic_wildcard_subs": "wildcard topic subscriptions (# or +, reads many devices)",
    "topic_max_depth": "deepest MQTT topic level used",
    "topic_is_command": "messages on a pump command topic",
    "arp_requests": "ARP requests (who-has)",
    "arp_replies": "ARP replies (is-at)",
    "arp_gratuitous": "unsolicited (gratuitous) ARP announcements",
    "arp_reply_rate": "ARP replies per second",
    "ip_mac_conflict": "IP address seen with a different MAC than before (possible ARP spoofing)",
    "vital_max_jump": "largest jump in a vital sign vs the device's recent readings",
    "vital_jump_count": "number of sudden vital-sign jumps",
    "implausible_vital": "physiologically implausible vital-sign value",
    "nf_active_flows": "active network flows",
    "nf_new_flows": "new network flows started",
    "nf_mean_flow_age": "average age of active flows",
}


# ---------------------------------------------------------------- calibration helpers (pure)

def _logit(s):
    s = np.clip(np.asarray(s, dtype=float), EPS, 1 - EPS)
    return np.log(s / (1 - s))


def ece(y, p, n_bins=15):
    """Expected calibration error, equal-width bins over the predicted attack probability."""
    return float(reliability_bins(y, p, n_bins)["ece"])


def reliability_bins(y, p, n_bins=15):
    y = np.asarray(y).astype(float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows, total = [], 0.0
    for i in range(n_bins):
        k = b == i
        n = int(k.sum())
        conf = float(p[k].mean()) if n else float("nan")
        acc = float(y[k].mean()) if n else float("nan")
        if n:
            total += abs(acc - conf) * n
        rows.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": n,
                     "mean_predicted": conf, "observed_attack_rate": acc})
    return {"ece": total / max(len(p), 1), "bins": rows}


def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def select_method(score, y, groups, methods=("platt", "isotonic")):
    """Cross-fit over validation groups (sessions): fit on all-but-one group, ECE on the held-out
    group; return (winner, {method: mean ECE}). Ties go to the first method listed (Platt)."""
    score, y, groups = np.asarray(score, dtype=float), np.asarray(y).astype(int), np.asarray(groups)
    uniq = sorted(set(groups.tolist()))
    if len(uniq) < 2:
        raise ValueError("need >= 2 validation groups to cross-fit the calibration method")
    res = {}
    for m in methods:
        errs = []
        for g in uniq:
            fit, hold = groups != g, groups == g
            if len(set(y[fit])) < 2:
                raise ValueError(f"group(s) other than {g} lack both classes")
            errs.append(ece(y[hold], Calibrator(m).fit(score[fit], y[fit]).predict(score[hold])))
        res[m] = float(np.mean(errs))
    best = min(methods, key=lambda m: (res[m], methods.index(m)))
    return best, res


def confidence_cutoffs(y, p, target_precision=0.95, target_recall=0.95):
    """(lo, hi) on calibrated probabilities, fitted on VALIDATION.
    hi = lowest unique p whose precision among rows with p >= hi is >= target_precision;
    lo = highest unique p such that rows with p >= lo still hold >= target_recall of the attacks.
    lo is capped at hi so the unsure band is never negative."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    P = int(y.sum())
    if P == 0 or P == len(y):
        raise ValueError("cut-offs need both classes")
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp, n = np.cumsum(ys), np.arange(1, len(ys) + 1)
    last = np.append(np.flatnonzero(np.diff(ps) != 0), len(ps) - 1)   # end of each tied block
    thr, prec, rec = ps[last], tp[last] / n[last], tp[last] / P
    ok_p = np.flatnonzero(prec >= target_precision)
    hi = float(thr[ok_p.max()]) if len(ok_p) else float("inf")         # lowest qualifying cut-off
    ok_r = np.flatnonzero(rec >= target_recall)
    lo = float(thr[ok_r.min()]) if len(ok_r) else float(thr[-1])        # highest qualifying cut-off
    return min(lo, hi), hi


def band(p, lo, hi):
    """0 = confident benign (p < lo), 1 = unsure, 2 = confident attack (p >= hi)."""
    p = np.asarray(p, dtype=float)
    return np.where(p >= hi, 2, np.where(p < lo, 0, 1))


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


# ---------------------------------------------------------------- model + SHAP

def load_model(name):
    if name == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier()
        clf.load_model(str(MODEL_DIR / "xgboost_binary_seed0.json"))
        return clf
    import joblib
    return joblib.load(MODEL_DIR / "random_forest_binary_seed0.joblib")


def shap_values(name, model, X):
    """TreeSHAP attributions for the ATTACK class, shape (n, n_features), and the base value."""
    import shap
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    base = ex.expected_value
    if isinstance(sv, list):                      # older shap: one array per class
        sv, base = sv[1], np.atleast_1d(base)[1]
    elif np.ndim(sv) == 3:                        # (n, f, classes)
        sv, base = sv[:, :, 1], np.atleast_1d(base)[1]
    return np.asarray(sv), float(np.atleast_1d(base)[0] if np.ndim(base) else base)


def top_reasons(shap_row, x_row, feats, k=3):
    """Top-k plain-language reasons pushing this window TOWARDS 'attack' (positive SHAP), falling
    back to the largest |SHAP| if fewer than k push that way."""
    order = np.argsort(-shap_row)
    pos = [i for i in order if shap_row[i] > 0][:k]
    if len(pos) < k:
        rest = [i for i in np.argsort(-np.abs(shap_row)) if i not in pos]
        pos += rest[:k - len(pos)]
    return [{"feature": feats[i], "value": float(x_row[i]), "shap": float(shap_row[i]),
             "text": f"{FEATURE_TEXT.get(feats[i], feats[i])} = {x_row[i]:g} "
                     f"({'raises' if shap_row[i] > 0 else 'lowers'} attack likelihood)"}
            for i in pos]


class CalibratedDetector:
    """What the triage layer uses: calibrated probability, band and plain-language reasons."""

    def __init__(self, name, model, calibrator, config):
        self.name, self.model, self.calibrator, self.config = name, model, calibrator, config
        self.features = config["features"]
        self._explainer = None

    @classmethod
    def load(cls, name):
        import joblib
        cfg = json.loads((MODEL_DIR / f"{name}_triage_config.json").read_text())
        return cls(name, load_model(name), joblib.load(MODEL_DIR / f"{name}_calibrator.joblib"), cfg)

    def probability(self, X):
        return self.calibrator.predict(self.model.predict_proba(np.atleast_2d(X))[:, 1])

    def band(self, X):
        return band(self.probability(X), self.config["lo"], self.config["hi"])

    def explain(self, x, k=3):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
        sv = self._explainer.shap_values(np.atleast_2d(x))
        if isinstance(sv, list):
            sv = sv[1]
        elif np.ndim(sv) == 3:
            sv = sv[:, :, 1]
        return top_reasons(np.asarray(sv)[0], np.atleast_2d(x)[0], self.features, k)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(GRIDS), required=True)
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--shap-benign-sample", type=int, default=3000,
                    help="benign test windows explained for the global SHAP summary")
    ap.add_argument("--for-report", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    name = args.model
    import joblib

    df, fl = mdata.load_dataset(DATASET, V2 / "feature_list.json")
    ds = mdata.Dataset(df, fl, scale=False)
    feats = ds.feature_names
    leaked = set(feats) & (set(common.META_COLUMNS) | set(common.METADATA_EXTRAS))
    missing_text = [f for f in feats if f not in FEATURE_TEXT]
    if leaked:
        raise SystemExit(f"identifier/metadata columns among model inputs: {sorted(leaked)}")
    if missing_text:
        raise SystemExit(f"FEATURE_TEXT lacks: {missing_text}")
    model = load_model(name)

    Xva, yva, Xte, yte = ds.X("val"), ds.y("val"), ds.X("test"), ds.y("test")
    s_va, s_te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    sess_va = ds.meta("val")["session"].astype(str).to_numpy()
    cls_te = multiclass_targets(ds.meta("test"))

    # ---- calibration: choose by cross-fit on validation sessions, refit on all validation
    method, cv_ece = select_method(s_va, yva, sess_va)
    cal = Calibrator(method).fit(s_va, yva)
    p_va, p_te = cal.predict(s_va), cal.predict(s_te)
    other = "isotonic" if method == "platt" else "platt"
    p_te_other = Calibrator(other).fit(s_va, yva).predict(s_te)       # reported, not selected on
    print(f"[{name}] calibration cross-fit ECE on validation sessions: {cv_ece} -> {method}")
    cal_metrics = {
        "test_ece_raw": ece(yte, s_te), "test_ece_calibrated": ece(yte, p_te),
        f"test_ece_{other}_for_reference": ece(yte, p_te_other),
        "test_brier_raw": brier(yte, s_te), "test_brier_calibrated": brier(yte, p_te),
        **{f"val_crossfit_ece_{m}": v for m, v in cv_ece.items()},
    }
    print(f"  test ECE raw {cal_metrics['test_ece_raw']:.5f} -> calibrated "
          f"{cal_metrics['test_ece_calibrated']:.5f} | Brier {cal_metrics['test_brier_raw']:.5f} -> "
          f"{cal_metrics['test_brier_calibrated']:.5f}")

    # ---- confidence bands: cut-offs on validation, then test once
    lo, hi = confidence_cutoffs(yva, p_va, args.target_precision, args.target_recall)
    b_te = band(p_te, lo, hi)
    print(f"  cut-offs (validation): confident benign p < {lo:.4f} | confident attack p >= {hi:.4f}")
    band_rows = {}
    for i, bname in enumerate(BANDS):
        k = b_te == i
        band_rows[bname] = {"windows": int(k.sum()), "attacks": int(yte[k].sum()),
                            "benign": int((yte[k] == 0).sum()),
                            "attack_share": float(yte[k].mean()) if k.any() else float("nan")}
    per_class = {}
    for c in sorted(set(cls_te) - {"benign"}):
        k = cls_te == c
        per_class[c] = {bname: int((b_te[k] == i).sum()) for i, bname in enumerate(BANDS)}
        per_class[c]["n"] = int(k.sum())
    n_ben = max(int((yte == 0).sum()), 1)
    band_metrics = {
        "lo": lo, "hi": hi,
        "test_attacks_confident_attack_share": band_rows["confident_attack"]["attacks"] / max(int(yte.sum()), 1),
        "test_attacks_missed_confident_benign_share": band_rows["confident_benign"]["attacks"] / max(int(yte.sum()), 1),
        "test_precision_confident_attack": band_rows["confident_attack"]["attack_share"],
        "test_fpr_confident_attack": band_rows["confident_attack"]["benign"] / n_ben,
        "test_unsure_windows": band_rows["unsure"]["windows"],
    }
    for bname, r in band_rows.items():
        print(f"  test {bname:17s} windows {r['windows']:6d} | attacks {r['attacks']:4d} | "
              f"benign {r['benign']:6d}")
    for c, r in per_class.items():
        print(f"    {c:13s} n={r['n']:4d}  " + "  ".join(f"{b}={r[b]}" for b in BANDS))

    # ---- SHAP on test: all attacks + a fixed benign sample
    rng = np.random.default_rng(0)
    ben_idx = np.flatnonzero(yte == 0)
    ben_idx = rng.choice(ben_idx, size=min(args.shap_benign_sample, len(ben_idx)), replace=False)
    idx = np.sort(np.concatenate([np.flatnonzero(yte == 1), ben_idx]))
    sv, base = shap_values(name, model, Xte[idx])
    mean_abs = np.abs(sv).mean(axis=0)
    global_rank = sorted(zip(feats, mean_abs.tolist()), key=lambda kv: -kv[1])
    print("  SHAP global top 10:", [(f, round(v, 3)) for f, v in global_rank[:10]])
    cls_idx = cls_te[idx]
    per_class_shap = {}
    for c in sorted(set(cls_idx) - {"benign"}):
        m = np.abs(sv[cls_idx == c]).mean(axis=0)
        per_class_shap[c] = [(feats[i], float(m[i])) for i in np.argsort(-m)[:5]]
        print(f"    {c:13s} top: {[f for f, _ in per_class_shap[c][:3]]}")

    imp = dict(importances(name, model, feats))
    sanity = {"no_identifier_inputs": True,
              "spearman_shap_vs_model_importance": spearman(mean_abs, [imp[f] for f in feats])}
    if name == "xgboost":
        margin = model.predict(Xte[idx], output_margin=True)
        sanity["additivity_max_abs_error"] = float(np.max(np.abs(base + sv.sum(axis=1) - margin)))
    else:
        prob = model.predict_proba(Xte[idx])[:, 1]
        sanity["additivity_max_abs_error"] = float(np.max(np.abs(base + sv.sum(axis=1) - prob)))
    print(f"  SHAP sanity: {sanity}")

    # ---- example explanations (first confident-attack test window of each class), for the report
    examples = {}
    pos_in_idx = {j: n for n, j in enumerate(idx)}
    for c in sorted(set(cls_te) - {"benign"}):
        cand = [j for j in np.flatnonzero((cls_te == c) & (b_te == 2)) if j in pos_in_idx]
        if cand:
            j = cand[0]
            examples[c] = {"calibrated_p": float(p_te[j]),
                           "reasons": top_reasons(sv[pos_in_idx[j]], Xte[j], feats, 3)}

    # ---- latency: one window at a time, as the live path does
    lat_score = mmetrics.single_window_latency_ms(
        lambda x: cal.predict(model.predict_proba(x.reshape(1, -1))[:, 1]), Xte[:500])
    import shap
    ex = shap.TreeExplainer(model)
    lat_explain = mmetrics.single_window_latency_ms(
        lambda x: (cal.predict(model.predict_proba(x.reshape(1, -1))[:, 1]),
                   ex.shap_values(x.reshape(1, -1))), Xte[:500])
    print(f"  latency p95: calibrated score {lat_score['p95_ms']:.2f} ms | "
          f"score + SHAP {lat_explain['p95_ms']:.2f} ms")

    # ---- persist: artifacts for triage + run record
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, MODEL_DIR / f"{name}_calibrator.joblib")
    (MODEL_DIR / f"{name}_triage_config.json").write_text(json.dumps({
        "model": name, "calibration": method, "lo": lo, "hi": hi, "features": feats,
        "target_precision": args.target_precision, "target_recall": args.target_recall,
        "bands": list(BANDS)}, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    det_path = OUT_DIR / f"{name}_details.json"
    det_path.write_text(json.dumps({
        "calibration_method": method, "crossfit_ece": cv_ece,
        "reliability_test_raw": reliability_bins(yte, s_te),
        "reliability_test_calibrated": reliability_bins(yte, p_te),
        "bands_test": band_rows, "bands_test_per_class": per_class,
        "shap_global": global_rank, "shap_per_class_top5": per_class_shap,
        "shap_sanity": sanity, "example_explanations": examples,
        "latency_score": lat_score, "latency_score_plus_shap": lat_explain}, indent=2, default=str))
    metrics = {**cal_metrics, **band_metrics,
               "shap_spearman_vs_importance": sanity["spearman_shap_vs_model_importance"],
               "shap_additivity_max_abs_error": sanity["additivity_max_abs_error"],
               "latency_score_p95_ms": lat_score["p95_ms"],
               "latency_score_plus_shap_p95_ms": lat_explain["p95_ms"]}
    rec = tracking.build_run_record(
        experiment="triage-calibration", model=name, role="classifier",
        params={"calibration": method, "selection": "cross-fit ECE over val sessions",
                "target_precision": args.target_precision, "target_recall": args.target_recall,
                "base_model": "Experiment A seed-0 binary", "shap_benign_sample": len(ben_idx)},
        metrics=metrics, tags={"dataset": "v2", "details": str(det_path)},
        seed=0, split_path=str(SPLIT), dataset_path=str(DATASET), for_report=args.for_report)
    tracking.log_run(rec, json_path=str(OUT_DIR / f"{name}.json"), use_mlflow=not args.no_mlflow)
    print(f"[{name}] saved {MODEL_DIR / (name + '_calibrator.joblib')} and triage config; "
          f"record -> {OUT_DIR / (name + '.json')}")


if __name__ == "__main__":
    main()
