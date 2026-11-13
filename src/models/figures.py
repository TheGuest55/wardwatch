"""Report figures for the XGBoost / Random Forest results (static PNG + PDF, print-ready).

    python -m src.models.figures

Reads only logged run outputs (never re-trains), so every figure traces to a run record:
  runs/bootstrap/xgb_rf_time_random_details.json   (block-bootstrap CIs)
  runs/calibration/{xgboost,random_forest}_details.json
  runs/predictions/A_{xgboost,random_forest}.parquet
Writes data/processed/v2/figures/*.png (300 dpi) and *.pdf.

Style: validated palette (blue #2a78d6 = XGBoost, orange #eb6834 = Random Forest, the same in every
figure; ordinal blue ramp for the three confidence bands), thin marks, hairline solid grid,
one y-axis per chart, legend whenever there are >= 2 series, values labelled selectively.
"""
import json

import numpy as np
import pandas as pd

from .train_tree import V2

RUNS = V2 / "runs"
OUT = V2 / "figures"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
MODEL_COLOR = {"xgboost": "#2a78d6", "random_forest": "#eb6834"}
MODEL_NAME = {"xgboost": "XGBoost", "random_forest": "Random Forest"}
BAND_COLOR = ("#86b6ef", "#2a78d6", "#104281")     # ordinal: confident benign -> unsure -> attack
BAND_NAME = ("Confident benign", "Unsure", "Confident attack")
CLASS_NAME = {"arp_mitm": "ARP-spoofing MITM", "bruteforce": "Broker brute-force",
              "eavesdrop": "Wildcard eavesdrop", "mqtt_flood": "MQTT flood",
              "pump_cmd": "Forged pump command", "recon_scan": "Network scan",
              "spoof_vitals": "Spoofed vitals", "syn_flood": "SYN flood"}
SINGLE_COL, DOUBLE_COL = 3.5, 7.16                   # IEEE column widths (inches)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
        "legend.frameon": False, "lines.linewidth": 1.5})
    return plt


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    return OUT / f"{name}.png"


def _load(path):
    return json.loads(path.read_text())


# ------------------------------------------------------------------ figures

def fig_split_gap(plt, boot):
    """F1 with 95% block-bootstrap CI: time split vs random split, per model."""
    iv = boot["intervals"]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 1.7))
    rows = [("xgboost", 1.0), ("random_forest", 0.0)]
    split_style = {"A": ("Time split (realistic)", "o", 0.12),
                   "random": ("Random split (conventional)", "s", -0.12)}
    for model, y in rows:
        for split, (label, marker, dy) in split_style.items():
            r = iv[f"{split}_{model}"]["f1"]
            c = MODEL_COLOR[model]
            ax.errorbar(r["point"], y + dy, xerr=[[r["point"] - r["lo"]], [r["hi"] - r["point"]]],
                        fmt=marker, ms=5, color=c, mfc=c if split == "A" else "white", mew=1.2,
                        elinewidth=1.2, capsize=0)
            ax.annotate(f"{r['point']:.3f}", (r["hi"], y + dy), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks([y for _, y in rows], [MODEL_NAME[m] for m, _ in rows])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Binary F1 on test (95% block-bootstrap CI)")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=INK2, mfc=INK2, ms=5, label=split_style["A"][0]),
         Line2D([], [], marker="s", ls="", color=INK2, mfc="white", ms=5, label=split_style["random"][0])]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, handletextpad=0.3)
    return _save(fig, "fig_split_gap")


def fig_model_ci(plt, boot):
    """XGBoost vs RF on Experiment A: F1, precision, recall, PR-AUC with CIs (one 0-1 axis)."""
    iv = boot["intervals"]
    metrics = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"), ("pr_auc", "PR-AUC")]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))
    for j, (k, label) in enumerate(metrics):
        y = len(metrics) - 1 - j
        for model, dy in (("xgboost", 0.13), ("random_forest", -0.13)):
            r = iv[f"A_{model}"][k]
            ax.errorbar(r["point"], y + dy, xerr=[[r["point"] - r["lo"]], [r["hi"] - r["point"]]],
                        fmt="o", ms=4.5, color=MODEL_COLOR[model], elinewidth=1.2, capsize=0,
                        label=MODEL_NAME[model] if j == 0 else None)
    ax.set_yticks(range(len(metrics)), [lbl for _, lbl in metrics][::-1])
    ax.set_ylim(-0.6, len(metrics) - 0.4)
    ax.set_xlim(0.78, 1.0)
    ax.set_xlabel("Experiment A test score (95% block-bootstrap CI)")
    ax.grid(axis="y", visible=False)
    d = boot["comparisons"]["A_xgboost - A_random_forest"]
    ax.set_title(f"XGBoost − RF: F1 {d['f1']['diff']:+.3f} [{d['f1']['lo']:+.3f}, {d['f1']['hi']:+.3f}]"
                 f",  PR-AUC {d['pr_auc']['diff']:+.3f} [{d['pr_auc']['lo']:+.3f}, "
                 f"{d['pr_auc']['hi']:+.3f}]", fontsize=6.5, color=INK2, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handletextpad=0.2)
    return _save(fig, "fig_model_ci")


def fig_reliability(plt, cal):
    """Reliability diagram on test, before vs after calibration (XGBoost)."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8, zorder=1)
    ax.annotate("perfect calibration", (0.40, 0.33), rotation=38, fontsize=6, color=MUTED)
    series = (("reliability_test_raw", "Raw score", "#eb6834"),
              ("reliability_test_calibrated", f"Calibrated ({cal['calibration_method']})", "#2a78d6"))
    for key, label, c in series:
        bins = [b for b in cal[key]["bins"] if b["n"] >= 10]
        x = [b["mean_predicted"] for b in bins]
        y = [b["observed_attack_rate"] for b in bins]
        ax.plot(x, y, "-o", color=c, ms=4, mec="white", mew=0.8, label=f"{label}, "
                f"ECE {cal[key]['ece']:.4f}", zorder=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted attack probability (bin mean)")
    ax.set_ylabel("Observed attack rate")
    ax.legend(loc="upper left")
    ax.text(1.0, 0.02, "bins with ≥ 10 test windows", ha="right", fontsize=6, color=MUTED)
    return _save(fig, "fig_reliability_xgboost")


def fig_bands_per_class(plt, cal, model):
    """Where each attack class's test windows land: confident benign / unsure / confident attack."""
    per = cal["bands_test_per_class"]
    order = sorted(per, key=lambda c: per[c]["confident_attack"] / per[c]["n"])
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    keys = ("confident_benign", "unsure", "confident_attack")
    for i, c in enumerate(order):
        left = 0.0
        for k, color in zip(keys, BAND_COLOR):
            w = per[c][k] / per[c]["n"]
            if w > 0:
                ax.barh(i, w, left=left, height=0.62, color=color, edgecolor="white", linewidth=1.0)
            left += w
        ax.text(1.01, i, f"{per[c]['confident_attack']}/{per[c]['n']}", va="center", fontsize=6.5,
                color=INK2)
    ax.set_yticks(range(len(order)), [CLASS_NAME.get(c, c) for c in order])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of the class's attack windows (test)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(BAND_COLOR, BAND_NAME)],
              loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3, handlelength=1, columnspacing=0.8)
    ax.text(1.01, len(order) - 0.35, "conf. attack / all", fontsize=6, color=MUTED)
    return _save(fig, f"fig_bands_per_class_{model}")


def fig_shap_global(plt, cal, model, k=12):
    rank = cal["shap_global"][:k][::-1]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    ax.barh([f for f, _ in rank], [v for _, v in rank], height=0.6, color=MODEL_COLOR[model])
    ax.set_xlabel("Mean |SHAP| on test (log-odds)" if model == "xgboost"
                  else "Mean |SHAP| on test (probability)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{MODEL_NAME[model]}: top {k} features overall", loc="left", fontsize=8)
    return _save(fig, f"fig_shap_global_{model}")


def fig_shap_per_class(plt, cal, model):
    """Heatmap: which features drive each attack class (share of the class's top-5 |SHAP|)."""
    per = cal["shap_per_class_top5"]
    classes = sorted(per)
    feats = []
    for c in classes:
        for f, _ in per[c][:3]:
            if f not in feats:
                feats.append(f)
    M = np.zeros((len(classes), len(feats)))
    for i, c in enumerate(classes):
        tot = sum(v for _, v in per[c]) or 1.0
        for f, v in per[c]:
            if f in feats:
                M[i, feats.index(f)] = v / tot
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "blue_seq", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(min(DOUBLE_COL, 2.2 + 0.36 * len(feats)), 2.6))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(0.6, M.max()), aspect="auto")
    ax.set_xticks(range(len(feats)), feats, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), [CLASS_NAME.get(c, c) for c in classes])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(feats)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(classes)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(len(classes)):                     # label only each class's strongest driver
        j = int(M[i].argmax())
        ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=5.5,
                color="white" if M[i, j] > 0.35 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=6, color=MUTED)
    cb.outline.set_visible(False)
    cb.set_label("share of class's top-5 |SHAP|", fontsize=6.5, color=INK2)
    ax.set_title(f"{MODEL_NAME[model]}: drivers per attack class", loc="left", fontsize=8)
    return _save(fig, f"fig_shap_per_class_{model}")


def fig_pr_curves(plt, preds):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    for model, df in preds.items():
        y, s, p = df["y"].to_numpy(), df["score"].to_numpy(), df["pred"].to_numpy()
        pr, rc, _ = precision_recall_curve(y, s)
        c = MODEL_COLOR[model]
        ax.plot(rc, pr, color=c, label=f"{MODEL_NAME[model]} (PR-AUC {average_precision_score(y, s):.3f})")
        tp = ((p == 1) & (y == 1)).sum()
        op_p, op_r = tp / max(p.sum(), 1), tp / max(y.sum(), 1)
        ax.plot(op_r, op_p, "o", ms=6, color=c, mec="white", mew=1.2, zorder=4)
    ax.annotate("● operating point\n(threshold set on validation)", (0.02, 0.06),
                xycoords="axes fraction", fontsize=6, color=MUTED)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.16))
    return _save(fig, "fig_pr_curves_expA")


def fig_confusion(plt, df, model):
    y, p = df["y"].to_numpy(), df["pred"].to_numpy()
    M = np.array([[((y == 0) & (p == 0)).sum(), ((y == 0) & (p == 1)).sum()],
                  [((y == 1) & (p == 0)).sum(), ((y == 1) & (p == 1)).sum()]])
    rate = M / M.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(2.4, 2.1))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#ffffff", "#cde2fb", "#2a78d6", "#0d366b"])
    ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:,}\n{rate[i, j]:.1%}", ha="center", va="center", fontsize=7,
                    color="white" if rate[i, j] > 0.5 else INK)
    ax.set_xticks([0, 1], ["Benign", "Attack"])
    ax.set_yticks([0, 1], ["Benign", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{MODEL_NAME[model]}, Experiment A test", loc="left", fontsize=8)
    return _save(fig, f"fig_confusion_expA_{model}")


def fig_expd_iat(plt):
    """Experiment D: macro-F1 on CIC's official split with vs without the IAT column (XGBoost),
    against the published best (RF, Dadkhah et al. Table 8)."""
    from .exp_d import PUBLISHED
    full = _load(RUNS / "expD" / "xgboost.json")["metrics"]
    noiat = _load(RUNS / "expD" / "xgboost_noiat.json")["metrics"]
    tasks = [("2", "Binary"), ("6", "6 categories"), ("19", "19 classes")]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 1.9))
    for j, (t, label) in enumerate(tasks):
        y = len(tasks) - 1 - j
        pub = PUBLISHED[t]["RF"][3]
        a, b = full[f"{t}c_f1_macro_mean"], noiat[f"{t}c_f1_macro_mean"]
        ax.plot([b, a], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.plot(pub, y, "|", ms=11, mew=1.6, color=INK2, zorder=2, label="Published best (RF)" if j == 0 else None)
        ax.plot(a, y, "o", ms=6, color="#2a78d6", mec="white", mew=1, zorder=3,
                label="XGBoost, all 45 features" if j == 0 else None)
        ax.plot(b, y, "o", ms=6, color="#eb6834", mec="white", mew=1, zorder=3,
                label="XGBoost, IAT removed" if j == 0 else None)
        ax.annotate(f"{b - a:+.2f}", (b, y), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(tasks)), [lbl for _, lbl in tasks][::-1])
    ax.set_ylim(-0.6, len(tasks) - 0.4)
    ax.set_xlim(0.62, 1.0)
    ax.set_xlabel("Macro-F1 on CICIoMT2024 official test split")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=3, handletextpad=0.2,
              columnspacing=0.8, fontsize=6.3)
    return _save(fig, "fig_expD_iat")


def main():
    plt = _style()
    boot = _load(RUNS / "bootstrap" / "xgb_rf_time_random_details.json")
    cal = {m: _load(RUNS / "calibration" / f"{m}_details.json") for m in MODEL_COLOR}
    preds = {m: pd.read_parquet(RUNS / "predictions" / f"A_{m}.parquet") for m in MODEL_COLOR}
    made = [fig_split_gap(plt, boot), fig_model_ci(plt, boot), fig_reliability(plt, cal["xgboost"]),
            fig_pr_curves(plt, preds)]
    for m in MODEL_COLOR:
        made += [fig_bands_per_class(plt, cal[m], m), fig_shap_global(plt, cal[m], m),
                 fig_shap_per_class(plt, cal[m], m), fig_confusion(plt, preds[m], m)]
    if (RUNS / "expD" / "xgboost_noiat.json").exists():
        made.append(fig_expd_iat(plt))
    plt.close("all")
    for p in made:
        print(f"wrote {p} (+ .pdf)")


if __name__ == "__main__":
    main()
