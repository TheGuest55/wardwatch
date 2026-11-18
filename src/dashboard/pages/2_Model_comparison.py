"""Model comparison: per experiment and per class, LSTM ablations, bootstrap CIs."""
import plotly.express as px
import streamlit as st

from src.dashboard import loaders, metrics_view as mv, ui

root, view, df, reason = ui.page("Model comparison")
if df is None:
    ui.na(reason, "run records")
    st.stop()

tab_a, tab_b, tab_c, tab_s, tab_d, tab_ci = st.tabs(["Experiment A", "Experiment B (leave one attack out)",
                                                    "Experiment C (cross-dataset)", "Frame-size sensitivity",
                                                    "Experiment D (indicative)", "Bootstrap CIs"])

with tab_a:
    t = mv.a_by_config(df, view)
    if t.empty:
        ui.na("no report-grade Experiment A records", "Experiment A")
    else:
        st.dataframe(t[[c for c in t.columns if not c.startswith("_")]], use_container_width=True, hide_index=True)
        fig = px.bar(t, x="config", y="_F1", color="model", barmode="group", labels={"_F1": "F1 (mean over seeds)"})
        st.plotly_chart(fig, use_container_width=True)
        lstm = t[t["model"] == "lstm"]
        if len(lstm) > 1:
            st.markdown("**LSTM ablations** (k = windows of history; *no is_present* = presence flag removed)")
            st.dataframe(lstm[["config", "seeds", "F1", "PR-AUC", "FPR @ 90% recall", "macro-F1"]],
                         use_container_width=True, hide_index=True)
    ui.provenance(df, only=None)

with tab_b:
    t = mv.b_per_class(df, view)
    if t.empty:
        ui.na("no report-grade Experiment B records", "Experiment B")
    else:
        st.caption("F1 on the test split with one attack class withheld from training and validation; "
                   "stolen-credential classes (eavesdrop, pump_cmd, spoof_vitals) are expected to be hardest.")
        st.dataframe(t, use_container_width=True)
    ui.provenance(df)

with tab_c:
    t = mv.c_by_direction(df, view)
    if t.empty:
        ui.na("no report-grade Experiment C records (only smoke runs, if any)", "Experiment C")
    else:
        st.dataframe(t, use_container_width=True, hide_index=True)
    ui.provenance(df)

with tab_s:
    t = mv.c_sensitivity(df, view)
    st.caption("Sensitivity analysis, NOT a selection input: Experiment C as trained (mean of both directions) with and without "
               "the seven frame-size features. These records are excluded from every headline table.")
    if t.empty:
        ui.na("no frame-size sensitivity records", "frame-size sensitivity")
    else:
        st.dataframe(t, use_container_width=True, hide_index=True)
    ui.provenance(df, label="Provenance: records behind the other tabs (sensitivity records appear as excluded)")

with tab_d:
    st.caption("Experiment D (indicative benchmark): CIC's own feature CSVs and official train/test split, compared with the published "
               "baselines (Dadkhah et al., Table 8). NOT comparable to Experiments A-C (different features, no host column, no "
               "attacker-excluded view) and never part of the headline or the selection. Mean ± std over seeds.")
    t = mv.d_table(df)
    if t.empty:
        ui.na("no report-grade Experiment D records (runs/expD/)", "Experiment D")
    else:
        st.dataframe(t, use_container_width=True, hide_index=True)
        st.caption("Without IAT = the leakage re-run (working rule 3): IAT is the known time-proxy suspect in CIC's sister dataset.")
    ui.provenance(df, label="Provenance: records behind the other tabs (D records appear as excluded from the headline)")

with tab_ci:
    b = loaders.load_bootstrap(root)
    if not b.ok:
        ui.na(b.reason, "bootstrap CIs")
    else:
        st.caption("Block bootstrap over whole attack runs and benign time chunks (never single windows).")
        for tag, d in b.data.items():
            iv = d.get("intervals") if isinstance(d, dict) else None
            if not iv:
                continue                                       # e.g. a run record (selection.json) with no intervals
            st.markdown(f"**{tag}**")
            flat = {k: v for k, v in iv.items() if isinstance(v, dict) and "point" in v}
            if flat:                                           # {"B/xgboost": {point, lo, hi}, ...} (selection_details)
                rows = [{"statistic / model": k, "point": v["point"], "lo": v["lo"], "hi": v["hi"]} for k, v in flat.items()]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            for name, metrics in iv.items():                   # {"name": {"metric": {point, lo, hi}}} (bootstrap_ci)
                if name in flat or not isinstance(metrics, dict):
                    continue
                rows = [{"metric": k, "point": v["point"], "lo": v["lo"], "hi": v["hi"]} for k, v in metrics.items()
                        if isinstance(v, dict) and "point" in v]
                st.markdown(f"_{name}_")
                st.dataframe(rows, use_container_width=True, hide_index=True)
            for key in ("comparisons", "differences"):
                if isinstance(d, dict) and d.get(key):
                    st.markdown(f"_{key}_")
                    st.json(d[key], expanded=False)
