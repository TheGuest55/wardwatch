"""Model comparison: per experiment and per class, LSTM ablations, bootstrap CIs."""
import plotly.express as px
import streamlit as st

from src.dashboard import loaders, metrics_view as mv, ui

root, view, df, reason = ui.page("Model comparison")
if df is None:
    ui.na(reason, "run records")
    st.stop()

tab_a, tab_b, tab_c, tab_ci = st.tabs(["Experiment A", "Experiment B (leave one attack out)",
                                       "Experiment C (cross-dataset)", "Bootstrap CIs"])

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

with tab_ci:
    b = loaders.load_bootstrap(root)
    if not b.ok:
        ui.na(b.reason, "bootstrap CIs")
    else:
        st.caption("Block bootstrap over whole attack runs and benign time chunks (never single windows).")
        for tag, d in b.data.items():
            st.markdown(f"**{tag}**")
            for name, metrics in (d.get("intervals") or {}).items():
                rows = [{"metric": k, "point": v["point"], "lo": v["lo"], "hi": v["hi"]} for k, v in metrics.items()]
                st.markdown(f"_{name}_")
                st.dataframe(rows, use_container_width=True, hide_index=True)
            if d.get("comparisons"):
                st.json(d["comparisons"], expanded=False)
