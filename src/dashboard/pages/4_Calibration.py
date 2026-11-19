"""Reliability diagrams and ECE, where calibrated predictions exist (logged calibration runs)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import loaders, ui

root, view, df, reason = ui.page("Calibration")
st.caption("Only models with a logged calibration run appear here (isotonic/Platt fitted on validation, scored "
           "once on test). Raw scores that were never calibrated are not shown as if they were.")
found = False
for model in loaders.MODELS:
    why = loaders.not_applicable(model, "calibration")
    if why:
        st.markdown(f"**{model}**")
        st.info(f"calibration: not applicable. {why}.")
        continue
    d = loaders.load_calibration_details(model, root)
    if not d.ok:
        st.markdown(f"**{model}**")
        skipped = loaders.not_run(model, "calibration")
        if skipped:                                   # deliberately not run (not "missing"): say so
            st.info(f"calibration: not run: {skipped}.")
        else:
            ui.na(d.reason, "calibration")
        continue
    found = True
    det = d.data
    st.subheader(model)
    raw, cal = det.get("reliability_test_raw"), det.get("reliability_test_calibrated")
    cols = st.columns(3)
    cols[0].metric("Method", str(det.get("calibration_method", "not available")))
    if raw and cal:
        cols[1].metric("ECE raw", f"{raw['ece']:.4f}")
        cols[2].metric("ECE calibrated", f"{cal['ece']:.4f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot"), name="perfect"))
        for name, rel in (("raw", raw), ("calibrated", cal)):
            b = pd.DataFrame(rel["bins"]).dropna(subset=["mean_predicted"])
            fig.add_trace(go.Scatter(x=b["mean_predicted"], y=b["observed_attack_rate"], mode="lines+markers",
                                     name=name, text=b["n"], hovertemplate="predicted %{x:.2f}<br>"
                                     "observed %{y:.2f}<br>n=%{text}<extra></extra>"))
        fig.update_layout(xaxis_title="mean predicted attack probability", yaxis_title="observed attack rate",
                          height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        ui.na("details file has no reliability bins", "reliability diagram")
    if det.get("bands_test"):
        st.markdown("Confidence bands on test (cut-offs set on validation)")
        st.json(det["bands_test"], expanded=False)
    if df is not None:
        cal_rec = df[(df["model"] == model) & (df["experiment"] == "triage-calibration")]
        if not cal_rec.empty:
            ui.provenance(cal_rec, f"Provenance: {model} calibration run")
if not found:
    st.warning("No calibration outputs exist yet (runs/calibration/<model>_details.json).")
