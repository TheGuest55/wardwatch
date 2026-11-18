"""Why was it flagged: LSTM occlusion, XGBoost/RF TreeSHAP, autoencoder reconstruction error."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import loaders, ui
from src.dashboard.feature_text import plain

root, view, df, reason = ui.page("Explanations")
st.caption("Top contributing features per attack class, in plain language. Each model explains itself its own "
           "way, so the numbers are not comparable across models. All values are read from saved explanation files.")
tab_l, tab_x, tab_a = st.tabs(["LSTM (occlusion)", "XGBoost / Random Forest (TreeSHAP)",
                               "Autoencoder (reconstruction error)"])

with tab_l:
    e = loaders.load_lstm_explanations(root)
    if not e.ok:
        ui.na(e.reason, "LSTM explanations")
    else:
        d = e.data
        st.caption("Computed after the alert, outside the 50 ms scoring path. A feature (or group) is set to its "
                   "training mean over the last windows; the drop in the attack log-odds is its contribution.")
        rows = []
        for cls, c in d["classes"].items():
            v = (c.get("views", {}).get("last3", {}) or {}).get("verdict", {})
            rows.append({"class": cls, "example run": c.get("run_id"), "status": c.get("status"),
                         "top group": v.get("top_group"), "group expected": v.get("top_group_expected"),
                         "top feature": plain(v.get("top_feature")) if v.get("top_feature") else None,
                         "feature expected": v.get("top_feature_expected")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        cls = st.selectbox("Alert", list(d["classes"]))
        c = d["classes"][cls]
        if c.get("status") != "ok":
            ui.na(str(c.get("status")), f"explanation for {cls}")
        else:
            st.markdown(f"Run `{c.get('run_id')}` · host role **{c.get('host_role')}** · attack score "
                        f"{c['score']:.3f} (threshold {c['threshold']:.3f})")
            views = c.get("views", {})
            name = st.radio("Windows occluded", list(views), horizontal=True,
                            format_func=lambda k: {"last3": "last 3 windows", "all12": "all 12 windows"}.get(k, k))
            v = views[name]
            l, r = st.columns(2)
            l.markdown("**Features**")
            l.markdown("\n".join(f"- {s}" for s in v.get("feature_top5", [])) or "not available")
            r.markdown("**Feature groups**")
            r.markdown("\n".join(f"- {s}" for s in v.get("group_top5", [])) or "not available")
            exp = d.get("verdict_rule", {}).get("expected_groups", {}).get(cls)
            if exp:
                st.caption("Groups expected to matter for this class (pre-registered): " + ", ".join(exp))

with tab_x:
    shown = False
    for model in ("xgboost", "random_forest"):
        c = loaders.load_calibration_details(model, root)
        st.subheader(model)
        if not c.ok:
            ui.na(c.reason, "SHAP")
            continue
        shown = True
        det = c.data
        g = det.get("shap_global")
        if g:
            gd = pd.DataFrame(g[:15], columns=["feature", "mean |SHAP|"])
            gd["meaning"] = gd["feature"].map(plain)
            st.plotly_chart(px.bar(gd[::-1], x="mean |SHAP|", y="feature", orientation="h", hover_data=["meaning"],
                                   title="Global top features"), use_container_width=True)
        pc = det.get("shap_per_class_top5") or {}
        if pc:
            st.markdown("**Top 5 per attack class**")
            st.dataframe(pd.DataFrame([{"class": k, "rank": i + 1, "feature": f, "meaning": plain(f), "mean |SHAP|": v}
                                       for k, lst in pc.items() for i, (f, v) in enumerate(lst)]),
                         use_container_width=True, hide_index=True)
        ex = det.get("example_explanations") or {}
        for cls, item in ex.items():
            reasons = item.get("reasons") if isinstance(item, dict) else item
            if isinstance(reasons, list) and reasons and isinstance(reasons[0], dict):
                with st.expander(f"Example alert: {cls}"):
                    st.markdown("\n".join(f"- {r.get('text', r.get('feature'))}" for r in reasons))
        if det.get("shap_sanity"):
            st.caption("SHAP sanity checks (additivity, rank agreement with model importances, no identifier inputs)")
            st.json(det["shap_sanity"], expanded=False)
    if not shown:
        st.caption("TreeSHAP outputs have not been produced yet (runs/calibration/<model>_details.json).")

with tab_a:
    a = loaders.load_ae_details(root)
    if not a.ok:
        ui.na(a.reason, "autoencoder reconstruction error")
    else:
        st.caption("Share of the autoencoder's reconstruction error carried by each feature (seed 0, test rows). "
                   "A single identifier-like feature dominating would be a leakage warning.")
        ad = pd.DataFrame(a.data["per_feature_error_seed0"][:15], columns=["feature", "share of error"])
        ad["meaning"] = ad["feature"].map(plain)
        st.plotly_chart(px.bar(ad[::-1], x="share of error", y="feature", orientation="h", hover_data=["meaning"]),
                        use_container_width=True)
        st.dataframe(ad, use_container_width=True, hide_index=True)
ui.provenance(df)
