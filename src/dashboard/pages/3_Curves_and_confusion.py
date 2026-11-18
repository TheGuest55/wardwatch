"""PR/ROC curves and confusion matrices, derived from a run's saved predictions."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import derived, loaders, metrics_view as mv, ui

root, view, df, reason = ui.page("Curves and confusion")
if df is None:
    ui.na(reason, "run records")
    st.stop()
ok = loaders.eligible(df)
ok = ok[ok["experiment"].isin(["A", "B"])]
if ok.empty:
    ui.na("no report-grade Experiment A or B records", "predictions")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
exp = c1.selectbox("Experiment", sorted(ok["experiment"].unique()))
model = c2.selectbox("Model", sorted(ok[ok["experiment"] == exp]["model"].unique()))
sub = ok[(ok["experiment"] == exp) & (ok["model"] == model)]
key = "config" if exp == "A" else "held_out_class"
default = mv.selected_a_config(sub, root) if exp == "A" else None
opts = sorted(sub[key].dropna().unique())
pick = c3.selectbox("Configuration" if exp == "A" else "Held-out class", opts,
                    index=opts.index(default) if default in opts else 0)
sub = sub[sub[key] == pick]
seed = c4.selectbox("Seed", sorted(sub["seed"].unique()))
rec = sub[sub["seed"] == seed].iloc[0]

p = loaders.load_predictions(rec, root)
if not p.ok:
    ui.na(p.reason, f"predictions for {rec['run_name']}")
    ui.provenance(df, only=[model])
    st.stop()
pred = p.data
if view == "noatk":
    na = loaders.drop_attacker(pred)
    if not na.ok:
        ui.na(na.reason, "attacker-excluded view")
        st.stop()
    pred = na.data
ui.derived_note(pred.attrs.get("source") or p.data.attrs.get("source"))
st.caption(f"{rec['run_name']} · {len(pred):,} test rows · {int(pred['y'].sum()):,} attack · "
           f"threshold {rec['threshold']:.3f} (chosen on validation)" if rec["threshold"] == rec["threshold"]
           else f"{rec['run_name']} · {len(pred):,} test rows")

left, right = st.columns(2)
with left:
    c = derived.pr(pred)
    if c.ok:
        st.plotly_chart(px.line(c.data, x="recall", y="precision", title="Precision-recall"), use_container_width=True)
    else:
        ui.na(c.reason, "PR curve")
with right:
    c = derived.roc(pred)
    if c.ok:
        fig = px.line(c.data, x="fpr", y="tpr", title="ROC")
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot"), name="chance"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        ui.na(c.reason, "ROC curve")

st.subheader("Binary confusion matrix at the run's threshold")
cm = derived.confusion_binary(pred)
st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto"), use_container_width=True)

st.subheader("Multiclass confusion (classifiers, Experiment A)")
if loaders.not_applicable(model, "multiclass"):
    st.info(f"multiclass confusion: not applicable. {loaders.not_applicable(model, 'multiclass')}.")
elif exp != "A":
    ui.na("a withheld class cannot be named, so multiclass is Experiment A only")
else:
    mc = derived.confusion_multiclass(pred)
    if mc.ok:
        st.plotly_chart(px.imshow(mc.data, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                                  labels=dict(x="predicted", y="actual")), use_container_width=True)
    else:
        ui.na(mc.reason, "multiclass confusion")
ui.provenance(df, only=[model])
