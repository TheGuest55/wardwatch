"""Overview: dataset summary and one headline row per model. Entry point: streamlit run src/dashboard/app.py"""
import streamlit as st

from src.dashboard import loaders, metrics_view as mv, ui

root, view, df, reason = ui.page("Overview")
st.caption("Headline tables use only `for_report` run records (smoke runs and latency timings excluded). "
           "Mean ± std over seeds, test split. Nothing on this page is recomputed.")

st.subheader("Dataset")
ds = loaders.load_dataset_summary(root)
if ds.ok:
    st.dataframe(ds.data, use_container_width=True)
    st.caption("Rows per class and split in the processed corpus (testbed_windows.parquet): the rows models "
               "train and are scored on. Victim and boundary rows are excluded from it by construction.")
else:
    ui.na(ds.reason, "dataset summary")

st.subheader("Headline: " + ("all rows" if view == "all" else "attacker host excluded"))
if df is None:
    ui.na(reason, "run records")
head = mv.headline(df, view, root)
show = [c for c in head.columns if not c.startswith("_")]
st.dataframe(head[show], use_container_width=True, hide_index=True)
st.caption("A = time-based split (selected configuration); B = mean of per-class leave-one-attack-out F1; "
           "C = as-trained score averaged over both directions. FPR is at 90% recall.")
ui.provenance(df)

st.subheader("Why a cell is empty")
status = loaders.model_status(df)
for m, why in status.items():
    if why:
        st.markdown(f"- **{m}**: {why}")

st.subheader("Selection rule (CLAUDE.md)")
st.caption("Per role, highest binary F1 averaged over Experiments B and C (C as-trained, both directions). "
           "A winner is declared only when every candidate has both inputs.")
sel = mv.selection(head)
for role, s in sel.items():
    st.markdown(f"**{role}**")
    if s["winner"]:
        st.success(f"winner: {s['winner']} (score {s['scores'][s['winner']]:.3f})")
    else:
        st.warning("No winner declared. Missing inputs:")
        st.markdown("\n".join(f"- {m}" for m in s["missing"]) or "- none")
