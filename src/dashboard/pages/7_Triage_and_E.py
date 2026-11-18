"""Triage and Experiment E: offline triage of the held-out sessions, the withheld-attack analysis, the grouping
diagnosis and the live replay of the deployed pair (XGBoost + Isolation Forest). Records only; nothing is computed."""
import streamlit as st

from src.dashboard import triage_view as tv, ui

root, view, df, reason = ui.page("Triage and Experiment E")
st.caption("The deployed pair: XGBoost (calibrated, banded) + Isolation Forest, scored per host per 5 s window. Everything "
           "here is read from logged records (runs/triage, runs/expE); a missing file shows as not available.")

tab_t, tab_b, tab_g, tab_e = st.tabs(["Offline triage (A)", "Withheld attacks (B)", "Grouping diagnosis", "Experiment E (replay)"])

with tab_t:
    a = tv.load_triage_record("triage_A", root)
    if not a.ok:
        ui.na(a.reason, "offline triage")
    else:
        rec = a.data
        st.caption(tv.provenance_line(rec))
        for s in tv.sessions_in(rec):
            st.subheader(f"Held-out session {s}")
            st.markdown("**Verdict cells per class** (victim and boundary rows counted separately)")
            st.dataframe(tv.cell_table(rec, s), use_container_width=True)
            st.markdown("**Detection** (benign and attack windows)")
            st.dataframe(tv.detection_table(rec, s), use_container_width=True, hide_index=True)
            g = tv.grouping_summary(rec, s)
            st.markdown("**Incident grouping**")
            st.json(g, expanded=True)
            st.markdown("**Time to detect** (end of the first alerting attack window minus the run start)")
            st.dataframe(tv.ttd_table(rec, s), use_container_width=True, hide_index=True)
            st.markdown("**False-novel rate** (benign windows of the event host)")
            st.dataframe(tv.false_novel_table(rec, s), use_container_width=True, hide_index=True)
        st.markdown("**Grouping key and gap, chosen on validation only**")
        st.dataframe(tv.grouping_grid(rec), use_container_width=True, hide_index=True)
        if tv.validation_tied(rec):
            st.warning("Every validation setting has the same accuracy, so the selection had no signal: the tie-break "
                       "(simpler key, smaller gap) chose the setting. The rule was not changed after test.")
        ui.provenance(df, label="Provenance: model records (triage and E records are excluded from the headline tables)")

with tab_b:
    b = tv.load_triage_record("triage_B_withheld", root)
    if not b.ok:
        ui.na(b.reason, "withheld-attack analysis")
    else:
        st.caption(tv.provenance_line(b.data))
        st.markdown("Share of a withheld class's attack windows landing in each cell (mean ± std over 5 seeds). "
                    "XGBoost bands use isotonic calibration fitted on that fold's validation rows; Isolation Forest uses the fold's decision.")
        st.dataframe(tv.withheld_table(b.data), use_container_width=True, hide_index=True)
        loud = tv.loud_known(b.data)
        if loud:
            st.warning("Caveat: " + ", ".join(loud) + " land in \"Known attack (type named)\" at ~1.0. XGBoost still calls these loud "
                       "attacks confidently although it never saw the class, so the NAMED type is necessarily wrong. This is a limitation "
                       "of the verdict table for novel-but-loud attacks, not a success of novelty detection.")

with tab_g:
    d = tv.load_triage_record("grouping_diagnosis", root)
    if not d.ok:
        ui.na(d.reason, "grouping diagnosis")
    else:
        rec = d.data
        st.caption(tv.provenance_line(rec) + " · read-only: no rule changed; roles are ground truth, evaluation only")
        by_cause, by_trans = tv.diagnosis_totals(rec)
        st.markdown("**Extra incidents by cause** (attribution = the incident's source differs from the run's main incident)")
        st.json(by_cause, expanded=True)
        st.markdown("**By role transition** (main source role → extra source role)")
        st.json(by_trans, expanded=False)
        for s in tv.sessions_in(rec):
            st.subheader(s)
            st.dataframe(tv.diagnosis_runs(rec, s), use_container_width=True, hide_index=True)
            st.markdown("**False-alarm incidents** (no attack window)")
            st.json(tv.false_alarms(rec, s), expanded=False)

with tab_e:
    st.caption("Replay of a recorded held-out session at real speed through the deployed pair; not a live capture.")
    recs = tv.load_e_records(root)
    for s, r in recs.items():
        st.subheader(f"Session {s}")
        if not r.ok:
            ui.na(r.reason, f"Experiment E {s}")
            continue
        rec = r.data
        st.caption(tv.provenance_line(rec))
        summ = tv.e_summary(rec)
        over, budget = summ["ticks over budget"], summ["budget ms"]
        st.markdown(f"**Latency of the timed path** (budget {budget} ms per window)")
        st.dataframe(tv.e_latency_table(rec), use_container_width=True, hide_index=True)
        (st.warning if over else st.success)(f"{over} tick(s) over the {budget} ms budget")
        st.json(summ, expanded=False)
        st.markdown("**Detection**")
        st.dataframe(tv.e_detection(rec), use_container_width=True, hide_index=True)
        st.markdown("**Time to detect: offline vs live**")
        st.dataframe(tv.e_ttd(rec), use_container_width=True, hide_index=True)
        st.markdown("**Equivalence with the offline run**")
        st.json(tv.e_equivalence(rec), expanded=True)
