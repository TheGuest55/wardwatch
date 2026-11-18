"""Incidents / live: verdict counts, timeline, explanation and hashed evidence."""
import pandas as pd
import streamlit as st

from src.dashboard import incidents, loaders, ui

root, view, df, reason = ui.page("Incidents / live")
r = loaders.load_incidents(root)
FORMAT = """One JSON object per line in `live/incidents.jsonl`:
`incident_id`, `target_host`, `source`, `first_seen`, `last_seen`, `verdict` (one of the six below),
`top_features` [{feature, text, delta}], `pcap_slice`, `pcap_sha256`, `n_alerts`;
optional: `session`, `cell` (keeps the two "Detectors disagree" cells apart), `scope`, `attack_class` (the NAMED type, Known
attack only), `calibrated_p`, `timeline` [{t, event}]."""
if not r.ok:
    ui.na(r.reason, "incidents")
    with st.expander("Expected incident format"):
        st.markdown(FORMAT)
    st.markdown("**Verdicts shown here once incidents exist**")
    st.markdown("\n".join(f"- {v}" for v in incidents.VERDICTS))
    st.stop()

inc = r.data
sessions = ["All"] + (sorted(inc["session"].dropna().astype(str).unique()) if "session" in inc else [])
sel = st.selectbox("Session", sessions, help="Replayed held-out session; All shows every incident.")
if sel != "All":
    inc = inc[inc["session"].astype(str) == sel].reset_index(drop=True)
if inc.empty:
    ui.na(f"no incidents for session {sel}")
    st.stop()
if r.data.attrs.get("bad_lines"):
    st.warning(f"{r.data.attrs['bad_lines']} line(s) in incidents.jsonl could not be read and were skipped.")
st.subheader("Verdicts")
counts, cross = incidents.verdict_counts(inc)
st.dataframe(counts.to_frame(), use_container_width=True)
if cross is not None:
    st.caption("Incidents per NAMED attack class (Known attack only; the model's name, not ground truth) and verdict.")
    st.dataframe(cross, use_container_width=True)

st.subheader("Incident list")
cols = [c for c in ("incident_id", "session", "verdict", "cell", "target_host", "source", "first_seen", "last_seen", "n_alerts")
        if c in inc]
st.dataframe(inc[cols], use_container_width=True, hide_index=True)

pick = st.selectbox("Incident", list(inc["incident_id"]))
row = inc[inc["incident_id"] == pick].iloc[0]
st.markdown(f"**{row.get('verdict', 'unknown')}** · target `{row.get('target_host')}` · source "
            f"`{row.get('source')}` · {row.get('first_seen')} → {row.get('last_seen')}")
tl = row.get("timeline")
if isinstance(tl, list) and tl:
    st.markdown("**Timeline**")
    st.dataframe(pd.DataFrame(tl), use_container_width=True, hide_index=True)
st.markdown("**Why it was flagged**")
tf = row.get("top_features")
if isinstance(tf, list) and tf:
    st.markdown("\n".join("- " + (f.get("text") or f.get("feature") if isinstance(f, dict) else str(f)) for f in tf))
else:
    ui.na("no explanation attached to this incident")
st.markdown("**Evidence**")
status, detail = incidents.verify_hash(row.get("pcap_slice"), row.get("pcap_sha256"), root)
st.code(f"slice:  {row.get('pcap_slice')}\nsha256: {row.get('pcap_sha256')}")
{"verified": st.success, "MISMATCH": st.error, "refused": st.error}.get(status, st.info)(
    f"hash {status}" + ("" if status == "verified" else f": {detail}"))
