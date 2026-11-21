"""Live alerts: the alert stream of a running replay (live) or of a recorded one (playback), refreshed every 2 seconds."""
import pandas as pd
import streamlit as st

from src.dashboard import live, ui

root, view, df, reason = ui.page("Live alerts")
path = f"{root}/live/alerts_stream.jsonl"
HOWTO = ("Start a replay with `make live-demo SESSION=r05 SPEED=10` (see USER_MANUAL, \"Watch alerts live\"): a live replay "
         "when the PCAPs and models are present, otherwise a playback of the recorded stream.")


@st.fragment(run_every="2s")
def show():
    rd = st.session_state.get("live_reader")
    if rd is None or str(rd.path) != path:
        rd = st.session_state["live_reader"] = live.StreamReader(path)
    if not rd.refresh():
        st.info("Waiting for a replay: no alert stream yet. " + HOWTO)
        return
    kind, text = live.banner(rd.header)
    {"success": st.success, "info": st.info, "warning": st.warning}[kind](text)
    c = st.columns(5)
    c[0].metric("Alerts", sum(rd.counts.values()))
    for col, v in zip(c[1:4], live.VERDICTS):
        col.metric(v, rd.counts.get(v, 0))
    c[4].metric("Open incidents", len(rd.open_incidents()))
    st.caption(f"Window time of the latest event: {rd.last_ts or 'none yet'} · {rd.n_events} events"
               + (f" · {rd.bad_lines} unreadable line(s) skipped" if rd.bad_lines else ""))
    st.subheader("Open incidents")
    op = rd.open_incidents()
    if op:
        st.dataframe(pd.DataFrame(op)[[k for k in ("incident_id", "verdict", "cell", "n_alerts", "first_seen", "last_seen")
                                        if k in op[0]]].sort_values("first_seen", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("none open")
    st.subheader("Newest alerts first")
    al = rd.newest_alerts(50)
    if al:
        t = pd.DataFrame(al)
        cols = [k for k in ("ts", "verdict", "class", "p", "host_ip", "incident_id", "cell") if k in t]
        st.dataframe(t[cols], use_container_width=True, hide_index=True)
    else:
        st.caption("no alert yet")


show()
