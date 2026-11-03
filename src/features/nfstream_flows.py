"""Optional NFStream enrichment: a flow-level view that contributes a small per-host-window
feature set (active flows, new flows, mean active-flow duration).

NFStream is a soft dependency (see services/analysis/Dockerfile): the per-window features in
windows.py are computed directly from packets and fully cover the modelling granularity, so if
NFStream is missing the pipeline still runs and these columns are left NaN. The caller logs a
[DATA] note when that happens, per the "never quietly drop" rule.
"""
from .. import common

try:
    from nfstream import NFStreamer  # noqa: F401
    AVAILABLE = True
except Exception:  # pragma: no cover - import guarded on purpose
    AVAILABLE = False


def window_flow_features(pcap_path):
    """Return {(mac, window_epoch): {nf_active_flows, nf_new_flows, nf_dur_sum}} for one pcap, or
    None if NFStream is unavailable. Each flow contributes to both its source and destination host
    windows for every window it overlaps; 'new' counts only the window the flow starts in."""
    if not AVAILABLE:
        return None
    from nfstream import NFStreamer

    agg = {}
    streamer = NFStreamer(source=str(pcap_path), statistical_analysis=False, n_dissections=0)
    for flow in streamer:
        start = flow.bidirectional_first_seen_ms / 1000.0
        end = flow.bidirectional_last_seen_ms / 1000.0
        dur = max(0.0, end - start)
        w0 = common.window_start(start)
        w1 = common.window_start(end)
        macs = [m for m in (flow.src_mac, flow.dst_mac) if m]
        win = w0
        while win <= w1:
            is_new = (win == w0)
            for mac in macs:
                cell = agg.setdefault((mac, win), {"nf_active_flows": 0, "nf_new_flows": 0,
                                                   "nf_dur_sum": 0.0})
                cell["nf_active_flows"] += 1
                cell["nf_dur_sum"] += dur
                if is_new:
                    cell["nf_new_flows"] += 1
            win += int(common.WINDOW_SECONDS)
    return agg
