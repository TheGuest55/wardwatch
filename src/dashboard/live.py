"""Incremental reader for live/alerts_stream.jsonl (written while a replay or a playback runs). Standard library only.

The page keeps one StreamReader in st.session_state: each refresh reads only the bytes added since the last one (tail read), and
restarts when the file shrank or its header changed (a new replay)."""
import json
from collections import deque
from pathlib import Path

# the verdicts an ALERTING window can carry (the stream holds alerts only); the strings are triage.verdict's
VERDICTS = ("Known attack (type named)", "Possible novel attack", "Detectors disagree → review")


class StreamReader:
    def __init__(self, path, keep=200):
        self.path = Path(path)
        self.reset()
        self.keep = keep

    def reset(self):
        self.offset, self.header, self.n_events, self.bad_lines = 0, None, 0, 0
        self.counts, self.alerts, self.last_ts = {}, deque(maxlen=getattr(self, "keep", 200)), None
        self.incidents = {}                                   # incident_id -> latest event
        self.latest = {}                                      # host ip -> its latest alert (the ward map)
        self._carry = b""

    def refresh(self):
        """Read what was appended. Returns False when the file is absent."""
        if not self.path.exists():
            if self.offset or self.header:
                self.reset()
            return False
        size = self.path.stat().st_size
        if size < self.offset:                                # truncated: a new replay started
            self.reset()
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        self.offset += len(data)
        data = self._carry + data
        *lines, self._carry = data.split(b"\n")               # the last piece is an unfinished line (or empty)
        for raw in lines:
            if raw.strip():
                self._take(raw)
        return True

    def _take(self, raw):
        try:
            e = json.loads(raw)
        except ValueError:
            self.bad_lines += 1
            return
        if "header" in e:
            if self.header is not None and e["header"] != self.header:
                self.reset_keep_offset()
            self.header = e["header"]
            return
        self.n_events += 1
        self.last_ts = e.get("ts") or self.last_ts
        ev = e.get("event")
        if ev == "alert":
            v = e.get("verdict", "unknown")
            self.counts[v] = self.counts.get(v, 0) + 1
            self.alerts.append(e)
            if e.get("host_ip"):
                self.latest[e["host_ip"]] = e
        elif ev in ("incident_open", "incident_update", "incident_close"):
            self.incidents[e["incident_id"]] = {**self.incidents.get(e["incident_id"], {}), **e}

    def reset_keep_offset(self):
        off = self.offset
        self.reset()
        self.offset = off

    def newest_alerts(self, n=50):
        return list(self.alerts)[::-1][:n]

    def open_incidents(self):
        return [i for i in self.incidents.values() if i.get("event") != "incident_close"]


def banner(header):
    """(kind, text) for the mode banner, from the stream's header line."""
    if not header:
        return "info", "Live alerts: no stream header yet"
    mode, session = header.get("mode"), header.get("session", "?")
    if mode == "live":
        return "success", f"Live: replaying session {session} from PCAPs"
    if mode == "playback":
        rec = (header.get("recorded_utc") or header.get("started_utc") or "?")[:10]
        sha = (header.get("git_commit") or "unknown")[:7]
        return "info", (f"Playback: recorded alert stream from a real replay of {session} (recorded {rec}, commit {sha})")
    return "warning", f"Alert stream mode {mode!r} for session {session}"


# ---- ward map (display only: the stream's host_ip -> the ward's device, from config/ward.json via common.ip_role_map)
TILE_STYLE = {                                  # latest verdict -> (background, border); text stays dark for contrast
    "Known attack (type named)": ("#f4b6b0", "#b3261e"),
    "Possible novel attack": ("#f9d6a5", "#b26a00"),
    "Detectors disagree → review": ("#fbeea6", "#8a7a00"),
    None: ("#eceff1", "#90a4ae"),
}


def ward_tiles(cfg, latest):
    """Rows of tiles: one row per bed (monitor, pump, wearable), then the nurse station, the broker and, once seen, the attacker.
    Each tile: label, ip, verdict (latest alert's, or None), p, ts, background and border colours. Roles come from the config's
    IP -> role map; nothing here is read back into any decision."""
    from .. import common
    roles = common.ip_role_map(cfg)                           # {ip: broker | nurse | attacker | monitor | pump | wearable}
    by_role = {}
    for ip, r in roles.items():
        by_role.setdefault(r, []).append(ip)

    def tile(label, ip):
        e = latest.get(ip)
        v = e["verdict"] if e else None
        bg, bd = TILE_STYLE.get(v, TILE_STYLE[None])
        return {"label": label, "ip": ip, "verdict": v, "p": (e or {}).get("p"), "ts": (e or {}).get("ts"), "bg": bg, "border": bd}

    bed = lambda ip: int(ip.rsplit(".", 1)[1]) // 10                 # the config's addressing: 10 * bed + 1 / 2 / 3
    ips = {(bed(ip), r): ip for r in ("monitor", "pump", "wearable") for ip in by_role.get(r, [])}
    rows = [[tile(f"bed {b} {k}", ips[(b, k)]) for k in ("monitor", "pump", "wearable")] for b in sorted({b for b, _ in ips})]
    last = [tile("nurse station", by_role["nurse"][0]), tile("broker", by_role["broker"][0])]
    if by_role["attacker"][0] in latest:
        last.append(tile("attacker", by_role["attacker"][0]))
    rows.append(last)
    return rows


def tile_html(t):
    v = t["verdict"] or "no alert yet"
    p = "" if t["p"] is None else f" · p={float(t['p']):.2f}"
    ts = "" if not t["ts"] else f"<br><span style='font-size:0.75em'>{t['ts'][11:19]}</span>"
    return (f"<div style='background:{t['bg']};border:2px solid {t['border']};border-radius:8px;padding:6px 8px;color:#1b1b1b;"
            f"min-height:4.2em'><b>{t['label']}</b><br><span style='font-size:0.8em'>{t['ip']}</span><br>"
            f"<span style='font-size:0.85em'>{v}{p}</span>{ts}</div>")
