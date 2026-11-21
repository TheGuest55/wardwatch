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
