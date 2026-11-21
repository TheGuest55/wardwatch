"""Page 8 (Live alerts) and its incremental stream reader; playback pacing; the alert sink."""
import json

import pytest

from src.dashboard import live
from tests.dashboard.helpers import run

PAGE = "src/dashboard/pages/8_Live_alerts.py"
KNOWN, REVIEW = "Known attack (type named)", "Detectors disagree → review"


def head(mode="playback", **kw):
    return {"header": {"mode": mode, "session": "r05", "speed": 10, "git_commit": "abcdef1234567", "recorded_utc": "2026-11-21T09:00:00Z", **kw}}


def alert(ts, verdict=KNOWN, iid="r05-x-1", cls="syn_flood", p=0.99):
    return {"ts": ts, "event": "alert", "session": "r05", "host_mac": "aa", "host_ip": "172.28.10.10", "verdict": verdict,
            "cell": "known", "class": cls, "p": p, "incident_id": iid}


def inc(ev, ts, iid="r05-x-1", n=1):
    return {"ts": ts, "event": ev, "session": "r05", "incident_id": iid, "verdict": KNOWN, "cell": "known", "n_alerts": n,
            "first_seen": "2026-11-09T00:00:00Z", "last_seen": ts}


def write(root, lines, mode="w"):
    d = root / "live"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "alerts_stream.jsonl", mode) as f:
        for l in lines:
            f.write((l if isinstance(l, str) else json.dumps(l)) + "\n")


def texts(at):
    return " | ".join([*(x.value for x in at.info), *(x.value for x in at.success), *(x.value for x in at.warning),
                       *(x.value for x in at.markdown), *(m.label + "=" + str(m.value) for m in at.metric)])


def test_waiting_state_when_there_is_no_stream(tmp_path, monkeypatch):
    at = run(PAGE, tmp_path, monkeypatch)
    assert not at.exception
    assert "Waiting for a replay" in texts(at)


def test_playback_banner_names_session_date_and_commit(tmp_path, monkeypatch):
    write(tmp_path, [head("playback"), alert("2026-11-09T00:00:05Z")])
    at = run(PAGE, tmp_path, monkeypatch)
    assert not at.exception
    assert "Playback: recorded alert stream from a real replay of r05 (recorded 2026-11-21, commit abcdef1)" in texts(at)


def test_live_banner(tmp_path, monkeypatch):
    write(tmp_path, [head("live", recorded_utc=None), alert("2026-11-09T00:00:05Z")])
    at = run(PAGE, tmp_path, monkeypatch)
    assert "Live: replaying session r05 from PCAPs" in texts(at)


def test_counts_newest_first_and_open_incidents(tmp_path, monkeypatch):
    write(tmp_path, [head(), alert("2026-11-09T00:00:05Z"), inc("incident_open", "2026-11-09T00:00:05Z"),
                     alert("2026-11-09T00:00:10Z", REVIEW, "r05-y-2"), alert("2026-11-09T00:00:15Z"),
                     inc("incident_update", "2026-11-09T00:00:15Z", n=2), inc("incident_close", "2026-11-09T00:01:00Z", "r05-y-2")])
    at = run(PAGE, tmp_path, monkeypatch)
    vals = {m.label: m.value for m in at.metric}
    assert vals["Alerts"] == "3" or vals["Alerts"] == 3
    assert str(vals[KNOWN]) == "2" and str(vals[REVIEW]) == "1"
    assert str(vals["Open incidents"]) == "1"                       # r05-x-1 open; r05-y-2 closed
    newest = at.dataframe[-1].value                                  # the alerts table
    assert list(newest["ts"])[0] == "2026-11-09T00:00:15Z"           # newest first
    assert list(newest["ts"]) == sorted(newest["ts"], reverse=True)


def test_two_reads_show_a_growing_count(tmp_path, monkeypatch):
    write(tmp_path, [head(), alert("2026-11-09T00:00:05Z")])
    at = run(PAGE, tmp_path, monkeypatch)
    assert str({m.label: m.value for m in at.metric}["Alerts"]) == "1"
    write(tmp_path, [alert("2026-11-09T00:00:10Z"), alert("2026-11-09T00:00:15Z")], mode="a")
    at.run()
    assert str({m.label: m.value for m in at.metric}["Alerts"]) == "3"


class TestReader:
    def test_reads_only_appended_bytes_and_keeps_a_partial_line(self, tmp_path):
        write(tmp_path, [head(), alert("2026-11-09T00:00:05Z")])
        r = live.StreamReader(tmp_path / "live" / "alerts_stream.jsonl")
        assert r.refresh() and r.n_events == 1
        off = r.offset
        with open(tmp_path / "live" / "alerts_stream.jsonl", "a") as f:
            f.write(json.dumps(alert("2026-11-09T00:00:10Z"))[:30])          # a half-written line
        r.refresh()
        assert r.n_events == 1 and r.offset > off
        with open(tmp_path / "live" / "alerts_stream.jsonl", "a") as f:
            f.write(json.dumps(alert("2026-11-09T00:00:10Z"))[30:] + "\n")
        r.refresh()
        assert r.n_events == 2 and r.bad_lines == 0

    def test_a_truncated_file_restarts_the_reader(self, tmp_path):
        write(tmp_path, [head(), alert("2026-11-09T00:00:05Z"), alert("2026-11-09T00:00:10Z")])
        r = live.StreamReader(tmp_path / "live" / "alerts_stream.jsonl")
        r.refresh()
        assert r.n_events == 2
        write(tmp_path, [head("live"), alert("2026-11-09T00:00:05Z")])
        r.refresh()
        assert r.n_events == 1 and r.header["mode"] == "live"

    def test_bad_lines_are_counted_not_fatal(self, tmp_path):
        write(tmp_path, [head(), "{not json", alert("2026-11-09T00:00:05Z")])
        r = live.StreamReader(tmp_path / "live" / "alerts_stream.jsonl")
        r.refresh()
        assert r.bad_lines == 1 and r.n_events == 1


class TestWardMap:
    CFG = {"network": {"subnet": "172.28.10.0/24"}, "broker": {"ip": "172.28.10.10"}, "nurse": {"ip": "172.28.10.20"},
           "attacker": {"ip": "172.28.10.66"}, "beds": 4}

    def test_tiles_cover_four_beds_nurse_broker_and_the_attacker_only_once_seen(self):
        rows = live.ward_tiles(self.CFG, {})
        assert [len(r) for r in rows] == [3, 3, 3, 3, 2]
        labels = [t["label"] for r in rows for t in r]
        assert labels[:3] == ["bed 1 monitor", "bed 1 pump", "bed 1 wearable"] and "nurse station" in labels and "broker" in labels
        assert "attacker" not in labels
        assert all(t["verdict"] is None for r in rows for t in r)
        seen = live.ward_tiles(self.CFG, {"172.28.10.66": alert("2026-11-09T00:00:05Z") | {"host_ip": "172.28.10.66"}})
        assert seen[-1][-1]["label"] == "attacker"

    def test_each_tile_takes_the_latest_verdict_and_p(self):
        latest = {"172.28.10.21": alert("2026-11-09T00:00:05Z", KNOWN, p=0.91) | {"host_ip": "172.28.10.21"},
                  "172.28.10.22": alert("2026-11-09T00:00:10Z", REVIEW, p=0.2) | {"host_ip": "172.28.10.22"}}
        t = {x["label"]: x for r in live.ward_tiles(self.CFG, latest) for x in r}
        assert t["bed 2 monitor"]["verdict"] == KNOWN and t["bed 2 monitor"]["p"] == 0.91
        assert t["bed 2 pump"]["verdict"] == REVIEW
        assert t["bed 2 monitor"]["bg"] != t["bed 2 pump"]["bg"] != t["bed 1 monitor"]["bg"]

    @pytest.mark.parametrize("mode", ["live", "playback"])
    def test_the_page_shows_the_map_in_both_modes(self, tmp_path, monkeypatch, mode):
        write(tmp_path, [head(mode), alert("2026-11-09T00:00:05Z") | {"host_ip": "172.28.10.21"}])
        at = run(PAGE, tmp_path, monkeypatch)
        assert not at.exception
        html = " ".join(m.value for m in at.markdown)
        assert "bed 2 monitor" in html and "bed 4 wearable" in html and "nurse station" in html and "broker" in html
        assert "Known attack (type named) · p=0.99" in html                      # bed 2 monitor's tile (172.28.10.21)
        assert "attacker" not in html.replace("attacker's tile", "")
