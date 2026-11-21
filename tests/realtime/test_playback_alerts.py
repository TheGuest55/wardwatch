"""src/realtime/playback.py (stdlib pacing, header) and src/realtime/alerts.py (the sink); run_e writes nothing extra without one."""
import json
import tempfile
import unittest
from pathlib import Path

from src.realtime import alerts, playback

HEAD = {"mode": "recorded", "session": "r05", "speed": 10, "git_commit": "abc", "recorded_utc": "2026-11-21T09:00:00Z"}
EV = [{"ts": "2026-11-09T00:00:05Z", "event": "alert"}, {"ts": "2026-11-09T00:00:15Z", "event": "alert"},
      {"ts": "2026-11-09T00:00:15Z", "event": "incident_open"}, {"ts": "2026-11-09T00:00:25Z", "event": "incident_close"}]


def write_recording(d):
    p = Path(d) / "rec.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [{"header": HEAD}, *EV]) + "\n")
    return p


class TestPlayback(unittest.TestCase):
    def test_delays_are_window_ts_gaps_divided_by_speed(self):
        self.assertEqual(playback.delays(EV, 10.0), [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(playback.delays(EV, 1.0), [0.0, 10.0, 0.0, 10.0])

    def test_the_ts_format_the_run_writes_is_understood(self):
        ev = [{"ts": "2026-11-09T00:04:00+00:00"}, {"ts": "2026-11-09T00:04:30+00:00"}, {"ts": "2026-11-09T00:04:30Z"}]
        self.assertEqual(playback.delays(ev, 10.0), [0.0, 3.0, 0.0])

    def test_plays_every_event_in_order_with_the_recorded_header_and_mode_playback(self):
        with tempfile.TemporaryDirectory() as d:
            sleeps = []
            dest = Path(d) / "live" / "alerts_stream.jsonl"
            n = playback.play(write_recording(d), dest, 10.0, sleep=sleeps.append)
            lines = [json.loads(l) for l in dest.read_text().splitlines()]
            self.assertEqual(n, 4)
            self.assertEqual(lines[1:], EV)
            h = lines[0]["header"]
            self.assertEqual(h["mode"], "playback")
            self.assertEqual((h["git_commit"], h["recorded_utc"], h["session"]), ("abc", "2026-11-21T09:00:00Z", "r05"))
            self.assertEqual(sleeps, [1.0, 1.0])

    def test_missing_recording_says_where_it_looked(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as e:
                playback.main(["--data-root", d, "--session", "r05"])
            self.assertIn("alerts_stream_r05.jsonl", str(e.exception))


class FakeTracker:
    key = "source"
    open = {}


class FakeTr:
    session = "r05"
    tracker = FakeTracker()
    records = []
    members = {}


class TestSink(unittest.TestCase):
    def test_header_first_and_file_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            p.write_text("old\n")
            s = alerts.AlertSink(p, alerts.header("live", "r05", 10.0))
            s.f.close()
            lines = p.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["header"]["mode"], "live")

    def test_a_non_alerting_window_writes_no_alert(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            s = alerts.AlertSink(Path(d) / "s.jsonl", {"mode": "live"})
            res = {"cell": np.array(["no_alert"], dtype=object), "named": np.array([None], dtype=object), "p": np.array([0.1])}
            s.tick(FakeTr(), 1_700_000_000, ["aa"], ["1.2.3.4"], res, {})
            s.f.close()
            self.assertEqual(len((Path(d) / "s.jsonl").read_text().splitlines()), 1)         # the header only


class TestRunEWithoutSink(unittest.TestCase):
    def test_replay_session_signature_defaults_write_no_alert_stream(self):
        import inspect
        from src.experiments import run_e
        sig = inspect.signature(run_e.replay_session)
        self.assertIsNone(sig.parameters["alerts"].default)
        self.assertTrue(sig.parameters["with_post"].default)


if __name__ == "__main__":
    unittest.main()


from tests.needs import needs  # noqa: E402


@needs("RAW", "MODELS", "DATASET")
class TestDemoWritesOnlyTheStream(unittest.TestCase):
    def test_a_short_demo_writes_alerts_stream_and_nothing_under_runs_or_incidents(self):
        from src.experiments import run_e
        with tempfile.TemporaryDirectory() as d:
            live = Path(d) / "live"
            run_e.demo("r05", speed=0, max_seconds=40, live=live)
            names = sorted(p.relative_to(d).as_posix() for p in Path(d).rglob("*") if p.is_file())
            self.assertEqual(names, ["live/alerts_stream.jsonl"])                       # no slices, no incidents file, no record
            self.assertEqual(json.loads((live / "alerts_stream.jsonl").read_text().splitlines()[0])["header"]["mode"], "live")
