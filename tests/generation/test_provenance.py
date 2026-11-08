"""Regression test for review M9: a session's config.json records the git commit and the digest of
each built image, so provenance is recoverable (tags alone were all 0.1.0). docker is stubbed.
"""
import json
import os
import pathlib
import tempfile
import unittest

import record_provenance  # scripts/ on PYTHONPATH (added by the test runner path below)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _docker_stub(dirpath):
    p = pathlib.Path(dirpath) / "docker"
    # `docker image inspect <name> --format {{.Id}}` -> a deterministic fake digest per name.
    p.write_text('#!/bin/sh\nprintf "sha256:deadbeef-%s" "$3"\n')
    p.chmod(0o755)
    return str(p)


class TestProvenance(unittest.TestCase):
    def test_records_commit_and_digests(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = pathlib.Path(d) / "config.json"
            cfg.write_text(json.dumps({"session": "r01", "images": {
                "device": "wardwatch-device:0.1.0", "attacker": "wardwatch-attacker:0.1.0"}}))
            out = record_provenance.record(str(cfg), "abc123", docker=_docker_stub(d))
            self.assertEqual(out["git_commit"], "abc123")
            self.assertEqual(len(out["image_digests"]), 2)
            self.assertIn("wardwatch-device:0.1.0", out["image_digests"])
            # persisted to disk
            self.assertEqual(json.loads(cfg.read_text())["git_commit"], "abc123")

    def test_missing_image_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = pathlib.Path(d) / "config.json"
            cfg.write_text(json.dumps({"images": {"x": "nope:0"}}))
            # a docker that fails for every image
            stub = pathlib.Path(d) / "docker"
            stub.write_text("#!/bin/sh\nexit 1\n"); stub.chmod(0o755)
            out = record_provenance.record(str(cfg), "c", docker=str(stub))
            self.assertEqual(out["image_digests"], {})
            self.assertEqual(out["git_commit"], "c")

    def test_run_session_invokes_recorder(self):
        self.assertIn("record_provenance.py", (REPO / "scripts/run_session.sh").read_text())


if __name__ == "__main__":
    unittest.main()
