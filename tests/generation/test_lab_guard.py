"""Regression test for review M1 isolation guards (scripts/lib/lab_guard.sh).

Exercises the predicates run_session.sh uses to refuse starting on top of a lingering stack or
alongside a second capture. docker is stubbed; the pcap check uses a temp dir. Runs on the host.
"""
import os
import pathlib
import subprocess
import tempfile
import textwrap
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
GUARD = REPO / "scripts" / "lib" / "lab_guard.sh"


def _run(body, env=None):
    """Source the guard lib, run `body`, return the shell exit code."""
    script = f'. "{GUARD}"\n{body}\n'
    e = {**os.environ, **(env or {})}
    return subprocess.run(["bash", "-c", script], cwd=REPO, env=e,
                          capture_output=True, text=True).returncode


def _docker_stub(dirpath, prints):
    p = pathlib.Path(dirpath) / "docker"
    p.write_text(f'#!/bin/sh\nprintf "%s" "{prints}"\n')
    p.chmod(0o755)
    return str(p)


class TestLabGuard(unittest.TestCase):
    def test_pcap_changed_within_true_for_fresh_file(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "ward.pcap").write_text("x")
            self.assertEqual(_run(f'pcap_changed_within "{d}" 300'), 0)

    def test_pcap_changed_within_false_for_old_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "ward.pcap"
            f.write_text("x")
            subprocess.run(["touch", "-d", "2 hours ago", str(f)], check=True)
            self.assertEqual(_run(f'pcap_changed_within "{d}" 300'), 1)

    def test_pcap_changed_within_false_for_missing_dir(self):
        self.assertEqual(_run('pcap_changed_within /no/such/dir 300'), 1)

    def test_project_containers_present_true_when_docker_lists_ids(self):
        with tempfile.TemporaryDirectory() as d:
            stub_dir = pathlib.Path(d) / "bin"
            stub_dir.mkdir()
            _docker_stub(stub_dir, "abc123\\ndef456")
            self.assertEqual(_run('project_containers_present',
                                  env={"DOCKER": str(stub_dir / "docker")}), 0)

    def test_project_containers_present_false_when_none(self):
        with tempfile.TemporaryDirectory() as d:
            stub_dir = pathlib.Path(d) / "bin"
            stub_dir.mkdir()
            _docker_stub(stub_dir, "")
            self.assertEqual(_run('project_containers_present',
                                  env={"DOCKER": str(stub_dir / "docker")}), 1)

    def test_run_session_sources_guard_and_tears_down(self):
        # The runner must source the guard, stop the whole stack (not just capture), and down it.
        src = (REPO / "scripts" / "run_session.sh").read_text()
        self.assertIn("lab_guard.sh", src)
        self.assertIn("project_containers_present", src)
        self.assertIn("docker compose stop", src)
        self.assertIn("docker compose down --remove-orphans", src)
        # syntactically valid bash
        self.assertEqual(subprocess.run(["bash", "-n", str(REPO / "scripts" / "run_session.sh")]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
