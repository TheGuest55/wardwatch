#!/usr/bin/env python3
"""Augment a session's config.json with provenance (review M9).

Tags alone don't distinguish versions (every image was `0.1.0` even though the attacker code changed
between sessions). This records the exact `git_commit` and the content digest (image Id) of each
built image, so a session's code + images are identifiable later. Called by run_session.sh right
after `docker compose up --build`, before the manifest hashes config.json.

Stdlib only. `docker` is a parameter so the writer is testable with a stub.
"""
import json
import subprocess
import sys


def image_id(name, docker="docker"):
    """Return the content digest (image Id) of a local image, or None if absent/unknown."""
    try:
        r = subprocess.run([docker, "image", "inspect", name, "--format", "{{.Id}}"],
                           capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        return out if r.returncode == 0 and out else None
    except Exception:
        return None


def record(config_path, commit, docker="docker"):
    """Add `git_commit` and `image_digests` (name -> Id) to the config.json at config_path."""
    with open(config_path) as f:
        cfg = json.load(f)
    digests = {}
    for name in cfg.get("images", {}).values():
        d = image_id(name, docker)
        if d:
            digests[name] = d
    cfg["git_commit"] = commit
    cfg["image_digests"] = digests
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


if __name__ == "__main__":
    record(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "docker")
