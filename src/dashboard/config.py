"""Where the dashboard reads from. Everything is configurable by environment variable.

WARDWATCH_DATA_ROOT     processed corpus root (default: the MAIN checkout's data/processed/v2, because a
                        worktree has no data/ of its own)
WARDWATCH_RESULTS_ROOT  optional extra run directory (default: <repo>/results/runs, if it exists)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _main_checkout(repo=REPO):
    parts = repo.parts
    if ".worktrees" in parts:
        return Path(*parts[:parts.index(".worktrees")])
    return repo


def data_root():
    env = os.environ.get("WARDWATCH_DATA_ROOT")
    return Path(env) if env else _main_checkout() / "data" / "processed" / "v2"


def run_roots(root=None):
    """Directories searched (recursively) for run records. results/runs/<model>/ is optional."""
    root = Path(root) if root is not None else data_root()
    roots = [root / "runs"]
    extra = os.environ.get("WARDWATCH_RESULTS_ROOT")
    for cand in ([Path(extra)] if extra else [REPO / "results" / "runs", _main_checkout() / "results" / "runs"]):
        if cand.is_dir() and cand not in roots:
            roots.append(cand)
    return roots


def incidents_path(root=None):
    return (Path(root) if root is not None else data_root()) / "live" / "incidents.jsonl"
