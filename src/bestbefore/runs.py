"""Run records. Every run writes down what it ran, from which commit, with
which seed, so any number in `results/` can be traced back and rerun.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from .config import REPO_ROOT

# Per-image outputs live here, never at the top of results/: the gitignore
# lets .md and .csv files there into git.
RUNS_DIR = REPO_ROOT / "results" / "runs"

# Versions worth recording when installed; the rest of the env rarely matters.
PACKAGES = ("numpy", "pillow", "rapidocr-onnxruntime", "onnxruntime",
            "ultralytics", "torch")


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    return out.stdout.strip()


def _versions() -> dict[str, str]:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def start_run(name: str, config: dict, seed: int | None = None) -> Path:
    """Make `results/runs/<timestamp>-<name>/` holding run.json."""
    now = datetime.now()
    run_dir = RUNS_DIR / f"{now:%Y%m%d-%H%M%S}-{name}"
    run_dir.mkdir(parents=True)
    record = {
        "name": name,
        "started": now.isoformat(timespec="seconds"),
        "command": sys.argv,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "seed": seed,
        "config": config,
        "python": platform.python_version(),
        "packages": _versions(),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
    return run_dir
