"""Run records. Every run writes down what it ran, from which commit, with
which seed, so any number in `results/` can be traced back and rerun.
"""
from __future__ import annotations

import hashlib
import json
import os
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
            "ultralytics", "torch", "fonttools")


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    return out.stdout.strip()


def git_state() -> dict:
    """The commit the code is at, and whether it has uncommitted changes."""
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def write_json(path: Path, data) -> None:
    """Write JSON atomically: into a temporary file, then renamed over
    `path`. A reader sees the old file or the new one, never half of one
    (M2's two GPU jobs read each other's records while they are written)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def _versions() -> dict[str, str]:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def start_run(name: str, config: dict, seed: int | None = None,
              root: Path | None = None) -> Path:
    """Make `<root>/<timestamp>-<name>/` holding run.json. The root is
    results/runs/ unless given, e.g. data/store/runs/ for store photos."""
    now = datetime.now()
    run_dir = (root or RUNS_DIR) / f"{now:%Y%m%d-%H%M%S}-{name}"
    run_dir.mkdir(parents=True)
    record = {
        "name": name,
        "started": now.isoformat(timespec="seconds"),
        "command": sys.argv,
        **git_state(),
        "seed": seed,
        "config": config,
        "python": platform.python_version(),
        "packages": _versions(),
    }
    write_json(run_dir / "run.json", record)
    return run_dir


def update_run(run_dir: Path, **fields) -> None:
    """Add fields to a run's run.json (e.g. where its checkpoint went)."""
    path = run_dir / "run.json"
    record = json.loads(path.read_text())
    record.update(fields)
    write_json(path, record)


def latest_run(name: str, root: Path | None = None) -> Path | None:
    """The newest run directory made by start_run(name), if any."""
    found = sorted((root or RUNS_DIR).glob(f"*-{name}"))
    return found[-1] if found else None


def code_sha256(*modules) -> dict[str, str]:
    """SHA-256 of each module's source file, so a run's numbers can be tied
    to the exact code that made them, committed or not."""
    hashes = {}
    for module in modules:
        with open(module.__file__, "rb") as f:
            hashes[module.__name__] = hashlib.file_digest(f, "sha256").hexdigest()
    return hashes
