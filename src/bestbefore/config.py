"""Where the data lives, read from `.env`.

Each path is looked up only when asked for, so a machine that has ExpDate
but not the store photos (the cluster) can still run the ExpDate scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def _path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def expdate_dir() -> Path:
    """The unpacked ExpDate download."""
    return _path("EXPDATE_DIR")


def store_photos_dir() -> Path:
    """The folder holding samples1/, samples2/ and samples3/."""
    return _path("STORE_PHOTOS_DIR")


def scanify_dir() -> Path:
    """The Scanify repo, for baseline B."""
    return _path("SCANIFY_DIR")


def checkpoint_dir() -> Path:
    """Where training writes model weights and plots (never into git)."""
    return _path("CHECKPOINT_DIR")
