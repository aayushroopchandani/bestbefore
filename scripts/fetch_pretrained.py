"""Fetch everything a training job would otherwise download, ahead of time.

    python scripts/fetch_pretrained.py

Compute nodes have no internet, so this runs on a machine that does (the
cluster's login node, or the laptop). Into $CHECKPOINT_DIR/pretrained/:
  yolo11n.pt, yolo11s.pt   the COCO-pretrained models M1 fine-tunes
  yolo26n.pt               only used by Ultralytics' mixed-precision check
It also fetches the font Ultralytics draws its plots with, and points
Ultralytics' weights_dir setting at the same folder, so its own lookups
(the mixed-precision check) find the files instead of downloading them.
"""
from __future__ import annotations

import hashlib

from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_font
from ultralytics.utils.downloads import attempt_download_asset

from bestbefore.config import checkpoint_dir

WEIGHTS = ("yolo11n.pt", "yolo11s.pt", "yolo26n.pt")


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    target = checkpoint_dir() / "pretrained"
    target.mkdir(parents=True, exist_ok=True)
    for name in WEIGHTS:
        path = target / name
        if not path.exists():
            attempt_download_asset(str(path))
        if not path.exists():
            raise SystemExit(f"could not fetch {name}")
        print(f"{name:12s} {path.stat().st_size / 1e6:5.1f} MB  sha256 {sha256(path)[:16]}…")
    print("font:", check_font("Arial.ttf"))
    SETTINGS.update({"weights_dir": str(target)})
    print("Ultralytics weights_dir ->", SETTINGS["weights_dir"])


if __name__ == "__main__":
    main()
