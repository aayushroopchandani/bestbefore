"""Baseline B: Scanify's reader, the rule-based system this project grew out
of (docs/m0-data-and-test-set.md, C2).

It runs in Scanify's own venv, as a subprocess (scripts/scanify_runner.py),
so its dependencies stay out of this project and none of its code is
copied here. The run record pins Scanify's git commit.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path

from ..config import REPO_ROOT, scanify_dir
from ..runs import code_sha256
from ..samples import Expiry
from ..scoring import NONE, RIGHT, WRONG, tally
from ..store_import import SCANIFY_TRUTH, read_scanify_truth

RUNNER = REPO_ROOT / "scripts" / "scanify_runner.py"
# What Scanify measured on its own 44 photos with a known expiry
# (its experiments/blur_score/README.md): B must reproduce it.
SCANIFY_MEASURED = {"n": 44, RIGHT: 32, WRONG: 0, NONE: 12}


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(scanify_dir()), *args],
                          capture_output=True, text=True).stdout.strip()


def provenance() -> dict:
    return {"code_sha256": code_sha256(sys.modules[__name__]),
            "scanify_commit": _git("rev-parse", "HEAD"),
            "scanify_dirty": bool(_git("status", "--porcelain"))}


def parse_iso(text: str | None) -> Expiry | None:
    """'2027-08-31' -> Expiry(2027, 8, 31). Scanify always gives a day."""
    if not text:
        return None
    d = date.fromisoformat(text)
    return Expiry(d.year, d.month, d.day)


def read(paths: Iterable[Path]) -> Iterator[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        path_list = Path(tmp) / "paths.txt"
        path_list.write_text("\n".join(str(p) for p in paths) + "\n")
        python = scanify_dir() / ".venv" / "bin" / "python"
        with subprocess.Popen([str(python), str(RUNNER), str(scanify_dir()), str(path_list)],
                              stdout=subprocess.PIPE, text=True) as proc:
            for line in proc.stdout:
                row = json.loads(line)
                scanify = row.pop("expiry") or {}
                yield {**row, "expiry": parse_iso(scanify.get("expiry")),
                       "scanify": scanify}
        if proc.returncode:
            raise RuntimeError(f"scanify_runner.py exited with {proc.returncode}")


def reproduction(rows: list[dict], truth: dict[str, Expiry | None] | None = None) -> dict:
    """Scanify's own check on its own photos: the date read equals its
    ground truth exactly (month-only packs as their month's last day).
    Rows are run_baseline's, keyed by photo id."""
    if truth is None:
        truth = read_scanify_truth(scanify_dir() / SCANIFY_TRUTH)
    read = {r["id"]: r["read"] for r in rows}
    outcomes = []
    for photo, expected in truth.items():
        if expected is None or photo not in read:
            continue
        got = read[photo]
        outcomes.append(NONE if got is None else RIGHT if got == str(expected) else WRONG)
    counts = tally(outcomes)
    return {**counts, "matches_scanify": counts == SCANIFY_MEASURED}
