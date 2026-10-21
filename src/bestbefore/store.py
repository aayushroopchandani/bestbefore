"""The store test set: the retailer's photos, labelled by hand
(docs/m0-data-and-test-set.md, Part B).

The photos and their labels belong to the retailer. They stay in
`$STORE_PHOTOS_DIR` and `data/store/` (gitignored); only counts and the
frozen file's hash go into git (decision 7).

    data/store/labels.json     written by scripts/import_store_labels.py (B2)
    data/store/test_set.json   the same labels plus each photo's SHA-256,
                               written once by scripts/freeze_store_test_set.py (B3)

Both map a photo's path relative to `$STORE_PHOTOS_DIR` to its label:

    expiry      "YYYY-MM-DD", or "YYYY-MM" when the pack prints no day;
                None when this photo doesn't show a readable expiry
    derived     True when the expiry was worked out from mfg + shelf life
    mfg         the manufacture date, same format, or None
    format      how the expiry is printed, e.g. "DD/MM/YYYY"
    tag         the failure tag, one of TAGS
    unreadable  True when the expiry is in frame but can't be read, even zoomed in
    boxes       [{"label": one of BOX_LABELS, "xyxy": [x1, y1, x2, y2]}], 0..1
                of the upright photo
    note        free text
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from .config import REPO_ROOT, store_photos_dir
from .images import upright_size
from .samples import Box, Expiry, Sample

DATA_DIR = REPO_ROOT / "data" / "store"
LABELS_PATH = DATA_DIR / "labels.json"
TEST_SET_PATH = DATA_DIR / "test_set.json"
# Anything drawn from the store photos (contact sheets, galleries of misses)
# goes here, never into results/.
RUNS_DIR = DATA_DIR / "runs"

# In the order they're tried: a photo gets the first that applies (decision 5).
TAGS = ("other-face", "sideways", "curved", "dot-matrix", "small-print", "clean")
BOX_LABELS = ("expiry", "mfg", "other_date")

# SHA-256 of data/store/test_set.json, pasted in when it was frozen (B3).
# v1, frozen 2026-10-21. A new version replaces it: results/store_test_set.md.
TEST_SET_SHA256: str | None = "7b9a4043886c9f4fcb37cba339962261acaef19e2993f7fba365a75f49b588f4"

_DATE = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?")


def parse_date(text: str) -> Expiry:
    """'2026-11-20', or '2026-11' when no day is printed."""
    match = _DATE.fullmatch(text.strip())
    if not match:
        raise ValueError(f"{text!r} is not YYYY-MM-DD or YYYY-MM")
    year, month, day = match.groups()
    return Expiry(int(year), int(month), int(day) if day else None)


def photos_on_disk(root: Path | None = None) -> list[str]:
    """Every store photo, as a path relative to `root`, sorted."""
    root = root or store_photos_dir()
    return sorted(p.relative_to(root).as_posix() for p in root.glob("samples*/*")
                  if p.suffix.lower() in (".jpg", ".jpeg"))


def sha256(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def to_samples(labels: dict[str, dict], root: Path | None = None) -> list[Sample]:
    root = root or store_photos_dir()
    samples = []
    for photo in sorted(labels):
        label, path = labels[photo], root / photo
        w, h = upright_size(path)
        samples.append(Sample(
            id=photo, image=path, width=w, height=h,
            boxes=tuple(Box(b["label"], tuple(b["xyxy"])) for b in label["boxes"]),
            expiry=parse_date(label["expiry"]) if label["expiry"] else None,
            tag=label["tag"]))
    return samples


def freeze(labels: dict[str, dict], version: str, date: str,
           root: Path | None = None) -> dict:
    """The test-set file's content: every label plus its photo's SHA-256."""
    root = root or store_photos_dir()
    return {"version": version, "frozen": date,
            "photos": {p: {**labels[p], "sha256": sha256(root / p)}
                       for p in sorted(labels)}}


def read_test_set(path: Path = TEST_SET_PATH, root: Path | None = None,
                  expected: str | None = None) -> dict:
    """The frozen test set, once its file and every photo are shown unchanged."""
    expected = expected or TEST_SET_SHA256
    if expected is None:
        raise RuntimeError("the store test set isn't frozen yet: "
                           "run scripts/freeze_store_test_set.py")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(
            f"{path} has changed since it was frozen. A label change is a new "
            "version (docs/m0-data-and-test-set.md, B3).")
    test_set = json.loads(data)
    root = root or store_photos_dir()
    changed = [p for p, label in test_set["photos"].items()
               if sha256(root / p) != label["sha256"]]
    if changed:
        raise RuntimeError(f"{len(changed)} store photos have changed since "
                           f"freezing: {', '.join(changed)}")
    return test_set


def read_labels(path: Path = LABELS_PATH) -> dict[str, dict]:
    """labels.json as imported, not yet frozen."""
    return json.loads(path.read_text())


def labels() -> dict[str, dict]:
    """The frozen test set's labels. Before B3 freezes them, labels.json."""
    if TEST_SET_SHA256 is None:
        print(f"note: the store test set isn't frozen yet; reading {LABELS_PATH}",
              file=sys.stderr)
        return read_labels()
    return read_test_set()["photos"]


def load_samples() -> list[Sample]:
    return to_samples(labels())


def tag_counts(labels: dict[str, dict]) -> list[dict[str, str]]:
    """One row per tag, then the total: what the test set holds."""
    def row(name: str, group: list[dict]) -> dict[str, str]:
        expiry = [lab["expiry"] for lab in group if lab["expiry"]]
        return {"tag": name, "photos": f"{len(group)}",
                "with an expiry": f"{len(expiry)}",
                "to the day": f"{sum(len(e) == 10 for e in expiry)}",
                "to the month": f"{sum(len(e) == 7 for e in expiry)}",
                "derived": f"{sum(lab['derived'] for lab in group)}",
                "unreadable": f"{sum(lab['unreadable'] for lab in group)}"}

    rows = [row(t, [lab for lab in labels.values() if lab["tag"] == t]) for t in TAGS]
    return rows + [row("**all**", list(labels.values()))]
