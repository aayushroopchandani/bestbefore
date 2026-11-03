"""Store set v1 as training data (docs/m2b-store-and-size.md, "The data"):
pack design ids for v1 and v2, and v1 split by design.

    data/store/designs.csv    photo,design, one row per photo of v1 and v2,
                              named from the photos (S4)
    data/store/designs.json   the same ids, frozen by scripts/store_designs.py
                              --freeze; its SHA-256 is store.DESIGNS_SHA256

One design is one brand, product line and date-print layout. The ids come
from looking at the photos, never from a model's output. The split is
worked out from the frozen ids and v1's frozen labels each time, the same
every time (seed 0), and checked against SPLIT_SHA256. v2 never enters it.
Like the photos, the ids stay in data/store/: committed files hold counts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from . import store

DESIGNS_CSV = store.DATA_DIR / "designs.csv"
DESIGNS_PATH = store.DATA_DIR / "designs.json"
SEED = 0
# SHA-256 of the split (split_digest), pasted in when results/store_designs.md
# was first written (S4, 2026-11-03).
SPLIT_SHA256: str | None = "4418c65be61a57e1f949b7beb3013b7843eeec2a349f70496bcce6a7d2af02ca"


def read_designs_csv(path: Path | None = None) -> list[list[str]]:
    """designs.csv's rows after its `photo,design` header, each field
    stripped; blank lines left out. A spreadsheet's byte-order mark is fine."""
    path = path or DESIGNS_CSV
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [[field.strip() for field in row] for row in csv.reader(f)]
    if not rows or rows[0][:2] != ["photo", "design"]:
        raise ValueError(f"{path} doesn't start with the header photo,design")
    return [row for row in rows[1:] if any(row)]


def _key(design: str) -> str:
    return "".join(design.casefold().split())


def check_designs(rows: list[list[str]], photos: list[str]) -> list[str]:
    """Problems with the typed ids: every photo in `photos` exactly once,
    nothing else, no empty id, and no two ids that differ only in case or
    spaces (one design typed two ways would let it sit on both sides of the
    split)."""
    problems, seen, ids = [], Counter(), {}
    for n, row in enumerate(rows, start=2):
        if len(row) != 2:
            problems.append(f"line {n}: {len(row)} fields, not 2 (a comma in a design?)")
            continue
        photo, design = row
        seen[photo] += 1
        if photo not in photos:
            problems.append(f"line {n}: not a photo of v1 or v2: {photo}")
        if not design:
            problems.append(f"line {n}: no design for {photo}")
        else:
            ids.setdefault(_key(design), set()).add(design)
    problems += [f"listed {k} times: {p}" for p, k in sorted(seen.items()) if k > 1]
    problems += [f"not listed: {p}" for p in sorted(set(photos) - set(seen))]
    problems += [f"one design typed {len(v)} ways? {' / '.join(sorted(v))}"
                 for v in ids.values() if len(v) > 1]
    return problems


def freeze(rows: list[list[str]], test_sets: dict[str, str], date: str) -> dict:
    """designs.json's content: each photo's id, and the hashes of the label
    files the ids were typed against."""
    return {"frozen": date, "test_sets": test_sets,
            "designs": {photo: design for photo, design in sorted(rows)}}


def read_designs(path: Path | None = None, expected: str | None = None) -> dict[str, str]:
    """photo -> design, once designs.json is shown unchanged since freezing."""
    path = path or DESIGNS_PATH
    expected = expected or store.DESIGNS_SHA256
    if expected is None:
        raise RuntimeError("the store designs aren't frozen yet: run scripts/store_designs.py --freeze")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"{path} has changed since it was frozen")
    return json.loads(data)["designs"]


def design_tag(tags: list[str]) -> str:
    """A design's tag: its photos' commonest, `other-face` only if every
    photo is. A tie goes to the tag first in store.TAGS, the order photos
    take their tags in."""
    counts = Counter(t for t in tags if t != "other-face") or Counter(tags)
    return max(store.TAGS, key=lambda t: (counts[t], -store.TAGS.index(t)))


def val_count(n: int) -> int:
    """How many of a tag's n designs go to validation: 30%, rounded half
    up (exactly, in integers), at least 1 when there are 2 or more."""
    return max((3 * n + 5) // 10, 1 if n >= 2 else 0)


def split_v1(designs: dict[str, str], labels: dict[str, dict], seed: int = SEED) -> dict[str, str]:
    """photo -> "train" or "val". Each design takes its photos' commonest
    tag (other-face only if every photo is). Within each tag, the designs,
    sorted by id and shuffled with `seed`, give round(0.3 x n) of themselves
    to validation (at least 1 where a tag has 2 or more designs).

    `labels` are v1's; a photo from outside v1's folders is refused, so v2
    can't enter the split. One generator, seeded with `seed`, shuffles the
    tags in store.TAGS order."""
    outside = sorted(p for p in labels if p.split("/")[0] not in store.V1.folders)
    if outside:
        raise ValueError(f"only store set v1 is split; {len(outside)} photos from outside "
                         f"its folders: {', '.join(outside)}")
    missing = sorted(set(labels) - set(designs))
    if missing:
        raise ValueError(f"{len(missing)} v1 photos have no design: {', '.join(missing)}")
    photos_of: dict[str, list[str]] = {}
    for photo in sorted(labels):
        photos_of.setdefault(designs[photo], []).append(photo)
    by_tag: dict[str, list[str]] = {}
    for design in sorted(photos_of):
        by_tag.setdefault(design_tag([labels[p]["tag"] for p in photos_of[design]]), []).append(design)
    rng, val = random.Random(seed), set()
    for tag in store.TAGS:
        group = sorted(by_tag.get(tag, []))
        rng.shuffle(group)
        val |= set(group[:val_count(len(group))])
    return {p: "val" if designs[p] in val else "train" for p in sorted(labels)}


def split_digest(split: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()


def v1_split() -> dict[str, str]:
    """v1's split from the frozen ids and labels, checked against SPLIT_SHA256."""
    split = split_v1(read_designs(), store.labels(store.V1), SEED)
    if SPLIT_SHA256 is not None and split_digest(split) != SPLIT_SHA256:
        raise RuntimeError("v1's split differs from the one recorded in results/store_designs.md")
    return split
