"""Turn the Label Studio export of the store photos into labels.json, and
check it before it's trusted (docs/m0-data-and-test-set.md, B2).

The export is Label Studio's JSON-MIN: one dict per photo, holding the
`image` URL, the `box` rectangles (x, y, width and height as percentages of
`original_width` x `original_height`) and one key per text or choice field
of the labelling interface.

Every check returns a list of problems, one line per photo, so the import
script can show them all at once instead of stopping at the first.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .expdate import normalise
from .samples import Expiry, parse_year
from .store import BOX_LABELS, TAGS, parse_date

DERIVED = "derived from mfg + shelf life"   # the choice's text in the interface
EXPIRY_YEARS = range(2024, 2036)
# A date shorter than this, as a fraction of the photo's long side, is small
# print (the B1 tagging rule, applied from the boxes since 2026-10-21).
SMALL_PRINT = 0.015


# --- Converting -----------------------------------------------------------

def photo_id(image_url: str) -> str:
    """'/data/local-files/?d=samples2/sample%201.jpg' -> 'samples2/sample 1.jpg'."""
    return parse_qs(urlparse(image_url).query)["d"][0]


def _text(value) -> str | None:
    """A text field as one stripped string; None when blank."""
    if isinstance(value, list):
        value = " ".join(value)
    return (value or "").strip() or None


def _box(rect: dict) -> dict:
    x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
    # Percentages are pixels of a 100 x 100 image.
    xyxy = normalise((x, y, x + w, y + h), 100, 100)
    return {"label": rect["rectanglelabels"][0], "xyxy": [round(v, 6) for v in xyxy]}


def from_export(tasks: list[dict]) -> dict[str, dict]:
    """Photo id -> label, in the labels.json schema (see store.py)."""
    labels = {}
    for task in tasks:
        expiry, note = _text(task.get("expiry")), _text(task.get("note")) or ""
        labels[photo_id(task["image"])] = {
            "expiry": expiry,
            "derived": task.get("derived") == DERIVED,
            "mfg": _text(task.get("mfg")),
            "format": _text(task.get("format")),
            "tag": task.get("tag"),
            "unreadable": expiry is None and "unreadable" in note.lower(),
            "boxes": [_box(r) for r in task.get("box", [])],
            "note": note,
        }
    return labels


def drawn_sizes(tasks: list[dict]) -> dict[str, tuple[int, int]]:
    """The image size Label Studio drew the boxes on, per photo with boxes."""
    return {photo_id(t["image"]): (t["box"][0]["original_width"],
                                   t["box"][0]["original_height"])
            for t in tasks if t.get("box")}


def read_scanify_truth(path: Path) -> dict[str, Expiry | None]:
    """Scanify's EXPIRY_GROUND_TRUTH, read as data: the file is parsed, never
    imported, so none of Scanify's code runs here.

    Its keys are file names in samples2/. Its values are 'DD/MM/YY', with a
    month-only pack resolved to the last day of the month, or None when the
    photo alone shows no expiry.
    """
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "EXPIRY_GROUND_TRUTH" for t in node.targets)):
            raw = ast.literal_eval(node.value)
            break
    else:
        raise ValueError(f"no EXPIRY_GROUND_TRUTH in {path}")
    truth = {}
    for name, value in raw.items():
        if value is not None:
            day, month, year = value.split("/")
            value = Expiry(parse_year(year), int(month), int(day))
        truth[f"samples2/{name}"] = value
    return truth


# --- Checks: each returns one line per problem ------------------------------

def check_export(tasks: list[dict]) -> list[str]:
    """Known tags and box labels, upright and non-empty boxes."""
    problems = []
    for task in tasks:
        photo = photo_id(task["image"])
        if task.get("tag") not in TAGS:
            problems.append(f"{photo}: tag {task.get('tag')!r} isn't one of {', '.join(TAGS)}")
        boxes = task.get("box", [])
        if len({(r["original_width"], r["original_height"]) for r in boxes}) > 1:
            problems.append(f"{photo}: boxes drawn on images of different sizes")
        for r in boxes:
            if len(r["rectanglelabels"]) != 1 or r["rectanglelabels"][0] not in BOX_LABELS:
                problems.append(f"{photo}: box labelled {r['rectanglelabels']}")
            if r.get("rotation", 0):
                problems.append(f"{photo}: a box rotated {r['rotation']} degrees; "
                                "draw an upright rectangle instead")
            if r["width"] <= 0 or r["height"] <= 0:
                problems.append(f"{photo}: an empty box")
    return problems


def check_photos(labelled: list[str], on_disk: list[str]) -> list[str]:
    """Check 1: every photo labelled exactly once, and nothing else."""
    counts = Counter(labelled)
    return ([f"{p}: labelled {n} times" for p, n in sorted(counts.items()) if n > 1]
            + [f"{p}: not labelled" for p in sorted(set(on_disk) - set(counts))]
            + [f"{p}: labelled, but not in $STORE_PHOTOS_DIR"
               for p in sorted(set(counts) - set(on_disk))])


def _not_after(a: Expiry, b: Expiry) -> bool:
    """a is on or before b, compared to the day only when both print one."""
    if a.day is None or b.day is None:
        return (a.year, a.month) <= (b.year, b.month)
    return (a.year, a.month, a.day) <= (b.year, b.month, b.day)


def check_dates(labels: dict[str, dict]) -> list[str]:
    """Check 2: dates parse, the expiry year is plausible, mfg comes first."""
    problems = []
    for photo, label in sorted(labels.items()):
        try:
            expiry = parse_date(label["expiry"]) if label["expiry"] else None
            mfg = parse_date(label["mfg"]) if label["mfg"] else None
        except ValueError as e:
            problems.append(f"{photo}: {e}")
            continue
        if expiry and expiry.year not in EXPIRY_YEARS:
            problems.append(f"{photo}: expiry year {expiry.year} isn't between "
                            f"{EXPIRY_YEARS.start} and {EXPIRY_YEARS.stop - 1}")
        if expiry and mfg and not _not_after(mfg, expiry):
            problems.append(f"{photo}: mfg {mfg} is after expiry {expiry}")
    return problems


def check_consistency(labels: dict[str, dict]) -> list[str]:
    """Check 3: the expiry, the tag and the boxes tell the same story."""
    problems = []
    for photo, label in sorted(labels.items()):
        kinds = {b["label"] for b in label["boxes"]}
        if label["tag"] == "other-face":
            if label["expiry"] or "expiry" in kinds:
                problems.append(f"{photo}: other-face, but has an expiry or an expiry box")
        elif label["derived"]:
            if not (label["expiry"] and label["mfg"] and "mfg" in kinds):
                problems.append(f"{photo}: derived, so it needs an expiry, a mfg date and a mfg box")
            if "expiry" in kinds:
                problems.append(f"{photo}: derived, but has an expiry box; box only the mfg date")
        elif label["expiry"]:
            if "expiry" not in kinds:
                problems.append(f"{photo}: has an expiry, but no expiry box")
        elif not label["unreadable"]:
            problems.append(f"{photo}: no expiry, but not other-face and not noted unreadable")
    return problems


def same_expiry(ours: Expiry | None, theirs: Expiry | None) -> bool:
    """Scanify turns a month-only expiry into the month's last day, so the
    day is compared only when our label has one."""
    if ours is None or theirs is None:
        return ours is None and theirs is None
    if (ours.year, ours.month) != (theirs.year, theirs.month):
        return False
    return ours.day is None or ours.day == theirs.day


def check_scanify(labels: dict[str, dict], truth: dict[str, Expiry | None]) -> list[str]:
    """Check 4: our expiries agree with Scanify's ground truth."""
    problems = []
    for photo, theirs in sorted(truth.items()):
        if photo not in labels:
            problems.append(f"{photo}: in Scanify's ground truth, but not labelled")
            continue
        try:
            text = labels[photo]["expiry"]
            ours = parse_date(text) if text else None
        except ValueError:
            continue    # reported by check_dates
        if not same_expiry(ours, theirs):
            problems.append(f"{photo}: ours {ours or 'none'}, Scanify's {theirs or 'none'}")
    return problems


def check_orientation(drawn: dict[str, tuple[int, int]],
                      upright: dict[str, tuple[int, int]]) -> list[str]:
    """Check 5: Label Studio drew on the upright photo, the frame `load_upright` gives."""
    problems = []
    for photo, (w, h) in sorted(drawn.items()):
        size = upright.get(photo)
        if size is None or size == (w, h):
            continue    # a photo missing from disk is reported by check_photos
        why = " (swapped: drawn on the raw frame)" if size == (h, w) else ""
        problems.append(f"{photo}: drawn on {w}x{h}, upright is {size[0]}x{size[1]}{why}")
    return problems


# --- The small-print rule ----------------------------------------------------

def print_height(label: dict, size: tuple[int, int]) -> float | None:
    """Height of the expiry's tallest box (the mfg's, for a derived expiry)
    as a fraction of the photo's long side. None without such a box."""
    kind = "mfg" if label["derived"] else "expiry"
    heights = [b["xyxy"][3] - b["xyxy"][1] for b in label["boxes"] if b["label"] == kind]
    return max(heights) * size[1] / max(size) if heights else None


def print_tag(label: dict, size: tuple[int, int]) -> str:
    """Check 6, applied rather than flagged: `small-print` vs `clean` comes
    from the box, not from the eye, so anyone can recompute it. The other
    tags come first in the fixed order and are kept."""
    height = print_height(label, size)
    if label["tag"] not in ("clean", "small-print") or height is None:
        return label["tag"]
    return "small-print" if height < SMALL_PRINT else "clean"


def apply_print_rule(labels: dict[str, dict], upright: dict[str, tuple[int, int]]
                     ) -> tuple[dict[str, dict], Counter]:
    """The labels with print_tag applied, and how many photos moved, e.g.
    {'small-print -> clean': 20}."""
    out, moved = {}, Counter()
    for photo, label in labels.items():
        tag = print_tag(label, upright[photo]) if photo in upright else label["tag"]
        if tag != label["tag"]:
            moved[f"{label['tag']} -> {tag}"] += 1
        out[photo] = {**label, "tag": tag}
    return out, moved
