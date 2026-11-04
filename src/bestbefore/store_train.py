"""Store set v1 as training data (docs/m2b-store-and-size.md, "The data"):
pack design ids for v1 and v2, v1 split by design, v1's keyword boxes, and
the export of v1 and v3 for training.

    data/store/designs.csv    photo,design, one row per photo of v1 and v2,
                              named from the photos (S4)
    data/store/designs.json   the same ids, frozen by scripts/store_designs.py
                              --freeze; its SHA-256 is store.DESIGNS_SHA256
    data/store/designs_v3.csv, designs_v3.json  store set v3's, the same way
                              (--version v3; S5b), store.DESIGNS_V3_SHA256
    data/store/keywords/      OCR's keyword proposals, the Label Studio tasks
                              and the reviewed export (S5)
    data/store/v1_keywords.json  the due, prod and code boxes on each v1
                              photo, written by scripts/import_store_keywords.py
    $STORE_TRAIN_DIR          v1 and v3 exported for training, numbered, by
                              scripts/export_store_yolo.py (S6)
    data/store/train_names.json  the map from the export's numbers back to
                              photo and design names: laptop only

One design is one brand, product line and date-print layout. The ids come
from looking at the photos, never from a model's output. The split is
worked out from the frozen ids and v1's frozen labels each time, the same
every time (seed 0), and checked against SPLIT_SHA256. v2 never enters it.
Like the photos, the ids stay in data/store/: committed files hold counts.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageCms

from . import derived, images, store
from .detection_eval import iou
from .images import load_upright
from .runs import code_sha256, git_state
from .samples import Box
from .store_import import check_photos, photo_id, rect_box

DESIGNS_CSV = store.DATA_DIR / "designs.csv"
DESIGNS_PATH = store.DATA_DIR / "designs.json"
DESIGNS_V3_CSV = store.DATA_DIR / "designs_v3.csv"
DESIGNS_V3_PATH = store.DATA_DIR / "designs_v3.json"
SEED = 0
# SHA-256 of the split (split_digest), pasted in when results/store_designs.md
# was first written (S4, 2026-11-03).
SPLIT_SHA256: str | None = "4418c65be61a57e1f949b7beb3013b7843eeec2a349f70496bcce6a7d2af02ca"
# v3's (split_v3), pasted in when its section of results/store_designs.md was
# first written (S5b).
SPLIT_V3_SHA256: str | None = "edd2c25cceeefa2a070c3b2bec0c19f17603380d173210560eae2763837858a8"


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


def check_designs(rows: list[list[str]], photos: list[str], known: set[str] = frozenset(),
                  sets: str = "v1 or v2") -> list[str]:
    """Problems with the typed ids: every photo in `photos` exactly once,
    nothing else, no empty id, and no two ids that differ only in case or
    spaces (one design typed two ways would let it sit on both sides of the
    split), here or against the ids already `known` (v1's and v2's, when
    v3's are typed). `sets` names the photos' sets, for the message."""
    problems, seen, ids = [], Counter(), {}
    for n, row in enumerate(rows, start=2):
        if len(row) != 2:
            problems.append(f"line {n}: {len(row)} fields, not 2 (a comma in a design?)")
            continue
        photo, design = row
        seen[photo] += 1
        if photo not in photos:
            problems.append(f"line {n}: not a photo of {sets}: {photo}")
        if not design:
            problems.append(f"line {n}: no design for {photo}")
        else:
            ids.setdefault(_key(design), set()).add(design)
    problems += [f"listed {k} times: {p}" for p, k in sorted(seen.items()) if k > 1]
    problems += [f"not listed: {p}" for p in sorted(set(photos) - set(seen))]
    problems += [f"one design typed {len(v)} ways? {' / '.join(sorted(v))}"
                 for v in ids.values() if len(v) > 1]
    known_of = {_key(k): k for k in known}
    problems += [f"{d} is the known design {known_of[_key(d)]} typed another way"
                 for v in ids.values() for d in sorted(v) if _key(d) in known_of and d != known_of[_key(d)]]
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


def _by_design(designs: dict[str, str], labels: dict[str, dict], seed: int) -> set[str]:
    """The designs S4's rule sends to validation: each design takes its
    photos' commonest tag; within each tag, sorted by id and shuffled by one
    generator seeded with `seed` (tags in store.TAGS order), round(0.3 x n)."""
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
    return val


def _only(store_set: store.StoreSet, labels: dict[str, dict], designs: dict[str, str]) -> None:
    outside = sorted(p for p in labels if p.split("/")[0] not in store_set.folders)
    if outside:
        raise ValueError(f"only store set {store_set.version} is split here; {len(outside)} photos "
                         f"from outside its folders: {', '.join(outside)}")
    missing = sorted(set(labels) - set(designs))
    if missing:
        raise ValueError(f"{len(missing)} {store_set.version} photos have no design: {', '.join(missing)}")


def split_v1(designs: dict[str, str], labels: dict[str, dict], seed: int = SEED) -> dict[str, str]:
    """photo -> "train" or "val". Each design takes its photos' commonest
    tag (other-face only if every photo is). Within each tag, the designs,
    sorted by id and shuffled with `seed`, give round(0.3 x n) of themselves
    to validation (at least 1 where a tag has 2 or more designs).

    `labels` are v1's; a photo from outside v1's folders is refused, so v2
    can't enter the split. One generator, seeded with `seed`, shuffles the
    tags in store.TAGS order."""
    _only(store.V1, labels, designs)
    val = _by_design(designs, labels, seed)
    return {p: "val" if designs[p] in val else "train" for p in sorted(labels)}


def split_v3(designs_v3: dict[str, str], labels_v3: dict[str, dict], designs: dict[str, str],
             split: dict[str, str], seed: int = SEED) -> dict[str, str]:
    """photo -> "train" or "val" for store set v3 (S5b, decision 6). A v3
    photo whose design v1 has takes that design's side in v1's `split`; v3's
    other designs are split among themselves by split_v1's rule. `designs`
    and `split` are v1's (and v2's) frozen ids and v1's split, which don't
    change; a photo from outside v3's folders is refused."""
    _only(store.V3, labels_v3, designs_v3)
    side = {designs[p]: part for p, part in split.items()}
    new = {p: lab for p, lab in labels_v3.items() if designs_v3[p] not in side}
    val = _by_design(designs_v3, new, seed)
    return {p: side.get(designs_v3[p]) or ("val" if designs_v3[p] in val else "train")
            for p in sorted(labels_v3)}


def split_digest(split: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()


def v1_split() -> dict[str, str]:
    """v1's split from the frozen ids and labels, checked against SPLIT_SHA256."""
    split = split_v1(read_designs(), store.labels(store.V1), SEED)
    if SPLIT_SHA256 is not None and split_digest(split) != SPLIT_SHA256:
        raise RuntimeError("v1's split differs from the one recorded in results/store_designs.md")
    return split


def v3_split() -> dict[str, str]:
    """v3's split from its frozen ids and labels and v1's split, checked
    against SPLIT_V3_SHA256."""
    if store.DESIGNS_V3_SHA256 is None:
        raise RuntimeError("v3's designs aren't frozen yet: run scripts/store_designs.py --version v3 --freeze")
    designs_v3 = read_designs(DESIGNS_V3_PATH, store.DESIGNS_V3_SHA256)
    split = split_v3(designs_v3, store.labels(store.V3), read_designs(), v1_split(), SEED)
    if SPLIT_V3_SHA256 is not None and split_digest(split) != SPLIT_V3_SHA256:
        raise RuntimeError("v3's split differs from the one recorded in results/store_designs.md")
    return split


# --- Keyword boxes (S5) ------------------------------------------------------
#
# The due, prod and code words on v1's photos, so that S6's export doesn't
# teach the detector an EXP or MFG word is background. RapidOCR reads each
# photo's lines (scripts/propose_keywords.py); find_keywords picks the
# keywords out of a line and span_box cuts the line's box down to them. The
# proposals go into a Label Studio project of their own, "Store v1
# keywords", beside v1's frozen date boxes for reference, and the owner
# keeps, moves, deletes or adds each one and submits every photo
# (scripts/import_store_keywords.py reads the export back). The box rule is
# the generator's (synth/text.py): the word or phrase as printed, with its
# colon or full stop, never the date; a batch or lot code with its label, or
# a time.

KEYWORD_LABELS = ("due", "prod", "code")
KEYWORDS_DIR = store.DATA_DIR / "keywords"      # proposals, tasks, exports: they name photos
KEYWORDS_PATH = store.DATA_DIR / "v1_keywords.json"
# The sets that get keyword boxes: v1 (S5) and v3 (S5b); never v2, the test set.
KEYWORD_SETS = ("v1", "v3")


def keywords_dir(store_set: store.StoreSet) -> Path:
    """A set's proposals, tasks and exports: v1's in data/store/keywords/,
    another's in its own folder (data/store/v3/keywords/)."""
    return KEYWORDS_DIR if store_set is store.V1 else store_set.data_dir / "keywords"


def keywords_path(store_set: store.StoreSet) -> Path:
    """A set's reviewed keyword boxes: v1_keywords.json, or keywords.json in its folder."""
    return KEYWORDS_PATH if store_set is store.V1 else store_set.data_dir / "keywords.json"


def keyword_set(version: str) -> store.StoreSet:
    """The store set a keyword tool works on, refusing v2 (and anything else)."""
    if version not in KEYWORD_SETS:
        raise SystemExit(f"keyword boxes are for store sets {', '.join(KEYWORD_SETS)} only, not {version}")
    return store.SETS[version]
# The labelling interface's names, and the predictions' in Label Studio.
IMAGE, KEYWORD_CONTROL, REFERENCE_CONTROL = "img", "keyword", "date"
MODEL_VERSION = "rapidocr-1.2.3-keywords"
# A keyword box this close to a date box (IoU) is the date boxed by mistake.
OVER_DATE = 0.5

_DATE_WORD = r"\s*\.?\s*(?:\(\s*date\s*\)|date|dt\.?)(?![a-z])"
_SEP = r"(?:\s*[:：.\-])?"
_BY = r"\s*\.?\s*(?:by|lic)\b"               # Mfd. By: (the maker's address), Mfg. Lic. No.
_VALUE = r"(?:\s*(?=[a-z0-9\-]*\d)[a-z0-9][a-z0-9\-]*)?"     # a code's value has a digit


def _word(words: str, unless: str = _BY) -> str:
    """A keyword: its words, then a 'date' word or anything but a letter,
    then its colon or full stop if printed; never with `unless` after it."""
    return rf"(?:{words})(?!{unless})(?:{_DATE_WORD}|(?![a-z])){_SEP}"


# Case is ignored. The plan's words and the store's own phrasings
# (MFD(Date):, Month & Year of Import:), with OCR's commonest slips: words
# run together, a first letter dropped (mport, linufacturing).
KEYWORD_PATTERNS = {
    "due": (_word(r"best\s*before(?:\s*end)?|use\s*(?:by|before)|exp(?:iry|iration)?"),),
    "prod": (_word(r"\w*th\s*(?:&|[ao]nd)\s*year\s*of\s*(?:manufactur\w*|mfg|import|packing)"),
             _word(r"date\s*of\s*(?:manufactur\w*|mfg|import|packing)"),
             _word(r"\w{0,2}nufactur(?:e|ing)|mf[gdc0]|pkd|packed\s*on|packing\s*date"),
             _word(r"[ijl]?mport(?:ed)?(?:\s*on)?", unless=r"\s*(?:&|and\b|by\b|marketed)")),
    "code": (rf"\b(?:lot|batch)(?:\s*/\s*(?:lot|batch))?(?:\s*(?:no|number|code))?\.?(?![a-z]){_SEP}{_VALUE}",
             rf"\bb\.\s*no\.?(?![a-z]){_SEP}{_VALUE}",
             r"(?<![\d:.])(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?![\d:])"),
}
_PATTERNS = [(label, re.compile(p, re.I)) for label, ps in KEYWORD_PATTERNS.items() for p in ps]


def find_keywords(text: str) -> list[tuple[str, int, int]]:
    """Each keyword in one OCR line as (label, start, end), in reading
    order. Where matches overlap, the one starting first wins, then the
    longer: 'Month & Year of Import:' is one prod, not an import word
    inside one. A prod word after 'from' (Best Before 12 months from Mfg.,
    24 months from date of manufacture) is a shelf life's, and gets no box."""
    found = sorted((m.start(), -m.end(), label) for label, pattern in _PATTERNS
                   for m in pattern.finditer(text)
                   if not (label == "prod" and re.search(r"from\s*(?:the\s*)?(?:date\s*of\s*)?$",
                                                         text[:m.start()], re.I)))
    keywords, end = [], 0
    for start, minus_end, label in found:
        if start >= end:
            keywords.append((label, start, -minus_end))
            end = -minus_end
    return keywords


def span_box(quad, start: int, end: int, n: int, turned: bool = False,
             flipped: bool = False) -> tuple[float, float, float, float]:
    """The upright box round characters start to end of an OCR line of n
    characters, in the quad's units. The quad is RapidOCR's, clockwise from
    its top left, and it is cut along the line in proportion to the
    characters: a guess, as characters differ in width, which the review
    fixes. A line whose crop RapidOCR turned a quarter (1.5 times taller
    than wide) reads from its top edge down; one its classifier turned 180°
    reads the other way."""
    t0, t1 = start / n, end / n
    if flipped:
        t0, t1 = 1 - t1, 1 - t0
    p0, p1, p2, p3 = quad
    edges = ((p0, p3), (p1, p2)) if turned else ((p0, p1), (p3, p2))
    points = [(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) for a, b in edges for t in (t0, t1)]
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def propose(lines: list[dict], size: tuple[int, int]) -> list[dict]:
    """One photo's keyword proposals from its OCR lines (each a quad in the
    pixels of an image of `size`, its text, and whether it was turned or
    flipped): label, 0..1 xyxy of the photo, the keyword's text and its
    line's, and an id (kw1, kw2, ... in reading order), which Label Studio
    keeps as the region's."""
    w, h = size
    boxes = []
    for line in lines:
        text = line["text"]
        for label, start, end in find_keywords(text):
            x1, y1, x2, y2 = span_box(line["quad"], start, end, len(text), line["turned"], line["flipped"])
            boxes.append({"id": f"kw{len(boxes) + 1}", "label": label,
                          "xyxy": [round(min(max(v, 0.0), 1.0), 6) for v in (x1 / w, y1 / h, x2 / w, y2 / h)],
                          "text": text[start:end], "line": text})
    return boxes


def ls_region(region_id: str, control: str, box: dict, size: tuple[int, int],
              readonly: bool = False) -> dict:
    """A box as a Label Studio rectangle on the upright photo of `size`, in
    percent of its width and height, the convention v1's date boxes were
    drawn in: store_import.rect_box reads it back. A box's six decimals are
    four in percent, so rounding to four keeps it exactly."""
    x1, y1, x2, y2 = box["xyxy"]
    region = {"id": region_id, "from_name": control, "to_name": IMAGE, "type": "rectanglelabels",
              "original_width": size[0], "original_height": size[1], "image_rotation": 0,
              "value": {"x": round(100 * x1, 4), "y": round(100 * y1, 4), "width": round(100 * (x2 - x1), 4),
                        "height": round(100 * (y2 - y1), 4), "rotation": 0, "rectanglelabels": [box["label"]]}}
    return {**region, "readonly": True} if readonly else region


def keyword_task(photo: str, size: tuple[int, int], proposals: list[dict], label: dict) -> dict:
    """One photo's Label Studio task: the photo's URL as v1's project has
    it, and one prediction holding OCR's proposals and, read-only, the
    photo's frozen date boxes for reference."""
    result = [ls_region(p["id"], KEYWORD_CONTROL, p, size) for p in proposals]
    result += [ls_region(f"date{k}", REFERENCE_CONTROL, b, size, readonly=True)
               for k, b in enumerate(label["boxes"], start=1)]
    return {"data": {"image": f"/data/local-files/?d={quote(photo)}"},
            "predictions": [{"model_version": MODEL_VERSION, "result": result}]}


# Reading the reviewed project back: Label Studio's full JSON export, which
# keeps each region's id (JSON-MIN, which v1's import reads, doesn't).

def task_photo(task: dict) -> str:
    return photo_id(task["data"]["image"])


def submitted(task: dict) -> list[dict]:
    return [a for a in task.get("annotations", []) if not a.get("was_cancelled")]


def regions(annotation: dict, control: str) -> list[dict]:
    return [r for r in annotation["result"]
            if r.get("from_name") == control and r.get("type") == "rectanglelabels"]


def check_reviewed(tasks: list[dict], photos: list[str], name: str = "store set v1") -> list[str]:
    """Every v1 photo once and nothing else, each with exactly one
    annotation, submitted (not skipped), and no draft left: Label Studio
    keeps a draft while boxes are drawn or changed, and only Submit (or
    Update) turns it into the annotation (M0, D5)."""
    problems = check_photos([task_photo(t) for t in tasks], photos, name)
    for task in tasks:
        photo, done, every = task_photo(task), len(submitted(task)), len(task.get("annotations", []))
        if done != 1 or every != 1:
            problems.append(f"{photo}: {done} submitted and {every - done} skipped annotations, not 1 submitted")
        if task.get("drafts"):
            problems.append(f"{photo}: {len(task['drafts'])} drafts not submitted")
    return problems


def check_keyword_boxes(tasks: list[dict], upright: dict[str, tuple[int, int]],
                        control: str = KEYWORD_CONTROL, known: tuple[str, ...] = KEYWORD_LABELS) -> list[str]:
    """Each keyword box (or, with `control` and `known`, each box of another
    layer): one known label, upright, not empty, drawn on the upright photo
    (the frame v1's date boxes are in) and inside it."""
    problems = []
    for task in tasks:
        photo = task_photo(task)
        for r in (r for a in submitted(task) for r in regions(a, control)):
            v, labels, box = r["value"], r["value"].get("rectanglelabels", []), f"{photo}: a {control} box"
            if len(labels) != 1 or labels[0] not in known:
                problems.append(f"{box} labelled {labels}")
            if v.get("rotation", 0):
                problems.append(f"{box} rotated {v['rotation']} degrees; draw an upright rectangle")
            if v["width"] <= 0 or v["height"] <= 0:
                problems.append(f"{box} is empty")
            if (r["original_width"], r["original_height"]) != upright.get(photo):
                problems.append(f"{box} drawn on {r['original_width']}x{r['original_height']}, "
                                f"not the upright photo's {upright.get(photo)}")
            if min(v["x"], v["y"]) < -1e-6 or max(v["x"] + v["width"], v["y"] + v["height"]) > 100 + 1e-6:
                problems.append(f"{box} reaches outside the photo")
    return problems


def keyword_boxes(tasks: list[dict]) -> dict[str, list[dict]]:
    """photo -> its reviewed keyword boxes (region id, label, 0..1 xyxy),
    from its submitted annotation."""
    return {task_photo(t): [{"id": r["id"], **rect_box(r["value"])}
                            for a in submitted(t)[:1] for r in regions(a, KEYWORD_CONTROL)]
            for t in tasks}


def check_off_dates(keywords: dict[str, list[dict]], labels: dict[str, dict]) -> list[str]:
    """No keyword box over one of its photo's date boxes at IoU OVER_DATE
    or more: that is the date itself boxed as a keyword."""
    return [f"{photo}: a {k['label']} box over its {d['label']} box (IoU {iou(k['xyxy'], d['xyxy']):.2f})"
            for photo, boxes in sorted(keywords.items()) for k in boxes
            for d in labels.get(photo, {}).get("boxes", []) if iou(k["xyxy"], d["xyxy"]) >= OVER_DATE]


def _reference(task: dict) -> list[dict]:
    """The date layer's boxes in a task's submitted annotation, unrounded:
    label, 0..1 xyxy and region id."""
    return [{"id": r["id"], "label": r["value"]["rectanglelabels"][0],
             "xyxy": [r["value"]["x"] / 100, r["value"]["y"] / 100,
                      (r["value"]["x"] + r["value"]["width"]) / 100,
                      (r["value"]["y"] + r["value"]["height"]) / 100]}
            for a in submitted(task) for r in regions(a, REFERENCE_CONTROL)]


def _same(a: dict, b: dict) -> bool:
    return a["label"] == b["label"] and max(abs(x - y) for x, y in zip(a["xyxy"], b["xyxy"])) <= 1e-6


def check_reference(tasks: list[dict], labels: dict[str, dict]) -> list[str]:
    """Every one of a photo's frozen date boxes is still in its submitted
    annotation, unchanged: none moved or deleted. A date box the reviewer
    added (a date v1's labels missed) is allowed; added_dates collects it."""
    problems = []
    for task in tasks:
        photo, held = task_photo(task), _reference(task)
        lost = [f for f in labels.get(photo, {}).get("boxes", []) if not any(_same(f, h) for h in held)]
        if lost:
            problems.append(f"{photo}: {len(lost)} of its frozen date boxes moved or deleted")
    return problems


def added_dates(tasks: list[dict], labels: dict[str, dict]) -> dict[str, list[dict]]:
    """photo -> the date boxes the reviewer added (label, 0..1 xyxy), for
    photos with any: dates v1's frozen labels missed, kept apart from them."""
    out = {}
    for task in tasks:
        photo = task_photo(task)
        frozen = labels.get(photo, {}).get("boxes", [])
        new_ids = {h["id"] for h in _reference(task) if not any(_same(f, h) for f in frozen)}
        new = [rect_box(r["value"]) for a in submitted(task)[:1]
               for r in regions(a, REFERENCE_CONTROL) if r["id"] in new_ids]
        if new:
            out[photo] = new
    return out


def ocr_help(proposed: dict[str, list[dict]], final: dict[str, list[dict]]) -> Counter:
    """What the review did to OCR's proposals, matched by region id (Label
    Studio keeps a predicted region's id in the annotation made from it):
    (outcome, label) -> count. A proposal is kept (the same label, its box
    within 1e-6), moved (its box or label changed; 'relabelled' counts the
    label changes among them) or deleted, each under its proposed label; a
    box with no proposal's id is added, under its own label."""
    counts = Counter()
    for photo in sorted(set(proposed) | set(final)):
        left = {b["id"]: b for b in final.get(photo, [])}
        for p in proposed.get(photo, []):
            b = left.pop(p["id"], None)
            if b is None:
                counts["deleted", p["label"]] += 1
            elif b["label"] == p["label"] and max(abs(x - y) for x, y in zip(b["xyxy"], p["xyxy"])) <= 1e-6:
                counts["kept", p["label"]] += 1
            else:
                counts["moved", p["label"]] += 1
                counts["relabelled", p["label"]] += b["label"] != p["label"]
        for b in left.values():
            counts["added", b["label"]] += 1
    return counts


def overlap_bands(proposed: dict[str, list[dict]], final: dict[str, list[dict]]) -> Counter:
    """How far each proposal that survived the review (kept or moved) is
    from its final box, by IoU: '0.7+', '0.5-0.7' or 'under 0.5'. A
    proposal at 0.5 or more found its keyword; its box only needed fixing."""
    bands = Counter()
    for photo, given in proposed.items():
        final_of = {b["id"]: b for b in final.get(photo, [])}
        for p in given:
            if p["id"] in final_of:
                v = iou(p["xyxy"], final_of[p["id"]]["xyxy"])
                bands["0.7+" if v >= 0.7 else "0.5-0.7" if v >= 0.5 else "under 0.5"] += 1
    return bands


def reviewed_on(tasks: list[dict]) -> str | None:
    """The day of the review's last change (its latest annotation, UTC), so
    the keyword file is dated by its export and rebuilds from it; None when
    the export carries no dates."""
    days = [a["updated_at"][:10] for t in tasks for a in submitted(t) if a.get("updated_at")]
    return max(days) if days else None


def keywords_file(boxes: dict[str, list[dict]], hashes: dict[str, str], date: str,
                  dates: dict[str, list[dict]] | None = None, version: str = "v1") -> dict:
    """v1_keywords.json's content (or another set's): every photo's keyword boxes (label and
    0..1 xyxy of the upright photo, as its date boxes in v1's labels), the
    date boxes the reviewer added where v1's labels missed a date (v1's
    frozen labels don't change), and the hashes of what they were made from."""
    return {"version": version, "imported": date, **hashes,
            "photos": {p: [{"label": b["label"], "xyxy": b["xyxy"]} for b in boxes[p]]
                       for p in sorted(boxes)},
            "added_dates": {p: dates[p] for p in sorted(dates or {})}}


def read_keywords(path: Path | None = None) -> dict[str, list[dict]]:
    """photo -> its keyword boxes, from v1_keywords.json."""
    return json.loads((path or KEYWORDS_PATH).read_text())["photos"]


# --- The export for training (S6) ----------------------------------------------
#
# Store sets v1 and v3 as Ultralytics reads training data ("The export").
# Each photo is decoded from the very bytes its frozen hash names, turned
# upright (its EXIF turn applied: Ultralytics would read the file without
# it), put into sRGB (v3's photos are Display P3, whose numbers a detector
# would read as washed-out sRGB), shrunk to at most MAX_SIDE with an area
# filter, and saved as a numbered JPEG without metadata (the EXIF of 20 of
# v1's photos holds a GPS position). Its dates become `date` boxes, its
# keyword boxes keep their class. The training part is also turned 90, 180
# and 270 degrees (derived.turn); validation stays as it is. Every file is
# read back and checked, and the manifest holds every file's SHA-256, so a
# copy (the cluster's) is checked against it by check_export. Photo and
# design names stay on the laptop, in NAMES_PATH, outside the export
# folder: the export names photos by number and designs d01, d02, ...
# v2, the test set, is never exported.

EXPORT_SETS = ("v1", "v3")
MAX_SIDE = 1600
# Each part's folders, and the quarter turns (anticlockwise, derived.turn) they hold.
FOLDERS = {"train": {"train_0": 0, "train_90": 1, "train_180": 2, "train_270": 3},
           "val": {"val": 0}}
LISTS = {"train": "train.txt", "val": "val.txt"}
MANIFEST = "manifest.json"
NAMES_PATH = store.DATA_DIR / "train_names.json"
# Every store date is a `date` to the detector: telling them apart is M3's job.
YOLO_LABEL = {"expiry": "date", "mfg": "date", "other_date": "date",
              "due": "due", "prod": "prod", "code": "code"}
# A box read back from its label file may be off by the file's rounding
# (6 decimals: under 0.005 px at 1,600 px); more is a box gone wrong.
BOX_TOLERANCE_PX = 0.01
# A turned copy, turned back, differs from its upright copy only by JPEG's
# noise: a mean absolute difference of at most 1.19 grey levels on v1 and
# v3; a 180-degree copy left unturned, a wrong turn, 3.4 to 99 (median 59;
# the least on a photo 97% white). Measured in S6.
TURN_NOISE = 2.0
# A box under this across on its exported photo holds no print: a slip of
# the pen, left out and counted (Ultralytics drops boxes of 2 px or less
# after augmentation too). v1's frozen labels hold one, 1.1 by 3.8 px, the
# only box under 7 px of 634 (S6); their file keeps it, as M0 to M2 were
# scored on it.
MIN_BOX_PX = 2.0


@dataclass(frozen=True)
class ExportPhoto:
    """One store photo as the export writes it."""
    name: str                   # v1-0001: its files' stem, numbered within its set in photo order
    photo: str                  # its path under $STORE_PHOTOS_DIR (laptop only)
    sha256: str                 # its frozen SHA-256 (laptop only)
    part: str                   # train or val
    design: str                 # d01, d02, ...: its design, numbered over every exported design
    design_name: str            # the design's id (laptop only)
    tag: str
    boxes: tuple[Box, ...]      # store labels: its frozen dates, the dates its review added, its keywords
    added: int = 0              # how many of its dates its keyword review added (other_date)
    dropped: tuple[str, ...] = ()   # the labels of its boxes under MIN_BOX_PX, left out (without_slips)


def _box_problem(photo: str, b: dict) -> str | None:
    x1, y1, x2, y2 = b["xyxy"]
    if b["label"] not in YOLO_LABEL:
        return f"{photo}: a box labelled {b['label']}"
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return f"{photo}: a {b['label']} box outside the photo or without area: {b['xyxy']}"
    return None


def export_plan(sets: dict[str, dict], designs: dict[str, str], split: dict[str, str]) -> list[ExportPhoto]:
    """Every photo to export, numbered, from the frozen files. `sets` maps
    each set's version to {"labels": its frozen labels, each with its
    photo's sha256, "keywords": its keyword file's content}; `designs` and
    `split` hold every photo's design and side. Refuses a set other than v1
    and v3, a photo outside its set's folders (above all a v2 photo), one
    without a design, a side or a keyword review, a design on both sides,
    and a box outside its photo or without area. Problems name photos, so
    they're for the screen only."""
    if unknown := sorted(set(sets) - set(EXPORT_SETS)):
        raise ValueError(f"only store sets {', '.join(EXPORT_SETS)} are exported for training, "
                         f"not {', '.join(unknown)}")
    problems, sides = [], {}
    for version, given in sets.items():
        labels, keywords = given["labels"], given["keywords"]
        for photo in sorted(labels):
            top = photo.split("/")[0]
            if top in store.V2.folders:
                problems.append(f"{photo}: a store set v2 photo; v2 is the test set, never exported")
            elif top not in store.SETS[version].folders:
                problems.append(f"{photo}: not in store set {version}'s folders")
            if photo not in designs:
                problems.append(f"{photo}: no design")
            if split.get(photo) not in FOLDERS:
                problems.append(f"{photo}: on neither side of the split")
            elif photo in designs:
                sides.setdefault(designs[photo], set()).add(split[photo])
            if photo not in keywords["photos"]:
                problems.append(f"{photo}: no keyword review")
                continue
            for b in labels[photo]["boxes"] + keywords["added_dates"].get(photo, []) + keywords["photos"][photo]:
                if why := _box_problem(photo, b):
                    problems.append(why)
        problems += [f"{p}: keyword boxes, but not labelled in store set {version}"
                     for p in sorted(set(keywords["photos"]) - set(labels))]
    problems += [f"design {d} on both sides of the split" for d, s in sorted(sides.items()) if len(s) > 1]
    if problems:
        raise ValueError(f"{len(problems)} problems, nothing exported:\n  " + "\n  ".join(problems))

    exported = sorted({designs[p] for given in sets.values() for p in given["labels"]})
    number = {d: f"d{k:02d}" for k, d in enumerate(exported, start=1)}
    plan = []
    for version in (v for v in EXPORT_SETS if v in sets):
        labels, keywords = sets[version]["labels"], sets[version]["keywords"]
        for k, photo in enumerate(sorted(labels), start=1):
            added = keywords["added_dates"].get(photo, [])
            boxes = labels[photo]["boxes"] + added + keywords["photos"][photo]
            plan.append(ExportPhoto(f"{version}-{k:04d}", photo, labels[photo]["sha256"], split[photo],
                                    number[designs[photo]], designs[photo], labels[photo]["tag"],
                                    tuple(Box(b["label"], tuple(b["xyxy"])) for b in boxes), len(added)))
    return plan


def to_srgb(image: Image.Image, icc: bytes | None) -> Image.Image:
    """The image in sRGB, from the colour profile its file carried (v3's
    are Display P3). Without one it is taken to be sRGB already, as v1's."""
    if not icc:
        return image
    return ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                     ImageCms.createProfile("sRGB"),
                                     renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB")


def shrink(image: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """The long side shrunk to at most max_side with an area filter (PIL's
    BOX), as M2's backgrounds (synth/compose.py); a smaller photo as it is.
    Boxes are fractions of the photo, so they don't move."""
    scale = min(1.0, max_side / max(image.size))
    size = (round(image.width * scale), round(image.height * scale))
    return image if size == image.size else image.resize(size, Image.Resampling.BOX)


def export_image(data: bytes) -> Image.Image:
    """A photo's file, as bytes, the way the export saves it: upright
    (load_upright reads anything PIL opens), in sRGB, shrunk."""
    with Image.open(io.BytesIO(data)) as im:
        icc = im.info.get("icc_profile")
    return shrink(to_srgb(load_upright(io.BytesIO(data)), icc))


def without_slips(p: ExportPhoto, size: tuple[int, int]) -> ExportPhoto:
    """`p` without its boxes under MIN_BOX_PX across on its exported photo
    of `size`, their labels in `dropped`; an added date left out leaves the
    count of added dates too."""
    keep = [(x2 - x1) * size[0] >= MIN_BOX_PX and (y2 - y1) * size[1] >= MIN_BOX_PX
            for x1, y1, x2, y2 in (b.xyxy for b in p.boxes)]
    dates = sum(b.label in store.BOX_LABELS for b in p.boxes)     # the frozen ones, then the added ones
    return replace(p, boxes=tuple(b for b, k in zip(p.boxes, keep) if k),
                   added=sum(keep[dates - p.added:dates]),
                   dropped=p.dropped + tuple(b.label for b, k in zip(p.boxes, keep) if not k))


def turn_back_error(upright: Path, turned: Path, quarter: int) -> tuple[float, float]:
    """A turned copy against its upright copy, both as written: the mean
    absolute difference of their pixels once the turned one is turned back,
    and the largest difference between their boxes, in pixels."""
    back, boxes = derived.turn(load_upright(turned), derived.read_boxes(turned), 4 - quarter)
    first = load_upright(upright)
    if back.size != first.size:
        return float("inf"), float("inf")
    mad = float(np.abs(np.asarray(back, np.int16) - np.asarray(first, np.int16)).mean())
    w, h = first.size
    off = [abs(p - q) * s for a, b in zip(boxes, derived.read_boxes(upright), strict=True)
           for p, q, s in zip(a.xyxy, b.xyxy, (w, h, w, h))]
    return mad, max(off, default=0.0)


def _counts(plan: list[ExportPhoto]) -> dict:
    """Photos per set and part, images per folder, designs, boxes per class,
    dates per label and the boxes left out as slips, by label, in each part
    (per photo: a training photo is in every training folder too)."""
    of = {part: [p for p in plan if p.part == part] for part in FOLDERS}
    return {
        "photos": {v: {part: sum(p.name.startswith(f"{v}-") for p in of[part]) for part in FOLDERS}
                   for v in EXPORT_SETS},
        "images": {folder: len(of[part]) for part in FOLDERS for folder in FOLDERS[part]},
        "designs": {part: len({p.design for p in of[part]}) for part in FOLDERS},
        "boxes": {part: dict(sorted(Counter(YOLO_LABEL[b.label] for p in of[part] for b in p.boxes).items()))
                  for part in FOLDERS},
        "dates": {part: {**{label: sum(b.label == label for p in of[part] for b in p.boxes)
                            for label in store.BOX_LABELS},
                         "added": sum(p.added for p in of[part])} for part in FOLDERS},
        "dropped": {part: dict(sorted(Counter(label for p in of[part] for label in p.dropped).items()))
                    for part in FOLDERS},
    }


def write_export(plan: list[ExportPhoto], out: Path, sources: dict, root: Path,
                 names_path: Path = NAMES_PATH, made: str | None = None) -> dict:
    """Write the export into `out`, which must be empty or absent, and the
    names map into `names_path`, which must be outside it; returns the
    manifest. A box under MIN_BOX_PX across is left out (without_slips).
    Each file is read back and checked: its boxes against what was meant,
    no EXIF (no turn, no GPS), and each turned copy, turned back, against
    its upright copy. The manifest goes last, once all of it holds."""
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} isn't empty: the export goes into an empty folder, "
                         "so no file of an earlier one stays")
    if names_path.resolve().is_relative_to(out.resolve()):
        raise SystemExit(f"{names_path} is inside {out}: the names map stays out of what goes to the cluster")
    worst = {"box_error_px": 0.0, "turn_back_mad": 0.0, "turn_back_box_px": 0.0}
    exported = []
    for p in plan:
        data = (root / p.photo).read_bytes()
        if hashlib.sha256(data).hexdigest() != p.sha256:
            raise RuntimeError(f"{p.photo} has changed since it was frozen")
        image = export_image(data)
        p = without_slips(p, image.size)
        exported.append(p)
        boxes = [Box(YOLO_LABEL[b.label], b.xyxy) for b in p.boxes]
        written = {}
        for folder, quarter in FOLDERS[p.part].items():
            sample = derived.write_yolo(out, folder, f"{p.name}.jpg", *derived.turn(image, boxes, quarter))
            error = derived.check_written(sample)
            if error > BOX_TOLERANCE_PX:
                raise RuntimeError(f"{sample.image}: a box read back {error:.3f} px from where it was meant")
            worst["box_error_px"] = max(worst["box_error_px"], error)
            with Image.open(sample.image) as im:
                if im.getexif():
                    raise RuntimeError(f"{sample.image}: written with EXIF")
            written[quarter] = sample.image
        for quarter, path in written.items():
            if quarter:
                mad, px = turn_back_error(written[0], path, quarter)
                if mad > TURN_NOISE or px > BOX_TOLERANCE_PX:
                    raise RuntimeError(f"{path}, turned back, isn't its upright copy: "
                                       f"{mad:.2f} grey levels, boxes {px:.3f} px apart")
                worst["turn_back_mad"] = max(worst["turn_back_mad"], mad)
                worst["turn_back_box_px"] = max(worst["turn_back_box_px"], px)
    for part, name in LISTS.items():
        lines = [f"./images/{folder}/{p.name}.jpg\n" for folder in FOLDERS[part] for p in exported if p.part == part]
        (out / name).write_text("".join(lines))
    files = {f.relative_to(out).as_posix(): store.sha256(f)
             for f in sorted(out.rglob("*")) if f.is_file()}
    manifest = {
        "made": made or date.today().isoformat(), **git_state(),
        "code_sha256": code_sha256(sys.modules[__name__], derived, images),
        "settings": {"max_side": MAX_SIDE, "resample": "PIL BOX", "colour": "sRGB",
                     "jpeg_quality": derived.JPEG_QUALITY, "folders": FOLDERS, "lists": LISTS,
                     "min_box_px": MIN_BOX_PX},
        "sources": sources, "counts": _counts(exported), "worst": {k: round(v, 4) for k, v in worst.items()},
        "photos": {p.name: {"part": p.part, "design": p.design, "tag": p.tag,
                            "labels": [b.label for b in p.boxes], "added_dates": p.added,
                            "dropped": list(p.dropped)} for p in exported},
        "files": files,
    }
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    names = {"made": manifest["made"], "export": str(out),
             "manifest_sha256": hashlib.sha256(text.encode()).hexdigest(),
             "photos": {p.name: {"photo": p.photo, "sha256": p.sha256} for p in exported},
             "designs": dict(sorted({p.design: p.design_name for p in exported}.items()))}
    names_path.write_text(json.dumps(names, indent=1, sort_keys=True) + "\n")
    (out / MANIFEST).write_text(text)
    return manifest


def check_export(folder: Path, names: tuple[str, ...] = ("names.json", "train_names.json")) -> list[str]:
    """Problems with an export as found in `folder`, the laptop's or a copy
    of it (the cluster's), against its manifest: every file it lists, with
    its SHA-256; no other file in its photo and label folders; image counts
    as the manifest's; lists naming only its photos; and no names map. Uses
    nothing but hashlib, json and pathlib, so its source runs as it is where
    this package isn't installed."""
    path = folder / "manifest.json"
    if not path.is_file():
        return [f"no manifest.json in {folder}"]
    manifest = json.loads(path.read_text())
    listed, problems = manifest["files"], []
    for rel, digest in sorted(listed.items()):
        f = folder / rel
        if not f.is_file():
            problems.append(f"missing: {rel}")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            problems.append(f"changed: {rel}")
    for part in manifest["settings"]["folders"].values():
        for name in part:
            for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
                d = folder / kind / name
                found = {f.relative_to(folder).as_posix() for f in d.rglob("*") if f.is_file()} if d.is_dir() else set()
                problems += [f"not in the manifest: {r}" for r in sorted(found - set(listed))]
                n = sum(r.endswith(ext) for r in found)
                if n != manifest["counts"]["images"][name]:
                    problems.append(f"{kind}/{name}: {n} {ext} files, the manifest counts "
                                    f"{manifest['counts']['images'][name]}")
    for name in manifest["settings"]["lists"].values():
        lines = (folder / name).read_text().splitlines() if (folder / name).is_file() else []
        problems += [f"{name} names {line}, not one of the export's photos" for line in lines
                     if not (line.startswith("./images/") and line.endswith(".jpg") and line[2:] in listed)]
    problems += [f"a names map in the export: {f.relative_to(folder)}" for n in names for f in folder.rglob(n)]
    return problems


def check_v2_frozen() -> None:
    """The export starts only once v2, the test set, is frozen: its photo
    list and labels hashed in store.py (M0 D3, D6). Only those hashes are
    read here; no v2 file is opened."""
    if store.PHOTOS_V2_SHA256 is None or store.TEST_SET_V2_SHA256 is None:
        raise SystemExit("store set v2 isn't frozen yet (M0 D6): no store photo is exported "
                         "for training before the test set is frozen")


def export_inputs() -> tuple[dict[str, dict], dict[str, str], dict[str, str], dict]:
    """The export's inputs, each from its frozen file and checked against
    its hash: v1's and v3's labels (every photo hashed), keyword files,
    designs and splits; and the hashes of all of them, for the manifest.
    Refuses to start while v2 isn't frozen, or if the designs were typed
    against other labels than v1's and v2's frozen ones."""
    check_v2_frozen()
    designs = read_designs()
    if json.loads(DESIGNS_PATH.read_bytes())["test_sets"] != {"v1": store.TEST_SET_SHA256,
                                                             "v2": store.TEST_SET_V2_SHA256}:
        raise SystemExit(f"{DESIGNS_PATH} was typed against other labels than v1's and v2's frozen ones")
    split = {**v1_split(), **v3_split()}
    designs_v3 = read_designs(DESIGNS_V3_PATH, store.DESIGNS_V3_SHA256)
    sets, sources = {}, {}
    for s in (store.V1, store.V3):
        path = keywords_path(s)
        data = path.read_bytes()
        keywords = json.loads(data)
        if keywords["version"] != s.version or keywords["test_set_sha256"] != s.test_set_sha256:
            raise SystemExit(f"{path} wasn't made against store set {s.version}'s frozen labels")
        sets[s.version] = {"labels": store.read_frozen(s)["photos"], "keywords": keywords}
        sources[s.version] = {"labels": s.test_set_sha256, "keywords": hashlib.sha256(data).hexdigest()}
    sources["v1"].update(designs=store.DESIGNS_SHA256, split=SPLIT_SHA256)
    sources["v3"].update(photo_list=store.PHOTOS_V3_SHA256, designs=store.DESIGNS_V3_SHA256, split=SPLIT_V3_SHA256)
    sources["v2"] = {"exported": False, "labels": store.TEST_SET_V2_SHA256, "photo_list": store.PHOTOS_V2_SHA256}
    photos = {p for given in sets.values() for p in given["labels"]}
    return sets, {p: d for p, d in {**designs, **designs_v3}.items() if p in photos}, split, sources
