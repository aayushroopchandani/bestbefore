"""Store set v1 as training data (docs/m2b-store-and-size.md, "The data"):
pack design ids for v1 and v2, v1 split by design, and v1's keyword boxes.

    data/store/designs.csv    photo,design, one row per photo of v1 and v2,
                              named from the photos (S4)
    data/store/designs.json   the same ids, frozen by scripts/store_designs.py
                              --freeze; its SHA-256 is store.DESIGNS_SHA256
    data/store/keywords/      OCR's keyword proposals, the Label Studio tasks
                              and the reviewed export (S5)
    data/store/v1_keywords.json  the due, prod and code boxes on each v1
                              photo, written by scripts/import_store_keywords.py

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
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from . import store
from .detection_eval import iou
from .store_import import check_photos, photo_id, rect_box

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


def check_reviewed(tasks: list[dict], photos: list[str]) -> list[str]:
    """Every v1 photo once and nothing else, each with exactly one
    annotation, submitted (not skipped), and no draft left: Label Studio
    keeps a draft while boxes are drawn or changed, and only Submit (or
    Update) turns it into the annotation (M0, D5)."""
    problems = check_photos([task_photo(t) for t in tasks], photos, "store set v1")
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


def keywords_file(boxes: dict[str, list[dict]], hashes: dict[str, str], date: str,
                  dates: dict[str, list[dict]] | None = None) -> dict:
    """v1_keywords.json's content: every v1 photo's keyword boxes (label and
    0..1 xyxy of the upright photo, as its date boxes in v1's labels), the
    date boxes the reviewer added where v1's labels missed a date (v1's
    frozen labels don't change), and the hashes of what they were made from."""
    return {"version": "v1", "imported": date, **hashes,
            "photos": {p: [{"label": b["label"], "xyxy": b["xyxy"]} for b in boxes[p]]
                       for p in sorted(boxes)},
            "added_dates": {p: dates[p] for p in sorted(dates or {})}}


def read_keywords(path: Path | None = None) -> dict[str, list[dict]]:
    """photo -> its keyword boxes, from v1_keywords.json."""
    return json.loads((path or KEYWORDS_PATH).read_text())["photos"]
