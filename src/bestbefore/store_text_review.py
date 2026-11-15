"""M3 S4: the store text, checked by the owner in Label Studio
(docs/m3-date-reader.md, decision 6, "Store text (S4)", S4).

scripts/propose_store_text.py makes one task per exported store photo
(store sets v1 and v3, training and validation; never v2): the photo
upright, and each of its date boxes in text.json (the frozen labels' and
the four the keyword review added, in the export's order) as a region
carrying OCR's read as its text and the turn OCR chose, as a prediction.
The owner corrects every text, sets every turn, and draws a `life` box
round every shelf-life phrase with its text and turn.
scripts/import_store_text.py reads Label Studio's full JSON export back
through the checks here and writes data/store/text.json.

The proposals follow S2's recipe (pad 0, read both ways): PP-OCRv6 small's
recogniser alone on the box, read at its two quarter turns (decision 14),
the read that holds a date winning, then the more confident.

A turn is picked by which way the print reads, never as degrees, and
mapped to quarter turns by store_text.TURNS. Everything here names
photos: what it writes stays under data/store/.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from urllib.parse import quote

from . import store, store_text, store_train
from .reader import parse
from .samples import Expiry
from .store_import import rect_box

IMAGE = store_train.IMAGE
DATE_CONTROL, LIFE_CONTROL = "date", "life"
TEXT_CONTROL, TURN_CONTROL = "text", "turn"
MODEL_VERSION = "ppocrv6-small-rec-alone-pad0-turned"
CHOICE = {k: c for c, k in store_text.TURNS.items()}
UNREADABLE = "#"            # typed for a character that can't be read even zoomed in
# Decision 10's characters, either case: digits, letters, space and . / - : , ( ) & |
CHARSET = frozenset("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ./-:,()&|")
SAME = 1e-6                 # a box unchanged: every edge within this, in 0..1 (M2b S5's check 4)


# --- Tasks ------------------------------------------------------------------------------

def _regions(region_id: str, control: str, label: str, xyxy, size: tuple[int, int],
             text: str | None, turn: int | None) -> list[dict]:
    """A box as Label Studio results sharing one region id: its rectangle,
    and its text and turn when given (per-region controls)."""
    rect = store_train.ls_region(region_id, control, {"label": label, "xyxy": xyxy}, size)
    shape = {k: rect["value"][k] for k in ("x", "y", "width", "height", "rotation")}
    common = {"id": region_id, "to_name": IMAGE, "original_width": size[0], "original_height": size[1],
              "image_rotation": 0}
    out = [rect]
    if text:
        out.append({**common, "from_name": TEXT_CONTROL, "type": "textarea", "value": {**shape, "text": [text]}})
    if turn is not None:
        out.append({**common, "from_name": TURN_CONTROL, "type": "choices",
                    "value": {**shape, "choices": [CHOICE[turn]]}})
    return out


def note(n_boxes: int, derived: bool) -> str:
    """The line above a task's photo: what it asks for."""
    if n_boxes:
        head = f"{n_boxes} date box{'es' if n_boxes > 1 else ''}: click each, check its text and which way it reads."
    else:
        head = "No date box on this photo."
    if derived:
        return head + " Its expiry is worked out from a shelf life: draw a life box round the phrase."
    return head + " Draw a life box round any shelf-life phrase, if one is printed."


def task(photo: str, size: tuple[int, int], boxes: list[dict], proposals: dict[int, dict],
         derived: bool) -> dict:
    """One photo's task. `boxes`: its date boxes as text.json lists them,
    with their 0..1 xyxy ({"index", "label", "xyxy"}); `proposals`: each
    box's index -> OCR's {"text", "turn"}. Region ids are date<index>."""
    result = []
    for b in boxes:
        p = proposals.get(b["index"], {})
        result += _regions(f"date{b['index']}", DATE_CONTROL, b["label"], b["xyxy"], size,
                           p.get("text"), p.get("turn"))
    return {"data": {"image": f"/data/local-files/?d={quote(photo)}", "note": note(len(boxes), derived)},
            "predictions": [{"model_version": MODEL_VERSION, "result": result}]}


# --- Reading the review back --------------------------------------------------------------

def _by_region(task_: dict) -> dict[str, dict]:
    """The submitted annotation's results grouped by region id: the
    rectangle, the texts and the turns each region carries."""
    out: dict[str, dict] = defaultdict(lambda: {"rects": [], "texts": [], "turns": [], "other": []})
    for a in store_train.submitted(task_)[:1]:
        for r in a["result"]:
            g = out[r.get("id")]
            if r.get("type") == "rectanglelabels" and r.get("from_name") in (DATE_CONTROL, LIFE_CONTROL):
                g["rects"].append(r)
            elif r.get("type") == "textarea" and r.get("from_name") == TEXT_CONTROL:
                g["texts"] += r["value"].get("text", [])
            elif r.get("type") == "choices" and r.get("from_name") == TURN_CONTROL:
                g["turns"] += r["value"].get("choices", [])
            else:
                g["other"].append(f"{r.get('from_name')}/{r.get('type')}")
    return out


def _same(a, b) -> bool:
    return max(abs(x - y) for x, y in zip(a, b)) <= SAME


def read_review(tasks: list[dict], planned: dict[str, list[dict]]) -> tuple[dict[str, dict], list[str]]:
    """Each photo's reviewed boxes and the problems found. `planned`: photo
    -> its date boxes ({"index", "label", "xyxy"}). Every planned date box
    must be there once, its place and label unchanged, and no other date
    box; every box (date or life) must carry one text, not blank, and one
    turn. Returns photo -> {"boxes": index -> {"text", "turn", "id"},
    "life": [{"xyxy", "text", "turn", "id"}]} (life boxes top to bottom)."""
    reviewed, problems = {}, []
    for t in tasks:
        photo = store_train.task_photo(t)
        boxes, life = {}, []
        for rid, g in sorted(_by_region(t).items(), key=lambda kv: str(kv[0])):
            where = f"{photo}: region {rid}"
            if g["other"]:
                problems.append(f"{where} carries results of another kind: {g['other']}")
            if len(g["rects"]) != 1:
                problems.append(f"{where}: {len(g['rects'])} boxes with one id (a text or turn without its box?)")
                continue
            rect = g["rects"][0]
            box = rect_box(rect["value"])
            texts = [s.strip() for s in g["texts"]]
            if len(texts) != 1 or not texts[0]:
                problems.append(f"{where} ({box['label']}): {len(texts)} texts {texts}, not one")
            if len(g["turns"]) != 1 or g["turns"][0] not in store_text.TURNS:
                problems.append(f"{where} ({box['label']}): turn {g['turns']}, not one of {list(store_text.TURNS)}")
            text = " ".join(texts[0].split()) if texts else ""
            turn = store_text.TURNS.get(g["turns"][0]) if len(g["turns"]) == 1 else None
            if rect["from_name"] == LIFE_CONTROL:
                life.append({"xyxy": box["xyxy"], "text": text, "turn": turn, "id": rid})
                continue
            match = [b for b in planned.get(photo, []) if b["label"] == box["label"] and _same(b["xyxy"], box["xyxy"])]
            if not match:
                problems.append(f"{where}: a {box['label']} box that isn't one of the photo's date boxes "
                                "(moved, relabelled or added)")
            elif match[0]["index"] in boxes:
                problems.append(f"{where}: a second box on date line {match[0]['index']}")
            else:
                boxes[match[0]["index"]] = {"text": text, "turn": turn, "id": rid}
        lost = [b["index"] for b in planned.get(photo, []) if b["index"] not in boxes]
        if lost:
            problems.append(f"{photo}: date lines {lost} lost (moved, relabelled or deleted)")
        reviewed[photo] = {"boxes": boxes, "life": sorted(life, key=lambda b: (b["xyxy"][1], b["xyxy"][0]))}
    return reviewed, problems


# --- Values ---------------------------------------------------------------------------------

_DOTTED = re.compile(r"(?<!\d)(\d{2})\.(\d{2})\.(\d{2})(?!\d)")


def _date(*parts) -> Expiry | None:
    try:
        return Expiry(*parts)
    except ValueError:
        return None


def value_reason(text: str, value: str) -> str | None:
    """None when the frozen parser reads `text` as the frozen `value` at the
    precision the pack prints (a day only if the value has one); otherwise
    why not, as far as it can tell."""
    want = store.parse_date(value)
    found = parse.read_dates(text)
    got = parse.crop_date(found)
    if got == want:
        return None
    if not found:
        return "the frozen parser finds no date in it"
    if any(f.date == want for f in found):
        return (f"the parser's one date is {got}, the value is another of its dates "
                "(it keeps the longest, then the leftmost)")
    m = _DOTTED.search(text)
    if m and _date(2000 + int(m.group(3)), int(m.group(2)), int(m.group(1))) == want:
        return f"read year first as {got}: DD.MM.YY, which the frozen parser reads YY.MM.DD (S2's known trap)"
    if (got.year, got.month) == (want.year, want.month):
        if want.day is None:
            return f"read as {got}: a day in the text, the frozen value has none"
        if got.day is None:
            return f"read as {got}: no day in the text, the frozen value has one"
    if got.day and want.day and (got.year, got.month, got.day) == (want.year, want.day, want.month):
        return f"read as {got}: day and month swapped"
    return f"read as {got}: another date (a typo, or a label to look at again)"


def life_reason(texts: list[str], mfg: str | None, expiry: str) -> str | None:
    """None when a shelf-life phrase plus the frozen mfg date gives the
    frozen expiry (a derived photo's); otherwise why not."""
    if not texts:
        return "no life box: a derived expiry needs its shelf-life phrase"
    if mfg is None:
        return "the photo has no frozen mfg date"
    months = [parse.shelf_life(t) for t in texts]
    if not any(months):
        return f"the frozen parser reads no shelf life in {texts}"
    want, start = store.parse_date(expiry), store.parse_date(mfg)
    if any(m and parse.add_months(start, m) == want for m in months):
        return None
    got = [f"{m} months: {parse.add_months(start, m)}" for m in months if m]
    return f"mfg {mfg} plus {', '.join(got)}, the frozen expiry is {expiry}"


def mismatches(texts: dict[str, dict], values: dict[str, list[dict]], labels: dict[str, dict]) -> list[dict]:
    """Every expiry or mfg text that doesn't read as its frozen value, and
    every derived photo whose shelf life doesn't give its frozen expiry.
    `texts`: read_review's; `values`: photo -> text.json's boxes (index,
    label, value). Keys: '<photo>#<line>', or '<photo>#life'."""
    out = []
    for photo in sorted(texts):
        for b in values.get(photo, []):
            if b["label"] not in store_text.VALUE_LABELS or b["value"] is None or b["index"] not in texts[photo]["boxes"]:
                continue
            text = texts[photo]["boxes"][b["index"]]["text"]
            if why := value_reason(text, b["value"]):
                out.append({"key": f"{photo}#{b['index']}", "label": b["label"], "text": text,
                            "value": b["value"], "reason": why})
        label = labels[photo]
        if label["derived"]:
            life = [b["text"] for b in texts[photo]["life"]]
            if why := life_reason(life, label["mfg"], label["expiry"]):
                out.append({"key": f"{photo}#life", "label": "life", "text": " | ".join(life),
                            "value": f"{label['mfg']} -> {label['expiry']}", "reason": why})
    return out


def unsettled(found: list[dict], settled: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """The mismatches the owner hasn't settled, and settled keys that no
    longer mismatch (stale: to take out of settled.json)."""
    keys = {m["key"] for m in found}
    return [m for m in found if m["key"] not in settled], sorted(set(settled) - keys)


# --- Notes: not refusals, shown to the owner -----------------------------------------------

def notes(texts: dict[str, dict], planned: dict[str, list[dict]], proposals: dict[str, dict]) -> dict[str, list[str]]:
    """Things to look at, by kind: a turn that doesn't fit its box's shape
    (a tall box read upright or upside down, a wide one sideways); a turn
    other than OCR's; a character outside decision 10's set; an unreadable
    character; an other date the parser reads no date in."""
    out: dict[str, list[str]] = defaultdict(list)
    for photo in sorted(texts):
        shape = {b["index"]: b for b in planned.get(photo, [])}
        for index, r in sorted(texts[photo]["boxes"].items()):
            b, where = shape[index], f"{photo}#{index} ({shape[index]['label']}) {r['text']!r}"
            x1, y1, x2, y2 = b["xyxy"]
            if r["turn"] is not None and (r["turn"] % 2 == 1) != ((y2 - y1) > (x2 - x1)):
                out["turn and shape differ (the photo's aspect aside)"].append(f"{where}: {CHOICE[r['turn']]}")
            ocr = proposals.get(photo, {}).get(index)
            if ocr and ocr.get("turn") is not None and r["turn"] is not None and ocr["turn"] != r["turn"]:
                out["turn other than OCR's"].append(f"{where}: {CHOICE[r['turn']]}, OCR's {CHOICE[ocr['turn']]}")
            if b["label"] == "other_date" and not parse.read_dates(r["text"]):
                out["other date with no date the parser reads"].append(where)
        for k, r in enumerate(list(texts[photo]["boxes"].values()) + texts[photo]["life"]):
            if UNREADABLE in r["text"]:
                out["unreadable characters (#)"].append(f"{photo}: {r['text']!r}")
            elif odd := sorted(set(r["text"]) - CHARSET):
                out["characters outside decision 10's set"].append(f"{photo}: {r['text']!r} {odd}")
    return dict(out)


def ocr_help(texts: dict[str, dict], planned: dict[str, list[dict]], proposals: dict[str, dict]) -> Counter:
    """How OCR's proposals fared, by date label: (label, "text kept"),
    (label, "text changed"), (label, "turn kept"), (label, "turn changed")."""
    out: Counter = Counter()
    for photo in sorted(texts):
        label = {b["index"]: b["label"] for b in planned.get(photo, [])}
        for index, r in texts[photo]["boxes"].items():
            ocr = proposals.get(photo, {}).get(index, {})
            out[label[index], "text kept" if ocr.get("text") == r["text"] else "text changed"] += 1
            out[label[index], "turn kept" if ocr.get("turn") == r["turn"] else "turn changed"] += 1
    return out


def text_file(plan: list[store_train.ExportPhoto], reviewed: dict[str, dict], values: dict[str, list[dict]],
              months: dict[str, str | None], sources: dict, review: dict, date: str) -> dict:
    """data/store/text.json's content: every exported photo by its name,
    with its number, part and month, each date box (the export's line, its
    label, 0..1 xyxy, frozen value, text and turn) and each shelf-life box;
    what it was made from. Dated by the review's last change, so it rebuilds
    byte for byte from the same export."""
    photos = {}
    for p in plan:
        r, boxes = reviewed[p.photo], []
        for b, v in zip((b for b in p.boxes if b.label in store.BOX_LABELS), values[p.photo], strict=True):
            t = r["boxes"][v["index"]]
            boxes.append({"index": v["index"], "label": v["label"], "xyxy": [round(x, 6) for x in b.xyxy],
                          "value": v["value"], "text": t["text"], "turn": t["turn"]})
        photos[p.photo] = {"name": p.name, "part": p.part, "month": months[p.photo], "boxes": boxes,
                           "life": [{k: b[k] for k in ("xyxy", "text", "turn")} for b in r["life"]]}
    return {"version": "text", "reviewed": date, "sources": sources, "review": review,
            "photos": dict(sorted(photos.items()))}
