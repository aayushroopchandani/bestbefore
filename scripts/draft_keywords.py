"""Draft keyword boxes on a store set for the owner's review: OCR's
proposals snapped to their ink, then Claude's edits from looking at each
photo (docs/m2b-store-and-size.md, S5; changed 2026-11-03 at the owner's
request, as S4's design ids were; v3 in S5b).

    python scripts/draft_keywords.py                 # v1: data/store/keywords/draft.json, draft_predictions.json
    python scripts/draft_keywords.py --sheet         # and one picture per photo in data/store/runs/
    python scripts/draft_keywords.py --version v3    # v3: the same in data/store/v3/keywords/

Reads the set's keywords folder: lines.json and proposals.json (propose_keywords.py)
and edits.json, Claude's edits per photo, a list of:

    {"op": "delete", "id": "kw3"}                       OCR's box is wrong
    {"op": "label", "id": "kw2", "label": "prod"}       right place, wrong label
    {"op": "extend", "id": "kw1", "line": 14}           grow it over OCR line 14 (a code's value)
    {"op": "add", "label": "due", "line": 7, "chars": [0, 4]}   characters of an OCR line
    {"op": "add", "label": "code", "xyxy": [x1, y1, x2, y2]}    a box drawn by eye, 0..1
                                                                (or, with "src": "ocr2", by the second OCR pass)
    {"op": "box", "id": "kw2", "xyxy": [x1, y1, x2, y2]}        OCR's box redrawn by eye

A box drawn by eye can say "exact": true, to be kept as drawn: where a
tilted or crowded line would pull the snapping onto its neighbour.

Every box is then snapped to its ink (`tighten`): the line's padding
trimmed, and each end moved to the gap nearest it, so a keyword cut from
its line stops before its date. A box OCR proposed keeps its id (kw1, ...),
so S5 can measure OCR's proposals, Claude's draft and the owner's boxes
against each other; one added by eye gets cl1, cl2, ..., one from the second OCR pass
(src "ocr2": RapidOCR again on the photo's dates area, enlarged, with
lower detection thresholds) o1, o2, ... The draft goes to
Label Studio as a second prediction (MODEL_VERSION below) beside OCR's.
Only store sets v1 and v3, never v2; everything written names photos, so it
stays in data/store/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

from bestbefore import store, store_train
from bestbefore.config import store_photos_dir
from bestbefore.images import load_upright
from bestbefore.runs import code_sha256, start_run
from bestbefore.sheets import FONT

MODEL_VERSION = "claude-draft-1"
SHEET_WIDTH = 1300
COLOURS = {"due": (255, 127, 14), "prod": (44, 160, 44), "code": (148, 103, 189),
           "expiry": (214, 39, 40), "mfg": (31, 119, 180), "other_date": (127, 127, 127)}


def otsu(values: np.ndarray) -> float:
    """Otsu's threshold of grey levels 0..255."""
    hist = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(float)
    total, levels = hist.sum(), np.arange(256)
    w0, m0 = np.cumsum(hist), np.cumsum(hist * levels)
    w1, mean = total - w0, m0[-1] / total
    between = (mean * w0 - m0) ** 2 / np.maximum(w0 * w1, 1e-9)
    return float(np.argmax(between))


def _runs(flags: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Runs of True as (start, end), joined across gaps of up to `gap`."""
    runs = []
    for i in np.flatnonzero(flags):
        if runs and i - runs[-1][1] <= gap + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b + 1) for a, b in runs]


def tighten(gray: np.ndarray, box: tuple[int, int, int, int], along_x: bool,
            snap: bool = True, stops: list[tuple[float, float]] = ()) -> tuple[int, int, int, int]:
    """A box in pixels of `gray` shrunk to its ink. Across the line: the
    band of rows holding ink (the largest, if several). Along it: each end
    moved to the faintest column within half a character of it, then to
    the ink beside that. `stops` are the spans along the line of the date
    boxes next to it (drawn by hand, tight): an end facing one goes to the
    last ink before it, so a keyword keeps its colon and stops short of
    its date. Ink is the minority side of an Otsu split inside the box, so
    light print on dark works too. A box that would shrink to almost
    nothing is kept as it was."""
    if not along_x:
        y1, x1, y2, x2 = tighten(gray.T, (box[1], box[0], box[3], box[2]), True, snap, stops)
        return x1, y1, x2, y2
    H, W = gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return box
    h = y2 - y1
    inside = gray[y1:y2, x1:x2]
    t = otsu(inside)
    dark = inside <= t
    ink_dark = dark.mean() <= 0.5
    e = max(2, round(0.3 * h))
    cx1, cx2 = max(0, x1 - e), min(W, x2 + e)
    crop = gray[y1:y2, cx1:cx2]
    mask = (crop <= t) if ink_dark else (crop > t)
    rows = mask[:, x1 - cx1:x2 - cx1].mean(axis=1) > 0.02
    bands = _runs(rows, max(1, round(0.08 * h)))
    if not bands:
        return box
    a, b = max(bands, key=lambda r: (mask[r[0]:r[1]].sum(), -r[0]))
    cols = mask[a:b].sum(axis=0)
    inked = np.flatnonzero(cols > max(1, 0.06 * (b - a)))     # more than a speck of shadow
    if len(inked) == 0:
        return box

    def end_at(cut: int, first: bool) -> int:
        """The ink edge next to the faintest column within e of `cut`."""
        lo, hi = max(0, cut - e), min(len(cols), cut + e + 1)
        if not snap:
            gap = cut
        else:
            window = cols[lo:hi]
            candidates = np.flatnonzero(window == window.min()) + lo
            gap = int(candidates[np.argmin(np.abs(candidates - cut))])
        if first:
            after = inked[inked >= gap]
            return int(after[0]) if len(after) else cut
        before = inked[inked < gap] if snap else inked[inked <= gap]
        return int(before[-1]) + 1 if len(before) else cut

    nx1, nx2 = end_at(x1 - cx1, True), end_at(x2 - cx1, False)
    for s0, s1 in ((s0 - cx1, s1 - cx1) for s0, s1 in stops):
        if s0 >= (nx1 + nx2) / 2 and s0 < nx2 + 1.5 * (b - a):         # the date follows
            before = inked[(inked < s0 - 1) & (inked >= nx1)]
            nx2 = int(before[-1]) + 1 if len(before) else nx2
        elif s1 <= (nx1 + nx2) / 2 and s1 > nx1 - 1.5 * (b - a):       # it comes first (upside down)
            after = inked[(inked > s1 + 1) & (inked < nx2)]
            nx1 = int(after[0]) if len(after) else nx1
    if nx2 - nx1 < 3 or (nx2 - nx1) < 0.25 * (x2 - x1):
        return box
    return cx1 + nx1, y1 + a, cx1 + nx2, y1 + b


def line_angle(quad) -> float:
    """The angle of a line's reading direction from the x axis, in degrees."""
    (x0, y0), (x1, y1) = quad[0], quad[1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def beside(xyxy, dates: list[dict], along_x: bool) -> list[tuple[float, float]]:
    """The spans along the line (0..1) of the date boxes on the same line as
    a box: across it, they overlap by half the smaller of the two."""
    a, b = (1, 3) if along_x else (0, 2)        # across the line
    u, v = (0, 2) if along_x else (1, 3)        # along it
    out = []
    for d in dates:
        d = d["xyxy"]
        overlap = min(xyxy[b], d[b]) - max(xyxy[a], d[a])
        if overlap > 0.5 * min(xyxy[b] - xyxy[a], d[b] - d[a]):
            out.append((d[u], d[v]))
    return out


def snap_box(gray: np.ndarray, xyxy, along_x: bool, tilted: bool, dates: list[dict] = ()) -> list[float]:
    """0..1 xyxy snapped to its ink: tilted print only loses its empty margins."""
    H, W = gray.shape
    px = (xyxy[0] * W, xyxy[1] * H, xyxy[2] * W, xyxy[3] * H)
    side = W if along_x else H
    stops = [(s0 * side, s1 * side) for s0, s1 in beside(xyxy, list(dates), along_x)]
    x1, y1, x2, y2 = tighten(gray, px, along_x, snap=not tilted, stops=stops)
    return [round(min(max(v, 0.0), 1.0), 6) for v in (x1 / W, y1 / H, x2 / W, y2 / H)]


def line_box(line: dict, size: tuple[int, int], chars=None) -> list[float]:
    """An OCR line's box, or that of characters [start, end) of it, 0..1."""
    start, end = chars or (0, len(line["text"]))
    x1, y1, x2, y2 = store_train.span_box(line["quad"], start, end, len(line["text"]),
                                          line["turned"], line["flipped"])
    return [x1 / size[0], y1 / size[1], x2 / size[0], y2 / size[1]]


def union(a, b) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def line_index(proposal: dict, read: list[dict], size: tuple[int, int]) -> int:
    """The OCR line a proposal was cut from: the line with its text whose
    box holds the proposal's centre."""
    cx, cy = ((proposal["xyxy"][0] + proposal["xyxy"][2]) / 2, (proposal["xyxy"][1] + proposal["xyxy"][3]) / 2)
    same = [k for k, line in enumerate(read) if line["text"] == proposal["line"]]
    for k in same:
        x1, y1, x2, y2 = line_box(read[k], size)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return k
    raise SystemExit(f"no OCR line holds proposal {proposal['id']} ({proposal['line']!r})")


def draft(photo: str, lines: dict, proposals: list[dict], edits: list[dict], gray: np.ndarray,
          dates: list[dict] = ()) -> list[dict]:
    """One photo's draft: OCR's proposals with Claude's edits, each snapped
    to its ink. A box is one or more parts (a code's label and, on another
    OCR line, its value), each snapped on its own and then joined."""
    size, read = tuple(lines["size"]), lines["lines"]
    line_of = {p["id"]: line_index(p, read, size) for p in proposals}
    boxes = {p["id"]: {"label": p["label"], "parts": [(p["xyxy"], line_of[p["id"]])]} for p in proposals}
    added = Counter()
    for op in edits:
        kind, box = op["op"], boxes.get(op.get("id"))
        if kind == "delete":
            del boxes[op["id"]]
        elif kind == "label":
            box["label"] = op["label"]
        elif kind == "extend":
            box["parts"].append((line_box(read[op["line"]], size, op.get("chars")), op["line"]))
        elif kind == "box":
            box["parts"] = [(op["xyxy"], "exact" if op.get("exact") else None)]
        elif kind == "add":
            prefix = "o" if op.get("src") == "ocr2" else "cl"
            added[prefix] += 1
            part = ((op["xyxy"], "exact" if op.get("exact") else None) if "xyxy" in op
                    else (line_box(read[op["line"]], size, op.get("chars")), op["line"]))
            boxes[f"{prefix}{added[prefix]}"] = {"label": op["label"], "parts": [part]}
        else:
            raise ValueError(f"{photo}: unknown edit {op}")
    out = []
    for box_id, b in boxes.items():
        snapped = []
        for xyxy, k in b["parts"]:
            if k == "exact":
                snapped.append([round(v, 6) for v in xyxy])
                continue
            if k is not None:
                angle, along_x = line_angle(read[k]["quad"]), not read[k]["turned"]
                off = abs(angle) % 180
                tilted = min(off, 180 - off) > 6 if along_x else abs(off - 90) > 6
            else:
                along_x = (xyxy[2] - xyxy[0]) * gray.shape[1] >= (xyxy[3] - xyxy[1]) * gray.shape[0]
                tilted = False
            snapped.append(snap_box(gray, xyxy, along_x, tilted, dates))
        xyxy = snapped[0]
        for other in snapped[1:]:
            xyxy = union(xyxy, other)
        out.append({"id": box_id, "label": b["label"], "xyxy": xyxy,
                    "from": {"kw": "ocr", "o": "ocr2", "cl": "claude"}[box_id.rstrip("0123456789")]})
    return out


def sheet_picture(im: Image.Image, read: list[dict], ocr_size, boxes: list[dict], dates: list[dict],
                  photo: str) -> tuple[Image.Image, dict]:
    """The part of the photo holding its dates and keywords (the whole photo
    when it has neither), with OCR's lines numbered, the dates thin and the
    draft's boxes thick with their ids, at SHEET_WIDTH px; and where the
    picture sits in the photo (crop, 0..1), to turn a box drawn on it back."""
    W, H = im.size
    near = [b["xyxy"] for b in boxes + dates]
    if near:
        x1, y1 = min(b[0] for b in near), min(b[1] for b in near)
        x2, y2 = max(b[2] for b in near), max(b[3] for b in near)
        mx, my = max(0.12, 0.6 * (x2 - x1)), max(0.12, 0.6 * (y2 - y1))
        x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(1, x2 + mx), min(1, y2 + my)
    else:
        x1, y1, x2, y2 = 0, 0, 1, 1
    crop = im.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))
    scale = SHEET_WIDTH / max(crop.width, crop.height * 1.2)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
    pen = ImageDraw.Draw(crop)

    def px(b):
        return ((b[0] - x1) * W * scale, (b[1] - y1) * H * scale, (b[2] - x1) * W * scale, (b[3] - y1) * H * scale)

    for k, line in enumerate(read):
        q = [((x / ocr_size[0] - x1) * W * scale, (y / ocr_size[1] - y1) * H * scale) for x, y in line["quad"]]
        if all(-50 < x < crop.width + 50 and -50 < y < crop.height + 50 for x, y in q):
            pen.polygon(q, outline=(0, 200, 220))
            pen.text((q[0][0] + 2, q[0][1] - 14), str(k), fill=(0, 140, 160), font=FONT)
    for d in dates:
        pen.rectangle(px(d["xyxy"]), outline=COLOURS[d["label"]], width=1)
    for b in boxes:
        r = px(b["xyxy"])
        pen.rectangle(r, outline=COLOURS[b["label"]], width=2)
        pen.text((r[0], r[3] + 1), b["id"], fill=COLOURS[b["label"]], font=FONT)
    out = Image.new("RGB", (crop.width, crop.height + 20), "white")
    out.paste(crop, (0, 0))
    ImageDraw.Draw(out).text((4, crop.height + 3), photo, fill="black", font=FONT)
    return out, {"photo": photo, "crop": [x1, y1, x2, y2], "picture": [crop.width, crop.height]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", default="v1", help="the store set: v1 or v3 (never v2)")
    parser.add_argument("--sheet", action="store_true", help="also one picture per photo, in data/store/runs/")
    parser.add_argument("--only", nargs="*", help="draw only these photos (all are drafted)")
    args = parser.parse_args()

    store_set = store_train.keyword_set(args.version)
    folder = store_train.keywords_dir(store_set)
    lines = json.loads((folder / "lines.json").read_text())
    proposals = json.loads((folder / "proposals.json").read_text())
    edits_path = folder / "edits.json"
    edits = json.loads(edits_path.read_text()) if edits_path.exists() else {}
    labels = store.labels(store_set)
    photos = store.photos_on_disk(store_set)
    if sorted(lines["photos"]) != photos or sorted(proposals["photos"]) != photos or set(edits) - set(photos):
        raise SystemExit(f"the OCR lines, proposals and edits must name exactly store set "
                         f"{store_set.version}'s photos")
    run_name = "keyword-draft" + ("" if store_set is store.V1 else f"-{store_set.version}")
    run_dir = start_run(run_name, vars(args), root=store.RUNS_DIR) if args.sheet else None
    drafts, results, crops = {}, [], {}
    for k, photo in enumerate(photos, start=1):
        im = load_upright(store_photos_dir() / photo)
        gray = np.asarray(im.convert("L"), np.float32)
        drafts[photo] = draft(photo, lines["photos"][photo], proposals["photos"][photo]["boxes"],
                              edits.get(photo, []), gray, labels[photo]["boxes"])
        size = tuple(proposals["photos"][photo]["size"])
        results.append({"photo": photo, "result": store_train.keyword_task(
            photo, size, drafts[photo], labels[photo])["predictions"][0]["result"]})
        if run_dir and (not args.only or photo in args.only):
            picture, crops[f"{k:03d}"] = sheet_picture(
                im, lines["photos"][photo]["lines"], lines["photos"][photo]["size"], drafts[photo],
                labels[photo]["boxes"], photo)
            picture.save(run_dir / f"{k:03d}.jpg", quality=88)
        print(f"\r{k}/{len(photos)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    record = {"model_version": MODEL_VERSION, "proposals_sha256": store.sha256(folder / "proposals.json"),
              "edits_sha256": store.sha256(edits_path) if edits_path.exists() else None,
              "code_sha256": code_sha256(store_train, sys.modules[__name__])}
    (folder / "draft.json").write_text(json.dumps({**record, "photos": drafts}, indent=1) + "\n")
    (folder / "draft_predictions.json").write_text(json.dumps(
        {"model_version": MODEL_VERSION, "predictions": results}) + "\n")
    boxes = [b for bs in drafts.values() for b in bs]
    print(f"{len(boxes)} boxes on {sum(bool(b) for b in drafts.values())} photos: "
          + ", ".join(f"{c} {n}" for c, n in sorted(Counter(b["label"] for b in boxes).items()))
          + f"; {sum(b['from'] == 'claude' for b in boxes)} added by edits; "
          f"{sum(len(e) for e in edits.values())} edits on {len(edits)} photos")
    if run_dir:
        (run_dir / "crops.json").write_text(json.dumps(crops, indent=1) + "\n")
        print(f"sheet: {run_dir}")


if __name__ == "__main__":
    main()
