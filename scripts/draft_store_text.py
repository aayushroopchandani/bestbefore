"""Draft the store text for the owner's check: OCR's proposals with Claude's
edits from looking at every date box (docs/m3-date-reader.md, S4; added
2026-11-15 at the owner's request, as M2b S5's keyword draft was).

    python scripts/draft_store_text.py --sheet                    # every box, as drafted, on sheets
    python scripts/draft_store_text.py --sheet --only v1-0001 v3-0002:1
    python scripts/draft_store_text.py --photos v1-0001 v1-0002   # whole photos, to find shelf-life phrases
    python scripts/draft_store_text.py                            # draft_predictions.json

Reads data/store/text/: proposals.json (scripts/propose_store_text.py) and
edits.json, Claude's edits by numbered photo:

    {"v1-0001": {"boxes": {"1": {"text": "EXP 08/2027", "turn": "reads bottom to top"}},
                 "life": [{"xyxy": [x1, y1, x2, y2], "text": "...", "turn": "upright"}]}}

A box's edit may give its text, its turn (a choice's name, store_text.TURNS)
or both; `life` lists the shelf-life boxes, 0..1 of the upright photo.

--sheet draws each date box (and life box) from the full-size upright photo,
turned by its drafted turn, so a box drafted the right way up reads upright
and one drafted wrong shows it; its box in red, its drafted text under it.
--photos draws whole photos with their boxes. Both go to an owner-only run
folder under data/store/runs/. Without either, the draft is written as
Label Studio predictions (MODEL_VERSION, beside OCR's), one per photo, to
data/store/text/draft_predictions.json. Only store sets v1 and v3, as
exported; never v2. Everything here names photos and stays under data/store/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bestbefore import store, store_text, store_text_review as review, store_train
from bestbefore.config import store_photos_dir, store_train_dir
from bestbefore.images import load_upright, upright_size
from bestbefore.runs import code_sha256, start_run

MODEL_VERSION = "claude-draft-1"
TEXT_H = 56                 # a box's short side on the sheet, in pixels (enlarged if smaller)
MARGIN = 0.6                # context round each box, as a share of its short side
TILE_W_MAX, SHEET_W, PER_SHEET = 1400, 1440, 14
FONT = ImageFont.load_default(size=22)
EDITS, PROPOSALS = "edits.json", "proposals.json"


def photos() -> list[dict]:
    """Every exported photo: its number, path, part, derivedness and date
    boxes ({"index", "label", "xyxy"}), as text.json lists them."""
    folder = store_train_dir()
    sets, designs, split, _ = store_train.export_inputs()
    labels = {p: lab for given in sets.values() for p, lab in given["labels"].items()}
    out = []
    for p in store_train.export_plan(sets, designs, split):
        first = next(iter(store_train.FOLDERS[p.part]))
        p = store_train.without_slips(p, upright_size(folder / "images" / first / f"{p.name}.jpg"))
        values = store_text.box_values(p, labels[p.photo])
        dated = [b for b in p.boxes if b.label in store.BOX_LABELS]
        out.append({"name": p.name, "photo": p.photo, "part": p.part, "derived": labels[p.photo]["derived"],
                    "boxes": [{"index": v["index"], "label": v["label"], "xyxy": list(b.xyxy)}
                              for b, v in zip(dated, values, strict=True)]})
    return out


def drafted(entries: list[dict]) -> dict[str, dict]:
    """name -> {"boxes": index -> {"text", "turn"}, "life": [...]}: OCR's
    proposals with Claude's edits applied; turns as quarter turns."""
    folder = store_text.REVIEW_DIR
    proposed = json.loads((folder / PROPOSALS).read_text())["photos"]
    edits = json.loads((folder / EDITS).read_text()) if (folder / EDITS).exists() else {}
    if unknown := sorted(set(edits) - {e["name"] for e in entries}):
        sys.exit(f"edits for photos not exported: {unknown}")
    out = {}
    for e in entries:
        mine, edit = proposed.get(e["name"], {}), edits.get(e["name"], {})
        if bad := sorted(set(edit.get("boxes", {})) - {str(b["index"]) for b in e["boxes"]}):
            sys.exit(f"{e['name']}: edits for date lines {bad} it doesn't have")
        boxes = {}
        for b in e["boxes"]:
            p, ed = mine[str(b["index"])], edit.get("boxes", {}).get(str(b["index"]), {})
            boxes[b["index"]] = {"text": ed.get("text", p["text"]),
                                 "turn": store_text.TURNS[ed["turn"]] if "turn" in ed else p["turn"],
                                 "edited": bool(ed)}
        life = [{"xyxy": b["xyxy"], "text": b["text"], "turn": store_text.TURNS[b["turn"]]}
                for b in edit.get("life", [])]
        out[e["name"]] = {"boxes": boxes, "life": life}
    return out


def tile(image: Image.Image, xyxy, turn: int, caption: str) -> Image.Image:
    """One box from the upright photo with MARGIN of context, turned by
    `turn` quarters anticlockwise, scaled so its short side is TEXT_H; its
    box in red; the caption above it."""
    W, H = image.size
    x1, y1, x2, y2 = xyxy[0] * W, xyxy[1] * H, xyxy[2] * W, xyxy[3] * H
    m = MARGIN * min(x2 - x1, y2 - y1)
    rect = (max(0, int(x1 - m)), max(0, int(y1 - m)), min(W, int(x2 + m) + 1), min(H, int(y2 + m) + 1))
    crop = image.crop(rect).copy()
    ImageDraw.Draw(crop).rectangle((x1 - rect[0] - 1, y1 - rect[1] - 1, x2 - rect[0], y2 - rect[1]),
                                   outline=(230, 30, 30), width=max(1, round(min(x2 - x1, y2 - y1) / 40)))
    crop = Image.fromarray(np.ascontiguousarray(np.rot90(np.asarray(crop), turn)))
    short = min(x2 - x1, y2 - y1)
    scale = TEXT_H / short
    if crop.width * scale > TILE_W_MAX:
        scale = TILE_W_MAX / crop.width
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                       Image.Resampling.LANCZOS)
    out = Image.new("RGB", (max(crop.width, 600), crop.height + 30), "white")
    ImageDraw.Draw(out).text((4, 3), caption, fill="black", font=FONT)
    out.paste(crop, (0, 30))
    return out


def stack(tiles: list[Image.Image]) -> Image.Image:
    h = sum(t.height + 12 for t in tiles)
    out = Image.new("RGB", (SHEET_W, h), (235, 235, 235))
    y = 0
    for t in tiles:
        out.paste(t, (0, y))
        y += t.height + 12
    return out


def sheets(entries: list[dict], draft: dict[str, dict], only: list[str], run) -> int:
    root, tiles, n = store_photos_dir(), [], 0
    for e in entries:
        wanted = [o.split(":")[1] for o in only if o.split(":")[0] == e["name"] and ":" in o]
        if only and e["name"] not in only and not wanted:
            continue
        boxes = [b for b in e["boxes"] if not wanted or str(b["index"]) in wanted]
        life = draft[e["name"]]["life"] if not wanted or "life" in wanted else []
        if not boxes and not life:
            continue
        image = load_upright(root / e["photo"])
        for b in boxes:
            d = draft[e["name"]]["boxes"][b["index"]]
            tiles.append(tile(image, b["xyxy"], d["turn"],
                              f"{e['name']}:{b['index']} {b['label']}  {review.CHOICE[d['turn']]}"
                              f"{'  (edited)' if d['edited'] else ''}   [{d['text']}]"))
        for k, b in enumerate(life):
            tiles.append(tile(image, b["xyxy"], b["turn"],
                              f"{e['name']}:life{k}  {review.CHOICE[b['turn']]}   [{b['text']}]"))
    for i in range(0, len(tiles), PER_SHEET):
        n += 1
        stack(tiles[i:i + PER_SHEET]).save(run / f"sheet-{n:02d}.png")
    return len(tiles)


def whole(entries: list[dict], draft: dict[str, dict], names: list[str], run) -> None:
    """Each named photo whole, at most 1,600 px, its date and life boxes drawn."""
    root = store_photos_dir()
    for e in entries:
        if e["name"] not in names:
            continue
        image = load_upright(root / e["photo"])
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        W, H = image.size
        pen = ImageDraw.Draw(image)
        for b in e["boxes"] + draft[e["name"]]["life"]:
            x1, y1, x2, y2 = b["xyxy"]
            pen.rectangle((x1 * W, y1 * H, x2 * W, y2 * H), outline=(255, 127, 14) if "label" not in b else (230, 30, 30),
                          width=3)
        image.save(run / f"{e['name']}.jpg", quality=90)


def predictions(entries: list[dict], draft: dict[str, dict]) -> list[dict]:
    """The draft as one Label Studio prediction per photo: each date box
    with its text and turn under its id date<index>, each life box as cl<k>."""
    root, out = store_photos_dir(), []
    for e in entries:
        size, d = upright_size(root / e["photo"]), draft[e["name"]]
        result = []
        for b in e["boxes"]:
            x = d["boxes"][b["index"]]
            result += review._regions(f"date{b['index']}", review.DATE_CONTROL, b["label"], b["xyxy"], size,
                                      x["text"], x["turn"])
        for k, b in enumerate(d["life"], start=1):
            result += review._regions(f"cl{k}", review.LIFE_CONTROL, store_text.LIFE, b["xyxy"], size,
                                      b["text"], b["turn"])
        out.append({"photo": e["photo"], "model_version": MODEL_VERSION, "result": result})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sheet", action="store_true", help="draw every box as drafted")
    parser.add_argument("--only", nargs="*", default=[], help="photos (v1-0001) or boxes (v1-0001:1, v1-0001:life)")
    parser.add_argument("--photos", nargs="*", help="draw these photos whole")
    args = parser.parse_args()
    os.umask(0o077)
    entries = photos()
    draft = drafted(entries)
    if args.sheet or args.photos:
        run = start_run("m3-store-text-draft", vars(args), root=store.RUNS_DIR)
        os.chmod(run, 0o700)
        if args.sheet:
            print(f"{sheets(entries, draft, args.only, run)} boxes on sheets", file=sys.stderr)
        if args.photos:
            whole(entries, draft, args.photos, run)
        print(run)
        return
    folder = store_text.REVIEW_DIR
    content = {"model_version": MODEL_VERSION,
               "code_sha256": code_sha256(review, store_text, sys.modules[__name__]),
               "edits_sha256": store.sha256(folder / EDITS) if (folder / EDITS).exists() else None,
               "predictions": predictions(entries, draft)}
    digest = store_text.write_private(folder / "draft_predictions.json", content)
    edited = sum(b["edited"] for d in draft.values() for b in d["boxes"].values())
    life = sum(len(d["life"]) for d in draft.values())
    print(f"draft: {edited} date boxes edited, {life} life boxes, {len(content['predictions'])} photos -> "
          f"{folder / 'draft_predictions.json'} (sha256 {digest[:12]})")


if __name__ == "__main__":
    main()
