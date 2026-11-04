"""Export store sets v1 and v3 for training (docs/m2b-store-and-size.md, S6, "The export").

    python scripts/export_store_yolo.py              # write the export into $STORE_TRAIN_DIR, then check it
    python scripts/export_store_yolo.py --check DIR  # check a copy (the cluster's) against its manifest
    python scripts/export_store_yolo.py --sheet      # contact sheets of the export, in data/store/runs/

Reads only frozen files, each checked against its hash: v1's and v3's
labels and photos, their keyword boxes (with the dates S5's review added),
the designs and both splits. It refuses to start while store set v2 isn't
frozen, and refuses any v2 photo: v2 is the test set and never leaves the
laptop. Every photo goes in upright, in sRGB, at most 1,600 px, as a
numbered JPEG (v1-0001.jpg, v3-0001.jpg) without metadata, any box under
2 px across left out as a slip and counted; the training part also turned
90, 180 and 270 degrees; validation as it is:

    $STORE_TRAIN_DIR/images/{train_0,train_90,train_180,train_270,val}/v1-0001.jpg
                     labels/.../v1-0001.txt     date, due, prod, code (yolo.CLASSES)
                     train.txt, val.txt         ./images/... lines, so a copy works as it is
                     manifest.json              counts, the frozen files' hashes, each photo's
                                                part, design number, tag and box labels, and
                                                every file's SHA-256

The map back to photo and design names is data/store/train_names.json, on
the laptop only, outside the export folder, so copying the folder can't
take it along. The export writes into an empty folder only. Problems name
photos, so they're printed on screen only.

--sheet draws the export as written, from its files: each photo with its
boxes and every box enlarged, and each training photo in its 4 turns.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from bestbefore import derived, store, store_train
from bestbefore.config import store_photos_dir, store_train_dir
from bestbefore.runs import read_json, start_run
from bestbefore.sheets import FONT

# By the store's label, so a sheet tells an expiry from a manufacture date.
COLOURS = {"expiry": (230, 30, 30), "mfg": (30, 90, 230), "other_date": (160, 60, 200),
           "due": (255, 140, 0), "prod": (20, 160, 60), "code": (0, 170, 190)}
WIDTH = 1600            # a sheet's width
THUMB = 420             # a photo's long side on the photo sheets
CROP_H = 64             # an enlarged box's height (a tall box's width)
PHOTOS_PER_SHEET = 4
TURN_THUMB = 200        # a turned copy's long side on the turn sheets
TURNS_PER_SHEET = 14


def export() -> None:
    store_train.check_v2_frozen()               # before anything else is read
    sets, designs, split, sources = store_train.export_inputs()
    out = store_train_dir()
    plan = store_train.export_plan(sets, designs, split)
    manifest = store_train.write_export(plan, out, sources, store_photos_dir())
    c = manifest["counts"]
    for v, parts in c["photos"].items():
        print(f"{v}: {parts['train']} photos train, {parts['val']} validate")
    print("images: " + ", ".join(f"{folder} {n}" for folder, n in c["images"].items()))
    print(f"designs: {c['designs']['train']} train, {c['designs']['val']} validate")
    for part in store_train.FOLDERS:
        print(f"{part} boxes per photo: " + ", ".join(f"{k} {n}" for k, n in c["boxes"][part].items())
              + "; dates: " + ", ".join(f"{k} {n}" for k, n in c["dates"][part].items()))
    print("worst: " + ", ".join(f"{k} {v}" for k, v in manifest["worst"].items()))
    slips = {n: e["dropped"] for n, e in manifest["photos"].items() if e["dropped"]}
    print(f"left out, under {store_train.MIN_BOX_PX:g} px across: "
          + (", ".join(f"{n} {' '.join(labels)}" for n, labels in slips.items()) or "none"))
    print(f"wrote {len(manifest['files'])} files into {out}, the names map into {store_train.NAMES_PATH}")
    check(out)


def check(folder: Path) -> None:
    problems = store_train.check_export(folder)
    for p in problems:
        print(f"    {p}")
    if problems:
        raise SystemExit(f"✗ {folder}: {len(problems)} problems")
    manifest = read_json(folder / store_train.MANIFEST)
    print(f"✓ {folder}: all {len(manifest['files'])} files as the manifest has them, nothing else, no names map")


# --- Contact sheets -----------------------------------------------------------------

def _draw(im: Image.Image, boxes, labels: list[str], width: int = 2) -> Image.Image:
    """A copy of `im` with each box drawn in its label's colour."""
    im = im.copy()
    draw, (w, h) = ImageDraw.Draw(im), im.size
    for b, label in zip(boxes, labels, strict=True):
        x1, y1, x2, y2 = b.xyxy
        draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=COLOURS[label], width=width)
    return im


def _crop(im: Image.Image, box, boxes, labels: list[str], label: str) -> Image.Image:
    """One box enlarged with its neighbourhood (every box drawn), captioned:
    a margin of its short side above and below and twice that beside it."""
    W, H = im.size
    x1, y1, x2, y2 = box.xyxy[0] * W, box.xyxy[1] * H, box.xyxy[2] * W, box.xyxy[3] * H
    tall = y2 - y1 > x2 - x1
    m = min(x2 - x1, y2 - y1) + 4
    mx, my = (m, 2 * m) if tall else (2 * m, m)
    region = (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(W, int(x2 + mx) + 1), min(H, int(y2 + my) + 1))
    crop = _draw(im, boxes, labels, width=1).crop(region)
    scale = CROP_H / (crop.width if tall else crop.height)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
    tile = Image.new("RGB", (max(crop.width, 60), crop.height + 16), "white")
    tile.paste(crop, (0, 0))
    ImageDraw.Draw(tile).text((2, crop.height + 1), label, fill=COLOURS[label], font=FONT)
    return tile


def _pack(tiles: list[Image.Image], width: int) -> Image.Image:
    """Tiles left to right in rows of at most `width`."""
    rows, row, x = [], [], 0
    for t in tiles:
        if row and x + t.width > width:
            rows.append(row)
            row, x = [], 0
        row.append(t)
        x += t.width + 6
    rows.append(row)
    out = Image.new("RGB", (width, sum(max(t.height for t in r) + 6 for r in rows if r) or 1), "white")
    y = 0
    for r in (r for r in rows if r):
        x = 0
        for t in r:
            out.paste(t, (x, y))
            x += t.width + 6
        y += max(t.height for t in r) + 6
    return out


def photo_row(path: Path, name: str, entry: dict) -> Image.Image:
    """A photo as exported, with its boxes, beside each box enlarged."""
    im, boxes = Image.open(path).convert("RGB"), derived.read_boxes(path)
    thumb = _draw(im, boxes, entry["labels"], width=3)
    thumb.thumbnail((THUMB, THUMB))
    crops = _pack([_crop(im, b, boxes, entry["labels"], label) for b, label in zip(boxes, entry["labels"])],
                  WIDTH - THUMB - 20)
    caption = f"{name}  {entry['part']}  {entry['tag']}  {entry['design']}  {im.width}x{im.height}"
    row = Image.new("RGB", (WIDTH, max(thumb.height + 18, crops.height) + 10), "white")
    draw = ImageDraw.Draw(row)
    draw.text((2, 1), caption, fill="black", font=FONT)
    row.paste(thumb, (0, 18))
    row.paste(crops, (THUMB + 14, 0))
    draw.line([(0, row.height - 3), (WIDTH, row.height - 3)], fill=(180, 180, 180), width=2)
    return row


def turn_tile(folder: Path, name: str, entry: dict) -> Image.Image:
    """A training photo in its 4 turns, each with its boxes."""
    shots = []
    for turned in store_train.FOLDERS["train"]:
        path = folder / "images" / turned / f"{name}.jpg"
        shot = _draw(Image.open(path).convert("RGB"), derived.read_boxes(path), entry["labels"], width=8)
        shot.thumbnail((TURN_THUMB, TURN_THUMB))
        shots.append(shot)
    tile = Image.new("RGB", (4 * (TURN_THUMB + 6), TURN_THUMB + 18), "white")
    ImageDraw.Draw(tile).text((2, 1), f"{name}  0 / 90 / 180 / 270", fill="black", font=FONT)
    for k, shot in enumerate(shots):
        tile.paste(shot, (k * (TURN_THUMB + 6), 18))
    return tile


def _stack(images: list[Image.Image]) -> Image.Image:
    sheet = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images)), "white")
    y = 0
    for i in images:
        sheet.paste(i, (0, y))
        y += i.height
    return sheet


def sheets(folder: Path) -> None:
    manifest = read_json(folder / store_train.MANIFEST)
    photos = manifest["photos"]
    run_dir = start_run("store-export-sheet", {"photos": len(photos)}, root=store.RUNS_DIR)
    names = sorted(photos)
    for k in range(0, len(names), PHOTOS_PER_SHEET):
        rows = [photo_row(folder / "images" / ("train_0" if photos[n]["part"] == "train" else "val") / f"{n}.jpg",
                          n, photos[n]) for n in names[k:k + PHOTOS_PER_SHEET]]
        _stack(rows).save(run_dir / f"photos-{k // PHOTOS_PER_SHEET + 1:02d}.jpg", quality=90)
    train = [n for n in names if photos[n]["part"] == "train"]
    for k in range(0, len(train), TURNS_PER_SHEET):
        tiles = [turn_tile(folder, n, photos[n]) for n in train[k:k + TURNS_PER_SHEET]]
        pairs = [_pack(tiles[i:i + 2], 2 * tiles[0].width + 20) for i in range(0, len(tiles), 2)]
        _stack(pairs).save(run_dir / f"turns-{k // TURNS_PER_SHEET + 1:02d}.jpg", quality=90)
    print(f"{len(names)} photos and {len(train)} training photos' turns drawn: {run_dir}")
    print("boxes by label: " + ", ".join(f"{k} {n}" for k, n in sorted(
        Counter(label for p in photos.values() for label in p["labels"]).items())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", type=Path, metavar="DIR", help="check a copy of the export against its manifest")
    mode.add_argument("--sheet", action="store_true", help="contact sheets of $STORE_TRAIN_DIR's export")
    args = parser.parse_args()
    if args.check:
        check(args.check)
    elif args.sheet:
        sheets(store_train_dir())
    else:
        export()


if __name__ == "__main__":
    main()
