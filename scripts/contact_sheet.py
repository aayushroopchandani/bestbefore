"""Draw a dataset's boxes, to check the labels by eye before trusting them.

    python scripts/contact_sheet.py --set real-train --n 50 --seed 0
    python scripts/contact_sheet.py --set store --n 99     # every store photo

Writes two sheets into a new folder under results/runs/ (data/store/runs/
for the store photos, which never go into results/):
  photos.jpg  the sampled photos, whole, with every box drawn
  dates.jpg   each date box zoomed in, with its neighbours (EXP / MFG marks)
              drawn too and the transcription underneath. This is the one
              that shows what exactly a date box covers.
"""
from __future__ import annotations

import argparse
import random

from PIL import Image, ImageDraw, ImageFont

from bestbefore.images import load_upright
from bestbefore.runs import start_run
from bestbefore.samples import DATE_LABELS, Sample
from bestbefore.sets import NAMES, load_set, runs_root

COLOURS = {
    "date": (230, 30, 30), "exp": (255, 140, 0), "due": (30, 90, 230),
    "prod": (20, 160, 60), "code": (140, 140, 140),
    "expiry": (230, 30, 30), "mfg": (30, 90, 230), "other_date": (140, 140, 140),
}
THUMB = 360            # longest side of a photo on the sheet
CROP_H = 110           # height of a zoomed date on the sheet
COLS = 10
FONT = ImageFont.load_default(size=13)


def _draw_boxes(draw: ImageDraw.ImageDraw, boxes, to_px, width: int = 2) -> None:
    for b in boxes:
        x1, y1, x2, y2 = to_px(b.xyxy)
        draw.rectangle([x1, y1, x2, y2],
                       outline=COLOURS.get(b.label, (255, 0, 255)), width=width)


def photo_tile(s: Sample, im: Image.Image) -> Image.Image:
    tile = im.copy()
    tile.thumbnail((THUMB, THUMB))
    w, h = tile.size
    draw = ImageDraw.Draw(tile)
    _draw_boxes(draw, s.boxes,
                lambda b: (b[0] * w, b[1] * h, b[2] * w, b[3] * h))
    out = Image.new("RGB", (THUMB, THUMB + 18), "white")
    out.paste(tile, ((THUMB - w) // 2, 0))
    caption = "  ".join(str(x) for x in (s.id, s.tag, s.expiry) if x)
    ImageDraw.Draw(out).text((4, THUMB + 2), caption, fill="black", font=FONT)
    return out


def date_tiles(s: Sample, im: Image.Image) -> list[Image.Image]:
    """One zoomed tile per date box: the box plus a margin of one box height."""
    W, H = im.size
    tiles = []
    for b in s.boxes:
        if b.label not in DATE_LABELS:
            continue
        x1, y1, x2, y2 = b.xyxy[0] * W, b.xyxy[1] * H, b.xyxy[2] * W, b.xyxy[3] * H
        m = min(x2 - x1, y2 - y1) + 4
        cx1, cy1 = max(0, x1 - 2 * m), max(0, y1 - m)
        cx2, cy2 = min(W, x2 + 2 * m), min(H, y2 + m)
        crop = im.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
        scale = CROP_H / crop.height
        crop = crop.resize((max(1, min(600, round(crop.width * scale))), CROP_H))
        sx, sy = crop.width / (cx2 - cx1), crop.height / (cy2 - cy1)
        draw = ImageDraw.Draw(crop)
        _draw_boxes(draw, s.boxes, lambda bb: (
            (bb[0] * W - cx1) * sx, (bb[1] * H - cy1) * sy,
            (bb[2] * W - cx1) * sx, (bb[3] * H - cy1) * sy), width=1)
        tile = Image.new("RGB", (crop.width, CROP_H + 18), "white")
        tile.paste(crop, (0, 0))
        # Store boxes carry no transcription: show the typed expiry instead.
        text = b.text or (s.expiry if b.label == "expiry" else None)
        caption = f"{b.label}: {text or ''}   {s.id}"
        ImageDraw.Draw(tile).text((2, CROP_H + 2), caption, fill="black", font=FONT)
        tiles.append(tile)
    return tiles


def grid(tiles: list[Image.Image], cols: int) -> Image.Image:
    w, h = tiles[0].size
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    return sheet


def rows_of(tiles: list[Image.Image], width: int = 1800) -> Image.Image:
    """Pack tiles of different widths left to right into rows."""
    rows, row, x = [], [], 0
    for t in tiles:
        if x + t.width > width and row:
            rows.append(row)
            row, x = [], 0
        row.append(t)
        x += t.width + 4
    rows.append(row)
    h = tiles[0].height + 4
    sheet = Image.new("RGB", (width, len(rows) * h), "white")
    for r, row in enumerate(rows):
        x = 0
        for t in row:
            sheet.paste(t, (x, r * h))
            x += t.width + 4
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", required=True, choices=NAMES)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples = load_set(args.set)
    picked = (samples if args.n >= len(samples)
              else random.Random(args.seed).sample(samples, args.n))
    run_dir = start_run(f"contact-{args.set}", vars(args), seed=args.seed,
                        root=runs_root(args.set))

    photos, dates = [], []
    for s in picked:
        im = load_upright(s.image)
        photos.append(photo_tile(s, im))
        dates.extend(date_tiles(s, im))

    grid(photos, COLS).save(run_dir / "photos.jpg", quality=88)
    if dates:
        rows_of(dates).save(run_dir / "dates.jpg", quality=88)
    print(run_dir)


if __name__ == "__main__":
    main()
