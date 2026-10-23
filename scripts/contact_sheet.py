"""Draw a dataset's boxes, to check the labels by eye before trusting them.

    python scripts/contact_sheet.py --set real-train --n 50 --seed 0
    python scripts/contact_sheet.py --set store --n 99     # every store photo
    python scripts/contact_sheet.py --set synth-val --per-tag 8   # a row per stratum

Writes two sheets into a new folder under results/runs/ (data/store/runs/
for the store photos, which never go into results/):
  photos.jpg  the sampled photos, whole, with every box drawn
  dates.jpg   each date box zoomed in, with its neighbours (EXP / MFG marks)
              drawn too and the transcription underneath. This is the one
              that shows what exactly a date box covers.

With --per-tag K: K photos of each tag (the store's failure tags,
synth-val's strata), one row per tag on photos.jpg, and on dates.jpg one
row per tag of its expiry boxes only (synth-val's targets).
"""
from __future__ import annotations

import argparse
import random

from PIL import Image, ImageDraw

from bestbefore.images import load_upright
from bestbefore.runs import start_run
from bestbefore.samples import DATE_LABELS, Sample
from bestbefore.sets import NAMES, load_set, runs_root
from bestbefore.sheets import FONT, grid, rows_of

COLOURS = {
    "date": (230, 30, 30), "exp": (255, 140, 0), "due": (30, 90, 230),
    "prod": (20, 160, 60), "code": (140, 140, 140),
    "expiry": (230, 30, 30), "mfg": (30, 90, 230), "other_date": (140, 140, 140),
}
THUMB = 360            # longest side of a photo on the sheet
CROP_H = 110           # height of a zoomed date on the sheet
COLS = 10


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


def date_tiles(s: Sample, im: Image.Image, labels=DATE_LABELS, tall: int = 1) -> list[Image.Image]:
    """One zoomed tile per date box: the box plus a margin of one box
    height. A box taller than wide gets a tile `tall` times as high."""
    W, H = im.size
    tiles = []
    for b in s.boxes:
        if b.label not in labels:
            continue
        x1, y1, x2, y2 = b.xyxy[0] * W, b.xyxy[1] * H, b.xyxy[2] * W, b.xyxy[3] * H
        m = min(x2 - x1, y2 - y1) + 4
        cx1, cy1 = max(0, x1 - 2 * m), max(0, y1 - m)
        cx2, cy2 = min(W, x2 + 2 * m), min(H, y2 + m)
        crop = im.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
        crop_h = CROP_H * (tall if y2 - y1 > x2 - x1 else 1)
        scale = crop_h / crop.height
        crop = crop.resize((max(1, min(600, round(crop.width * scale))), crop_h))
        sx, sy = crop.width / (cx2 - cx1), crop.height / (cy2 - cy1)
        draw = ImageDraw.Draw(crop)
        _draw_boxes(draw, s.boxes, lambda bb: (
            (bb[0] * W - cx1) * sx, (bb[1] * H - cy1) * sy,
            (bb[2] * W - cx1) * sx, (bb[3] * H - cy1) * sy), width=1)
        tile = Image.new("RGB", (crop.width if tall == 1 else max(crop.width, 150), crop_h + 18), "white")
        tile.paste(crop, (0, 0))
        # Store boxes carry no transcription: show the typed expiry instead.
        text = b.text or (s.expiry if b.label == "expiry" else None)
        caption = f"{b.label}: {text or ''}   {s.id}"
        ImageDraw.Draw(tile).text((2, crop_h + 2), caption, fill="black", font=FONT)
        tiles.append(tile)
    return tiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", required=True, choices=NAMES)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-tag", type=int, metavar="K", help="K photos of each tag, a row per tag")
    args = parser.parse_args()

    samples = load_set(args.set)
    run_dir = start_run(f"contact-{args.set}", vars(args), seed=args.seed,
                        root=runs_root(args.set))
    if args.per_tag:
        by_tag(samples, args.per_tag, args.seed, run_dir)
        print(run_dir)
        return
    picked = (samples if args.n >= len(samples)
              else random.Random(args.seed).sample(samples, args.n))

    photos, dates = [], []
    for s in picked:
        im = load_upright(s.image)
        photos.append(photo_tile(s, im))
        dates.extend(date_tiles(s, im))

    grid(photos, COLS).save(run_dir / "photos.jpg", quality=88)
    if dates:
        rows_of(dates).save(run_dir / "dates.jpg", quality=88)
    print(run_dir)


def by_tag(samples: list[Sample], k: int, seed: int, run_dir) -> None:
    """K photos of each tag, in the order the tags first appear: one row of
    photos per tag, and one row per tag of their expiry boxes, captioned
    (sideways ones three times as tall, to be readable)."""
    rng = random.Random(seed)
    tags = list(dict.fromkeys(s.tag for s in samples))
    photos, rows = [], []
    for tag in tags:
        of_tag = [s for s in samples if s.tag == tag]
        crops = []
        for s in rng.sample(of_tag, min(k, len(of_tag))):
            im = load_upright(s.image)
            photos.append(photo_tile(s, im))
            crops += date_tiles(s, im, labels={"exp", "expiry"}, tall=3)
        if crops:
            row = Image.new("RGB", (sum(c.width + 4 for c in crops), max(c.height for c in crops)), "white")
            x = 0
            for c in crops:
                row.paste(c, (x, 0))
                x += c.width + 4
            caption = Image.new("RGB", (row.width, 20), "white")
            ImageDraw.Draw(caption).text((4, 3), str(tag), fill="black", font=FONT)
            rows += [caption, row]
        photos += [Image.new("RGB", photos[-1].size, "white")] * (-len(photos) % k)   # each tag starts a row
    grid(photos, k).save(run_dir / "photos.jpg", quality=88)
    if rows:
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        sheet.save(run_dir / "dates.jpg", quality=90)


if __name__ == "__main__":
    main()
