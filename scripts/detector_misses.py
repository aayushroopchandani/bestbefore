"""Draw the store expiry dates a detector missed, to see why (M1, D3).

    python scripts/detector_misses.py --run n640

Reads the run's store predictions (saved by `eval_detector.py --set store`)
and the frozen store test set. One tile per expiry date not found at the
run's own threshold (IoU 0.5): the labelled box in green, the predicted
date boxes in red with their confidence, and the most confident box on the
date, if any, in the caption. Tiles are grouped by failure tag.

Written to data/store/runs/, never to results/: it shows store photos.
"""
from __future__ import annotations

import argparse
import json

from PIL import Image, ImageDraw

from bestbefore import store
from bestbefore.detection_eval import Prediction, iou, match_photo
from bestbefore.images import load_upright
from bestbefore.runs import latest_run, start_run
from bestbefore.samples import DATE_LABELS
from bestbefore.sets import load_set, runs_root
from bestbefore.sheets import FONT, rows_of

TILE_H = 140            # height of a tile's crop, px
MIN_W = 340             # wide enough for the caption
SHOW_CONF = 0.05        # predicted boxes are drawn from this confidence up
TRUTH, PRED = (20, 170, 60), (230, 30, 30)


def tile(im: Image.Image, truth, preds: list[Prediction], caption: str) -> Image.Image:
    """The labelled date with a margin around it, boxes drawn, caption below."""
    W, H = im.size
    x1, y1, x2, y2 = truth[0] * W, truth[1] * H, truth[2] * W, truth[3] * H
    m = 0.6 * max(x2 - x1, y2 - y1) + 20
    cx1, cy1, cx2, cy2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
    crop = im.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
    crop = crop.resize((max(1, min(700, round(crop.width * TILE_H / crop.height))), TILE_H))
    sx, sy = crop.width / (cx2 - cx1), crop.height / (cy2 - cy1)

    def px(b):
        return ((b[0] * W - cx1) * sx, (b[1] * H - cy1) * sy,
                (b[2] * W - cx1) * sx, (b[3] * H - cy1) * sy)

    draw = ImageDraw.Draw(crop)
    for p in preds:
        r = px(p.xyxy)
        draw.rectangle(r, outline=PRED, width=2)
        draw.text((r[0] + 2, r[1] + 2), f"{p.conf:.2f}", fill=PRED, font=FONT)
    draw.rectangle(px(truth), outline=TRUTH, width=2)
    out = Image.new("RGB", (max(crop.width, MIN_W), TILE_H + 18), "white")
    out.paste(crop, (0, 0))
    ImageDraw.Draw(out).text((2, TILE_H + 2), caption, fill="black", font=FONT)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="n640", help="an M1 run scored on the store photos")
    args = parser.parse_args()

    scored = latest_run(f"detector-store-{args.run}", store.RUNS_DIR)
    if scored is None:
        raise SystemExit(f"{args.run} isn't scored on the store photos yet: "
                         f"python scripts/eval_detector.py --set store --runs {args.run}")
    threshold = json.loads((scored / "run.json").read_text())["config"]["threshold"]
    boxes = json.loads((scored / "predictions.json").read_text())

    tiles = []
    for s in sorted(load_set("store"), key=lambda s: (store.TAGS.index(s.tag), s.id)):
        truths = [b for b in s.boxes if b.label in DATE_LABELS]
        preds = [Prediction("date", tuple(b[:4]), b[4]) for b in boxes.get(s.id, [])]
        _, taken = match_photo([p for p in preds if p.conf >= threshold],
                               [b.xyxy for b in truths], 0.5)
        missed = [b for b, conf in zip(truths, taken) if b.label == "expiry" and conf is None]
        if not missed:
            continue
        im = load_upright(s.image)
        shown = [p for p in preds if p.conf >= SHOW_CONF]
        for b in missed:
            best = max((p.conf for p in preds if iou(p.xyxy, b.xyxy) >= 0.5), default=None)
            note = f"best box {best:.2f}" if best is not None else "no box"
            tiles.append(tile(im, b.xyxy, shown, f"{s.tag}  {note}  {s.id}"))

    out = start_run(f"detector-misses-{args.run}", {**vars(args), "threshold": threshold},
                    root=runs_root("store"))
    if tiles:
        rows_of(tiles).save(out / "misses.jpg", quality=88)
    print(f"{len(tiles)} missed expiry dates (threshold {threshold}): {out / 'misses.jpg'}")


if __name__ == "__main__":
    main()
