"""Draw a detector's boxes on every store photo, to look at by eye (after M2 S8).

    python scripts/store_gallery.py                          # the M2 detector, m2-gen-rot-1280-s0
    python scripts/store_gallery.py --runs m2-gen-rot-1280-s0 m2-gen-rot-s0

Reads the per-photo predictions eval_detector.py kept for each run (every
box down to confidence 0.001) and the frozen store test set; nothing is
scored again. One full photo per store photo, in
<tag>/<outcome>/, with:

    green, thick   the labelled expiry box; green, thin: other labelled dates
    blue           a kept box (confidence >= the run's threshold) on a labelled date
    red            a kept box on no labelled date (a false box)
    orange         a labelled date's own box below the threshold, with its confidence

Outcome is the photo's worst expiry: found (kept, IoU >= 0.5),
seen-below-threshold (its matched box is under the threshold),
never-boxed (no box at IoU 0.5 at any confidence), or no-expiry-box
(other-face, or an expiry worked out from a mfg date). The matching is the
one S8's counts use: greedy, most confident first, over every box
(m2_results.store_causes), so the folders add up to store.json.

Written to data/store/runs/, never to results/: it shows store photos.
"""
from __future__ import annotations

import argparse
import csv
import json

from PIL import Image, ImageDraw, ImageFont

from bestbefore import store
from bestbefore.detection_eval import iou
from bestbefore.images import load_upright
from bestbefore.m2 import finished_run
from bestbefore.runs import latest_run, start_run
from bestbefore.samples import DATE_LABELS
from bestbefore.sets import load_set

LONG_SIDE = 1600        # photos are drawn at this size
EXPIRY, OTHER = (20, 200, 60), (120, 220, 140)
KEPT, FALSE, BELOW = (30, 110, 255), (235, 30, 30), (255, 150, 0)
OUTCOMES = ("found", "seen-below-threshold", "never-boxed")


def match(preds: list[list[float]], truths: list[tuple], iou_min: float = 0.5) -> tuple[list, list]:
    """detection_eval.match_photo, keeping which prediction took each date:
    (for each truth the index of its prediction or None, for each
    prediction whether it took a date)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][4])
    taken: list[int | None] = [None] * len(truths)
    hit = [False] * len(preds)
    for i in order:
        best, best_iou = None, iou_min
        for j, t in enumerate(truths):
            if taken[j] is None:
                v = iou(tuple(preds[i][:4]), t)
                if v >= best_iou:
                    best, best_iou = j, v
        if best is not None:
            taken[best], hit[i] = i, True
    return taken, hit


def font(size: int):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw(image, truths, labels, preds, taken, hit, threshold: float, header: str) -> Image.Image:
    im = load_upright(image).convert("RGB")
    im.thumbnail((LONG_SIDE, LONG_SIDE))
    W, H = im.size
    d = ImageDraw.Draw(im)
    f = font(max(14, W // 70))
    w = max(2, W // 400)

    def px(b):
        return (b[0] * W, b[1] * H, b[2] * W, b[3] * H)

    for b, label in zip(truths, labels):
        d.rectangle(px(b), outline=EXPIRY if label == "expiry" else OTHER,
                    width=2 * w if label == "expiry" else w)
    for p, ok in zip(preds, hit):
        if p[4] >= threshold:
            r = px(p)
            d.rectangle(r, outline=KEPT if ok else FALSE, width=w)
            d.text((r[0], max(0, r[1] - f.size - 2)), f"{p[4]:.2f}", fill=KEPT if ok else FALSE, font=f)
    for j, i in enumerate(taken):
        if i is not None and preds[i][4] < threshold:
            r = px(preds[i])
            d.rectangle(r, outline=BELOW, width=w)
            d.text((r[0], r[3] + 2), f"{preds[i][4]:.3f}", fill=BELOW, font=f)
    band = Image.new("RGB", (W, f.size + 12), "white")
    ImageDraw.Draw(band).text((6, 4), header, fill="black", font=f)
    out = Image.new("RGB", (W, H + band.height), "white")
    out.paste(band, (0, 0))
    out.paste(im, (0, band.height))
    return out


def gallery(run: str, samples, root) -> dict:
    scored = latest_run(f"detector-store-{run}", store.RUNS_DIR)
    if scored is None:
        raise SystemExit(f"{run} isn't scored on the store photos: no predictions kept")
    threshold = json.loads((scored / "run.json").read_text())["config"]["threshold"]
    boxes = json.loads((scored / "predictions.json").read_text())
    folder = root / run
    rows, counts = [], {o: 0 for o in OUTCOMES}
    for s in samples:
        dated = [b for b in s.boxes if b.label in DATE_LABELS]
        truths, labels = [b.xyxy for b in dated], [b.label for b in dated]
        preds = boxes.get(s.id, [])
        taken, hit = match(preds, truths)
        expiry = [j for j, label in enumerate(labels) if label == "expiry"]
        states = []
        for j in expiry:
            i = taken[j]
            state = ("never-boxed" if i is None else
                     "found" if preds[i][4] >= threshold else "seen-below-threshold")
            states.append(state)
            counts[state] += 1
            rows.append({"photo": s.id, "tag": s.tag, "outcome": state,
                         "confidence": "" if i is None else f"{preds[i][4]:.4f}",
                         "iou": "" if i is None else f"{iou(tuple(preds[i][:4]), truths[j]):.3f}",
                         "false_boxes": sum(p[4] >= threshold and not h for p, h in zip(preds, hit))})
        outcome = max(states, key=OUTCOMES.index) if states else "no-expiry-box"
        if not states:
            rows.append({"photo": s.id, "tag": s.tag, "outcome": outcome, "confidence": "", "iou": "",
                         "false_boxes": sum(p[4] >= threshold and not h for p, h in zip(preds, hit))})
        header = f"{run}  threshold {threshold:.3f}  {s.tag}  {outcome}  {s.id}"
        out = folder / s.tag / outcome / (s.id.replace("/", "__").rsplit(".", 1)[0] + ".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        draw(s.image, truths, labels, preds, taken, hit, threshold, header).save(out, quality=90)
    with (folder / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"threshold": threshold, **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", default=["m2-gen-rot-1280-s0"])
    args = parser.parse_args()
    samples = sorted(load_set("store"), key=lambda s: (store.TAGS.index(s.tag), s.id))
    root = start_run("store-gallery", vars(args), root=store.RUNS_DIR)
    (root / "LEGEND.txt").write_text(__doc__)
    for run in args.runs:
        c = gallery(run, samples, root)
        scores = json.loads((finished_run(run.removeprefix("m2-")) / "store.json").read_text())["iou0.5"]
        print(f"{run} (threshold {c['threshold']:.3f}): expiry found {c['found']}, "
              f"seen below threshold {c['seen-below-threshold']}, never boxed {c['never-boxed']} "
              f"(store.json: found {scores['expiry_found']} of {scores['expiry_dates']})")
    print(f"-> {root}")


if __name__ == "__main__":
    main()
