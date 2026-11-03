"""Propose keyword boxes on store set v1 with RapidOCR, as Label Studio
predictions (docs/m2b-store-and-size.md, S5).

    python scripts/propose_keywords.py            # data/store/keywords/: proposals, tasks, project
    python scripts/propose_keywords.py --sheet    # and a sheet of them in data/store/runs/

Each v1 photo goes upright (load_upright) and shrunk to 1,000 px on its
long side, as Scanify shrinks it (M0's best size for RapidOCR), through
RapidOCR 1.2.3 as baseline A runs it. Its lines are matched against
store_train.KEYWORD_PATTERNS, and each keyword's share of its line's box
becomes a due, prod or code proposal, 0..1 of the upright photo: the frame
and convention v1's date boxes are in. Written to data/store/keywords/:

    proposals.json  every photo's proposals, with OCR's text
    lines.json      every photo's OCR lines: quad (in the 1,000 px photo), text,
                    turned and flipped, for scripts/draft_keywords.py
    tasks.json      one Label Studio task per photo: the proposals and,
                    read-only, the photo's frozen date boxes, as a prediction
    project.json    the project's title, interface and instructions (the box rule)

The sheet draws the proposals and the date boxes on each upright photo,
and zoomed in round each date box, to check that both sit where they
should before the review. Only store set v1: its photos are its folders'
and must be its frozen labels'; nothing reads testset2/. Everything
written names photos, so it stays under data/store/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date

import numpy as np
from PIL import Image, ImageDraw

from bestbefore import store, store_train
from bestbefore.baselines import rapidocr_regex
from bestbefore.config import store_photos_dir
from bestbefore.detection_eval import iou
from bestbefore.images import load_upright
from bestbefore.runs import code_sha256, start_run
from bestbefore.sheets import FONT, grid, rows_of
from bestbefore.tables import markdown_table

LONG_SIDE = 1000
TITLE = "Store v1 keywords"

INTERFACE = """<View>
  <Header value="Box each keyword as printed, with its colon or full stop, never its date. due: EXP. / Best Before: / Expiry Date :   prod: MFG: / Mfd(Date): / IMPORT:   code: Batch No.: K2311A / 14:32. No box for prices, weights, phone numbers, barcode digits, addresses or 12 months from Mfg."/>
  <Image name="img" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="keyword" toName="img">
    <Label value="due" background="#ff7f0e"/>
    <Label value="prod" background="#2ca02c"/>
    <Label value="code" background="#9467bd"/>
  </RectangleLabels>
  <Header value="v1's date boxes, for reference only: leave them as they are"/>
  <RectangleLabels name="date" toName="img">
    <Label value="expiry" background="#d62728"/>
    <Label value="mfg" background="#1f77b4"/>
    <Label value="other_date" background="#7f7f7f"/>
  </RectangleLabels>
</View>
"""

INSTRUCTIONS = """<h3>Keyword boxes on store set v1</h3>
<p>OCR has proposed keyword boxes. On every photo: keep, move or delete each one, add any it missed,
then <b>Submit</b> (after changing a photo already submitted, <b>Update</b>). A photo with no keyword
is submitted with none.</p>
<p>The red, blue and grey boxes are v1's date boxes (expiry, mfg, other), for reference only: leave them
as they are.</p>
<table>
<tr><th>Label</th><th>Box</th><th>Examples</th></tr>
<tr><td><b>due</b></td><td>the expiry word or phrase as printed, with its colon</td>
<td>EXP. &middot; Best Before: &middot; Expiry Date : &middot; USE BY</td></tr>
<tr><td><b>prod</b></td><td>the manufacture, packing or import word or phrase</td>
<td>MFG: &middot; Mfd(Date): &middot; Month and year of manufacture: &middot; IMPORT:</td></tr>
<tr><td><b>code</b></td><td>a batch or lot code with its label, or a time</td>
<td>Batch No.: K2311A &middot; B.NO: 2307B &middot; 14:32</td></tr>
<tr><td>no box</td><td>prices, quantities, phone numbers, barcode digits, addresses
(Manufactured by:, Imported &amp; Marketed by:), 12 months from Mfg.</td>
<td>MRP: &#8377;450.00 &middot; USP Rs. 1.50/g</td></tr>
</table>
<p>Never box the date itself: a word glued to its date (EXP.FEB 2027) gets a box on EXP. only. One box
per keyword; a sideways keyword still gets an upright rectangle, as the dates did.</p>
"""

SHEET_THUMB, SHEET_CROP = 360, 120
COLOURS = {"due": (255, 127, 14), "prod": (44, 160, 44), "code": (148, 103, 189),
           "expiry": (214, 39, 40), "mfg": (31, 119, 180), "other_date": (127, 127, 127)}


def turned(quad) -> bool:
    """Whether RapidOCR 1.2.3 turns this line's crop a quarter before
    reading it (get_rotate_crop_image: 1.5 times taller than wide)."""
    q = np.asarray(quad, np.float32)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    return h >= 1.5 * w


def ocr_lines(image: Image.Image) -> list[dict]:
    """RapidOCR's lines as RapidOCR.__call__ makes them (the same models,
    steps and score filter), plus what __call__ drops and span_box needs:
    whether each line's crop was turned a quarter, and whether its
    classifier turned it 180°."""
    engine = rapidocr_regex._engine()
    img = np.asarray(image)
    h, w = img.shape[:2]
    if h <= engine.min_height or w / h > engine.width_height_ratio:
        raise ValueError(f"a {w}x{h} photo, which RapidOCR reads without finding lines")
    quads, _ = engine.text_detector(img)
    if quads is None or len(quads) == 0:
        return []
    quads = engine.sorted_boxes(quads)
    crops, turns, _ = engine.text_cls(engine.get_crop_img_list(img, quads))
    read, _ = engine.text_recognizer(crops)
    return [{"quad": np.asarray(q).tolist(), "text": text, "score": float(score), "turned": turned(q),
             "flipped": bool("180" in label and label_score > engine.text_cls.cls_thresh)}
            for q, (label, label_score), (text, score) in zip(quads, turns, read)
            if score >= engine.text_score]


def v1_photos() -> tuple[list[str], dict[str, dict]]:
    """v1's photos from its folders, and its frozen labels, which must name
    exactly the same photos."""
    photos, labels = store.photos_on_disk(store.V1), store.labels(store.V1)
    if sorted(labels) != photos or any(p.split("/")[0] not in store.V1.folders for p in photos):
        raise SystemExit("store set v1's folders and its frozen labels name different photos")
    return photos, labels


def draw(im: Image.Image, boxes: list[dict], to_px, width: int) -> None:
    pen = ImageDraw.Draw(im)
    for b in boxes:
        x1, y1, x2, y2 = to_px(b["xyxy"])
        keyword = b["label"] in store_train.KEYWORD_LABELS
        pen.rectangle([x1, y1, x2, y2], outline=COLOURS[b["label"]], width=width + keyword)


def sheet(photos: list[str], labels: dict[str, dict], proposed: dict[str, dict], run_dir) -> None:
    """photos.jpg: every photo with its date boxes and proposals; dates.jpg:
    a zoom round each date box with every box near it, to see each keyword
    box beside its date."""
    tiles, zooms = [], []
    for photo in photos:
        im = load_upright(store_photos_dir() / photo)
        W, H = im.size
        boxes = labels[photo]["boxes"] + proposed[photo]["boxes"]
        tile = im.copy()
        tile.thumbnail((SHEET_THUMB, SHEET_THUMB))
        w, h = tile.size
        draw(tile, boxes, lambda b: (b[0] * w, b[1] * h, b[2] * w, b[3] * h), 1)
        out = Image.new("RGB", (SHEET_THUMB, SHEET_THUMB + 18), "white")
        out.paste(tile, ((SHEET_THUMB - w) // 2, 0))
        ImageDraw.Draw(out).text((4, SHEET_THUMB + 2), f"{photo}  {len(proposed[photo]['boxes'])} proposed",
                                 fill="black", font=FONT)
        tiles.append(out)
        for b in labels[photo]["boxes"]:
            x1, y1, x2, y2 = b["xyxy"][0] * W, b["xyxy"][1] * H, b["xyxy"][2] * W, b["xyxy"][3] * H
            m = min(x2 - x1, y2 - y1)
            cx1, cy1, cx2, cy2 = (max(0, x1 - 6 * m), max(0, y1 - 3 * m), min(W, x2 + 6 * m), min(H, y2 + 3 * m))
            crop = im.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
            scale = SHEET_CROP / crop.height
            crop = crop.resize((max(1, min(900, round(crop.width * scale))), SHEET_CROP))
            sx, sy = crop.width / (int(cx2) - int(cx1)), crop.height / (int(cy2) - int(cy1))
            draw(crop, boxes, lambda bb: ((bb[0] * W - int(cx1)) * sx, (bb[1] * H - int(cy1)) * sy,
                                          (bb[2] * W - int(cx1)) * sx, (bb[3] * H - int(cy1)) * sy), 1)
            zoom = Image.new("RGB", (max(crop.width, 260), SHEET_CROP + 18), "white")
            zoom.paste(crop, (0, 0))
            ImageDraw.Draw(zoom).text((2, SHEET_CROP + 2), f"{b['label']}  {photo}", fill="black", font=FONT)
            zooms.append(zoom)
    grid(tiles, 6).save(run_dir / "photos.jpg", quality=85)
    rows_of(zooms, 2400).save(run_dir / "dates.jpg", quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sheet", action="store_true", help="also draw them, in data/store/runs/")
    args = parser.parse_args()

    photos, labels = v1_photos()
    root = store_photos_dir()
    proposed, read = {}, {}
    for k, photo in enumerate(photos, start=1):
        im = load_upright(root / photo)
        size = im.size
        im.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
        lines = ocr_lines(im)
        read[photo] = {"size": list(im.size), "lines": lines}
        proposed[photo] = {"size": list(size), "lines": len(lines),
                           "boxes": store_train.propose(lines, im.size)}
        print(f"\r{k}/{len(photos)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    out = store_train.KEYWORDS_DIR
    out.mkdir(parents=True, exist_ok=True)
    record = {"made": date.today().isoformat(), "long_side": LONG_SIDE, "model_version": store_train.MODEL_VERSION,
              "test_set_sha256": store.TEST_SET_SHA256,
              "code_sha256": code_sha256(store_train, rapidocr_regex, sys.modules[__name__])}
    (out / "proposals.json").write_text(json.dumps({**record, "photos": proposed}, indent=1) + "\n")
    (out / "lines.json").write_text(json.dumps({**record, "photos": read}) + "\n")
    tasks = [store_train.keyword_task(p, tuple(proposed[p]["size"]), proposed[p]["boxes"], labels[p])
             for p in photos]
    (out / "tasks.json").write_text(json.dumps(tasks) + "\n")
    project = {"title": TITLE, "label_config": INTERFACE, "expert_instruction": INSTRUCTIONS,
               "show_instruction": True, "show_skip_button": False, "show_collab_predictions": True,
               "model_version": store_train.MODEL_VERSION, "maximum_annotations": 1}
    (out / "project.json").write_text(json.dumps(project, indent=1) + "\n")

    boxes = [b for p in photos for b in proposed[p]["boxes"]]
    over = sum(iou(b["xyxy"], d["xyxy"]) >= store_train.OVER_DATE
               for p in photos for b in proposed[p]["boxes"] for d in labels[p]["boxes"])
    counts = Counter(b["label"] for b in boxes)
    print(markdown_table([{"label": c, "proposed": f"{counts[c]}",
                           "photos": f"{sum(any(b['label'] == c for b in proposed[p]['boxes']) for p in photos)}"}
                          for c in store_train.KEYWORD_LABELS], left=("label",)))
    print(f"\n{len(boxes)} proposals on {sum(bool(proposed[p]['boxes']) for p in photos)} of {len(photos)} "
          f"photos, from {sum(proposed[p]['lines'] for p in photos)} OCR lines; "
          f"{over} over a date box (IoU >= {store_train.OVER_DATE})")
    print(f"wrote {out}/: proposals.json, lines.json, tasks.json ({len(tasks)} tasks), project.json")
    if args.sheet:
        run_dir = start_run("keyword-proposals", vars(args), root=store.RUNS_DIR)
        sheet(photos, labels, proposed, run_dir)
        print(f"sheet: {run_dir}")


if __name__ == "__main__":
    main()
