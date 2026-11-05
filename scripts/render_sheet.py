"""Draw rendered text blocks on plain grey, every token's box drawn, to check
the renderer by eye (docs/m2-synthetic-generator.md, S4; gen-v2's text,
docs/m2b-store-and-size.md, S7).

    python scripts/render_sheet.py               # sheet.png
    python scripts/render_sheet.py --v2          # gen-v2's new text and print, in render-sheet-v2
    python scripts/render_sheet.py --time 1000   # also time 1000 random blocks

Rows are the print styles, dot-matrix once per row count (7, 9); --v2 adds
gen-v2's print (S8): the laser window, plain and dotified, and dot-matrix
with nozzle rows. Columns are the format families (--v2: gen-v2's new
forms, glued words, line-pairs and dense panels). Each cell is one block in a training font, at a normal
size (digits 22 px) above the same block as small print (digits 8 px,
shown 3x with its pixels kept). Boxes: date red, due blue, prod green, code
magenta; text with no box has none. A window's strokes are light on its
dark rectangle. Nothing from ExpDate is on it.

--time draws blocks as the spec does (kind, format, style, font) at print
heights from the plan's size bands on a photo of --long-side pixels, and
reports the milliseconds per block (render, shrink, boxes) and how many
boxed tokens had no ink left at their final size.

Writes into a new folder under results/runs/.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bestbefore.runs import start_run
from bestbefore.sheets import FONT
from bestbefore.synth import fonts, render as R, text as T
from bestbefore.synth.spec import STYLE, TEXT, TEXT_V2, V2

GREY = 170
COLOURS = {"date": (230, 30, 30), "due": (30, 90, 230), "prod": (20, 160, 60),
           "code": (200, 0, 200)}
NORMAL, SMALL, ZOOM = 22.0, 8.0, 3
CELL_W = 470
CELL_W_V2 = 900         # a line-pair is a long line

_NOT_RANGE = tuple(kind for kind in TEXT.kinds if kind[0] != "range")
FAMILIES = {      # the plan's format families, each forced through the text spec
    "numbers, day": replace(TEXT, kinds=_NOT_RANGE, month_word=0.0, day_numbers=1.0),
    "numbers, month only": replace(TEXT, kinds=_NOT_RANGE, month_word=0.0, day_numbers=0.0),
    "month, full name": replace(TEXT, kinds=_NOT_RANGE, month_word=1.0, full_name=1.0),
    "month, abbreviation": replace(TEXT, kinds=_NOT_RANGE, month_word=1.0, full_name=0.0),
    "range": replace(TEXT, kinds=(("range", 1.0),)),
}
ROWS = (("solid", STYLE), *((f"dot-matrix, {n} rows", replace(STYLE, dot_rows=(n,)))
                            for n in STYLE.dot_rows),
        ("label", STYLE), ("embossed", STYLE))
# gen-v2's print (M2b S8), each with the gen-v2 print it is forced through.
ROWS_V2 = (("window", STYLE, replace(V2.style, window_dots=0.0)),
           ("window, dotified", STYLE, replace(V2.style, window_dots=1.0)),
           *((f"dot-matrix, {n} rows, nozzle rows", replace(STYLE, dot_rows=(n,)), replace(V2.style, nozzle=1.0))
             for n in STYLE.dot_rows))
WINDOW_INK = 235.0      # a window's strokes, light grey


def _only(kind: str):
    return replace(TEXT_V2, kinds=((kind, 1.0),))


COLUMNS_V2 = {    # gen-v2's new forms (M2b S7), each forced through its text spec: (TextSpec, V2Text)
    "glued words: a pair": (replace(TEXT, kinds=(("pair", 1.0),)), replace(V2.text, glued=1.0)),
    "line-pair: mfg and expiry": (_only("line-pair"), replace(V2.text, pair_import=0.0)),
    "line-pair: mfg and import": (_only("line-pair"), replace(V2.text, pair_import=1.0)),
    "dense panel": (_only("dense-panel"), V2.text),
    "dense panel: import, distractors": (replace(_only("dense-panel"), panel_codes=0.0),
                                         replace(V2.text, dense_import=1.0)),
}


def paint(layer: R.Layer) -> Image.Image:
    """The layer over plain grey: a label's sticker or a window's dark
    rectangle, then the ink, black, light grey on a window, or for embossed
    white where lit and black in shade."""
    canvas = np.full((layer.height, layer.width, 3), GREY, np.float32)
    window = "window" in layer.params
    if layer.backing is not None:
        s = layer.backing
        h, w = s.alpha.shape
        under = canvas[s.y:s.y + h, s.x:s.x + w]
        colour = layer.params["window" if window else "label"]["colour"]
        under += (np.asarray(colour, np.float32) - under) * s.alpha[..., None]
    tone = R.tone(layer)
    colour = WINDOW_INK if window else 0.0 if tone is None else 255.0 * tone[..., None]
    canvas += (colour - canvas) * R.ink(layer)[..., None]
    return Image.fromarray(np.clip(canvas + 0.5, 0, 255).astype(np.uint8))


def tile(block: T.Block, layer: R.Layer, zoom: int = 1) -> Image.Image:
    im = paint(layer)
    if zoom > 1:
        im = im.resize((im.width * zoom, im.height * zoom), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    for token, box in zip(block.tokens, R.boxes(layer)):
        if token.cls and box:
            x1, y1, x2, y2 = (v * zoom for v in box)
            draw.rectangle((x1 - 1, y1 - 1, x2, y2), outline=COLOURS[token.cls])
    return im


def cell(block: T.Block, style: str, spec, family: fonts.Family, seed: list[int], v2=None,
         width: int = CELL_W) -> Image.Image:
    """The block at the normal size over the same block as small print."""
    normal = R.shrink(R.render(block, style, family, NORMAL, np.random.default_rng(seed), spec, v2))
    small = R.shrink(R.render(block, style, family, SMALL, np.random.default_rng(seed), spec, v2))
    parts = [tile(block, normal), tile(block, small, ZOOM)]
    caption = f"{family.name}, {normal.params['weight']}, {block.kind}"
    height = sum(p.height + 6 for p in parts) + 16
    out = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(out).text((2, 0), caption, fill="black", font=FONT)
    y = 16
    for p in parts:
        out.paste(p.crop((0, 0, min(p.width, width), p.height)), (0, y))
        y += p.height + 6
    return out


def sheet(seed: int, v2: bool = False) -> Image.Image:
    """M2's sheet (the format families) or, with v2, gen-v2's new forms
    drawn with gen-v2's text and print."""
    columns = COLUMNS_V2 if v2 else {name: (spec, None) for name, spec in FAMILIES.items()}
    width = CELL_W_V2 if v2 else CELL_W
    train = fonts.families("train")
    rows = []
    styles = [(name, spec, V2.style) for name, spec in ROWS] + list(ROWS_V2) if v2 else \
        [(name, spec, None) for name, spec in ROWS]
    for r, (name, spec, v2_style) in enumerate(styles):
        style = name.split(",")[0]
        cells = []
        for c, (text_spec, v2_text) in enumerate(columns.values()):
            rng = np.random.default_rng([seed, r, c])
            block = T.random_block(rng, text_spec, v2_text)
            family = train[(r * len(columns) + c) % len(train)]     # every font shows up
            cells.append(cell(block, style, spec, family, [seed, r, c, 1], v2_style, width))
        height = max(c.height for c in cells) + 22
        row = Image.new("RGB", (len(cells) * (width + 10), height), "white")
        ImageDraw.Draw(row).text((2, 2), name.upper(), fill="black", font=FONT)
        for i, c in enumerate(cells):
            row.paste(c, (i * (width + 10), 22))
        rows.append(row)
    header = Image.new("RGB", (rows[0].width, 20), "white")
    for i, column in enumerate(columns):
        ImageDraw.Draw(header).text((i * (width + 10) + 2, 3), column, fill="black", font=FONT)
    out = Image.new("RGB", (rows[0].width, header.height + sum(r.height for r in rows)), "white")
    out.paste(header, (0, 0))
    y = header.height
    for row in rows:
        out.paste(row, (0, y))
        y += row.height
    return out


def print_height(rng: np.random.Generator) -> float:
    """A date's height over the photo's long side, from the plan's bands:
    normal 70% (1.5 to 4.5%), small 30% (0.6 to 1.5%), log-uniform."""
    lo, hi = (0.015, 0.045) if rng.random() < 0.7 else (0.006, 0.015)
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def timing(n: int, long_side: int, seed: int) -> dict:
    """Milliseconds per block on one CPU, as one of S6's worker processes
    (one per CPU) would have it: OpenCV's own threads switched off."""
    cv2.setNumThreads(1)
    train = fonts.families("train")
    for family in train:                         # fonts loaded once, as a worker would
        for style in R.STYLES:
            R.render(T.Block(((T.Token("08/2027", "date"),),), "single"), style, family, 20,
                     np.random.default_rng(0))
    per_style: dict[str, list[float]] = {s: [] for s in R.STYLES}
    boxed = unboxed = 0
    for i in range(n):
        rng = np.random.default_rng([seed, 99, i])
        block = T.random_block(rng)
        style, family = R.random_style(rng), fonts.random_family(rng, "train")
        size = print_height(rng) * long_side
        start = time.perf_counter()
        layer = R.shrink(R.render(block, style, family, size, rng))
        found = R.boxes(layer)
        per_style[style].append(1000 * (time.perf_counter() - start))
        for token, box in zip(block.tokens, found):
            if token.cls:
                boxed += box is not None
                unboxed += box is None
    every = [ms for times in per_style.values() for ms in times]

    def summary(times: list[float]) -> dict:
        return {"blocks": len(times), "mean_ms": round(float(np.mean(times)), 1),
                "median_ms": round(float(np.median(times)), 1),
                "p95_ms": round(float(np.percentile(times, 95)), 1),
                "max_ms": round(float(np.max(times)), 1)}
    return {"long_side": long_side, "all": summary(every),
            "by_style": {s: summary(t) for s, t in per_style.items() if t},
            "boxed_tokens": boxed, "boxed_tokens_without_ink": unboxed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--v2", action="store_true", help="gen-v2's new text instead of M2's format families")
    parser.add_argument("--time", type=int, default=0, metavar="N", help="also time N random blocks")
    parser.add_argument("--long-side", type=int, default=1600,
                        help="the photo's long side for --time (1,600 is the largest background)")
    args = parser.parse_args()
    run_dir = start_run("render-sheet-v2" if args.v2 else "render-sheet", vars(args), seed=args.seed)
    sheet(args.seed, args.v2).save(run_dir / "sheet.png")
    print(run_dir / "sheet.png")
    if args.time:
        result = timing(args.time, args.long_side, args.seed)
        (run_dir / "timing.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
