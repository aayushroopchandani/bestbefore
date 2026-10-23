"""Does dotify keep a date legible at 5, 7 and 9 rows of dots?

    python experiments/m2-dot-rows/measure.py            # 160 dates per condition

Two measures, both on the generator's own renderer and the 16 training
fonts; nothing from ExpDate or the store photos:

1. Stem width in cells. The stem of each face's "1" (the instance dotify
   draws with), measured across at half the digits' height, as a share of
   the digit height; in cells of the grid that is that share times the
   rows (square cells). Dotify puts a dot where the text covers 40% of a
   cell, so a stroke that straddles two cells keeps its dots wherever it
   falls only if it is at least 0.8 of a cell wide.
2. Read by an OCR. 160 dates (formats drawn by the text spec, upper case,
   each training font 10 times) drawn solid and as dot-matrix at 5, 7 and
   9 rows: digits 28 px, no letter spacing, black on light grey in the
   middle of a 1000 x 400 canvas, read by RapidOCR 1.2.3 with its default
   settings (baseline A's reader). Exact reads (spaces ignored) and the
   mean similarity of what is read to the date (difflib's ratio).
   RapidOCR reads dot-matrix poorly in general (Scanify's misses), so only
   the differences between row counts mean anything.

Needs the fonts (scripts/fetch_fonts.py) and the baselines extra.
"""
from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR

from bestbefore.synth import fonts, render as R, text as T
from bestbefore.synth.spec import STYLE
from bestbefore.tables import markdown_table

OUT = Path(__file__).parent / "results"
ROWS = (5, 7, 9)                # the plan's rows as first written
KEEPS = 0.8                     # a stroke this many cells wide keeps its dots (40% rule)


def stem_share(face: fonts.Face) -> float:
    """The width of the "1"'s stem over the digits' height."""
    em = 1000
    font = fonts.load(face, em)
    top, bottom = fonts.digit_band(face)
    left, t, right, b = font.getbbox("1", anchor="ls")
    im = Image.new("L", (right - left + 4, b - t + 4))
    ImageDraw.Draw(im).text((2 - left, 2 - t), "1", font=font, fill=255, anchor="ls")
    row = np.asarray(im)[round(2 - t + em * (top + bottom) / 2)] >= 128
    edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(int), [0]))))
    return float(max(edges[1::2] - edges[::2])) / (em * (bottom - top))


def stems() -> list[dict]:
    rows = []
    for family in fonts.families("train"):
        share = stem_share(family.face(bold=True))
        rows.append({"family": family.name, "stem / digit height": round(share, 3),
                     **{f"cells at {n} rows": round(share * n, 2) for n in ROWS}})
    return rows


def legibility(n: int) -> list[dict]:
    ocr = RapidOCR()                    # default settings, as baseline A
    train = fonts.families("train")
    flat = replace(STYLE, letter_spacing=(0.0, 0.0))
    conditions = [("solid", flat)] + [(f"dot-matrix, {r} rows", replace(flat, dot_rows=(r,)))
                                      for r in ROWS]
    scores = {name: [] for name, _ in conditions}
    for i in range(n):
        rng = np.random.default_rng([7, i])
        token = T.date_token(T.random_date(rng), T.random_format(rng), "upper", "expiry")
        truth = token.text.replace(" ", "")
        for name, spec in conditions:
            layer = R.shrink(R.render(T.Block(((token,),), "single"), name.split(",")[0],
                                      train[i % len(train)], 28.0, np.random.default_rng([8, i]), spec))
            canvas = np.ones((400, 1000), np.float32)
            y, x = (400 - layer.height) // 2, (1000 - layer.width) // 2
            canvas[y:y + layer.height, x:x + layer.width] -= R.ink(layer)
            image = np.repeat((canvas * 215).astype(np.uint8)[..., None], 3, axis=2)
            result, _ = ocr(image)
            read = "".join(text for _, text, _ in result or []).replace(" ", "").upper()
            scores[name].append((read == truth, difflib.SequenceMatcher(None, truth, read).ratio()))
    return [{"drawn": name, "exact reads": f"{sum(e for e, _ in s)} / {n}",
             "mean similarity": f"{np.mean([r for _, r in s]):.2f}"} for name, s in scores.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=160)
    args = parser.parse_args()
    widths = stems()
    reads = legibility(args.n)
    OUT.mkdir(exist_ok=True)
    (OUT / "dot_rows.json").write_text(json.dumps({"stems": widths, "reads": reads}, indent=2))
    print(markdown_table([{k: str(v) for k, v in row.items()} for row in widths], left=("family",)))
    for n in ROWS:
        cells = [row[f"cells at {n} rows"] for row in widths]
        print(f"{n} rows: median stem {np.median(cells):.2f} cells; "
              f"{sum(c >= KEEPS for c in cells)} of {len(cells)} faces at least {KEEPS}")
    print()
    print(markdown_table(reads, left=("drawn",)))


if __name__ == "__main__":
    main()
