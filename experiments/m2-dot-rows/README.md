# M2 dotify: are dates legible at 5, 7 and 9 rows of dots?

## Question

The plan's dotify draws dot-matrix text by sampling a TrueType font on a
grid of 5, 7 or 9 rows across a digit's height, putting a dot where the
text covers at least 40% of a cell (docs/m2-synthetic-generator.md, "Print
styles"). On S4's render sheet the 5-row dates garbled: `2` read as `Z`,
`1` as `!`, Oswald's `EXP` as `#KP`. Is that a quirk of a few fonts, or of
5 rows?

## Method

`measure.py`, on the generator's own renderer and the 16 training fonts
(nothing from ExpDate or the store photos):

1. **Stem width in cells.** The stem of each face's `1` (the bold instance
   dotify draws with; VT323 and Share Tech Mono have only a regular),
   measured across at half the digits' height, as a share of the digit
   height. In cells of the grid that is the share times the rows (square
   cells). A stroke `w` cells wide that falls across two cells covers
   `w / 2` of each, so the 40% rule keeps its dots wherever it falls only
   if `w` is at least 0.8.
2. **Read by an OCR.** 160 dates (formats drawn by the text spec, upper
   case, each training font 10 times), drawn solid and as dot-matrix at 5,
   7 and 9 rows: digits 28 px, no letter spacing, black on light grey in
   the middle of a 1000 × 400 canvas, read by RapidOCR 1.2.3 with its
   default settings (baseline A's reader). Exact reads (spaces ignored),
   and the mean similarity of what is read to the date (difflib's ratio,
   0 when nothing is read). RapidOCR reads dot-matrix poorly in general
   (Scanify's misses), so only the differences between row counts mean
   anything.

```bash
python scripts/fetch_fonts.py
python experiments/m2-dot-rows/measure.py
```

## Result

Run on 2026-10-23, on the renderer at commit `ef9156e`, in 1.6 minutes on the
laptop.

| Stem of the `1` | Median | Range | Faces at 0.8 cells or more |
|---|---:|---:|---:|
| share of the digit height | 0.20 | 0.12 to 0.24 | |
| cells at 5 rows | 0.99 | 0.61 to 1.22 | 14 of 16 |
| cells at 7 rows | 1.40 | 0.85 to 1.71 | 16 of 16 |
| cells at 9 rows | 1.79 | 1.09 to 2.20 | 16 of 16 |

The thinnest are Share Tech Mono (0.12) and VT323 (0.16), the two
without a bold; then Lato (0.17).

| Drawn | Exact reads | Mean similarity |
|---|---:|---:|
| solid | 155 / 160 | 1.00 |
| dot-matrix, 5 rows | 1 / 160 | 0.07 |
| dot-matrix, 7 rows | 13 / 160 | 0.24 |
| dot-matrix, 9 rows | 41 / 160 | 0.45 |

- **At 5 rows a bold stroke is about one cell wide**, at the edge of what
  the 40% rule keeps. With the horizontal pitch up to 1.2 times the
  vertical, a stem can be under 0.85 of a cell across, and where it falls
  between two cells its dots go: that is how a `1` comes to read as `!`.
- **5 rows is also too coarse for a TrueType glyph's shape**: a `2`'s
  curve becomes the diagonal of a `Z`, and small counters close. Real
  5-dot printer fonts are drawn for their grid; sampling a font onto one
  doesn't give that.
- **The OCR agrees:** it reads almost nothing at 5 rows (1 of 160, against
  41 of 160 at 9 rows) where it reads solid text almost always.
- The plan's own reason for the rows names 7 and 9 ("5×7 and 7×9 are
  printer standards": 7 and 9 rows tall).

## Decision

Decided 2026-10-23 by the owner: 5 rows dropped. `dot_rows` in
`synth/spec.py` is now `(7, 9)`, and the plan says so. `measure.py`
still measures all three row counts, so the numbers above reproduce.
