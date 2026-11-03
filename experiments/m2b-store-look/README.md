# M2b / S3: the store's expiry dates, as the M2 detector sees them

## Question

The M2 detector (`gen-rot-1280`, seed 0) finds 39 of the 81 store expiry
dates in frame. Which of the rest does it box but doubt, and which does it
never box? What do they look like (contrast, sharpness, size) next to the
dates it finds, ExpDate's and the generator's? And are the generator's
dates as big as the store's?

These numbers first came from a scratch script and two snippets
(2026-10-28 and 2026-10-31) and shaped M2b's plan
([`docs/m2b-store-and-size.md`](../../docs/m2b-store-and-size.md), "What M2
and the store gallery tell us"). This experiment makes them reproducible
from a committed script.

## Method

`measure.py` reads store set v1 only (`store.V1`; v2, the test set, is
never opened), M2's predictions for its three `gen-rot-1280` seeds (kept by
`eval_detector.py --set store` in S8, every box down to confidence
0.001), real-val, gen-sample and the laptop's 5,000 generated photos
(gen50k's tasks 0 to 4, M2 S6's trial). Nothing is trained or scored
again; the seeds are used as they are, each at its own F2 threshold.
Other-face photos are left out: their expiry isn't in frame.

- **Outcome.** Each seed's boxes are matched to the labelled dates with
  `store_gallery.match`, the greedy matching S8's counts use (IoU 0.5,
  most confident first, over every box). An expiry is **kept** if its box
  reaches the threshold, **doubted** if it is boxed only below it, and
  **never boxed** otherwise. The script stops if any seed's counts differ
  from its `store.json`.
- **Contrast**: inside the box, luminance split by Otsu's threshold, the
  bright side's mean minus the dark side's (0 to 255). Under 40 is
  **faint**.
- **Sharpness**: the standard deviation of the Laplacian, with the crop
  first resized so its short side is 32 px, divided by the contrast. So it
  measures edges, not ink strength. **Soft** means under the kept
  expiries' median.
- **Size**: the box's short side over the photo's long side. This is the
  small-print measure made turn-neutral: for a sideways date, the short
  side is its print height. Generated dates are measured the same way,
  from the sidecars' ink boxes.
- **Month words**: the labelled format has `MON` in any case (as in M1's
  store-misses experiment). **Full month names** are M2 S8's cause
  (`m2_results.cause_of`): upright expiries whose format has `Month`.
- **Wide sideways boxes**: a sideways-tagged expiry whose box is wider
  than tall is print turned 180°, not 90°.

Counts, medians, quartiles and percentiles only: no photo, crop or photo
name.

```bash
python experiments/m2b-store-look/measure.py
```

## Result

Run on 2026-11-02 on the laptop's CPU (7 s), with the code as committed beside it. Pasted as the script
prints it:

Store set v1, other-face left out: 192 labelled dates, 81 of them expiries.

### 1. Outcome per seed (store_gallery.match, each seed at its own threshold)

| seed | threshold | kept | doubted | never boxed | boxed at any confidence | store.json: found, boxed |
|---|---:|---:|---:|---:|---:|---:|
| gen-rot-1280-s0 | 0.342 | 39 | 30 | 12 | 69 | 39, 69 |
| gen-rot-1280-s1 | 0.424 | 53 | 17 | 11 | 70 | 53, 70 |
| gen-rot-1280-s2 | 0.399 | 46 | 26 | 9 | 72 | 46, 72 |

Mean of 3 seeds: kept 46.0 (39–53), boxed at any confidence 70.3 (69–72).

### 2. The look inside each date's box: median [quartiles]

| expiries (s0's outcome) | n | contrast | sharpness | short side, px | short side, % of long side | faint (contrast < 40) |
|---|---:|---:|---:|---:|---:|---:|
| kept | 39 | 103.8 [84.0–129.8] | 0.42 [0.34–0.49] | 68.0 [32.0–102.5] | 2.04 [1.77–2.70] | 1 |
| doubted | 30 | 106.2 [94.9–126.3] | 0.34 [0.28–0.39] | 31.5 [20.2–58.8] | 1.44 [1.28–1.83] | 0 |
| never boxed | 12 | 80.9 [62.7–104.8] | 0.36 [0.27–0.39] | 49.5 [16.0–91.5] | 1.71 [1.00–2.27] | 0 |
| real-val | 187 | 121.8 [88.9–143.8] | 0.44 [0.34–0.52] | 27.0 [24.0–35.0] | – | 4.3% |
| gen-sample | 1764 | 69.1 [43.2–99.7] | 0.32 [0.22–0.45] | 33.0 [23.0–48.2] | – | 21.1% |

real-val and gen-sample rows are over every labelled date; their faint column is a share. Soft below means sharpness under the kept expiries' median (0.42); real-val is soft for 43.9% of its dates, gen-sample for 70.2%.

### 3. By tag

| tag | expiries | tall boxes | faint (< 40) | soft | small (under 1.5%) | kept by s0 |
|---|---:|---:|---:|---:|---:|---:|
| sideways | 15 | 13 | 0 | 12 | 8 | 3 |
| dot-matrix | 20 | 0 | 0 | 11 | 2 | 19 |
| small-print | 15 | 0 | 0 | 14 | 15 | 2 |
| clean | 31 | 0 | 1 | 15 | 0 | 15 |

Sideways expiries with wide boxes (turned 180°, not 90°): 2. The highest confidence any seed gives one: 0.005. At least as sharp as the kept median: 1; 2.5% or more of the long side: 1.

### 4. Month words, by s0's outcome

| expiries | n | kept | doubted | never boxed |
|---|---:|---:|---:|---:|
| month as a word | 36 | 9 | 20 | 7 |
| numbers only | 45 | 30 | 10 | 5 |

Of the 30 doubted, 20 have the month as a word.

### 5. The three seeds, over the 81 expiries

| expiries | n |
|---|---:|
| found by at least one seed | 55 |
| found by all 3 | 37 |
| boxed by none (never boxed by all 3) | 7 |

Full month names (upright, M2 S8's cause), kept by each seed; an unboxed one counts as confidence 0: gen-rot-1280-s0 6 of 19, median confidence 0.131; gen-rot-1280-s1 15 of 19, median confidence 0.700; gen-rot-1280-s2 9 of 19, median confidence 0.333.

### 6. Found rates by role and year, over the 3 seeds

| dates | years | n | found (seed-runs) | boxed at any confidence |
|---|---|---:|---:|---:|
| expiry | 2026 or earlier | 12 | 94.4% | 100.0% |
| expiry | 2027 or later | 69 | 50.2% | 84.5% |
| expiry | all | 81 | 56.8% | 86.8% |
| mfg | 2026 or earlier | 86 | 57.4% | 89.1% |
| mfg | all | 86 | 57.4% | 89.1% |
| other_date | all | 25 | 41.3% | 86.7% |
| sideways expiry | all | 15 | 17.8% | 62.2% |
| sideways mfg | all | 17 | 27.5% | 62.7% |
| sideways other_date | all | 7 | 28.6% | 100.0% |
| upright expiry | all | 66 | 65.7% | 92.4% |
| upright mfg | all | 69 | 64.7% | 95.7% |
| upright other_date | all | 18 | 46.3% | 81.5% |

Expiries in 2026 or earlier: 12; by tag clean 6, dot-matrix 5, small-print 1; 10 numbers only; by format DD/MM/YYYY 8, DDth Month YYYY 2, MM/YYYY 1, YYYYMMDD 1.

ExpDate real-train's date transcriptions with a year: 1,244; 2020 to 2023: 1,065; 2027 or later: 21.

### 7. Date sizes: box short side over the photo's long side

| dates | n | under 1.5% | 1.5–2.5% | 2.5% and up | median |
|---|---:|---:|---:|---:|---:|
| store v1, expiry dates | 81 | 30.9% | 46.9% | 22.2% | 1.83% |
| store v1, every date | 192 | 37.0% | 44.3% | 18.8% | 1.73% |
| generated (M2's spec, task 0: 1,000 photos) | 2,166 | 15.9% | 19.8% | 64.3% | 3.06% |
| generated (M2's spec, tasks 0–4: 5,000 photos) | 10,948 | 17.3% | 21.0% | 61.7% | 3.04% |
| generated, upright (5,000 photos) | 6,004 | 20.6% | 23.5% | 55.9% | 2.74% |
| generated, tilted (5,000 photos) | 1,680 | 4.7% | 11.7% | 83.6% | 5.23% |
| generated, sideways (5,000 photos) | 3,264 | 17.7% | 21.2% | 61.1% | 2.98% |

Store v1, every date, percentiles 5th 0.89%, 10th 1.12%, 25th 1.36%, 50th 1.73%, 75th 2.19%, 90th 2.80%, 95th 3.56%; smallest 0.07%, largest 5.70%.

Generated median over store median (task 0): 1.77 times every store date's, 1.67 times the expiries'.

Pending S4 (design ids): designs in v1, expiries per design.

## What it shows

Every number above matches the plan's within rounding, with one
exception, noted below.

- **Seen but doubted (Cause 1).** For the M2 detector, 30 of the 81
  expiries are boxed below its threshold of 0.342 and 12 are never boxed.
  Month words make up 20 of the 30 doubted; of all 36 month-word expiries,
  only 9 are kept. The seeds disagree: 55 expiries are found by at least
  one seed and 37 by all three. Seed 0 is the weakest on full month names
  (6 of 19 kept, median confidence 0.131, against 15 and 9 of 19 for
  seeds 1 and 2).
- **Store print isn't fainter; it's smaller and softer (Cause 3).** The
  median contrast is about the same for kept (104) and doubted (106)
  expiries; only 1 of 81 is faint, against 21% of generated dates. The
  doubted are softer (0.34 against 0.42) and smaller (31.5 against 68 px
  on the short side; 1.44% against 2.04% of the long side).
- **Sideways (Causes 2 and 3).** 13 of the 15 sideways expiries are tall
  boxes. The 2 wide ones are print turned 180°: no seed gives either more
  than 0.005, though one is at least as sharp as the kept median and 2.5%
  or more of the long side. 12 of the 15 are soft and none is faint.
  **New:** measured turn-neutrally, 8 of the 15 are under 1.5% of the long
  side. The scratch script measured box height, which for a sideways date
  is its length, and so counted none as small. The plan doesn't quote that
  count.
- **Generated dates are bigger than the store's (Cause 4).** Median 3.06%
  against 1.73% for every store date (1.77 times) and 1.83% for the
  expiries (1.67 times). Only 19.8% of generated dates fall in 1.5–2.5%,
  where 44.3% of store dates lie. Tilted generated dates are the largest
  (median 5.23%). Over all 5,000 generated photos instead of task 0's
  1,000, the bands move by under 3 points (median 3.04%).
- **The refuted hunches stay refuted.** Manufacture dates are found 57.4%
  of the time and expiries 56.8% (upright: 64.7% and 65.7%). The 12
  expiries from 2026 or earlier are found 94.4% of the time, against 50.2%
  for 2027 on. They are the easy kinds: 10 of them are numbers only and 5
  are dot-matrix. ExpDate real-train has 1,065 of its 1,244 dated
  transcriptions in 2020 to 2023 and 21 from 2027 on.
- **The one difference from the plan.** The plan's "mostly numeric (8)"
  for the 12 early expiries counts one format, `DD/MM/YYYY`. Numbers-only
  expiries number 10 (8 `DD/MM/YYYY`, 1 `MM/YYYY`, 1 `YYYYMMDD`). The
  claim, mostly numeric, holds either way. The plan now says "10, 8 of
  them `DD/MM/YYYY`" (owner's OK, S3).

## Pending (after S4)

- **Design counts**: designs in v1, and expiries per design. The plan's
  "four designs give about 30 of the 81; one pouch design about 10, and 6
  of the 12 never boxed" is by eye until S4 freezes the design ids.
- **The same print swinging between photos** (Cause 1: one pack's date at
  0.84 in one photo and 0.010 in another) needs the design ids too.

## Not measured here

- Cause 2's other layouts (a date glued to its keyword, two dates on a
  line, dense panels, laser windows, barcode digits, missing dot rows,
  worn print) were seen in the gallery. No label records them, so they
  stay observations by eye.
- Seed 0's confidences on single photos (the plan's "0.02 to 0.08 where
  seeds 1 and 2 give 0.6 to 0.85"): this script reports medians over the
  19 full month names, not per-photo values.

## Decision

Nothing in M2b's plan changes: the measured numbers are the ones it was
built on. The design counts join this README after S4.
