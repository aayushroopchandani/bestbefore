# M2 / the ablation: turned copies, generated dates, or both?

## Question

What does each kind of new training data add to the date detector:
ExpDate's training photos turned 90° (`rot`), 50,000 photos with generated
dates (`gen`), or both (`gen-rot`), against M1's data with M2's recipe
(`base`)? Measured on the validation suite's hard sets and, once, on the
store photos, next to M1. And which arm becomes the M2 detector?

## Why it matters

M1's detector found 97% of ExpDate test's expiry dates but 27% of the
store photos' (mean of 3 seeds). Four causes explain the misses
([`docs/m2-synthetic-generator.md`](../../docs/m2-synthetic-generator.md),
"What M1 tells us"): sideways dates were never in training; ExpDate's month
words are all short and big, the store's often full names and small;
small print is seen but doubted; and the threshold was chosen where every
date is easy. Turning photos is the cheap remedy for the first cause; the
generator targets the first three. If the cheap one does as well on
sideways dates, the generator has to earn its place on the others.

## Method

| Arm | Training photos (real ones repeated to about 7%) | Listed | Epochs |
|---|---|---:|---:|
| `base` | ExpDate real train + synthetic (the 14 copies of a validation photo left out) | 12,781 | 200 |
| `rot` | the same, plus one copy of each turned 90° or 270° | 25,562 | 100 |
| `gen` | `base`'s, plus 50,000 generated photos | 66,529 | 38 |
| `gen-rot` | `rot`'s, plus 50,000 generated photos | 79,310 | 32 |

- **Fixed for every arm:** `yolo11n` from COCO weights, 640 px, 2.56
  million training images, MuSGD with `lr0=0.01` and `momentum=0.937`, no
  early stopping; the model is the final weights (`last.pt`). Seeds 0, 1
  and 2.
- **Validation suite** (`scripts/score_suite.py`): `real-val`, ExpDate
  validation as it is (the guard); `real-val-rot`, the same photos turned;
  `real-val-small`, the same photos in 2 × 2 grids, every date at half its
  size; `synth-val`, 2,100 generated photos in 4 held-out fonts on
  validation photos, one target date each, in 7 strata that each change
  one thing from `clean`.
- **Threshold:** each run keeps the confidence with the best F2 for dates
  on the three real sets pooled, unchanged on the store photos.
- **Store photos** (S8, once per run, on the laptop): expiry dates found
  by failure tag and by cause: the month as a word (full name or
  abbreviation) and numeric dates by print height, for upright expiries,
  and sideways ones apart. Counts only.

**Decision rules, written before any run** (`bestbefore.m2`):

- **Pilot gate** (`pilot_passes`): a short `gen` run must box at least
  half of synth-val's targets at any confidence, or the generated arms
  stop.
- **The M2 detector** (`decide_detector`), over `base`, `rot`, `gen`,
  `gen-rot` and `gen-rot-1280` (from
  [`m2-input-size`](../m2-input-size/README.md)): an arm is eligible if
  its mean mAP@0.5 for dates on `real-val` is no more than 1 point under
  `base`'s; among those, the highest mean mAP@0.5 over the three hard
  sets, where any candidate within 1 point of it goes to the cheapest
  (listed cheapest first). The M2 detector is that arm's seed 0. The store
  photos judge every arm afterwards; none is picked by its store score.
- **Rotation:** no rule; read off `rot` against `base` and `gen-rot`
  against `gen`.

The chain applies the rules as their inputs complete and writes them to
`results/runs/m2-decisions.json`; `summarise.py` applies them again
through the same function (`m2.decisions()`) and warns if they differ.

**What we expected, written before any run:**

| Arm | sideways | month as a word | small print | ExpDate validation |
|---|---|---|---|---|
| `base` | still about 0 | low | slightly up (threshold) | as M1 |
| `rot` | up a lot | low | as `base` | as M1 |
| `gen` | up | up a lot | up | as M1, perhaps 1 point lower |
| `gen-rot` | up the most | up a lot | up | as M1 |

If `gen` scores high on synth-val month words but doesn't lift the store's
month-word dates, the generated dates don't look like real ones, and that
gets written up as a finding.

## Result

Rebuild with:

```bash
scripts/pull_cluster_results.sh
python experiments/m2-ablation/summarise.py --write
```

<!-- results:start -->
_Not run yet: the arms train in GPU jobs A (72914) and B (72915)._
<!-- results:end -->

## What it means

_Written once every run is scored on validation and the store photos (S8)._
