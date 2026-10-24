# M2 / S5's look check: do generated blocks land on the pack?

## Question

On S5's sample of generated photos, how many generated blocks lie mostly
off the pack, on hands, floors, tables, shelves and walls? And would a
simple placement rule keep them on it?

## Why it matters

The generator places a block on the plainest of 20 free places 70% of the
time. A hand or an out-of-focus shelf is plainer than printed packaging,
so crisp text can land on a blurred surface: a sign of pasting that a
model could learn instead of the date
([`docs/m2-synthetic-generator.md`](../../docs/m2-synthetic-generator.md),
S5, "For the owner (1)").

## Method

- **Labels, by eye, on 2026-10-23:** the 50 photos of S5's contact sheet
  (`python scripts/contact_sheet.py --set gen-sample --n 50`, seed 0),
  with each generated block outlined. A block is on the pack when most of
  it lies on the pack. `on_pack.csv` has one row per block: the photo, the
  block's index in its sidecar line, its place in pixels (`x, y, w, h`)
  and `on_pack` (`yes` or `no`). All 72 blocks of those photos are there.
- **The photos they belong to:** gen-sample as it is on disk, made at
  `d4de541` (after the floor on faint ink) and again, byte for byte, in
  S6. `recount.py` checks every place against that sample's sidecar and
  stops if one differs.
- **Two measures from S5's record:** an edge margin, which keeps a block
  whose centre is at least 0.15 of the photo's short side from every
  edge; and nearness to the background's labelled dates, the distance
  from a block's centre to the nearest `date` box, over the photo's long
  side.
- Counts only. The photos are generated on ExpDate backgrounds, so they
  stay in `data/`, never committed.

```bash
python experiments/m2-generated-look/recount.py
```

## Result

Recounted on 2026-10-24 (M2 S7):

| blocks | n | on the pack | off the pack | off |
|---|---:|---:|---:|---:|
| all, on 50 photos | 72 | 51 | 21 | 29% |
| centre at least 0.15 of the short side from every edge | 51 | 41 | 10 | 20% |
| centre within 0.2 of the long side of a labelled date | 22 | 16 | 6 | 27% |
| centre within 0.15 of the long side of a labelled date | 13 | 11 | 2 | 15% |

The edge margin would give up 10 of the 51 on-pack places (20%).

- **21 of 72 blocks are off the pack**, not the 20 that S5's record first
  gave. The record is corrected. No record survives of how the 20 was
  counted, or whether on an earlier version of the sample, so the labels
  here are the count.
- **An edge margin would only trim the off-pack share,** from 29% to
  20%, at the cost of one on-pack place in five. As S5 found.
- **Nearness to the dates doesn't tell pack from not-pack either.** Within
  0.2 of the long side, 27% of blocks are off the pack, against 29% of all.
  S5's record gives 7 of 25 and 4 of 15 for this measure. Its code wasn't
  kept, and none of several readings of its wording gives those counts
  (the centre or the whole block, the date boxes or every labelled box,
  over the long or the short side), so this script's reading is the one
  above. The two other signals in the record, colour like the pack round
  the dates (AUC 0.64) and the surface's detail, can't be rebuilt from
  this file and stay as first recorded.

## Decision

Unchanged: the plan's placement is kept, and a mask of the pack is in
the plan's Later (decided by the owner on 2026-10-23). The recount moves
no number past what that decision weighed.
