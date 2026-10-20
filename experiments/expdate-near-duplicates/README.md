# ExpDate: are there duplicates across the splits?

## Question

Does any photo in ExpDate's **validation** or **test** split also appear,
nearly unchanged, in the training data (Products-Real train or
Products-Synth)? And do validation or test share a *product* with train,
photographed separately?

## Why it matters

If a validation or test photo is also in training, the model partly
*recognises* it instead of reading a new photo, and that score comes out
too high. The two numbers at stake:

- **Validation** (165 photos): M1 chose its model size, input size,
  threshold and training data on it.
- **Test** (665 photos): M1's headline, 637 of 665 expiry dates found and
  mAP@0.5 0.993.

We made the validation split ourselves (15% of Products-Real train,
seed 0). ExpDate's authors made the train / test split, so we didn't know
how careful it was.

## Method

`find_duplicates.py` fingerprints all 13,625 photos (each opened upright
through `load_upright`) and compares four pairs of splits:

| Comparison | Why |
|---|---|
| validation vs train | our own split |
| test vs train + validation | ExpDate's split |
| validation vs synthetic | synthetic photos are training data too |
| test vs synthetic | the same |

Train vs synthetic isn't compared: both are training data, so a photo in
both leaks nothing.

Four checks, strictest first:

1. **Exact copy:** the same file bytes (SHA-256).
2. **Near copy:** the same photo resized or recompressed. Each photo gets
   a 256-bit *difference hash* (dHash): shrink it to 17 × 16 grey pixels,
   then record for each pixel whether it's brighter than its right-hand
   neighbour. Two copies of one photo differ in a few bits; unrelated
   photos differ in about half (128). A pair counts as near at **≤ 25
   bits**, a limit set before looking at any result.
3. **Reused photo:** a synthetic photo drawn on top of a real one. These
   keep the real photo's size, so every same-size pair is compared pixel
   by pixel on 64 × 64 grey thumbnails, after evening out brightness and
   contrast. It catches copies whose fake shapes and darkening move them
   too far for the hash.
4. **Same date text:** a *different* photo that may show the same product,
   found because both photos carry the same printed date.

Every check produces candidates, ranked. The contact sheets the script
writes to `results/runs/<run>/` (not committed: they're ExpDate images)
were then **checked by eye**: the closest 24 pairs of every check, and
all 65 validation / train pairs that share a date.

## Result

Rebuild with:

```bash
python experiments/expdate-near-duplicates/find_duplicates.py --write
```

<!-- results:start -->
| check | photos checked | exact copies | near copies (≤ 25 bits) | nearest photo, bits apart: min / median | reused (≥ 50% pixels match) | best pixel match | share a date text |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation vs train | 165 | 0 | 0 | 77 / 98 | 0 | 41% | 65 |
| test vs train + validation | 665 | 0 | 0 | 75 / 97 | 0 | 31% | 204 |
| validation vs synthetic | 165 | 0 | 1 | 6 / 92 | 1 | 99% | 18 |
| test vs synthetic | 665 | 0 | 0 | 71 / 92 | 0 | 33% | 83 |
<!-- results:end -->

**What the eye check found** (run on 2026-10-20):

- **Test: clean.** No test photo is a copy of a training or validation
  photo. The closest pairs by hash (71 bits or more) and by pixels (33% at
  most) are all different packs. Most of the high pixel matches are one
  popcorn-bag background the synthetic set reuses many times, next to
  unrelated test photos.
- **Validation vs real train: no copies.** The closest pair is 77 bits
  apart, and the best pixel match (41%) is two different tea boxes of the
  same brand.
- **Validation vs synthetic: 1 photo reused 14 times.** Validation photo
  `img_00277.jpg` (a tin with `EXPIRY DATE:2023/07/03`) is the background
  of **14 synthetic training photos**. Its real date was erased and fake
  dates printed on, sometimes with shifts, zooms, darkening or fake
  shapes on top. Their pixel matches run from 99% down to 17%, so only 7
  pass the 50% line; the other 7 were confirmed by eye. No other
  validation photo was reused.
- **Same product, separate photo: 2 clear pairs, 1 possible**, out of the
  65 validation photos that share a date with a train photo:
  `img_00449` / `img_00415` (two bottles of the same mayonnaise),
  `img_00203` / `img_00819` (two cups of the same snack), and possibly
  `img_00492` / `img_00102` (a yogurt drink). These are different
  physical packs photographed separately, not copies. A few more pairs
  are the same *brand* with a different product (for example
  `img_00034` / `img_00035`, two flavours of one biscuit). The rest share
  a date by coincidence. For test vs train, the closest 24 of the 204
  same-date pairs were checked, and none were the same product.

**A side finding:** Products-Synth is built partly on real Products-Real
photos. Besides `img_00277`, at least 8 *training* photos (tins, cans,
a tea caddy) are each the background of several synthetic photos. That's
harmless, since both are training data, but it means "synthetic" and
"real" in ExpDate aren't fully separate.

## What it means

- **M1's test result stands unchanged: 637 of 665, mAP@0.5 0.993.** No
  test photo was in training.
- **Validation is very slightly optimistic,** by at most one photo: 1 of
  165 photos, 1 of 187 dates (0.5 points of recall). M1 finds that date
  (confidence 0.88, IoU 0.97); it would probably have found it anyway, but
  we can't tell.
- **None of M1's choices change.**
  - **C1** (model and input size): all four candidates trained on the
    same synthetic data, so all four had the same head start on that
    photo. They were within 1 point of each other anyway.
  - **C2** (keep synthetic data or not): the run with synthetic data
    found 185 of 187 validation dates, real-only 179. Dropping the reused
    photo can change that by one date at most, which leaves synthetic
    still ahead, and the rule keeps synthetic data on a tie anyway.
  - **The threshold** (0.728) was chosen over 187 dates; one date doesn't
    move it.
- **The same-product pairs are normal and are left in.** ExpDate's real
  photos were taken in shops, so popular products turn up more than
  once. They make validation a little easier than a brand-new shop
  would. That's one more reason the store photos are the real test.

## Decision

**No retraining, and no change to M1's results.** The validation split
stays as it is (`val.txt`), so every M1 run stays comparable.

**For M2:** when the detector is retrained, leave the 14 synthetic copies
of `img_00277.jpg` out of the training list, so that validation photo is
clean from then on.

## Limits

- The hash and pixel checks find copies of a **whole photo**. A synthetic
  photo made from a **crop** of a real one, resized to a new shape, would
  be missed: crops change both the size and the hash. The reused copies
  found here all kept the original size, which suggests the generator
  keeps sizes, but it isn't proven.
- The same-date check only finds same-product pairs whose printed date
  was transcribed identically. A twin with a different date isn't found.
- The test vs train same-date pairs were only checked by eye for the
  closest 24 of 204.
