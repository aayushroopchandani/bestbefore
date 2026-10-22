# M2: Synthetic generator

Planned 2026-10-21. Status: **in progress**: S1 done 2026-10-22 (GPU jobs A
`72914` and B `72915` held). Next: S2 (the derived sets).
The learnings guide for M2 (`learnings/m2.md`) is written separately,
later.

**Goal.** Teach the date detector the dates M1 never learned: sideways,
month written as a word, and small. A generator draws such dates onto real
pack photos. An ablation measures what it adds on the store photos, next
to a cheap rival (the training photos turned 90°) and a larger input
(1280 px).

**Done when.** `results/m2_detector.md` reports, for every arm of the
ablation (3 seeds each):

- the **validation suite**: ExpDate validation (must not get worse) and
  three hard sets (real dates turned sideways, real dates made small,
  synthetic dates one difficulty at a time);
- the **store photos** by failure tag and by cause (month as a word;
  numbers by print size), next to M1;
- the **M2 detector**, chosen on validation by a rule written below,
  scored once on ExpDate test.

M2 is still judged only on **finding** dates. Reading them is M3's job.

---

## What M1 tells us

### The gap

| Expiry dates found | ExpDate test | Store photos |
|---|---:|---:|
| the M1 detector (`n640`, seed 0) | 637 / 665 | 18 / 81 |
| mean of 3 seeds | 97.1% | 26.7% |

By store tag (mean of 3 seeds): dot-matrix 73%, clean 23%, sideways 0 of
15, small print 0 of 15. Precision stayed at 0.93 to 0.95. The detector
rarely invents dates; it misses them. Four causes explain the misses.

### Cause 1: sideways dates were never in training

17.6% of store date boxes are taller than wide. ExpDate has 0% (real) and
0.5% (synthetic). No seed found a sideways expiry, and none even got a box
at low confidence. This is plain absence from the training data.

### Cause 2: month words were in training, but not like the store's

M1's analysis said ExpDate's dates are about 94% numbers. That's true of
the **real** photos. It isn't true of the **synthetic** photos M1 also
trained on. Counted for this plan, from the ExpDate files:

| | real train | synthetic |
|---|---:|---:|
| dates with the month as a word | 77 of 1,244 (6.2%) | **8,026 of 16,674 (48.1%)** |
| full month names (`April`, `December`) | 0 | 0 |
| ordinals (`19th`) | 0 | 0 |
| median height of those dates at 640 px | 16.0 px | 26.4 px |

So M1 saw about 8,000 month-word dates in training and still found 3% of
the store's (44% got a box at some confidence). And it does find
ExpDate-style month words: it found 185 of 187 validation dates, so at
least 10 of the 12 that have a month word.

The store's month-word dates differ from ExpDate's in three ways:

1. **Full names and ordinals.** 22 of the 36 store expiries with a month
   word spell it out (`Month YYYY` 19, `DDth Month YYYY` 2,
   `DD Month YYYY` 1; `results/m0_store_stats.md`). ExpDate has none:
   every month word is a three-letter abbreviation.
2. **Size.** ExpDate's synthetic month-word dates are big (26 px at
   640 px input). Store dates are small (median 11 px).
3. **Look.** ExpDate's synthetic dates are single, isolated lines, often
   with heavy noise and pasted black shapes (M0's synthetic contact
   sheet). A model can tie "month word" to that synthetic look instead of
   to the word.

So simply adding month-word formats would repeat what Products-Synth
already did. The generator has to fix all three. (The README's "6% of
ExpDate's dates" is corrected in S9.)

### Cause 3: small print is mostly seen, but doubted

| Numeric store expiries, by print height | Found | Boxed at any confidence |
|---|---:|---:|
| 2.5% of the long side and up | 87% | 92% |
| 1.5 to 2.5% | 46% | 91% |
| under 1.5% | 12% | 54% |

At 640 px input the median store date is 11.0 px tall (ExpDate 15.5 px),
and 17.6% are under 8 px. At 1280 px only 0.5% are. Two fixes: small
synthetic dates (data) and a 1280 px input (resolution).

### Cause 4: a threshold chosen where everything is easy

56% of store expiry dates got a box at some confidence; 27% cleared the
thresholds (0.728 for seed 0, about 0.665 for the others). Each threshold
maximised F1 on ExpDate validation, where nearly every date is found with
high confidence. On harder photos confidence drops, and a strict
threshold cuts real dates. The ExpDate
test split showed it too: seed 0's stricter threshold found 14 fewer
expiry dates than seeds 1 and 2, with the same mAP.

### What already works, and what M2 can't fix

- **Works:** numeric dot-matrix dates (73% found, 93% boxed), precision
  (0.93 to 0.95), and box placement (IoU 0.3 adds almost nothing).
- **Not M2's job:** `other-face` photos (a capture problem) and reading
  (M3).
- **The store set is small:** 15 sideways and 15 small-print expiries. A
  difference of 1 to 3 dates between two arms is noise; the report gives
  `k/n` and the range over seeds.
- **The store photos are less independent now.** M1's store result chose
  M2's targets, so M2's store numbers are a somewhat optimistic estimate
  for a new store. Two guards: nothing in this plan changes after a store
  score, and the generator is built from the failure *causes*, never by
  copying any store photo. A fresh set of store photos would be the clean
  test (Later).

## How M2 closes the gap

| Cause | What M2 changes | Measured on |
|---|---|---|
| sideways | generated sideways dates; rival: the training photos turned 90° or 270° | `real-val-rot`, synth-val `sideways`, store `sideways` |
| month as a word | full names, ordinals and abbreviations, drawn small and realistic, inside blocks of other text | synth-val `month-word`, store month-word dates |
| small print | small generated dates; a 1280 px arm | `real-val-small`, synth-val `small`, store `small-print` |
| strict threshold | threshold for recall (F2), chosen on validation photos that include hard cases | store: found vs boxed |
| validation blind to all of the above | a validation suite with real turned dates, real small dates and synthetic hard dates | — |

## What we expect (written before any run)

| Arm | sideways | month as a word | small print | ExpDate validation |
|---|---|---|---|---|
| `base`: M1's data, M2's recipe | still about 0 | low | slightly up (threshold) | as M1 |
| `rot`: + turned copies | up a lot | low | as `base` | as M1 |
| `gen`: + generated photos | up | up a lot | up | as M1, perhaps 1 point lower |
| `gen-rot`: both | up the most | up a lot | up | as M1 |
| `gen-rot-1280`: both, at 1280 px | as `gen-rot` | as `gen-rot` | up the most | as M1 |

If `gen` scores high on synth-val month words but doesn't lift the store's
month-word dates, the generated dates don't look like real ones. That is a
finding about synthetic realism, and it gets written up as one.

---

## What we start from

| Fact | Where it comes from | What it means for M2 |
|---|---|---|
| YOLO lists in `$EXPDATE_DIR/yolo/`: `train_real` 937, `train_synth` 11,858, `val` 165, `test` 665 | M0 A4 | the pools every arm draws from |
| background photos: real train 937 (930 at least 640 px on the long side, 711 at least 1,000 px); synthetic 11,844 once the 14 copies are out (7,255 at least 640 px, 3,143 at least 1,000 px); validation 165 (590 to 2,048 px) | measured for this plan | enough real pack surfaces to draw on |
| `n640` trained 1.28 M images (100 epochs × 12,795) in 66 min, at 22.1 batches of 16 a second (354 images/s) with 8 loader workers; `n1280` at 13.5 batches a second (216 images/s) | M1 GPU budget, job log | the speeds every estimate below starts from |
| `s640` trained as fast as `n640`, and the log warned of slow image access (17.7 MB/s) | M1, learnings §2.9 | loading photos, not the GPU, set the pace |
| `optimizer=auto` gave MuSGD (lr 0.01) to runs over 10,000 steps and AdamW to shorter ones, and set `warmup_bias_lr` to 0 | M1 job log; Ultralytics `trainer.py` | set it by hand, the same for every arm |
| with `val=False`, Ultralytics validates only after the last epoch; `patience=0` turns early stopping off | Ultralytics 8.4.177 source | how the fixed budget is run |
| Ultralytics reads list files without removing repeats | `data/base.py`, `get_img_files` | listing a file twice repeats its photos |
| Ultralytics checks free RAM for `cache="ram"` on the **node**, not the job's Slurm limit | `data/base.py`, `check_cache_ram` | our own check, or the job is killed |
| `opencv-python` 5.0 comes with Ultralytics; `fonttools` with matplotlib | the laptop venv | the generator needs no new heavy dependency |

**The 14 synthetic copies of validation photo `img_00277.jpg`**, read off
the near-duplicates contact sheet (`3-reused-1.jpg`, checked by eye on
2026-10-20) and confirmed today to be synthetic file names: `img_04501`,
`img_02041`, `img_03967`, `img_00948`, `img_04759`, `img_03871`,
`img_02734`, `img_05924`, `img_02301`, `img_04028`, `img_08236`,
`img_08164`, `img_05985`, `img_08250`. Until now no file listed them; S2
puts them in code.

**M1's Later items given to M2:**

| M1 Later item | In M2 |
|---|---|
| leave out the 14 copies | decision 18, S2 |
| pick the threshold for recall | decision 11 (F2) |
| set `optimizer` and `lr0` explicitly | decision 9 |
| train `n640` for longer | decision 7 (twice M1's budget) |
| faster data loading (RAM cache, more workers) | decision 15 (speed test) |
| `n1280` on the store photos | decision 13 (`gen-rot-1280`) |
| month-name dates for the generator | the generator's text spec |
| exact 90° turns as a rival to synthetic sideways dates | the `rot` arm |
| tiled inference | stays in Later: the 1280 arm asks the same question first |

---

## Decisions

Each comes with its reason. **(new)** marks a choice not written down
before. **(changes M1 n)** marks a departure from M1's decision n.

1. **(new) Backgrounds are ExpDate training photos: half real, half
   synthetic.** Real pack surfaces keep the photos realistic; the
   synthetic ones add variety (11,844 photos against 937). Validation and
   test photos are never used, and neither are the 14 copies. Only
   photos at least 640 px on the long side are used (930 real, 7,255
   synthetic), so no background is enlarged at 640 px input. Their own
   labelled boxes stay.
2. **(new) Generate 50,000 training photos, plus 2,100 for synth-val.**
   Generating is cheap (minutes on the CPU nodes); training is the cost.
   Variety in the causes matters more than count: each real background
   is used about 27 times and each synthetic one about 3 times, always
   with different text, places and effects. Whether more distinct photos
   help is measured (decision 14), not assumed.
3. **(new) The mix of difficulties is set by one rule, not fitted to the
   store photos:** each hard factor (sideways, month word, small print,
   dot-matrix) appears in about 30 to 45% of generated blocks, so the
   model sees plenty of each. The ExpDate photos and the backgrounds' own
   dates, all of the easy kind, keep easy dates at about half of
   everything it sees. Format weights are equal within a family, not the
   store's frequencies.
4. **(new) Every box comes from the ink that was drawn.** The generator
   knows each character it rendered, so a box is the tight rectangle
   around that token's ink after all bending and turning. No box is
   estimated.
5. **(new) A validation suite instead of ExpDate validation alone**
   (changes M1 3). ExpDate validation can't see sideways, month-name or
   small dates, so it can't tell the arms apart. The suite is in
   [Training and validation](#training-and-validation).
6. **The ablation** (M1 Later's none / 90° / synthetic / both, plus two):
   `base`, `rot`, `gen`, `gen-rot`, plus `gen10k` (data size) and
   `gen-rot-1280` (input size), **3 seeds each** (0, 1, 2). **(new)** M1's
   seeds differed by up to 7 of 81 store dates, so one seed per arm can't
   separate arms on the store photos.
7. **(new) Every arm trains on the same number of images: 2.56 million**
   (changes M1 4). An arm with more photos would otherwise also get more
   training steps, and we couldn't tell whether the photos or the extra
   steps helped. 2.56 M is twice `n640`'s 1.28 M, since `n640` was still
   improving at its last epoch. Epochs follow from the budget; warmup
   and the mosaic-off phase keep M1's shares of training (3% and 10%).
8. **(new) Real photos stay at about 7% of what every arm sees**, M1's
   share (937 of 12,795). They are listed several times when generated
   photos are added. Arms then differ only in what the other 93% is made
   of.
9. **(new) The optimiser is set by hand: MuSGD, `lr0=0.01`,
   `momentum=0.9`, `warmup_bias_lr=0.0`** (from M1 Later). That's exactly
   what `optimizer=auto` chose for `n640`, so `base` repeats M1's recipe,
   and no arm's optimiser depends on its data size (M1 C2's caveat).
10. **(new) The final weights are the model; no early stopping**
    (changes M1 4). `best.pt` is picked on ExpDate validation, which can't
    see the hard dates, so it would pick for ExpDate's kind of date.
    Training uses `val=False` and `patience=0`: no validation until the
    end, which also saves time. The run's model is `last.pt`.
11. **(new) Threshold: the best F2 on the three real validation sets
    pooled** (changes M1 5; M1 Later). F2 counts a missed date twice as
    much as a false box. The pool (`real-val`, `real-val-rot`,
    `real-val-small`) holds real photos with hard dates, so the threshold
    isn't set where everything is easy. Synthetic photos don't set it:
    their confidences may not carry over to real ones. Each run keeps its
    own threshold, unchanged on test and store photos.
12. **(new) The M2 detector is chosen by a rule written now, on
    validation only**, before any store score (rule in [Decision
    rules](#decision-rules)). The store photos then judge every arm; none
    is picked by its store score.
13. **(new) The 1280 px question uses one arm chosen now: `gen-rot`.** It
    has the most varied data. Choosing it in advance means the 1280 runs
    don't wait for any result, so no GPU waits either. Tiled inference,
    M1's other option, goes to Later.
14. **(new) Data size is measured by distinct photos, at the same mix:**
    `gen10k` lists 10,000 generated photos 5 times, `gen` lists 50,000
    once. Same share of generated photos, same training budget; only the
    number of different generated photos changes.
15. **Loader settings come from a speed test before the arms** (M1
    Later): 3 epochs of `base` with 8 workers, 24 workers, and 24 workers
    plus RAM cache. The fastest wins; within 10%, the simpler. RAM cache
    is used only when our own estimate fits in 60% of the job's memory.
16. **(new) Two GPU jobs share one work queue, and both are queued on
    day one, held** (changes M1's single holder job). See S1 and [The
    chain](#the-chain-two-gpu-jobs-one-work-queue). The cluster's caps
    (2 running and 5 submitted, counting every job of ours: S1) allow
    it, but the two run together only while nothing else of ours runs.
17. **Store photos never go to the cluster** (as M1 7). Store scoring
    runs on the laptop's CPU. ExpDate test is scored once, for the M2
    detector's 3 seeds only (as M1 6).
18. **The 14 synthetic copies of `img_00277.jpg` leave training** (M1
    Later), so validation is fully clean from M2 on.

---

## The generator

### One generated photo

1. **Background.** Pick an ExpDate training photo (decision 1), shrunk to
   at most 1,600 px on the long side. No model input is bigger than
   1,280 px, so more pixels would only cost disk and loading time.
2. **Blocks.** Draw 1 block (60%) or 2 (40%). A block is 1 to 5 lines
   printed together, like the panel on a pack:

   ```
   MFG: 12/2025          ← prod  + date (mfg)
   EXP: 11/2027          ← due   + date (expiry)
   BATCH NO: K2311       ← code
   MRP Rs. 450.00        ← no box: a distractor
   ```

3. For each block: choose its text, style, size, angle and surface;
   render it at 4× the final resolution; bend and turn it; shrink it to
   its final size; find a free place; blend it in.
4. **Camera effects** on the whole photo, so the new text and the
   background share the same blur, noise and compression.
5. **Save** the JPEG, its YOLO label file, and one JSON line describing
   everything drawn.

### Text (`synth/text.py`)

**Date values:** years 2018 to 2035, any valid day. In a block with two
dates the expiry is the mfg date plus 6 to 36 months.

**Formats:**

| Family | Formats |
|---|---|
| numbers, with a day | `DD/MM/YYYY` `DD-MM-YYYY` `DD.MM.YYYY` `DD MM YYYY` `DD/MM/YY` `DD.MM.YY` `DD-MM-YY` `YYYY.MM.DD` `YYYY/MM/DD` `YYYY-MM-DD` `YY.MM.DD` `YYYYMMDD` `DDMMYYYY` `DDMMYY` |
| numbers, month only | `MM/YYYY` `MM-YYYY` `MM.YYYY` `MM/YY` `YYYY.MM` `YYYY/MM` |
| month word, full name | `Month YYYY` `DD Month YYYY` `DDth Month YYYY` `Month DD, YYYY` |
| month word, abbreviation | `MON YYYY` `MON-YYYY` `MON.YYYY` `MON/YYYY` `MON YY` `DD MON YYYY` `DD-MON-YYYY` `DD/MON/YYYY` `DD.MON.YYYY` `DDMONYY` `DD MON YY` `DDth MON YYYY` `MON DD YYYY` `YYYY.MON.DD` `YYYY MON DD` |
| range | `MM/YYYY-MM/YYYY` `MM/YY-MM/YY`: two dates, mfg then expiry |

The list comes from `CLAUDE.md`'s store formats, ExpDate's own shapes and
how packs print dates generally. **Sampling:** numbers 55%, month word
45%; within month words, full name 50% and abbreviation 50%; a day is
printed on 70% of numeric dates and 40% of month-word ones; half of the
month-word dates with a day use an ordinal (`1st`, `2nd`, `3rd`, `4th` …
`11th`, `12th`, `13th` … `21st`, `22nd`, `23rd` … `31st`). Words are UPPER
50%, Title 45%, lower 5%. Within those choices the format is uniform.

**Other tokens:**

| Class | Examples |
|---|---|
| `due` (expiry words) | `EXP`, `EXP.`, `EXP:`, `EXP DATE`, `EXPIRY DATE:`, `Best Before`, `BEST BEFORE:`, `BB`, `B.B.`, `USE BY`, `Use before` |
| `prod` (mfg words) | `MFG`, `MFG:`, `MFD`, `Mfg. Date:`, `MFG DT`, `PKD`, `PKD:`, `Packed on:`, `DOM`, `Date of Mfg.` |
| `code` (lot and batch) | `LOT A23K7`, `B.No. 2307B`, `BATCH: K2311`, `L 21346` |
| no box (distractors) | `MRP Rs. 450.00`, `(Incl. of all taxes)`, `Net Wt. 1.2 kg`, `Net Qty 400 g`, `14:32`, `Best Before 12 months from Mfg.` (only `Best Before` is `due`), `1800 123 4567`, `Lic. No. 10012345678901` |

Distractors teach "digits are not always a date", which keeps precision
high next to prices, weights and licence numbers.

**Block kinds:**

| Kind | Share | Lines |
|---|---:|---|
| single | 35% | `[expiry word] date` (the word on 70% of lines) |
| pair | 35% | mfg line and expiry line, same format (mfg first 80% of the time) |
| panel | 20% | a pair plus 1 to 3 lines of codes and distractors, in the same font |
| range | 5% | `[word] MM/YYYY-MM/YYYY` |
| relative | 5% | an mfg line and `Best Before N months from Mfg.`: only the mfg date is a date |

Panels and pairs give the two-dates-per-photo layout of real packs (64%
of store photos show two dates, 9 to 13% of ExpDate's real photos), and
the **same font** for the date and its neighbours makes the model find a
date by what it says, not by looking different (cause 2).

### Print styles (`synth/render.py`)

| Style | Share | How |
|---|---:|---|
| solid | 40% | a TrueType font, regular or bold |
| dot-matrix / inkjet | 35% | "dotify": render the text, sample it on a grid, draw a dot where the grid point is inked (below) |
| label | 15% | a white or cream sticker behind the block, crisp black text (thermal printer look) |
| embossed | 10% | no ink: the text raises or presses the surface, lit from one side |

**Dotify**, the dot-matrix renderer:

```
mask = the text rendered in a bold font, 4× size
rows = 5, 7 or 9 dots per character height     (5×7 and 7×9 are printer standards)
pitch = character height / rows
for every grid point:                          (horizontal pitch × 0.8 to 1.2)
    if the mask covers ≥ 40% of the point's cell: draw a round dot there
dot radius 0.30 to 0.48 × pitch, position jitter σ = 0.06 × pitch,
2 to 8% of dots dropped, inkjet look: blur σ = 0.15 to 0.35 × pitch
```

It works on any font, so dot-matrix month names come for free. The pixel
font VT323 gives blocky glyphs close to a real coding printer's; other
fonts give inkjet-like ones.

**Rendering at 4×, then shrinking with an area filter**, gives small text
the soft, anti-aliased look a camera gives it, instead of sharp pixels.
Each token gets its own mask, so its box is known exactly.

**Fonts** (`synth/fonts.py`): about 20 families under the SIL Open Font
License or Apache 2.0, fetched from the Google Fonts repository at a
pinned commit and checked by SHA-256 (as `download_expdate.py` does). A
check with `fontTools` confirms every font has every character the text
spec uses. Candidates, pinned in S4:

| Split | Families |
|---|---|
| training (16) | Roboto, Open Sans, Lato, Montserrat, Source Sans 3, Noto Sans, Work Sans, Oswald, Roboto Condensed, Barlow Condensed, Noto Serif, Merriweather, Roboto Mono, Source Code Pro, Share Tech Mono, VT323 |
| synth-val only (4) | IBM Plex Sans, Archivo Narrow, PT Serif, IBM Plex Mono |

The 4 held-out families never appear in training photos, so synth-val
measures whether the model learned dates or memorised fonts.

### Geometry (`synth/warp.py`)

| Factor | Shares |
|---|---|
| orientation | upright (within ±5°) 55%; tilted 5° to 35° either way 15%; sideways 90° or 270° (±8°) 30% |
| surface | flat 75%; cylinder (a can or bottle: the text compresses towards the edges) 15%; bend (a cap or jar: the line arcs) 10% |
| perspective | a mild random homography on 40% of blocks |

Exact quarter turns use `numpy.rot90`, which loses nothing. Other angles
and the warps use OpenCV (`warpAffine`, `warpPerspective`, `remap`) on an
enlarged canvas, so nothing is cut off. After every transform, each
token's box is recomputed from its warped mask: the tightest upright
rectangle around its ink. Tilted dates therefore get the same kind of box
as the store labels: an upright rectangle around the date.

### Size

The **print height** is the height of the date's ink across its line,
divided by the photo's long side: the same measure as the store's
small-print rule (`store_import.print_height`).

| Band | Share | Range (log-uniform) |
|---|---:|---|
| normal | 70% | 1.5% to 4.5% |
| small | 30% | 0.6% to 1.5% |

0.6% is 8 px at 1,280 px input, about the smallest a date can be and still
be found; 4.5% covers ExpDate's large dates. Sizes are drawn before the
background: **a photo with a small block gets a background of at least
1,000 px**, so a small date has real pixels behind it instead of being
enlarged later at 1,280 px input.

### Placement and blending (`synth/compose.py`)

- **A free place:** inside the photo, clear of every existing labelled box
  and every block already placed, with a margin of half a line. Of 20
  random candidates, the one over the plainest surface (lowest mean
  gradient) is taken 70% of the time, a random one 30%. After 30 failed
  tries the block shrinks by 20%; then it's skipped and recorded.
- **Ink against the surface:** dark ink (black, dark blue, brown, dark
  red) on light surfaces, light ink (white, cream, silver, yellow) on
  dark ones; 15% colourful ink; 20% low contrast (luminance only 8 to 15%
  apart).
- **Blending:** dark ink multiplies the surface, so its texture and
  shading show through; light ink is laid over. The ink is also scaled by
  the surface's local brightness, so a shadow falls across text and
  background alike.
- **Ink defects:** opacity 75 to 100%, uneven density, a slight spread,
  and now and then a scratch through a character.

### Camera effects (`synth/camera.py`)

Applied to the whole photo, mild on purpose: heavy damage belongs to M4's
readability checker, which can reuse this module with stronger settings.

| Effect | Share of photos | Range |
|---|---:|---|
| defocus blur | 40% | Gaussian σ 0.3 to 1.2 px |
| motion blur | 10% | 3 to 7 px line kernel |
| sensor noise | 50% | Gaussian σ 2 to 6 (of 255) |
| glare | 15% | 1 or 2 soft bright ellipses, 5 to 25% of the short side, centred near a block half the time |
| uneven light | 30% | a linear or radial brightness ramp of ±25% |
| dim light | 10% | brightness × 0.5 to 0.8, plus noise |
| colour cast | 30% | white balance shifted up to 8% per channel |
| JPEG | all | quality 70 to 95, as the photo is saved |

### Labels and the sidecar

- **YOLO label lines** (`class cx cy w h`, classes `0 date`, `1 due`,
  `2 prod`, `3 code`, as `yolo.CLASSES`) for every token with a class,
  plus the background's own boxes.
- **The box rule is ExpDate's and the store's:** a date box covers the
  date value with its separators and ordinal (`19th December 2026` is one
  box), not the `EXP` word (`due`) and not a time or code after it. A
  range is two dates.
- **The sidecar**, one JSON line per photo: background, and for each block
  its kind, style, font, size, angle, surface, ink and camera effects; for
  each token its text, class, box, and for dates the role (expiry or mfg),
  the format, and the year, month and day. The label lines are written in
  the sidecar's token order, so the two always match. M3 can reuse the
  roles and parts.

### synth-val: the held-out synthetic check

2,100 photos, generated with a different seed, the 4 held-out font
families, and the **165 validation photos as backgrounds** (never trained
on). Each photo has **one** new date on a single line, the **target**,
loaded as an `exp` box, so the evaluator's expiry recall is the target
recall. The validation photos' own dates count as ordinary dates.

| Stratum | Photos | The target date |
|---|---:|---|
| `clean` | 300 | upright, numbers, normal size, solid or label style, flat |
| `sideways` | 300 | as `clean`, turned 90° or 270° |
| `month-word` | 300 | as `clean`, month as a word (full name and abbreviation, half each) |
| `small` | 300 | as `clean`, small band, on validation photos of at least 1,000 px |
| `dot-matrix` | 300 | as `clean`, dot-matrix style |
| `curved` | 300 | as `clean`, cylinder or bend |
| `mixed` | 300 | drawn with the training spec |

Each stratum changes one thing from `clean`, so recall per stratum shows
what the model learned, one difficulty at a time. Camera effects follow
the training spec in every stratum.

### Four rules for realism

1. **Real surfaces underneath:** backgrounds are pack photos, not flat
   colours.
2. **One camera for everything:** blur, noise, light and JPEG are applied
   to the finished photo, never to the text alone.
3. **Big, then small:** text is rendered at 4× and shrunk, as a camera
   sees it.
4. **Dates among look-alikes:** panels put the date in the same font as
   prices, weights and codes, so the model must learn what a date says.

And one rule against leaks: **nothing is designed from a store photo.**
The causes come from M1's aggregate analysis; the formats from
`CLAUDE.md`, ExpDate and general packaging practice.

### The spec (`synth/spec.py`)

All shares and ranges above live in one frozen dataclass, `GenSpec`, with
the synth-val strata as overrides of it. The spec's SHA-256 (of its JSON)
goes into every manifest and every training run's record, so each model
names exactly the data it saw. Photo `i` of a set is drawn from its own
random generator, `numpy.random.default_rng([set_seed, i])`, so the same
photo comes out however the work is split into jobs. Set seeds: `gen` 1,
`synth-val` 2, `gen-sample` 3.

### Code layout

```
src/bestbefore/synth/
  spec.py      GenSpec, the training spec and the synth-val strata
  fonts.py     font list (family, file, SHA-256, style, split), loading
  text.py      dates, formats, tokens, blocks
  render.py    block → 4× RGBA layer + one mask per token, in 4 styles
  warp.py      turn, tilt, perspective, cylinder, bend; boxes from masks
  compose.py   backgrounds, placement, ink colour, blending
  camera.py    whole-photo effects
  make.py      make_one(i, spec, ...) → image, boxes, sidecar line; write_one()
```

The data types, in `text.py`:

```python
@dataclass(frozen=True)
class Token:
    text: str
    cls: str | None                  # "date", "due", "prod", "code" or None (no box)
    role: str | None = None          # dates only: "expiry", "mfg" or "target"
    parts: tuple[int, int, int | None] | None = None   # dates only: year, month, day
    format: str | None = None        # dates only, e.g. "DDth Month YYYY"

@dataclass(frozen=True)
class Block:
    lines: tuple[tuple[Token, ...], ...]
    kind: str                # single, pair, panel, range, relative
```

---

## Training and validation

### The validation suite

| Set | Photos | Dates | Made from | Measures |
|---|---:|---:|---|---|
| `real-val` | 165 | 187 | ExpDate validation, as it is | no regression (the guard) |
| `real-val-rot` | 165 | 187 | each validation photo turned 90° or 270° | sideways, on real print |
| `real-val-small` | 41 | about 186 | the validation photos in 2 × 2 grids | small print, on real print |
| `synth-val` | 2,100 | 2,100 targets, plus the backgrounds' dates | the generator (held-out fonts, validation backgrounds) | each difficulty alone; month words |

A 2 × 2 grid halves every date's size relative to the photo: ExpDate's
median of 16 px at 640 px input becomes about 8 px, the size of store
small print. The pixels are still real print. 165 photos make 41 grids;
the one photo left over is named in the manifest.

**Scoring a run** (`scripts/score_suite.py`, written to `suite.json`):

- predictions on all four sets with `rect=False`, as M1 D3;
- the threshold: best F2 over `real-val`, `real-val-rot` and
  `real-val-small` pooled (decision 11);
- per set: found at IoU 0.5 and 0.3 at that threshold, boxed at any
  confidence, precision, false boxes per photo, and Ultralytics' mAP@0.5
  per class (one dataset file per set);
- synth-val per stratum: target recall at the threshold and at any
  confidence.

### The arms

R = ExpDate real train (937). E = ExpDate synthetic without the 14 copies
(11,844). `-rot` = one copy of each photo turned 90° or 270° (a seeded
coin per photo). G50k = the 50,000 generated photos; G10k = the first
10,000 of them.

| Arm | Real photos (repeated to about 7%) | Other photos | Photos listed | Epochs | Input |
|---|---|---|---:|---:|---:|
| `base` | R × 1 | E | 12,781 | 200 | 640 |
| `rot` | (R + R-rot) × 1 | E + E-rot | 25,562 | 100 | 640 |
| `gen` | R × 5 | E + G50k | 66,529 | 38 | 640 |
| `gen-rot` | (R + R-rot) × 3 | E + E-rot + G50k | 79,310 | 32 | 640 |
| `gen10k` | R × 5 | E + G10k × 5 | 66,529 | 38 | 640 |
| `gen-rot-1280` | as `gen-rot` | | 79,310 | 32 | 1,280 |

Each arm sees 2.53 to 2.56 million images. The code (`m2.py`):

```python
BUDGET = 2_560_000     # images per run: twice M1's n640 (100 epochs x 12,795)
REAL_SHARE = 0.07      # M1's share of real photos (937 of 12,795)

def real_repeats(n_real: int, n_other: int) -> int:
    """How often to list the real photos so they're about REAL_SHARE of the mix."""
    return max(1, round(REAL_SHARE * n_other / ((1 - REAL_SHARE) * n_real)))

def schedule(n_listed: int, share: float = 1.0) -> dict:
    """Epochs for the budget; warmup and mosaic-off keep M1's 3% and 10%."""
    epochs = max(1, round(share * BUDGET / n_listed))
    return dict(epochs=epochs, warmup_epochs=round(0.03 * epochs, 2),
                close_mosaic=max(1, round(0.10 * epochs)))
```

### The recipe

```python
RECIPE = dict(batch=16, fliplr=0.0, deterministic=True,   # as M1
              optimizer="MuSGD", lr0=0.01, momentum=0.9,   # what auto chose for n640
              warmup_bias_lr=0.0,                          # auto sets this too
              patience=0, val=False)                       # no early stopping; validate at the end
```

Everything else is Ultralytics' default, as in M1 (mosaic, HSV, ±10%
shift, ±50% zoom; no rotation, no flips). `rot` turns photos by exact
quarter turns in its data, never by Ultralytics' `degrees`, which loosens
the boxes of long, thin dates. M1's `detector.SETTINGS` stays as it is,
so M1's runs remain reproducible.

### Decision rules

Written before any run, as code in `m2.py`, applied by the chain and
re-checked by each experiment's `summarise.py` (as M1 did).

**1. Loader** (after the speed test):

```python
def loader_choice(speed: dict[str, float]) -> str:
    """Images per second in epoch 2 of each speed run: a = 8 workers,
    b = 24 workers, c = 24 workers + RAM cache. The fastest; within 10%
    of it, the simplest (a, then b, then c)."""
```

Epoch 2 is timed because epoch 1 includes start-up and caching, and the
last epoch includes the final validation (Ultralytics' `results.csv` has
a cumulative `time` column). An arm uses RAM cache only if 30 sampled
photos, scaled to the arm's size, need under 60% of `SLURM_MEM_PER_NODE`.

**2. Pilot gate** (before the generated arms):

```python
def pilot_passes(suite: dict) -> bool:
    """The generated labels are learnable: a short run on `gen` boxes at
    least half of synth-val's targets at any confidence (IoU 0.5)."""
    return suite["synth-val"]["any_conf"]["expiry_recall"] >= 0.5
```

A model nearly always learns its own training distribution, so failing
this means broken labels. The chain then stops the generated arms before
about 30 GPU-hours are spent on them.

**3. The M2 detector** (once every candidate's 3 seeds are scored):

```python
NOISE = 0.01
CANDIDATES = ("base", "rot", "gen", "gen-rot", "gen-rot-1280")   # cheapest first
HARD = ("real-val-rot", "real-val-small", "synth-val")

def decide_detector(suites: dict[str, list[dict]]) -> str:
    """Eligible: mean mAP@0.5 for dates on real-val no more than NOISE
    under base's. Among them, the highest mean mAP@0.5 over the HARD sets;
    any candidate within NOISE of it goes to the cheapest."""
```

mAP is used because it doesn't depend on a threshold. The M2 detector is
the chosen arm's **seed 0**, declared in advance as in M1, with its own
F2 threshold. `gen10k` isn't a candidate: it answers the data-size
question.

**Reading the side questions** (also fixed now):

- **Data size:** if `gen` beats `gen10k` by more than 1 point on the hard
  sets, more distinct photos help, and 200,000 becomes a Later candidate.
  Otherwise 50,000 is already enough.
- **Input size:** `gen-rot-1280` against `gen-rot`, on `real-val-small`
  first.
- **Rotation:** `rot` against `base`, and `gen-rot` against `gen`.

### The chain: two GPU jobs, one work queue

`scripts/m2_chain.py`, run by every GPU job. It walks a fixed task list
and runs the first task that is **ready** (its data and inputs exist),
**not done**, and **not claimed**:

```
stage 1: needs the derived sets (S2)
  m2-smoke              1 epoch on 2% of base                           ~3 min
  m2-speed-a, -b, -c    3 epochs of base each                           ~5 min each
  → loader decision
  m2-base-s0..s2, m2-rot-s0..s2                                         ~2 h each

stage 2: needs the generated sets (S6)
  m2-pilot              gen, seed 0, 10% of the budget, then scored     ~15 min
  → pilot gate
  m2-gen-rot-1280-s0..s2   (longest first)                              ~3.3 h each
  m2-gen-s0, m2-gen-rot-s0, m2-gen10k-s0, then the s1s, then the s2s    ~2 h each

scoring: needs a finished run and all four suite sets
  suite.json for each run                                               ~3 min each
  → the M2 detector, once every candidate is scored
```

- **Claims.** A task is claimed by creating `results/runs/m2-claims/<task>/`
  with `os.mkdir`, which is atomic: of two jobs trying at once, exactly
  one succeeds. The folder holds the Slurm job id. A claim whose job is no
  longer in `squeue` is stale, and is taken over.
- **Done** means the run's `run.json` says `finished` (training), or its
  `suite.json` exists (scoring). A run cut off midway is restarted from
  scratch in a new folder.
- **Order:** scoring tasks first (quick, and they unblock decisions);
  then longest first, so the two GPUs finish together; seed 0 of every
  arm before any seed 1, so early results already cover every arm.
- **Nothing to do:** the job exits, freeing the GPU, except while the
  other job is running the pilot, which unlocks most of stage 2 within
  minutes: then it waits up to 30 minutes, checking every 5.
- **Decisions** are written to `results/runs/m2-decisions.json` as soon as
  their inputs are complete. `--dry-run` prints every task with its state
  (done, claimed by job N, ready, waiting for X).
- **A job that ends with tasks left** (time limit, crash, or nothing ready
  yet) gets a replacement at once: held if its data isn't ready, released
  if it is. At most two GPU jobs are kept. The replacement counts toward
  the 5-job cap (S1), which other jobs of ours share, so if `sbatch`
  refuses, the chain logs it and exits normally.

---

## Slices

Slices are numbered in the order to do them. A slice that ends in an HPC
job comes as early as its inputs allow, so the job waits in the queue
while the next slice is written. Each slice says where it runs and how
it touches the cluster.

### S1. Queue the GPU jobs · cluster · *submits 2 GPU jobs*

**Goal:** two GPU jobs in the queue from day one, held so neither starts
before its work is ready.

**Why held.** In M1 the placeholder job had to exit if it started before
the chain existed, which would have lost its place. A job submitted with
`--hold` can't start; `scontrol release` makes it eligible when its work
is ready. It keeps its place because the queue is ordered by submit time
alone (every job has priority 1, every partition the same tier) and a
release keeps the submit time (both measured in step 3).

1. **Laptop:** write `slurm/m2_gpu.sbatch`, commit, push.

   ```bash
   #!/bin/bash
   # Holds a place in the GPU queue for M2's training (docs/m2-synthetic-generator.md, S1).
   #
   #   cd ~/bestbefore && sbatch --hold slurm/m2_gpu.sbatch   # twice: jobs A and B
   #   scontrol release <job id>                              # once its work is ready
   #
   # The queue is first come, first served by submit time, and a release
   # keeps the submit time, so a held job keeps its place without starting
   # early. Slurm copies this script at submission but reads
   # slurm/m2_chain.sh when the job starts, so the chain can be written
   # after submitting. If it isn't there yet, the job exits at once rather
   # than keep a GPU idle.
   #SBATCH --job-name=m2
   #SBATCH --partition=gpu
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=24
   #SBATCH --mem=96G
   #SBATCH --time=2-00:00:00
   #SBATCH --output=slurm/logs/%x-%j.out

   echo "started $(date '+%F %T') on $(hostname)"
   nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
   df -h /tmp | sed 's/^/local disk: /'
   if [ ! -f slurm/m2_chain.sh ]; then
       echo "slurm/m2_chain.sh is not there yet: exiting without using the GPU"
       exit 1
   fi
   exec bash slurm/m2_chain.sh
   ```

   24 CPUs and 96 GB are half a GPU node's CPUs and a fifth of its
   memory: room for 24 loader workers and a RAM cache of the smaller arms.
   They fit beside any GPU job queued on 2026-10-22 (8 to 23 CPUs, 110 to
   230 GB each). The plan said they couldn't be raised once the job
   waits; step 3 found they can (`scontrol update`), so S3 and S6 can
   adjust them before releasing a job. Only the script is fixed at
   submission, which is why it does nothing but start the chain.
2. **Cluster, login node:**

   ```bash
   cd ~/bestbefore && git pull && mkdir -p slurm/logs
   sbatch --hold slurm/m2_gpu.sbatch       # job A: stage 1, released in S3
   sbatch --hold slurm/m2_gpu.sbatch       # job B: stage 2, released by S6's check job
   squeue -u $USER -o "%.10i %.6j %.9T %.16r"
   ```

3. **Read the limits** and write them here. Measured 2026-10-22 with
   `sacctmgr`, `scontrol show config`, `sprio`, `sinfo`, and three
   throwaway jobs on `short` (held, never run, cancelled at once):

   | What | Measured | What it means for M2 |
   |---|---|---|
   | job caps | `limit_gpu_qos` is our account's only QOS, so it covers **every** job of ours on every partition, CPU jobs and other projects' jobs included: **2 running**, **5 submitted** (waiting + running) | A, B and every CPU job share 2 run slots and 5 queue slots |
   | job arrays | each task counts as one job: with 1 job queued, a 6-task array was refused (`QOSMaxSubmitJobPerUserLimit`), a 4-task one accepted | S6's 51-task array can't be submitted (note in S6) |
   | queue order | every job has priority 1 (`priority/multifactor`, all weights 0), every partition priority tier 1; Slurm 23.11, backfill | first come, first served by submit time |
   | hold, then release | a held job kept its submit time after `scontrol release`, at priority 1 | a released job keeps its place |
   | changing a waiting job | `scontrol update` raised a held job's CPUs, memory and time limit | S1's resources aren't final |
   | CPU partitions | `short`: `cn1`–`cn4`, 6 h; `defaultq`: `cn1`–`cn4`, 3 days; 48 CPUs and about 500 GB per node | S2's derive job (16 CPUs, 1 h) fits `short` |

   **Can 2 GPU jobs wait while 2 CPU jobs run?** Yes: that is 4 of the
   5 submitted. But only 2 jobs of ours run at once, so while 2 CPU jobs
   run neither GPU job can start, and A and B run together only while
   nothing else of ours runs. A job from another project of ours was
   running on 2026-10-22 (ends 2026-10-23), so A, B and S2's derive job
   bring us to 4 of 5.

   **The GPU queue on 2026-10-22:** all four H100s are in jobs of up to
   7 days; the first ends about 2026-10-25 22:45. 10 GPU jobs from 3
   other users, submitted earlier, wait ahead of A and B (each user runs
   at most 2). So job A is unlikely to start before 2026-10-26 and may
   wait a week or more. S2 and S3 have time; releasing A as soon as S3 is
   done costs nothing.

**Check:** both jobs `PENDING`, reason `JobHeldUser`. Their ids go into
this plan.
**Commit:** `feat: m2 GPU job script, submitted held`

**Done 2026-10-22.** Job A is **`72914`**, job B **`72915`**: both
`PENDING`, reason `JobHeldUser`, submitted at 20:41:43 with 24 CPUs,
96 GB and 2 days, logging to `slurm/logs/m2-<id>.out`. The script on the
cluster has the same SHA-256 as the committed one. `sbatch --test-only`
gave an unheld copy a start of 2026-10-27 on `cn5`. That estimate leaves
out the jobs ahead that are waiting only on their owners' caps, so the
real start is likely later.

### S2. Derived sets · laptop, then a CPU job · *submits 1 CPU job*

**Goal:** the turned training copies (for `rot`), the two real hard
validation sets, and the training list without the 14 copies, on both
machines.

**Code:**

- `config.py`: `generated_dir()`, read from `GENERATED_DIR` (laptop
  `data/generated`, cluster `~/bestbefore-data/generated`); added to
  `.env.example`.
- `src/bestbefore/derived.py`:
  - `turn(image, boxes, quarter)`: exact quarter turns with
    `Image.transpose`. A box `(x1, y1, x2, y2)` in 0..1 becomes
    `(y1, 1−x2, y2, 1−x1)` for 90° anticlockwise and
    `(1−y2, x1, 1−y1, x2)` for 270°.
  - `grid(four samples)`: each photo centred in a C × C cell (grey 114
    padding, C = the longest side of the four), on a 2C × 2C canvas;
    boxes scaled and shifted.
  - `write_yolo(...)`: the JPEG (quality 95) into `images/<set>/`, the
    label lines (`yolo.label_lines`) into `labels/<set>/`.
  - `load_yolo_set(list_path, sidecar=None) -> list[Sample]`: samples
    from images and label files, so the evaluator, `contact_sheet.py` and
    `box_stats.py` work on every derived and generated set. With a
    sidecar, `target` dates load as `exp` and the stratum becomes the
    sample's `tag`.
- `src/bestbefore/m2.py` (begun): `REUSED_VAL_COPIES`, the 14 ids, with a
  comment pointing to the near-duplicates experiment; the
  `$GENERATED_DIR/m2/` layout.
- `scripts/derive_sets.py [--only val] [--workers N]`: writes
  `m2/derived/`, the lists (`train_synth_clean`, `real_rot`, `synth_rot`,
  `real_val_rot`, `real_val_small`) and `m2/manifests/derived.json`
  (counts, seed, git commit). The turn for each photo comes from a seeded
  generator keyed by its id. It stops if any of the 14 ids is missing
  from the synthetic list.
- `sets.py`: `real-val-rot`, `real-val-small`.
- `slurm/m2_derive.sbatch`: partition `short`, 16 CPUs, 16 GB, 1 hour;
  loads the Python module and the venv as `m1_chain.sh` does, then runs
  `derive_sets.py --workers $SLURM_CPUS_PER_TASK`.
- `tests/test_derived.py`: a made-up 200 × 100 image with one red
  rectangle; after each turn the mapped box covers the red pixels, which
  catches a formula turning the wrong way; four turns give back the
  original; grid boxes land in the right cell; label lines round-trip.

**Steps:**

1. Laptop: `python scripts/derive_sets.py --only val`, then
   `contact_sheet.py --set real-val-rot` and `--set real-val-small`: are
   the boxes on the dates?
2. Commit and push. Cluster: add `GENERATED_DIR` to `.env`, `git pull`,
   `sbatch slurm/m2_derive.sbatch`.

**Check:** `real_rot` 937, `synth_rot` 11,844, `train_synth_clean` 11,844
(11,858 − 14), `real-val-rot` 165 photos and 187 dates, `real-val-small`
41 photos.
**Commits:** `feat: GENERATED_DIR and the derived sets (turned copies, 2×2 grids)`,
then `feat: slurm job for the derived sets`

### S3. The M2 recipe and the chain · laptop · *releases GPU job A*

**Goal:** everything stage 1 needs, so job A can be released.

**Code:**

- `detection_eval.best_threshold(m, beta=1.0)`: F-beta instead of F1.
  M1's calls don't change (beta 1).
- `detector.py`:
  - `dataset_yaml(...)` also takes absolute list paths, repeats
    (`train_real*5` lists the file five times) and a chosen `val` list.
    M1's calls write the same file as before (a test).
  - `train(..., keep="best")`: `"last"` returns `last.pt`.
  - `score_suite(checkpoint, imgsz, device) -> dict`, as described in
    [The validation suite](#the-validation-suite). A set that doesn't
    exist yet is marked missing, not skipped silently.
- `m2.py`: `ARMS`, `BUDGET`, `REAL_SHARE`, `real_repeats()`,
  `schedule()`, `RECIPE`, `loader_choice()`, the RAM check,
  `pilot_passes()`, `decide_detector()`, `NOISE`.
- `scripts/train_detector.py`: `--milestone m1|m2` (default `m1`, which
  changes nothing). With `m2`: `--arm <name>` gives the lists, repeats,
  input size and schedule; `--budget-share` (0.1 for the pilot);
  `--epochs` and `--fraction` still override, for smoke and speed tests;
  `--workers`, `--cache`. The run record adds the arm's lists and counts,
  the schedule, the loader settings and the manifests' hashes.
  Ultralytics' output goes to `$CHECKPOINT_DIR/m2/<id>/`.
- `scripts/score_suite.py --run <name>`: writes `results/runs/<run>/suite.json`.
- `scripts/m2_chain.py` and `slurm/m2_chain.sh` (as M1's, calling
  `m2_chain.py`).
- `tests/test_m2_rules.py`: repeats and epochs for every arm (the table
  above), `decide_detector` on made-up numbers (a tie, an ineligible
  winner), `loader_choice` within 10%, a stale claim found with a fake
  `squeue`, task readiness.
- `tests/test_detection_eval.py`: on a small example, F2 picks a lower
  threshold than F1.

**Steps:**

1. Laptop smoke: `python scripts/train_detector.py --milestone m2 --arm
   base --epochs 1 --fraction 0.02 --device cpu --name smoke-laptop`,
   then `python scripts/score_suite.py --run smoke-laptop` (synth-val
   shows as missing).
2. `python scripts/m2_chain.py --dry-run` on the laptop.
3. Commit, push. Cluster: `git pull`; once S2's job has finished,
   `python scripts/m2_chain.py --dry-run` must show stage 1 as ready and
   stage 2 as waiting for the generated data. Then `scontrol release 72914`
   and note its expected start (`squeue --start -j 72914`).

**Check:** the laptop smoke writes `run.json`, `data.yaml`, `last.pt` and
`suite.json`.
**Commits:** `feat: F-beta threshold in the detection evaluator`,
`feat: m2 arms, budget and decision rules`,
`feat: training script runs the m2 recipe and keeps the final weights`,
`feat: validation suite scoring with a pooled F2 threshold`,
`feat: m2 chain, two GPU jobs sharing one work queue`

### S4. Generator, part 1: fonts, text and rendering · laptop

**Goal:** any block of text drawn in any of the four styles, with an
exact box per token.

**Code:**

- `pyproject.toml`: a `synth` extra with `fonttools`; OpenCV comes from
  the `train` extra (Ultralytics), so generation runs in the training
  environment. `runs.PACKAGES` adds `opencv-python` and `fonttools`.
- `synth/fonts.py` and `scripts/fetch_fonts.py`: the font table (family,
  file, licence folder, SHA-256, style, split), downloaded from the
  Google Fonts repository at a pinned commit into `$GENERATED_DIR/fonts/`,
  hashes checked, and every character of the text spec checked in each
  font's character map with `fontTools`.
- `synth/text.py`: the formats, tokens and block kinds above.
- `synth/render.py`: layout from font metrics (letter spacing, line
  spacing 1.2 to 1.6), the four styles, dotify, 4× rendering, one mask
  per token.
- `scripts/render_sheet.py`: a sheet of rendered blocks on plain grey,
  every style × format family, with each token's box drawn. Nothing from
  ExpDate is on it.
- `tests/test_synth_text.py`: every format turns a known date into the
  right string, and back with a format-specific pattern; ordinal suffixes
  (`1st`, `2nd`, `3rd`, `4th`, `11th`, `12th`, `13th`, `21st`, `22nd`,
  `23rd`, `31st`); month-only formats drop the day; a pair's expiry
  falls 6 to 36 months after its mfg date; sampled shares within 3
  points of the spec over 10,000 blocks.
- `tests/test_synth_render.py`: every box holds all of its token's ink
  and touches it on all four sides (±1 px); dotify draws dots on a
  character and none on a blank; same seed, same pixels.

**Check by eye:** the render sheet. Month words legible, dot-matrix that
looks printed, boxes hugging the ink.
**Commits:** `feat: font list and fetch script with hashes`,
`feat: date strings and text blocks for the generator`,
`feat: text renderer: solid, dot-matrix, embossed and label styles`

### S5. Generator, part 2: geometry, composing, camera · laptop

**Goal:** complete generated photos with labels, and a 500-photo sample
that passes a look check.

**Code:**

- `synth/warp.py`, `synth/compose.py`, `synth/camera.py`, as described
  above.
- `synth/spec.py`: `GenSpec` with the training shares, and its hash.
- `synth/make.py`: `make_one(i, spec, pool, fonts, set_seed)` returns the
  image, the boxes and the sidecar line; `write_one()` saves all three.
- `scripts/generate.py --set gen-sample --count 500 --workers 4` (the
  scale-out comes in S6).
- `sets.py`: `gen-sample`, read back through `derived.load_yolo_set`.
- `tests/test_synth_warp.py`: a quarter turn maps boxes exactly; after a
  tilt, perspective, cylinder or bend, each box still holds all of its
  token's ink.
- `tests/test_synth_compose.py`: a placed block never overlaps an
  existing box; every box lies inside the photo; YOLO lines round-trip;
  the same `i` gives the same JPEG bytes.

**Look check** (written down in this plan when done):

- `contact_sheet.py --set gen-sample --n 50`: boxes on all drawn text
  that should have one, no box on anything else, nothing cut off;
- `box_stats.py gen-sample`: shares of tall boxes, month words, small
  print and dates per photo, against the spec (not against the store
  photos);
- the date crops, side by side with real ExpDate dates: does anything
  look obviously pasted?

**Commits:** `feat: geometry for text layers (turn, tilt, perspective, curve)`,
`feat: compose generated dates onto ExpDate backgrounds`,
`feat: camera effects for generated photos`

### S6. Generator, part 3: generate at scale · laptop, then a CPU job array · *releases GPU job B*

**Goal:** the 50,000 training photos and the 2,100-photo synth-val on the
cluster, checked, with GPU job B released the moment they're complete.

> **To revise when S6 is written (found in S1).** Each array task counts
> as one job toward the 5-job cap, and jobs A and B already take 2 of the
> 5, so the 51-task array below can't be submitted. Recommended instead:
> one job on `short` using a whole node (48 CPUs, `--workers 48`), still
> split into the same `--task` ranges inside the job, so any split gives
> the same photos. At S6's estimate of 0.2 s per photo per CPU, that's
> about 4 minutes.

**Code:**

- `synth/spec.py`: the synth-val strata and the held-out fonts.
- `scripts/generate.py`:
  - `--task N`: tasks 0 to 49 write `gen` photos `1000·N` to `1000·N + 999`;
    task 50 writes synth-val. Each task uses `--workers` processes, and
    each photo its own random generator, so any split gives the same
    photos.
  - `--check`: counts the photos, opens every image, parses every label
    file, checks label lines against the sidecars, measures the shares
    against the spec, then writes the lists (`gen50k`, `gen10k` = photos
    0 to 9,999, `synth_val`) and `manifests/{gen,synth-val}.json` with
    `complete: true`, the spec hash, the git commit and the font hashes.
    It saves the same numbers as a run record in `results/runs/`, so
    `pull_cluster_results.sh` brings them to the laptop. It exits with an
    error if anything is missing.
- `slurm/m2_generate.sbatch`: an array `0-50`, partition `short`, 4 CPUs,
  8 GB, 2 hours per task.
- `slurm/m2_generate_check.sbatch`: runs `generate.py --check`; if it
  succeeds and `GPU_JOB` is set, `scontrol release "$GPU_JOB"`.
- `docs/datasets.md`: a "Generated data (M2)" section: what is generated,
  the spec, counts, sizes, fonts and their licences.

**Steps:**

1. Laptop: `generate.py --set gen-sample --count 1000 --workers 4`,
   timed, to size the tasks (expected: about 0.2 s per photo per CPU, so
   a task takes about a minute).
2. Commit, push. Cluster, login node: `git pull`,
   `python scripts/fetch_fonts.py`.
3. Submit, from the login node:

   ```bash
   ARRAY=$(sbatch --parsable slurm/m2_generate.sbatch)
   sbatch --dependency=afterok:$ARRAY --export=ALL,GPU_JOB=72915 slurm/m2_generate_check.sbatch
   ```

**Check:** the manifests say `complete: true` with 50,000 and 2,100
photos; the measured shares are within 3 points of the spec; job B shows
as eligible (`squeue -j 72915` reason no longer `JobHeldUser`).
**Commits:** `feat: synth-val strata with held-out fonts`,
`feat: generation job array and completeness check`,
`docs: generated data in the dataset notes`

### S7. Scoring and report code · laptop · *while the GPUs run*

**Goal:** everything S8 and S9 run, written and tested before the results
exist, against M1's runs and validation only.

**Code:**

- `scripts/eval_detector.py`: knows `m1-` and `m2-` runs. An M2 run uses
  `last.pt` and the threshold in its `suite.json`; an M1 run works as
  before. It still refuses to score a run twice on `store` or `real-test`.
- `m2.py`: the tables. The suite table per arm (mean and range over
  seeds), synth-val per stratum, store by tag per arm, and **store by
  cause**: month word (full name or abbreviation) and numbers by print
  band, for upright expiries, plus sideways. The cause split reuses the
  logic of `experiments/m1-store-misses/analyse.py`, which stays as it is
  so M1's numbers don't move.
- `scripts/m2_report.py`: writes `results/m2_detector.md` from the run
  records alone, as `m1_report.py` does.
- Experiment folders, each with a README (question, why, method, decision
  rule, results between the markers, what it means) and a `summarise.py`
  that rebuilds the results and re-applies the rule:
  - `experiments/m2-loader-speed/`
  - `experiments/m2-ablation/` (the four main arms, validation and store)
  - `experiments/m2-data-size/` (`gen` vs `gen10k`)
  - `experiments/m2-input-size/` (`gen-rot` vs `gen-rot-1280`)
- `scripts/pull_cluster_results.sh`: also lists `-m2-` runs and pulls
  `suite.json` and the generation check records.
- Tests: the tables on made-up results; `eval_detector` picks the right
  weights and threshold for each milestone.

**Commits:** `feat: m2 scoring by arm, tag and cause, and the report script`,
then `feat: m2 experiment folders`

### S8. Score the store photos and ExpDate test · laptop · *after every run is scored*

**Goal:** every arm judged on the store photos, and the M2 detector on
ExpDate test, each exactly once.

1. `scripts/pull_cluster_results.sh`. The chain's `m2-decisions.json`
   and every `summarise.py` must agree; a disagreement stops the slice.
2. Copy the 18 runs' `last.pt` (about 5.5 MB each at `n` size):
   `rsync -a bbhpc:bestbefore-data/checkpoints/m2/<run>/weights/last.pt …`.
3. **Device check:** re-score the M2 detector's suite on the laptop's CPU.
   Its numbers must match the GPU's to within a date or two (M1 D2 saw
   one date move, from half precision).
4. **Store photos**, all 18 runs, once each:
   `python scripts/eval_detector.py --set store --runs <all 18>` (about 30
   minutes on the CPU; the 1280 runs are slowest).
5. **ExpDate test**, the M2 detector's 3 seeds only, once.
6. `python experiments/m2-*/summarise.py --write`, and the store-by-cause
   tables.

**Commits:** `feat: M2 scored on the store photos and ExpDate test` (if
code changed), then `docs: m2 experiment results`

### S9. Report · laptop

**Goal:** M2's deliverable, `results/m2_detector.md`, and the README.

- `python scripts/m2_report.py` → `results/m2_detector.md`: the suite,
  the ablation on the store photos by tag and by cause next to M1, data
  size, input size, ExpDate test for the M2 detector, GPU minutes and
  images per second per run, and the generated data's measured shares.
- `README.md`: M2 marked done, its headline, what it shows (including a
  result that doesn't beat M1, if that's what happens), how to reproduce;
  the month-word line corrected (6% of ExpDate's *real* dates; 48% of its
  synthetic ones, all abbreviations); optionally a figure of generated
  photos (ExpDate backgrounds, credited).
- This plan: status, measured times, decisions as they came out, Later.

**Commit:** `docs: M2 results`

---

## Order and sessions

```
S1 ─► GPU jobs A and B wait, held
S2 ─► [CPU: derived sets] ──┐
S3 ─────────────────────────┴─► release A ─► [GPU: stage 1: smoke, speed, base ×3, rot ×3]
S4 ─► S5 ─► S6 ─► [CPU array: generate] ─► [check] ─► release B ─► [GPU: stage 2: pilot, 1280 ×3, gen, gen-rot, gen10k ×3]
                                                   S7 (while the GPUs run)
                                                   S8 (once every run is scored) ─► S9
```

| Session | Slices | Ends with |
|---|---|---|
| 1 | S1, S2 | two held GPU jobs; the derive job submitted |
| 2 | S3 | GPU job A released |
| 3 | S4 | the render sheet looked at |
| 4 | S5 | the 500-photo sample looked at |
| 5 | S6 | generation submitted; job B released by the check job |
| 6 | S7 | scoring and report code ready |
| 7 | S8, S9 | M2's results |

If job A finishes stage 1 before the generated data exists, it exits
with stage 2 untouched; submit a replacement at once, held (the chain's
rule). Job B alone can finish stage 2 within its 2 days, even at M1's
speed (about 28 GPU-hours).

## Compute budget

Estimates scaled from M1's measured speed (354 images/s at 640 px, 216
at 1280 px). The speed test may halve the 640 px times; the measured
column is filled in as runs finish.

| Work | Runs | Each | GPU hours at M1's speed | Measured |
|---|---:|---|---:|---|
| smoke + speed test | 4 | 3 to 5 min | 0.3 | |
| `base`, `rot` (stage 1) | 6 | 2.0 h | 12 | |
| pilot | 1 | 15 min | 0.25 | |
| `gen`, `gen-rot`, `gen10k` | 9 | 2.0 h | 18 | |
| `gen-rot-1280` | 3 | 3.3 h | 10 | |
| suite scoring | 19 | about 3 min | 1 | |
| **total** | | | **about 42** (about 21 h on 2 GPUs) | |

| CPU work | Size | Estimate |
|---|---|---|
| derived sets | 12,987 photos written | about 15 CPU-minutes (minutes on 16 CPUs) |
| generation | 52,100 photos | about 3 CPU-hours (minutes of wall time as an array) |
| check | reads all of them | under 10 minutes |

**Storage** on the cluster: generated photos about 12.5 GB, turned copies
about 1.5 GB, synth-val about 0.5 GB, checkpoints under 1 GB. About
15 GB, against 950 GB of quota (65 GB used).

## Files at the end of M2

```
docs/              m2-synthetic-generator.md; datasets.md gains "Generated data (M2)"
src/bestbefore/    m2.py, derived.py, synth/{spec,fonts,text,render,warp,compose,camera,make}.py;
                   changed: config.py, detector.py, detection_eval.py, sets.py, runs.py
scripts/           derive_sets.py, score_suite.py, m2_chain.py, fetch_fonts.py, render_sheet.py,
                   generate.py, m2_report.py; changed: train_detector.py, eval_detector.py,
                   pull_cluster_results.sh
slurm/             m2_gpu.sbatch, m2_chain.sh, m2_derive.sbatch, m2_generate.sbatch,
                   m2_generate_check.sbatch
experiments/       m2-loader-speed/, m2-ablation/, m2-data-size/, m2-input-size/
tests/             test_derived.py, test_m2_rules.py, test_synth_text.py, test_synth_render.py,
                   test_synth_warp.py, test_synth_compose.py; changed: test_detection_eval.py,
                   test_detector.py
results/           m2_detector.md   (committed; runs/ is not)
```

## Risks

- **The model learns the synthetic look instead of dates.** The realism
  rules make that harder; the decision rule leans on real pixels
  (`real-val-rot`, `real-val-small`); the store photos judge. If
  synth-val is high and the real sets stay low, that's the finding.
- **Wrong generated labels.** Boxes come from the ink, the tests check
  them, the sample is looked at by eye, and the pilot gate stops the
  generated arms before most of their GPU time is spent.
- **New text pasted over unlabelled text in a background.** Placement
  avoids every labelled box and prefers plain surfaces; ExpDate labels
  every date, so no date is hidden under a new block.
- **A released hold may not keep its place** on this cluster. S1's test
  says it does (the queue is ordered by submit time, and a release keeps
  it), but check `squeue --start` after releasing job A. If it doesn't,
  the next milestone goes back to M1's unheld placeholder.
- **The GPU queue is long** (S1: all four H100s in jobs of up to 7 days,
  10 GPU jobs ahead of ours on 2026-10-22). Job A may wait a week or
  more. The two jobs keep their place; nothing else is changed for it.
- **RAM cache bigger than the job's memory.** Ultralytics checks the
  node's free memory, not the job's limit, so the job would be killed
  mid-run. Our own check keeps the cache under 60% of
  `SLURM_MEM_PER_NODE`.
- **A GPU job ends with work left.** The chain is resumable, and a
  replacement is submitted at once.
- **38 epochs may be too few for 50,000 new photos.** The pilot, the
  data-size arm and the training loss curves show it; a longer budget is
  a Later item, not a change after seeing the store photos.
- **The store set is small and now less independent.** Counts with `k/n`,
  ranges over seeds, and the caveat in the report.
- **Font licences.** Only OFL or Apache 2.0 fonts, recorded with their
  hashes; the font files are fetched, never committed.
- **Rendering can differ between machines** (Pillow and FreeType
  builds). All training data is generated on the cluster; the tests
  check boxes and geometry, not exact pixels across machines.
- **Ultralytics is AGPL-3.0**, as in M1. Fine for this open repo.

## Later

- **Tiled inference** for very small dates, if 1280 px isn't enough.
- **A fresh set of store photos** (a new store visit, labelled and frozen
  as v2) for a test that M2's design never saw.
- **200,000 generated photos**, if `gen` clearly beats `gen10k`.
- **Superscript ordinals** (`19ᵗʰ`), **upside-down packs** (180°), and
  **Hindi** label words.
- **A crop mode for M3:** the generator already records each date's text,
  role and day / month / year, so it can write reader training crops.
- **A realism check:** a small classifier trained to tell generated date
  crops from real ones; the less it can, the better the generator.
- **More backgrounds** from public pack photos, licences checked.
- **Oriented boxes** (YOLO-OBB) for tilted dates.
- **A longer budget** if the loss curves are still falling at the end.
- From M1, still open: YOLO26 or a permissively licensed detector
  (RT-DETR); RapidOCR's own text boxes as a no-training detection
  baseline; retraining on train + validation once settings are fixed;
  ONNX export for Scanify.
