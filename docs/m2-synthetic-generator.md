# M2: Synthetic generator

Planned 2026-10-21. Status: **in progress**: S1 done 2026-10-22 (GPU jobs A
`72914` and B `72915` held); S2 done 2026-10-22 (the derived sets, on both
machines); S3 done 2026-10-22 (the M2 recipe and the chain; job A released,
expected to start between 2026-10-25 and about 2026-11-09); S4 done
2026-10-23 (fonts, text and rendering; the owner's two questions
answered the same day: 7 and 9 dot rows, times boxed as `code`). Next:
S5 (geometry, composing, camera).
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
| `optimizer=auto` gave MuSGD (lr 0.01) to runs over 10,000 steps and AdamW to shorter ones, and set `warmup_bias_lr` to 0. The momentum 0.9 it printed lasted only until warmup, which moves momentum to the `momentum` argument (default 0.937) for the rest of training (found in S3) | M1 job log and checkpoint; Ultralytics `trainer.py` | set it by hand, the same for every arm |
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
   `momentum=0.937`, `warmup_bias_lr=0.0`** (from M1 Later). That's
   exactly what `n640` trained with under `optimizer=auto`, so `base`
   repeats M1's recipe, and no arm's optimiser depends on its data size
   (M1 C2's caveat). **(changed 2026-10-22, S3)** The plan said
   `momentum=0.9`, read off M1's log line `MuSGD(lr=0.01, momentum=0.9)`.
   That is only the starting value: Ultralytics' warmup moves every
   parameter group's momentum from 0.8 to the `momentum` argument, which
   auto leaves at its default of 0.937, and never resets it. M1's
   checkpoint records `momentum: 0.937`, and two tiny CPU runs ended
   warmup exactly where that predicts (S3).
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
| `code` (lot, batch and time) | `LOT A23K7`, `B.No. 2307B`, `BATCH: K2311`, `L 21346`, `14:32` |
| no box (distractors) | `MRP Rs. 450.00`, `(Incl. of all taxes)`, `Net Wt. 1.2 kg`, `Net Qty 400 g`, `Best Before 12 months from Mfg.` (only `Best Before` is `due`), `1800 123 4567`, `Lic. No. 10012345678901` |

Distractors teach "digits are not always a date", which keeps precision
high next to prices, weights and licence numbers. **(changed 2026-10-23,
S4)** A time like `14:32` is a `code`, not a distractor: ExpDate's own
photos, real and synthetic, box the time printed beside a date as
`code`, and they are half of what every arm sees (S4's record).

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
rows = 7 or 9 dots per character height        (5×7 and 7×9 printer fonts are 7 and 9 dots tall)
pitch = character height / rows
for every grid point:                          (horizontal pitch × 0.8 to 1.2)
    if the mask covers ≥ 40% of the point's cell: draw a round dot there
dot radius 0.30 to 0.48 × pitch, position jitter σ = 0.06 × pitch,
2 to 8% of dots dropped, inkjet look: blur σ = 0.15 to 0.35 × pitch
```

It works on any font, so dot-matrix month names come for free. The pixel
font VT323 gives blocky glyphs close to a real coding printer's; other
fonts give inkjet-like ones. **(changed 2026-10-23, S4)** The plan said
5, 7 or 9 rows. At 5 rows a font's strokes are about one cell wide and
its dates garble (`experiments/m2-dot-rows/`), and the standards named
are 7 and 9 dots tall, so 5 rows was dropped.

**Rendering at 4×, then shrinking with an area filter**, gives small text
the soft, anti-aliased look a camera gives it, instead of sharp pixels.
Each token gets its own mask, so its box is known exactly.

**Fonts** (`synth/fonts.py`): about 20 families under the SIL Open Font
License or Apache 2.0, fetched from the Google Fonts repository at a
pinned commit and checked by SHA-256 (as `download_expdate.py` does). A
check with `fontTools` confirms every font has every character the text
spec uses. Pinned in S4 (google/fonts `e7db6ca`, all 20 in its `ofl/`
folder):

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
    space_before: bool = True        # False: printed against the token before (a range; S4)

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
RECIPE = dict(batch=16, fliplr=0.0, deterministic=True,     # as M1
              optimizer="MuSGD", lr0=0.01, momentum=0.937,   # what n640 trained with (decision 9)
              warmup_bias_lr=0.0,                            # auto sets this too
              patience=0, val=False)                         # no early stopping; validate at the end
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

Epoch 2 is timed because epoch 1 includes the loader workers' start-up
and the first, uncached read of every photo, and the last epoch includes
the final validation (Ultralytics' `results.csv` has a cumulative `time`
column). **(corrected in S3)** The RAM cache is filled before that
clock starts, so it isn't in epoch 1's time either; it is in the wait
for the first row, which the watchdog counts separately. An arm uses RAM
cache only if 30 sampled photos, scaled to the arm's size, need under 60%
of `SLURM_MEM_PER_NODE`.

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
  m2-smoke              1 epoch on 2% of base, then scored at once      ~3 min
                        (synth-val missing), so scoring is tested on the GPU too
  m2-speed-a, -b, -c    3 epochs of base each                           ~5 min each
  → loader decision
  m2-base-s0, m2-rot-s0, then the s1s, then the s2s                     ~2 h each

stage 2: needs the generated sets (S6)
  m2-pilot              gen, seed 0, 10% of the budget, then scored     ~15 min
  → pilot gate
  m2-gen-rot-1280-s0..s2   (longest first)                              ~3.3 h each
  m2-gen-s0, m2-gen-rot-s0, m2-gen10k-s0, then the s1s, then the s2s    ~2 h each

scoring: needs a finished run and all four suite sets (the smoke: the three real ones)
  suite.json for each run but the speed runs                            ~3 min each
  → the M2 detector, once every candidate is scored
```

- **Claims.** A task is claimed by creating `results/runs/m2-claims/<task>/`
  with `os.mkdir`, which is atomic: of two jobs trying at once, exactly
  one succeeds. The folder holds the Slurm job id. A claim whose job is no
  longer in `squeue` is stale, and is taken over. **(added in S3)** So is
  a claim holding this job's own id: it dates from before a Slurm requeue
  (this cluster requeues a job whose node fails, `JobRequeue=1`). Of two
  jobs taking over the same stale claim, one wins: each first creates a
  takeover marker named after that claim, again with `os.mkdir`.
- **Done** means the run's `run.json` says `finished` (training), or its
  `suite.json` exists (scoring). A run cut off midway is restarted from
  scratch in a new folder.
- **Order:** scoring tasks first (quick, and they unblock decisions);
  then longest first, so the two GPUs finish together; seed 0 of every
  arm before any seed 1, so early results already cover every arm.
- **Nothing to do:** the job exits, freeing the GPU, except while the
  other job is running the pilot, which unlocks most of stage 2 within
  minutes: then it waits up to 30 minutes, checking every 5.
- **Watchdog** (added 2026-10-22, with the 3-day limit): the chain stops
  a training run when its `results.csv` gains no epoch for 2 hours (the
  longest epoch, `gen-rot-1280`'s, is about 6 minutes at M1's speed),
  marks it failed in its `run.json`, and moves on. A hung loader or a
  stuck read on the shared filesystem doesn't crash, so without this it
  would hold the GPU until the time limit. **(added in S3)** Training
  starting counts as progress too: a callback records it in `run.json`
  once the label scan and the RAM cache are done. So caching has its own
  2 hours and never counts against the first epoch, and a read stuck
  during caching is still caught. `results.csv` gains a row every epoch
  with `val=False` too (checked in S3).
- **Failures** (added in S3): a task that fails twice (an error, or
  stopped by the watchdog) is left for a person. The dry run lists it,
  and it doesn't keep a job alive or earn a replacement.
- **Time left** (added in S3): a task starts only if twice its estimate
  fits before the job's end (`SLURM_JOB_END_TIME`). Otherwise the job
  ends with tasks left and gets a replacement, so the time limit doesn't
  cut runs short in normal running. A run cut off anyway (a much slower
  run, a `scancel`) is restarted from scratch by a later job, once its
  claim is stale.
- **Decisions** are written to `results/runs/m2-decisions.json` as soon as
  their inputs are complete. `--dry-run` prints every task with its state
  (done, claimed by job N, ready, waiting for X).
- **A job that ends with tasks left** (time limit, crash, or nothing ready
  yet) gets a replacement at once: held if its data isn't ready, released
  if it is. At most two GPU jobs are kept. The replacement counts toward
  the 5-job cap (S1), which other jobs of ours share, so if `sbatch`
  refuses, the chain logs it and exits normally. **(added in S3)** An
  error in the chain itself ends the job without a replacement, which
  would only fail the same way.

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
   #SBATCH --time=3-00:00:00
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

**Changed 2026-10-22: 3 days, not 2.** A GPU is hard to get here, and a
job cut off by its limit means queueing again. 3 days is the `gpu`
partition's maximum, so the partition stays the same; stage 2 alone
(about 29 GPU-hours at M1's speed) then fits even if the generated photos
load 2.5 times slower. The two held jobs were raised on 2026-10-22 with
`scontrol update JobId=72914 TimeLimit=3-00:00:00` (and the same for
72915; S1's check showed a waiting job's limit can be raised). `squeue`
then showed both still `PENDING`, reason `JobHeldUser`, with a time limit
of `3-00:00:00` and their submit time unchanged (2026-10-22 20:41:43),
so they kept their place in the queue. The script now says 3 days for
any replacement. `long` would allow 7 days, but nothing in M2 needs more
than 3.

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
    boxes scaled and shifted. **(new)** The photos go into grids in fours
    of similar size (sorted by long side), so every grid halves its
    dates' size nearly exactly; the largest photo is the one left over.
  - `write_yolo(...)`: the JPEG (quality 95) into `images/<set>/`, the
    label lines (`yolo.label_lines`) into `labels/<set>/`; then
    `check_written` reads every one back (size, classes, boxes).
  - `load_yolo_set(list_path) -> list[Sample]`: samples from images and
    label files, so the evaluator, `contact_sheet.py` and `box_stats.py`
    work on every derived and generated set. **(changed)** The `sidecar`
    argument (`target` dates as `exp`, the stratum as `tag`) moves to S6,
    which writes synth-val. Its sidecar's format is only fixed in S5, so
    writing the reader now would mean guessing that format.
- `src/bestbefore/m2.py` (begun): `REUSED_VAL_COPIES`, the 14 ids, with a
  comment pointing to the near-duplicates experiment; the
  `$GENERATED_DIR/m2/` layout: `derived/images/<set>/` and
  `derived/labels/<set>/`, `lists/<name>.txt`, `manifests/<name>.json`.
- `scripts/derive_sets.py [--only val] [--workers N]`: writes
  `m2/derived/`, the lists (`train_synth_clean`, `real_rot`, `synth_rot`,
  `real_val_rot`, `real_val_small`) and `m2/manifests/derived.json`
  (counts, seed, git commit). It stops if any of the 14 ids is missing
  from the synthetic list. Details fixed while writing it:
  - **Sources** are ExpDate's YOLO lists and label files, exactly what M1
    trained and validated on, not the JSON annotations.
  - **The coin** for each photo's turn is seeded with its set and file
    name, not the file name alone: real and synthetic photos share names
    (`img_00003.jpg` is in both).
  - **Lists** hold absolute paths, one per line, written on each machine.
    `train_synth_clean` lists ExpDate's own `yolo/images/synth/` paths
    without resolving their symlinks, because Ultralytics finds a label
    by turning `/images/` into `/labels/` in the path as listed.
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

**Done 2026-10-22.** All from commit `7b3fc0b`, clean tree, seed 0:

| Set | Photos | Dates | Turned 90° / 270° | Worst box error after reading back |
|---|---:|---:|---:|---:|
| `real_val_rot` | 165 | 187 | 75 / 90 | 0.0 px |
| `real_val_small` | 41 | 186 | | 0.002 px |
| `train_synth_clean` | 11,844 | | | (a list) |
| `real_rot` | 937 | 1,057 | 442 / 495 | 0.0 px |
| `synth_rot` | 11,844 | 16,652 | 5,910 / 5,934 | 0.0 px |

- **The boxes are right.** Breaking the formulas on purpose (turning the
  wrong way, mirroring, swapping a grid's rows and columns, dropping
  the centring) fails `tests/test_derived.py` every time. The laptop's
  contact sheets show every turned and gridded box on its date.
- **The sets measure what they should** (`box_stats.py`): every date box
  in `real-val-rot` is tall (0% in `real-val`), and in `real-val-small`
  the median date's short side at 640 px drops from 16.0 to 8.0 px, with
  51.6% under 8 px (0% at 1280 px). The left-over photo is
  `img_00500.jpg`, the largest.
- **The same on both machines:** the validation sets' label files are
  byte-identical on the laptop and the cluster.
- **Checked on the cluster:** every list matches its image and label
  folders, none of the 24,831 listed photos lacks a label file, and none
  of the 14 copies is left in either synthetic list.
- **Cost:** CPU job `72917` took 2.5 minutes on 16 CPUs (`short`, `cn2`).
  The photos take **2.8 GB**, not the 1.5 GB estimated: re-saving at
  quality 95 makes `synth_rot` (2.4 GB) twice the size of the synthetic
  originals. That's fine against 950 GB of quota.

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
- Details fixed while writing it:
  - **The recipe is checked, not assumed.** At train start (after
    `optimizer=auto` would have been resolved) a callback records what
    Ultralytics actually uses: its arguments, the optimiser, the number
    of training images (repeats included) and whether the RAM cache was
    filled. The run stops there if any of it differs from the recipe,
    the schedule, the input size, the seed or the fraction. The loader is
    only recorded: Ultralytics uses no workers on a CPU, on purpose (the
    laptop smoke run's first try stopped on that). At the end it records
    the momentum after warmup.
    (`detector.train(..., keep, callbacks)`, `detector.applied(trainer)`,
    `m2.recipe_problems()`.)
  - **`--epochs` recomputes warmup and mosaic-off** from the epochs
    given, so a 3-epoch speed run keeps mosaic on in epoch 2 (with
    `base`'s 200-epoch schedule, `close_mosaic=20` would turn it off from
    the start).
  - `yolo.split_repeats("train_real*5")` and `m2.mix()`: an arm's lists
    with the real photos' repeats, from the lists' line counts.
    `m2.list_missing()`: a list is usable once its file exists and the
    manifest that made it names it (S6's must also say complete).
  - `detection_eval.pooled()`: several sets' matches as one, for the
    pooled threshold.
  - `score_suite(checkpoint, imgsz, out_dir, device)`: one dataset file
    per set, kept in `out_dir/suite/`. The threshold is stored
    unrounded, for S7 to use exactly (see S7).
  - `src/bestbefore/slurm.py` (new): `squeue`, `sbatch`, the job's time
    left, and the claims, apart from the chain so a test can fake
    `squeue`. `tests/test_slurm.py` holds those tests.
  - `runs.update_run` writes `run.json` atomically (a temporary file,
    then `os.replace`): the two jobs read each other's records, and half
    a file would stop the other job's chain.
  - `scripts/derive_sets.py --only lists`: writes `train_synth_clean`
    without the 2.4 GB of turned copies, merged into the manifest as
    `--only val` is. The laptop's smoke run needs it.
  - `tests/test_detector.py`: M1 unchanged (`SETTINGS`, M1's
    `data.yaml` text, and `train_detector.py`'s m1 path: same run
    record, same calls); Ultralytics keeps a list's repeats.

**Steps:**

1. Laptop smoke: `python scripts/derive_sets.py --only lists` (the
   same 11,844 names as the cluster's list, none of the 14 copies), then
   `python scripts/train_detector.py --milestone m2 --arm
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

**Done 2026-10-22.** Code from `95a2261` to `814d825`, 193 tests pass.
Job A (`72914`) was released at 22:24 (step 3); job B (`72915`) stays
held for S6. What was measured:

- **Job A after its release** (`squeue --start`, `scontrol show job`,
  22:25): `PENDING`, reason `Priority` (no longer `JobHeldUser`), and its
  submit time is still 20:41:43. So it kept its place in the queue: the
  risk that a released hold loses it didn't happen. Slurm's estimated
  start is **2026-10-25 22:45** (end 2026-10-28 22:45). That is only when
  the first H100 frees up, and Slurm gives four of mehul's jobs the same
  time. It leaves out the six jobs waiting only on their owners' 2-job
  caps, as S1 found. Ten GPU jobs submitted earlier wait ahead of A (shovik
  3, meera 3, mehul 4). Taking them first come, first served, with each
  user's cap, job A gets about the 11th GPU to free up. If every job runs
  to its time limit (7 days for most), that is about **2026-11-09**. It
  will be earlier by however much those jobs finish early. Jobs submitted
  later queue behind A and B. Slurm's backfill may slip a short one in,
  but only where that doesn't delay them.

  Who gets each H100 as it frees up, in that worst case (first come,
  first served, at most 2 running jobs per user, every job running to its
  time limit):

  | GPU frees up | It goes to |
  |---|---|
  | 2026-10-25 22:45 | shovik's `72861` |
  | 2026-10-26 10:27 | shovik's `72864` (meera's older `72863` waits: meera already runs 2) |
  | 2026-10-29 02:10 | meera's `72863` |
  | 2026-10-29 11:29 | mehul's `72868` (shovik already runs 2) |
  | 2026-11-01 22:45 | shovik's `72867` (a 12-hour job) |
  | 2026-11-02 10:27 | mehul's `72869` |
  | 2026-11-02 10:45 | meera's `72886` |
  | 2026-11-05 02:10 | meera's `72889` |
  | 2026-11-05 11:29 | mehul's `72870` |
  | 2026-11-09 10:27 | mehul's `72888` |
  | **2026-11-09 10:45** | **job A** |

  Job B, released by then, would get the next one, about 2026-11-12.

- **The recipe, on a real trainer.** The laptop smoke run's log says
  `optimizer: MuSGD(lr=0.01, momentum=0.937)`, with no `optimizer=auto`
  line, and its record of what Ultralytics applied matches the recipe and
  schedule: `lr0` 0.01, `momentum` 0.937, `warmup_bias_lr` 0, `patience` 0,
  `val` False, `batch` 16, `fliplr` 0. The first try stopped at train
  start, as the check is meant to: Ultralytics sets `workers` to 0 on a
  CPU. The loader is now recorded, not checked (`3efe889`).
- **Momentum (decision 9's correction).** M1's `n640` checkpoint
  records `momentum: 0.937`. Two CPU runs with 4 warmup batches ended
  warmup at 0.9028 (argument 0.937) and 0.875 (argument 0.9), which is
  0.8 + (momentum − 0.8) × 3/4 both times. So the `momentum` argument,
  not the logged 0.9, sets the momentum training runs at.
- **`val=False`** still writes one `results.csv` row per epoch (3 rows
  for 3 epochs), with the cumulative `time` column.
- **Repeats.** A laptop run on `train_real*5` + `train_synth_clean`,
  stopped before epoch 1, logged `Using 16529 train, 165 val images`
  (5 × 937 + 11,844) from 12,781 distinct photos; its label scan counted
  16,529 photos, 0 corrupt. Stage 1 lists nothing twice, so every M2 run
  checks the count at train start.
- **The laptop's list.** `derive_sets.py --only lists` took 0.3 s:
  11,844 names in the same order as the cluster's list (the same SHA-256
  of the names, `71f49c3a…`), each under its own machine's
  `yolo/images/synth/`, and none of the 14 copies.
- **Laptop smoke** (`smoke-laptop`, CPU): 256 photos for 1 epoch in
  1.7 minutes; `run.json` (finished), `data.yaml`, `last.pt`; then
  `score_suite.py` in 1.0 minute, writing `suite.json` with synth-val
  marked missing. A 1-epoch model's scores mean nothing.
- **The suite on a real model** (M1's `n640`, laptop CPU, validation
  sets only). Pooled F2 threshold 0.182, against the F1 threshold of 0.728
  M1 chose:

  | Set | Dates found | Boxed at any confidence | Precision | mAP@0.5 date |
  |---|---:|---:|---:|---:|
  | `real-val` | 187 / 187 (M1 at F1: 184) | 187 | 0.921 | 0.991 |
  | `real-val-rot` | 0 / 187 | 0 | | 0.000 |
  | `real-val-small` | 167 / 186 | 179 | 0.729 | 0.898 |

  Ultralytics' own recall at the threshold agrees within 1.1 points. The
  hard sets see what M1 missed (no sideways date boxed at all), and F2
  lowers the threshold, as decision 11 intends.
- **The chain's dry run**, on the laptop and on the cluster (at
  `814d825`): smoke and speed runs ready; `base` and `rot` waiting only
  for the loader decision (on the laptop, `rot` also waits for its turned
  copies, which stay on the cluster); stage 2 waiting for `gen50k` and
  `gen10k` (S6); every scoring task waiting for its run. `squeue --me`
  answers as the chain reads it. The cluster's Slurm tracks a job's
  processes by cgroup (`ProctrackType=proctrack/cgroup`), so a training
  run in its own process group still ends with its job; `KillWait` is
  30 s, `JobRequeue` 1.
- **Ultralytics' label cache.** The cache sits beside the first label
  file in sorted order, and every dataset here starts in ExpDate's
  `labels/real-train/`, so they all share `labels/real-train.cache`. Each
  run rebuilds it (3 s for 16,529 photos on the laptop). Ultralytics
  checks the cache's hash and rescans on any mismatch or unreadable file,
  so two jobs at once only cost a rescan.
- **A 2% smoke run trains on real photos only.** With `--fraction`,
  Ultralytics keeps the first round(n × fraction) photos of the *sorted*
  list (`data/base.py`, `get_img_files`). ExpDate's `real-train` paths
  sort before `synth` on both machines, so `m2-smoke`'s 256 photos are
  all real. That's fine for a test of the plumbing, and only smoke runs
  use a fraction (the pilot shortens its epochs instead).
- **The mAP comes from batches grouped by shape.** `suite.json`'s mAP,
  the decision rule's measure, comes from Ultralytics' `model.val()`.
  That call sets `rect=True` by default (`engine/model.py`), so it pads
  photos less than our own predictions do, exactly as for M1's mAP. Only
  our matching, the threshold and the recall use `rect=False`. mAP needs
  no threshold, and every arm is measured the same way, so the rule
  stays fair. But a set's mAP and its recall come from slightly different
  predictions.
- **The RAM cache, in detail.** Ultralytics caches every *listed* photo
  separately, so a photo listed five times takes five times the memory;
  `m2.ram_check` scales its 30-photo sample to the whole list for that
  reason. Ultralytics' own check (`check_cache_ram`) also wants twice its
  estimate free on the node. When the node is that busy, it trains
  without the cache and only warns in the log. Each run's record says
  whether the cache was really filled (`applied.ram_cache`).
- **What `val=False` leaves behind.** Ultralytics still writes `best.pt`
  every epoch: with no validation, the epoch's score and the best score
  are both empty, and it treats them as equal. At the end `best.pt` is
  the same as `last.pt`. It also validates twice at the end, after the
  last epoch and again when it checks `best.pt`, which takes seconds on
  a GPU. Harmless: M2's model is `last.pt`.

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
- Details fixed while writing it:
  - **Text before fonts in the commits** (the plan listed fonts first):
    the font check needs the characters the text can print
    (`text.CHARSET`), so the text commit comes first and every commit
    runs on its own.
  - **`synth/spec.py` begins in S4**, with `TextSpec` and `StyleSpec`:
    the shares S4 uses. S5 adds the rest, `GenSpec` and its hash. Every
    number has one home from the start.
  - **`Token.space_before`** (default `True`): a range prints
    `05/2025-04/2027` with no spaces, as two date tokens and a dash.
  - **Ink is alpha ≥ 0.25** (`render.INK`, the one definition): at least a
    quarter of a pixel covered. A faint fringe doesn't stretch a box, and
    the thin strokes of small print still count. It is measured on the
    drawn alpha, before S5 picks the ink's colour and opacity.
  - **Each token is a sprite:** its own ink, styled, on a patch of the
    block. Sprites sit on the 4× grid (positions and sizes multiples of
    4), so shrinking a sprite gives exactly the pixels that shrinking the
    whole block would, and a box is the tight rectangle around its
    sprite's ink at any size.
  - **Size is the digits' height:** `render(..., size)` takes the height
    of the digits in final pixels, measured per face as the median top
    and bottom of the ten digits (Merriweather's `6` and `9` stand out
    from the rest), so a date is the same size in every font.
  - **Basic layout, no shaping:** glyph positions don't depend on whether
    a machine's Pillow has libraqm. No kerning; coding printers don't
    kern either. Pillow has no letter spacing, so a token is stamped
    glyph by glyph from bitmaps cached per block; a test compares it with
    Pillow drawing the whole string.
  - **Left open by the plan, fixed by rule** (in `spec.py`; nothing looked
    at the store photos):
    - the case share applies to month words and ordinal suffixes
      (`4TH DECEMBER`, `4th December`, `4th december`); the `due`,
      `prod` and code words print as listed;
    - numbers are zero-padded (`DD` is `04`), ordinals never (`4th`);
    - both lines of a pair carry their word (`MFG`, `EXP`): without them
      the two dates can't be told apart;
    - a panel's 1 to 3 extra lines (uniform) are each a code or a
      distractor, half each, put anywhere among the lines; the pair keeps
      its order;
    - a range has an expiry word on 70% of lines, as a single date; its
      two dates record their own format (`MM/YYYY`), the block's kind
      says it was a range;
    - a relative block's shelf life is 6, 9, 12, 18, 24 or 36 months, its
      mfg line first 80% of the time, as a pair's;
    - every date is any real day of 2018 to 2035, all equally likely; a
      pair's mfg date is drawn only up to where its expiry still falls
      in 2035;
    - solid and label text is bold half the time; dotify and emboss
      always use the bold instance (dotify as the plan says, emboss so
      the relief has room);
    - letter spacing 0 to 0.12 em after every character; every line
      starts at the same left edge, as a coding printer prints;
    - label: the sticker's margin 0.3 to 0.6 em, its corners rounded 0 to
      0.3 em, white or cream;
    - embossed: the relief is the text blurred by 0.02 to 0.05 em, lit
      from any direction, raised or pressed half each, its steepest
      edge at alpha 0.5 to 0.9.

**Check by eye:** the render sheet. Month words legible, dot-matrix that
looks printed, boxes hugging the ink.
**Commits:** `feat: date strings and text blocks for the generator`,
`feat: font list and fetch script with hashes`,
`feat: text renderer: solid, dot-matrix, embossed and label styles`

**Done 2026-10-23.** Code from `757149d` to `846410a`, the owner's two
decisions included (`fa8fe86`, `846410a`); 329 tests pass (136 new:
`test_synth_text.py` 70, `test_synth_fonts.py` 23, `test_synth_render.py`
43), and each commit's tests pass in a checkout of that commit alone.
**(changed)** The font tests have their own file,
`tests/test_synth_fonts.py`, so each of the three modules has one. The
render sheet, drawn after the decisions, is
`results/runs/20261023-134925-render-sheet/sheet.png` (seed 0; drawn
again, it is byte-identical). What was found:

- **Fonts.** 24 files, 15.6 MB, from google/fonts at `e7db6ca` (main on
  2026-10-23). Each download matched the git blob hash GitHub lists for
  its path; the SHA-256s are in `fonts.py`. On the laptop
  `fetch_fonts.py` fetched two of the files itself and checked all 24:
  hash, the 67 characters the text can print (every face has all of
  them), axes and licence.
  - **All 20 families are SIL OFL**, in the repository's `ofl/` folder:
    Roboto, Roboto Condensed and Roboto Mono, once Apache 2.0, have moved
    there. Roboto Condensed's own name table still says Apache 2.0;
    either is allowed.
  - **Four variable fonts default to lighter than regular:** Montserrat
    Thin (100), Source Sans 3 and Source Code Pro ExtraLight (200),
    Merriweather Light (300). Loaded without their axes set, they would
    draw hairline dates. Every instance names all its axes (`wght` 400 or
    700, `wdth` 100, Merriweather's `opsz` 18, its default); a test
    checks that Montserrat's regular draws over 1.5 times the ink of its
    default.
  - VT323 and Share Tech Mono have no bold. They draw regular wherever
    bold is asked, and the layer's record says so.
  - A loaded face at a new size costs 0.03 to 0.07 ms (FreeType reads the
    file lazily), so faces are cached per size, and each face's axis
    order and digit height once.
  - On the laptop's connection the two largest files (Merriweather
    4.6 MB, Noto Serif 1.9 MB) took over a minute each. `fetch_fonts.py`
    waits up to 60 s between reads, not for a whole file, and moves a
    file into place only once it is complete.
- **Text.** Over 10,000 blocks every share is within 3 points of the
  spec, the plan's and those fixed in S4 (a test; the range shares over
  10,000 ranges).
- **Boxes come from the ink, checked another way.** Each box is made from
  its token's own shrunk sprite; the tests find the block's ink a second
  way, all sprites on one 4× canvas shrunk with a plain numpy mean. In
  every style, at 6 to 40 px, over 160 random blocks and 60 lone tokens,
  every box holds all of its token's ink and touches it on all four sides
  within 1 px, and no ink lies outside every box. No sprite has ink on its
  edge, so nothing is cut off.
- **Size.** `size` is the digits' height: `2031` comes out within −1 to
  +2 px of 8, 20 and 45 px in every training font (a test). Measuring the
  ten digits together instead made Merriweather's dates about 8% too
  small: its `6` rises above the other digits and its `9` drops below the
  baseline.
- **The tests catch what they should.** 20 deliberate bugs, one at a
  time: 9 in the text (`11st`, no day clamp, unpadded days, month-only
  dates keeping a day, a pair running past 2035, month words at 40%, mfg
  always first, a spaced range dash, `Month` printed as `MON`) and 11 in
  the renderer (sprites off the 4× grid, a shrunk sprite 1 px off, boxes
  from any alpha instead of `INK`, an inclusive box end, a dot kernel cut
  at its radius, a relief margin of 1 px and of 2 sigma, dots without
  soft edges, dot-matrix left undotted, variable axes not set, the
  digits' height over all ten). All are caught. Two first got through,
  the cut kernel and the 1 px margin, because OpenCV's blur reflects at
  an image's border. Dots and relief now blur with a zero border (outside
  a sprite is flat surface), and two new tests catch both.
- **By eye, the render sheet** (each style by format family, a normal size
  over small print shown 3×): solid, label and embossed text looks
  printed, month words are legible at both sizes in those styles, and the
  boxes hug the ink, the light and shade of embossed text included.
  Dot-matrix at 7 and 9 rows looks like a coding printer next to ExpDate
  validation dates (`contact_sheet.py --set real-val --n 80 --seed 1`):
  distinct round dots, mostly dark, some faint. Oswald, condensed and
  bold, stays hard to read at 7 rows.
- **Dot-matrix at 5 rows garbles** (`2` reads as `Z`, `1` as `!`, Oswald's
  `EXP` as `#KP`). Measured in `experiments/m2-dot-rows/`: at 5 rows a
  bold stem is about one cell wide (median 0.99, at the edge of what the
  40% rule keeps), 5 rows is too coarse for a TrueType glyph's curves,
  and RapidOCR reads 1 of 160 dates at 5 rows against 13 at 7 and 41 at
  9 (155 when drawn solid). The plan's reason for its rows names 7 and 9
  ("5×7 and 7×9 are printer standards"). **Decided 2026-10-23
  (owner): 5 rows dropped**, so `dot_rows` is `(7, 9)`.
- **Dots lighten as they blur, as the plan's ranges make them.** A dot's
  darkest point is 1 − exp(−r²/2σ²): from 0.31 (radius 0.30, blur 0.35)
  to 0.99 (radius 0.48, blur 0.15) of full ink. Kept as the plan says:
  spread ink is lighter, and faint dot-matrix is among the validation
  dates too.
- **ExpDate's other labels**, seen by eye on 24 crops each of `code`, `due`
  and `prod` boxes from real-train and from the synthetic set (seed 0;
  counts not kept): a `code` box holds a time printed after a date
  (`10:13`, `22:54`) or a short lot code (`L20640`, `A1`), never a word
  like LOT; a `due` or `prod` box holds the key word only (`EXP`, `PROD`,
  `BEFORE` of `BEST BEFORE`, `BY` of `BEST BY`), sometimes with its
  colon. The plan had times as distractors with no box, the opposite of
  ExpDate's photos, which are half of what every arm sees. **Decided
  2026-10-23 (owner): generated times are `code`**, as in ExpDate.
  Phrases like `Best Before` keep one `due` box, as the plan says; M2
  scores dates only.
- **Codes and times print on lines of their own**, never right after a
  date on its line (`EXP 11/2027 14:32`): the plan gives no share for
  that, so none was invented. A date box stopping before a time on the
  same line is therefore taught by ExpDate's photos only (Later).
- **Speed**, on one CPU (OpenCV's own threads off, as in S6's workers):
  1,000 blocks drawn as the spec draws them, print heights from the
  plan's size bands (`render_sheet.py --time 1000 --long-side N`):

  | Photo's long side | Mean per block | Median | 95th percentile | Slowest | Dot-matrix mean |
  |---|---:|---:|---:|---:|---:|
  | 1,600 px | 10.7 ms | 4.7 ms | 44.9 ms | 146 ms | 20.2 ms |
  | 1,000 px | 5.2 ms | 3.2 ms | 17.2 ms | 41 ms | 8.7 ms |

  At 1.4 blocks a photo, that's at most about 15 ms of S6's 0.2 s per
  photo. Large dot-matrix dates are the slow tail: their dots are drawn at
  4× with kernels up to 159 px wide (`filter2D`). Measuring each cell's
  coverage with `cv2.integral` instead of numpy's `cumsum` saved 15%;
  dropping 5 rows, whose dots were the largest, another 12% of
  dot-matrix's time. No boxed token lost all its ink at its final size
  (3,265 of 3,265). Measured with nothing else running: a run beside the
  dot-rows experiment came out about 40% slower.
- **A slip:** looking for real dates to compare with, I opened
  `docs/assets/expdate_dates.jpg` before noticing it is made from the test
  split. Nothing was designed from it; the comparison used validation
  photos only.
- **Not checked:** that the cluster's Pillow and FreeType draw the same
  pixels as the laptop's (Pillow 12.3.0, FreeType 2.14.3; see S6),
  `fetch_fonts.py` on the cluster's login node (S6), and how a block looks
  turned, on a real surface and through a camera (S5).
- **For S5:** `render.render()` gives the block at 4× (a `Layer`, one
  sprite per token in `Block.tokens` order): warp the sprites at 4×, then
  `render.shrink()`, then `render.boxes()`, with `INK` the one definition
  of ink. Training text takes its font from `fonts.random_family(rng,
  "train")`, synth-val's from `"held-out"` (S6): the held-out test checks
  that function, so it must be the one used. `size` is the digits' height in final pixels: the print height
  times the photo's long side. A boxed token whose ink is gone at its
  final size gets no box (`None`); none did here, but S5 must redraw or
  skip such a block rather than write a date without its box. The ink's
  colour and opacity are S5's to choose: the boxes measure the drawn
  alpha. A label's text is black on its sticker (`params["label"]`), and
  embossed ink is white where lit, black in shade (`render.tone`).

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
>
> **Also to revise (found in S3).** A GPU job that runs out of ready
> tasks before S6 submits a held replacement (the chain's rule). So the
> check job should release every held M2 job
> (`squeue --me --name=m2 --states=PENDING`), not only `72915`.
>
> **And (found in S4).** Draw the render sheet on the cluster too
> (`render_sheet.py`, seed 0) and compare it with the laptop's byte for
> byte. If the two differ, nothing breaks (every box comes from the ink
> drawn on the machine that drew it), but the laptop's look checks no
> longer show the exact pixels trained on; say so in this plan.

**Code:**

- `derived.load_yolo_set(list_path, sidecar=None)` (moved from S2): with
  a sidecar, `target` dates load as `exp` and the stratum becomes the
  sample's `tag`, so the evaluator's expiry recall is synth-val's target
  recall per stratum. `sets.py`: `synth-val`.

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
  **(noted in S3)** `suite.json` keeps the threshold unrounded (a full
  float), so test and store photos are cut exactly where validation
  chose. Use it as stored, never rounded. M1's `val.json` kept 4
  decimals, and rounding up can move the cut past the box it was chosen
  at. Its display copies (`conf` in each summary) are rounded; don't
  read the threshold from those.
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
rule). Job B alone can finish stage 2 within its 3 days, even at M1's
speed (about 29 GPU-hours).

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
src/bestbefore/    m2.py, derived.py, slurm.py, synth/{spec,fonts,text,render,warp,compose,camera,make}.py;
                   changed: config.py, detector.py, detection_eval.py, sets.py, runs.py, yolo.py
scripts/           derive_sets.py, score_suite.py, m2_chain.py, fetch_fonts.py, render_sheet.py,
                   generate.py, m2_report.py; changed: train_detector.py, eval_detector.py,
                   pull_cluster_results.sh
slurm/             m2_gpu.sbatch, m2_chain.sh, m2_derive.sbatch, m2_generate.sbatch,
                   m2_generate_check.sbatch
experiments/       m2-loader-speed/, m2-ablation/, m2-data-size/, m2-input-size/, m2-dot-rows/
tests/             test_derived.py, test_m2_rules.py, test_slurm.py, test_synth_text.py, test_synth_fonts.py,
                   test_synth_render.py, test_synth_warp.py, test_synth_compose.py;
                   changed: test_detection_eval.py, test_detector.py
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
- **A time or lot code on the date's own line** (`EXP 11/2027 14:32`,
  `21/05/26 L2307`), so generated photos also teach a date box to stop
  before it (S4 prints codes and times on lines of their own).
- **More date variants, for M3's reader** (found in S4, not in the plan's
  format table): a day without its leading zero outside ordinals
  (`4 Dec 2026`), the four-letter `Sept`, `DDMONYYYY` (`19DEC2026`),
  ranges with month words (`JUN 2025-DEC 2026`), and CLAUDE.md's shelf-life
  wording (`Best Before 1 Year from Mfg. Dt.`; S4 prints `Best Before 12
  months from Mfg.`). To a detector they look almost like formats it
  already has, so M2 goes without them; a reader has to read each one.
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
