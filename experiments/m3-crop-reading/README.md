# M3 / S2: reading a tight date box, untrained

## Question

Fed a tight date box, how well does an untrained reader read it, and does
skipping its own text detector fix what Scanify saw?

CLAUDE.md's lesson says crops hurt off-the-shelf OCR: RapidOCR read 25 of
44 store dates from hand-drawn crops against 31 of 44 from the whole photo.
But what Scanify measured was **RapidOCR's full pipeline** (its own text
detector, angle classifier, then recogniser) on **loose date panels** (the
whole batch, MFG and expiry block). RapidOCR 1.2.3's detector enlarges any
image whose short side is under 736 px until it is 736 px, so a panel is
blown up about five times past the text sizes it was trained on. Nobody has
given a **tight** date box straight to the **recogniser alone**. This
experiment does, before anything is trained, because the answer decides
how every crop in M3 is cut (R1) and gives the untrained readers' honest
numbers (the plan's "untrained" rows, and baseline C's recipe).

## Method

`scripts/crop_experiment.py`, in three steps, run by
`slurm/m3_cpu.sbatch crop-experiment` on a CPU node of `short`:

1. **cut** (main venv): every crop below, at each padding, saved as PNG in
   an owner-only run folder on the cluster;
2. **read** (`~/venvs/ocr-cpu`, which holds one OpenCV, 4.10): every crop
   through every configuration, one crop at a time;
3. **score** (main venv): every read through M3's parser, frozen before the
   job (`reader/parse.py`, `PARSE_SHA256` in `m3.py`), scored, R1 applied.

### The crops: 385, none of them test

| Set | Crops | What they are | Truth |
|---|---:|---|---|
| real-val | 187 | every `date` box of ExpDate's 165 validation photos (all wide) | its transcription |
| store, training (v1-train, v3-train) | 138 | every `expiry` (64) and `mfg` (74) box | the photo's frozen expiry or mfg value |
| store, validation (v1-val, v3-val) | 60 | every `expiry` (29) and `mfg` (31) box | the same |

Store crops by the photo's tag and the box's shape: clean 74 wide;
dot-matrix 40 wide; small-print 34 wide; sideways 41 tall and 4 wide;
other-face 5 wide (mfg dates on photos whose expiry is on another face).
Every tall box is on a sideways photo. Every expiry and mfg box of v1 and
v3 has a value; none is on a derived photo (a derived expiry is never
boxed). **Store set v2 is never used, and ExpDate test never opened.**

Every photo is read as M3 reads it (decision 3): upright, its long side at
most 1,600 px (`store_train.shrink`). The store crops come from the
cluster's export of v1 and v3 (`$STORE_TRAIN_DIR`, upright copies, already
that size); which box holds which value comes from `text.json`, written on
the laptop by `scripts/export_store_text.py` from the frozen labels and
copied beside the export (numbered photos, no names).

A crop is the box plus **padding** on every side, as a share of the box's
short side (for a sideways date, its print height), clipped to the photo:
**0, 0.15 or 0.4** (`m3.PADS`). A box wider than tall is **wide**, else
**tall**.

### The configurations: 18

| Factor | Levels |
|---|---|
| reader | RapidOCR 1.2.3 (PP-OCRv3, baselines A and B's), PP-OCRv6 small (`paddleocr` 3.7) |
| what runs | **full**: its own detector, angle classifier and recogniser on the crop; **alone**: the recogniser only (RapidOCR's `text_recognizer`, `paddleocr`'s `TextRecognition`) |
| padding | 0, 0.15, 0.4 |
| turning | **as cut**; **turned** (decision 14), alone only: a wide crop read at 0° and 180°, a tall one at 90° and 270°; the read that holds a date wins, if both do the more confident, if neither the more confident |

2 readers × 3 paddings × (full + alone as cut + alone turned) = 18.

- **Full** is RapidOCR's default settings (as baselines A and B ran it)
  and PaddleOCR's pipeline with the PP-OCRv6 small detector and
  recogniser and the textline angle classifier, page orientation and
  unwarping off (as S1's check). The lines it returns are joined with a
  space, in its own order. RapidOCR's full pipeline drops a line scoring
  under 0.5 (its default `text_score`); the recogniser alone keeps every
  read, an empty one included.
- Each reader gets what it would make of an image file: RapidOCR an RGB
  array (it opens files with PIL; baseline A gives it the same), PaddleOCR
  a BGR one (it opens files with OpenCV).
- **One crop at a time**, so no read depends on the others (both
  recognisers pad a batch to its widest crop).
- **Threads:** every ONNX Runtime session (all of RapidOCR's) gets
  `intra_op_num_threads` set to the job's CPUs; PaddleOCR runs on Paddle
  with `cpu_threads` set the same. Every model from local files, offline.
- **Confidence:** what each reader returns, the mean of its characters'
  probabilities (decision 15's least-sure character needs our own decoder,
  S5 and S6).

### Scoring

- **The parser** is `reader/parse.py`, written from training data only
  (ExpDate's real-train dates without real-val and Products-Synth's, the
  expiry formats of v1-train and v3-train, gen-v2's format list) and
  frozen before the job: `m3.PARSE_SHA256` =
  `026755390a0392eca42dbf01391b9628f0ac27b434743b655fc3b576244613ea`. On
  top of `dates.py` (baseline A's, unchanged) it reads ordinal days,
  `Month DD, YYYY`, a day glued to its month word, and, in a crop, DDMMYYYY
  and a whole read of one month-and-2-digit-year date (`08/27`, `MAR 27`),
  a range of two, or DDMMYY. The score step refuses to run on any other
  parser.
- **The crop's date:** the parser's dates in the read; if it finds more
  than one (padding can reach a neighbouring date), the longest match,
  then the leftmost. **right** if it is the truth (`scoring.score`: year,
  month, and day when the truth has one), **wrong** if it is another date,
  **none** if the parser finds none. Wrong is kept apart from none.
- **real-val's truth date** is the parser's date in the transcription, by
  the same rule; a transcription that gives no date (a month and day with
  no year, say) is left out of date scores and counted.
- **Exact text** (real-val only: the store's printed text is typed in S4):
  the read equals the transcription, case folded and runs of spaces folded
  to one (decision 10).
- Reported per configuration: right, wrong and none, by set (real-val,
  store training, store validation), by tag and by wide or tall; exact
  text on real-val; seconds a crop. Per-crop reads of store crops stay in
  the owner-only run folder on the cluster and in `data/store/runs/` on the
  laptop, never in git.
- **Against the whole photo** (the lesson's own comparison): on v1's
  expiry boxes, the crop's read against baselines A and B's reads of the
  whole photo (M0's per-photo records), by tag. Not re-run.

### R1, the rule (`m3.crop_recipe`, written before the job)

- **pad:** the most right reads, summed over both untrained readers
  **alone with turning on**, over real-val and every store crop (training
  and validation; none of them test). A tie goes to the smaller pad.
- **turn:** kept if, at that pad, on tall crops (both readers summed) it
  adds right reads and adds no wrong read.

The result goes into `results/runs/m3-decisions.json`, and from there into
every M3 crop (S3 on) and baseline C.

### Skipped, with reasons

- Enlarging crops first: the recogniser resizes every crop to 48 px high
  anyway.
- Sharpening and thresholding: measured by Scanify (thresholding cost 12 of
  44, mild sharpening gained 2).
- Grayscale against colour, resize filters: small effects; with a few
  hundred crops a 1 or 2 date difference is noise, and many knobs find a
  lucky one.

## What we expect (the plan's table, before any run)

| Reader | real-val crops | store-val crops | store sideways | store dot-matrix |
|---|---|---|---|---|
| RapidOCR full pipeline on crops | middling | low | low | low |
| RapidOCR recogniser alone (PP-OCRv3) | high | middling | up with turning | low |
| PP-OCRv6 small alone | high | above v3 | up with turning | low |

## The owner's guess (before the job)

Given on 2026-11-13, after reading the method above and before any code
for the job was written:

- **Better read alone than through the full pipeline:** `clean` only.
  `dot-matrix`, `small-print` and `sideways` stay the same or get worse.
- **The better reader** alone with turning on, on store crops: PP-OCRv6
  small.
- **R1's padding:** 0.4.

## Result

Job `73956` (`slurm/m3_cpu.sbatch crop-experiment`, commit `4d3166e`), on
`cn1` of `short` with 8 CPUs and 16 GB, 2026-11-13, 18:55 to 19:34 (38 min
55 s). `text.json`'s SHA-256 `16866710…` (`4d3166e`, the export's manifest
`842395a5…`). Every number below comes from `summarise.py`, which scores
every read again with the frozen parser and checks the job's summary and
R1's record against it (they agree). Cells are right / wrong / none.
Pasted as the script prints it:

Run 20261113-185541-m3-crop-experiment: 187 real-val crops, 198 store crops, 7176 reads in 2240 s on 8 threads of cn1; parser 026755390a03.

#### 1. Right / wrong / none by configuration

| reader | runs | pad | real-val | store | store-train | store-val | real-val exact | ms a crop |
|---|---|---|---:|---:|---:|---:|---:|---:|
| rapidocr | full | 0 | 96 / 4 / 83 | 84 / 3 / 111 | 51 / 3 / 84 | 33 / 0 / 27 | 89 | 265.7 |
| rapidocr | full | 0.15 | 41 / 17 / 125 | 95 / 11 / 92 | 62 / 8 / 68 | 33 / 3 / 24 | 30 | 377.4 |
| rapidocr | full | 0.4 | 57 / 35 / 91 | 104 / 18 / 76 | 70 / 11 / 57 | 34 / 7 / 19 | 27 | 405.5 |
| rapidocr | alone | 0 | 147 / 7 / 29 | 127 / 0 / 71 | 84 / 0 / 54 | 43 / 0 / 17 | 138 | 40.1 |
| rapidocr | alone | 0.15 | 154 / 6 / 23 | 121 / 2 / 75 | 80 / 1 / 57 | 41 / 1 / 18 | 143 | 34.2 |
| rapidocr | alone | 0.4 | 134 / 12 / 37 | 90 / 4 / 104 | 65 / 4 / 69 | 25 / 0 / 35 | 113 | 27.1 |
| rapidocr | turned | 0 | 147 / 7 / 29 | 165 / 0 / 33 | 110 / 0 / 28 | 55 / 0 / 5 | 137 | 84.8 |
| rapidocr | turned | 0.15 | 154 / 6 / 23 | 158 / 2 / 38 | 106 / 1 / 31 | 52 / 1 / 7 | 143 | 70.0 |
| rapidocr | turned | 0.4 | 134 / 13 / 36 | 114 / 5 / 79 | 78 / 5 / 55 | 36 / 0 / 24 | 113 | 57.4 |
| ppocrv6 | full | 0 | 132 / 3 / 48 | 159 / 2 / 37 | 112 / 2 / 24 | 47 / 0 / 13 | 129 | 1211.0 |
| ppocrv6 | full | 0.15 | 146 / 3 / 34 | 156 / 2 / 40 | 106 / 2 / 30 | 50 / 0 / 10 | 144 | 1390.2 |
| ppocrv6 | full | 0.4 | 138 / 10 / 35 | 155 / 5 / 38 | 105 / 4 / 29 | 50 / 1 / 9 | 124 | 1495.8 |
| ppocrv6 | alone | 0 | 172 / 4 / 7 | 142 / 2 / 54 | 95 / 1 / 42 | 47 / 1 / 12 | 169 | 101.6 |
| ppocrv6 | alone | 0.15 | 174 / 7 / 2 | 138 / 1 / 59 | 92 / 1 / 45 | 46 / 0 / 14 | 172 | 88.5 |
| ppocrv6 | alone | 0.4 | 158 / 12 / 13 | 121 / 4 / 73 | 81 / 3 / 54 | 40 / 1 / 19 | 147 | 87.4 |
| ppocrv6 | turned | 0 | 172 / 5 / 6 | 181 / 4 / 13 | 124 / 3 / 11 | 57 / 1 / 2 | 169 | 151.0 |
| ppocrv6 | turned | 0.15 | 174 / 7 / 2 | 179 / 2 / 17 | 122 / 2 / 14 | 57 / 0 / 3 | 172 | 132.8 |
| ppocrv6 | turned | 0.4 | 159 / 12 / 12 | 157 / 7 / 34 | 105 / 6 / 27 | 52 / 1 / 7 | 147 | 125.8 |

real-val: 4 of 187 transcriptions give no date (left out of right / wrong / none); 1 gives one only by the whole-read rule.
real-val crops about 30 px high or less (RapidOCR's full pipeline skips its detector), by pad: 0: 112, 0.15: 46, 0.4: 2.

#### 2. Store crops by tag

| reader | runs | pad | other-face (5) | sideways (45) | dot-matrix (40) | small-print (34) | clean (74) |
|---|---|---|---:|---:|---:|---:|---:|
| rapidocr | full | 0 | 5 / 0 / 0 | 3 / 0 / 42 | 10 / 1 / 29 | 31 / 0 / 3 | 35 / 2 / 37 |
| rapidocr | full | 0.15 | 5 / 0 / 0 | 3 / 0 / 42 | 12 / 3 / 25 | 29 / 2 / 3 | 46 / 6 / 22 |
| rapidocr | full | 0.4 | 3 / 0 / 2 | 5 / 2 / 38 | 15 / 5 / 20 | 25 / 4 / 5 | 56 / 7 / 11 |
| rapidocr | alone | 0 | 5 / 0 / 0 | 0 / 0 / 45 | 24 / 0 / 16 | 31 / 0 / 3 | 67 / 0 / 7 |
| rapidocr | alone | 0.15 | 5 / 0 / 0 | 0 / 0 / 45 | 19 / 2 / 19 | 31 / 0 / 3 | 66 / 0 / 8 |
| rapidocr | alone | 0.4 | 5 / 0 / 0 | 0 / 0 / 45 | 14 / 1 / 25 | 23 / 1 / 10 | 48 / 2 / 24 |
| rapidocr | turned | 0 | 5 / 0 / 0 | 38 / 0 / 7 | 24 / 0 / 16 | 31 / 0 / 3 | 67 / 0 / 7 |
| rapidocr | turned | 0.15 | 5 / 0 / 0 | 37 / 0 / 8 | 19 / 2 / 19 | 31 / 0 / 3 | 66 / 0 / 8 |
| rapidocr | turned | 0.4 | 5 / 0 / 0 | 24 / 1 / 20 | 14 / 1 / 25 | 23 / 1 / 10 | 48 / 2 / 24 |
| ppocrv6 | full | 0 | 5 / 0 / 0 | 31 / 2 / 12 | 26 / 0 / 14 | 30 / 0 / 4 | 67 / 0 / 7 |
| ppocrv6 | full | 0.15 | 5 / 0 / 0 | 26 / 1 / 18 | 23 / 1 / 16 | 33 / 0 / 1 | 69 / 0 / 5 |
| ppocrv6 | full | 0.4 | 5 / 0 / 0 | 27 / 2 / 16 | 23 / 1 / 16 | 33 / 0 / 1 | 67 / 2 / 5 |
| ppocrv6 | alone | 0 | 5 / 0 / 0 | 2 / 0 / 43 | 31 / 2 / 7 | 33 / 0 / 1 | 71 / 0 / 3 |
| ppocrv6 | alone | 0.15 | 5 / 0 / 0 | 1 / 0 / 44 | 29 / 1 / 10 | 33 / 0 / 1 | 70 / 0 / 4 |
| ppocrv6 | alone | 0.4 | 5 / 0 / 0 | 0 / 0 / 45 | 22 / 4 / 14 | 33 / 0 / 1 | 61 / 0 / 13 |
| ppocrv6 | turned | 0 | 5 / 0 / 0 | 41 / 2 / 2 | 31 / 2 / 7 | 33 / 0 / 1 | 71 / 0 / 3 |
| ppocrv6 | turned | 0.15 | 5 / 0 / 0 | 42 / 1 / 2 | 29 / 1 / 10 | 33 / 0 / 1 | 70 / 0 / 4 |
| ppocrv6 | turned | 0.4 | 5 / 0 / 0 | 36 / 3 / 6 | 22 / 4 / 14 | 33 / 0 / 1 | 61 / 0 / 13 |

#### 3. Store crops by shape and label

| reader | runs | pad | wide (157) | tall (41) | expiry (93) | mfg (105) |
|---|---|---|---:|---:|---:|---:|
| rapidocr | full | 0 | 83 / 3 / 71 | 1 / 0 / 40 | 37 / 2 / 54 | 47 / 1 / 57 |
| rapidocr | full | 0.15 | 93 / 11 / 53 | 2 / 0 / 39 | 40 / 7 / 46 | 55 / 4 / 46 |
| rapidocr | full | 0.4 | 101 / 16 / 40 | 3 / 2 / 36 | 49 / 11 / 33 | 55 / 7 / 43 |
| rapidocr | alone | 0 | 127 / 0 / 30 | 0 / 0 / 41 | 61 / 0 / 32 | 66 / 0 / 39 |
| rapidocr | alone | 0.15 | 121 / 2 / 34 | 0 / 0 / 41 | 60 / 0 / 33 | 61 / 2 / 42 |
| rapidocr | alone | 0.4 | 90 / 4 / 63 | 0 / 0 / 41 | 38 / 3 / 52 | 52 / 1 / 52 |
| rapidocr | turned | 0 | 131 / 0 / 26 | 34 / 0 / 7 | 81 / 0 / 12 | 84 / 0 / 21 |
| rapidocr | turned | 0.15 | 125 / 2 / 30 | 33 / 0 / 8 | 78 / 0 / 15 | 80 / 2 / 23 |
| rapidocr | turned | 0.4 | 92 / 4 / 61 | 22 / 1 / 18 | 49 / 4 / 40 | 65 / 1 / 39 |
| ppocrv6 | full | 0 | 131 / 0 / 26 | 28 / 2 / 11 | 77 / 1 / 15 | 82 / 1 / 22 |
| ppocrv6 | full | 0.15 | 132 / 1 / 24 | 24 / 1 / 16 | 77 / 0 / 16 | 79 / 2 / 24 |
| ppocrv6 | full | 0.4 | 129 / 3 / 25 | 26 / 2 / 13 | 78 / 2 / 13 | 77 / 3 / 25 |
| ppocrv6 | alone | 0 | 142 / 2 / 13 | 0 / 0 / 41 | 70 / 0 / 23 | 72 / 2 / 31 |
| ppocrv6 | alone | 0.15 | 138 / 1 / 18 | 0 / 0 / 41 | 66 / 1 / 26 | 72 / 0 / 33 |
| ppocrv6 | alone | 0.4 | 121 / 4 / 32 | 0 / 0 / 41 | 59 / 4 / 30 | 62 / 0 / 43 |
| ppocrv6 | turned | 0 | 144 / 2 / 11 | 37 / 2 / 2 | 90 / 0 / 3 | 91 / 4 / 10 |
| ppocrv6 | turned | 0.15 | 141 / 1 / 15 | 38 / 1 / 2 | 87 / 1 / 5 | 92 / 1 / 12 |
| ppocrv6 | turned | 0.4 | 125 / 4 / 28 | 32 / 3 / 6 | 78 / 4 / 11 | 79 / 3 / 23 |

Turned reads kept, store crops, all pads: ppocrv6 tall at 90°: 38; ppocrv6 tall at 270°: 85; ppocrv6 wide at 0°: 454; ppocrv6 wide at 180°: 17; rapidocr tall at 90°: 39; rapidocr tall at 270°: 84; rapidocr wide at 0°: 450; rapidocr wide at 180°: 21. Reads holding more than one date: 8 of 6930.

#### 4. R1

Right reads, both readers alone with turning, all 385 crops, by pad: 0.0: 665, 0.15: 665, 0.4: 564. **Pad 0.**
Tall crops at that pad, both readers: as cut 0 / 0 / 82, turned 71 / 2 / 9. **Turning not kept.**
Used by M3 (m3.crop_recipe_used): pad 0, turning kept; the owner's departure from R1 on turn.
Wrong reads on store crops, alone with turning at that pad, by label: ppocrv6 mfg 4.

#### 5. v1's expiry crops against the whole photo (M0's baselines), pad 0

| v1 expiries, tag | n | A: whole photo | B: whole photo | rapidocr/full, pad 0 | rapidocr/turned, pad 0 | ppocrv6/full, pad 0 | ppocrv6/turned, pad 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sideways | 15 | 6 / 4 / 5 | 9 / 0 / 6 | 2 / 0 / 13 | 15 / 0 / 0 | 13 / 0 / 2 | 15 / 0 / 0 |
| dot-matrix | 20 | 7 / 0 / 13 | 7 / 0 / 13 | 4 / 0 / 16 | 13 / 0 / 7 | 16 / 0 / 4 | 18 / 0 / 2 |
| small-print | 15 | 14 / 0 / 1 | 14 / 0 / 1 | 14 / 0 / 1 | 14 / 0 / 1 | 13 / 0 / 2 | 15 / 0 / 0 |
| clean | 31 | 22 / 3 / 6 | 27 / 2 / 2 | 14 / 2 / 15 | 28 / 0 / 3 | 27 / 0 / 4 | 30 / 0 / 1 |
| **all** | 81 | 49 / 7 / 25 | 57 / 2 / 22 | 34 / 2 / 45 | 70 / 0 / 11 | 69 / 0 / 12 | 78 / 0 / 3 |

### What it shows

- **The lesson holds for the full pipeline only.** Given the tight box,
  RapidOCR's own pipeline reads 34 of v1's 81 expiries (2 wrong), fewer
  than baseline A from the whole photo (49, 7 wrong). Its **recogniser
  alone**, sideways crops turned, reads **70 with none wrong**, and
  PP-OCRv6 small's **78 with none wrong**, against B's 57 (2 wrong). The
  box was drawn by hand, so this is a reader's ceiling given a perfect
  detector, as Scanify's 25 of 44 was.
- **Why the full pipeline fails, consistent with Scanify's cause:** on
  real-val, RapidOCR's pipeline reads 96 dates at pad 0 but 41 at 0.15. At
  pad 0, 112 of the 187 crops are about 30 px high or less, so RapidOCR
  reads them whole without its detector; at 0.15 only 46 are, and the
  rest go through the detector, which enlarges a crop's short side to
  736 px. PaddleOCR's detector, which enlarges only under 64 px, loses far
  less (PP-OCRv6 full 159 store dates against 181 alone, turned).
- **Turning is what reads sideways dates.** Read as cut, a tall crop gives
  nothing (0 of 41, both readers, every pad). Read at 90° and 270°, 34
  (RapidOCR) and 37 (PP-OCRv6) of 41 at pad 0. The 180° read of wide crops
  adds 4 and 2 right reads, and no wrong one.
- **Padding: none or a little.** Both readers alone with turning read 665
  dates at pad 0 and at 0.15, and 564 at 0.4. At 0.4 a crop reaches
  neighbouring lines: wrong reads rise on real-val (RapidOCR 7, 6, 13;
  PP-OCRv6 5, 7, 12, by pad).
- **PP-OCRv6 small reads more, RapidOCR is more careful.** On the store
  crops at pad 0, turned: PP-OCRv6 181 / 4 / 13, RapidOCR 165 / 0 / 33; on
  real-val 172 / 5 / 6 against 147 / 7 / 29, and exact text on 169 of 187
  against 137. PP-OCRv6's 4 wrong store reads are all mfg dates: two are
  one print's `FEB` read as `FE8` (the parser then takes `8 2025` as
  August), one a digit (`2024` as `2034`), one a day (18 as 19). No expiry
  crop was read wrong by either reader alone at pad 0.
- **Speed on `cn1`** (8 CPUs, one crop at a time): the recognisers alone
  take 27 to 40 ms (RapidOCR) and 87 to 102 ms (PP-OCRv6) a crop, the full
  pipelines 266 to 406 ms and 1.2 to 1.5 s. The laptop's smoke (Apple M2,
  4 threads, 8 other crops, so roughly) read 3 to 4 times faster with the
  recognisers alone and about 20 times faster with PP-OCRv6's full
  pipeline; why wasn't looked into (Paddle's oneDNN meeting crops of every
  size, likely).

### Against the guess

| | guessed | measured (alone with turning against full, pad 0, store crops) |
|---|---|---|
| clean | better | better: RapidOCR 35 → 67 right, PP-OCRv6 67 → 71 |
| dot-matrix | same or worse | better: RapidOCR 10 → 24, PP-OCRv6 26 → 31 (2 wrong, from 0) |
| small-print | same or worse | the same for RapidOCR (31), better for PP-OCRv6 (30 → 33) |
| sideways | same or worse | worse as cut (0 to 2 right), far better turned: RapidOCR 3 → 38, PP-OCRv6 31 → 41 |
| the better reader | PP-OCRv6 small | PP-OCRv6 small: 181 store dates against 165, but 4 wrong against 0 |
| R1's pad | 0.4 | 0 (tied with 0.15 at 665; 0.4 the worst, 564) |

## Decision

- **R1, as written:** pad **0**, turning **not kept** (on tall crops it adds
  71 right reads and 2 wrong ones). In `results/runs/m3-decisions.json`.
- **Used by M3, the owner's departure (2026-11-13): pad 0, turning kept.**
  Both wrong reads that turned R1 against it are one misread in the right
  orientation, and without turning M3 reads no sideways crop; a misread is
  R4's threshold's job. Recorded in `m3.CROP_DEPARTURES`, tied to this run
  (a rerun of R1 doesn't inherit it); `m3.crop_recipe_used()` gives what
  every M3 crop (S3 on) and baseline C use.
- **CLAUDE.md's crop lesson is corrected** with the owner's OK: crops hurt
  RapidOCR's full pipeline, not its recogniser.
- **Baseline C** keeps PP-OCRv6 small's recogniser alone (decision 19),
  at this recipe.
- For later, not changed now (the parser is frozen): a month number glued
  to letters (`FE8 2025`) made a wrong date; a later parser version could
  refuse a numeric part with a letter touching it.

## Reproduce

The reads stay off git: they name numbered store photos. Copy the run's
folder from the cluster into an owner-only folder on the laptop, then run
the script (seconds):

```bash
scripts/pull_cluster_results.sh
```

```bash
(umask 077; mkdir -p data/store/runs/20261113-185541-m3-crop-experiment)
```

```bash
rsync -a --exclude crops/ bbhpc:bestbefore-data/generated/m3/crop-experiment/20261113-185541-m3-crop-experiment/ data/store/runs/20261113-185541-m3-crop-experiment/
```

```bash
python experiments/m3-crop-reading/summarise.py
```
