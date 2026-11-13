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

*After the job.*

## Decision

*After the job.*
