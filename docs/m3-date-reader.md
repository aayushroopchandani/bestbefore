# M3: The date reader

Planned 2026-11-11, after a discussion with the owner the same day (the
three readers, the data, the baselines, the crop experiment). Status:
**in progress**; S1 done on the cluster 2026-11-12 (GPU jobs `73946` and
`73947` held; the main venv extended with M2b unchanged; `ocr-cpu`; the
models fetched and hashed; llama.cpp built; the CPU check passed on
`cn1`, its AMX variant set aside); S2 done 2026-11-13 (the crop
experiment, job `73956`: pad 0, turning kept by the owner's departure from
R1); S3 done 2026-11-13 (the crop shards, job `73958`: 698,988 crops in
8 piles, 4.7 GiB, every record checked). Next: S4. M2b is still in
progress (S10 next there); every slice here that needs one of M2b's
outputs says so. The learnings
guide (`learnings/m3.md`) is written separately, later, as for M2.

**Goal.** Read the expiry date on a photo: find every date (M2b's
detector), read each one (a reader trained on cropped dates), tell the
expiry from the other dates (rules on the detector's keyword boxes), and
do it with **no more wrong reads than baseline B**, the reader this project
grew out of. Three readers, each with its own job:

| Reader | Its job | Why |
|---|---|---|
| **CRNN + CTC**, written here, walked through with the owner | learn how text reading works; the reader built from nothing | a fair "from scratch" arm, and the code the others reuse |
| **PP-OCRv6 small**, fine-tuned | the reader for the phone | the same family as the baselines' own reader, four generations on, pretrained on real text |
| **Qwen3-VL-2B**, untrained and LoRA fine-tuned | the heavy comparison for the portfolio | reads the whole photo, needs no detector, and can work out a shelf-life expiry |

**Done when.** `results/m3_reader.md` reports, from the run records
alone:

- **the crop experiment** (S2): the untrained readers on hand-drawn date
  boxes, the full pipeline against the recogniser alone, and the padding
  and orientation chosen;
- **the readers on the crop suites**: the CRNN's three data arms (3 seeds
  each), PP-OCRv6 off the shelf and fine-tuned, by stratum and tag, with
  size and milliseconds a crop;
- **the VLM**, untrained and fine-tuned, on the validation photos;
- **the M3 reader end to end on store set v2, scored once**: right, wrong
  and none by tag, with derived expiries and `other-face` photos apart,
  by pack design seen or unseen, next to baselines A, B and C (C: the
  detector plus the untrained PP-OCRv6), with a table of where each photo
  was lost; and once on ExpDate test.

---

## What earlier milestones tell us

### The bar: baselines A and B

From M0 (`results/m0_baselines.md`): **right** is the exact date, **wrong**
any other date, **none** no date.

| | n | A right | A wrong | A none | B right | B wrong | B none |
|---|---:|---:|---:|---:|---:|---:|---:|
| ExpDate test (marked expiry) | 655 | 502 | 28 | 125 | 57 | 6 | 592 |
| Store v1, readable expiry | 87 | 49 | 12 | 26 | 60 | 2 | 25 |

M0's per-photo records, split by M2b's v1 split (read 2026-11-11,
`data/store/runs/`, nothing re-run):

| v1 part | scored | A right / wrong / none | B right / wrong / none |
|---|---:|---|---|
| v1-train | 62 | 28 / 10 / 24 | 40 / 0 / 22 |
| v1-val | 25 | 21 / 2 / 2 | 20 / 2 / 3 |

- **B is careful.** Two wrong reads in 87, none on v1-train. A wrong date
  is worse than none, so M3's bar is B's wrong-read rate first and its
  right reads second.
- **v1-val is small and easy.** A reads 21 of its 25 expiries. One photo
  moves it by 4 points, so it can't separate readers by itself; the crop
  suites below carry those choices, and v1-val sets operating points.
- On `other-face` photos (no expiry in frame) A returned a date for 4 of 12
  on v1, B for none: any date there is a wrong read.

### What the detector hands the reader

On v1's 81 expiry dates in frame (M2, `results/m2_detector.md`), M2's
detector (`gen-rot-1280`, mean of 3 seeds) keeps **46** (39–53) at its
threshold and boxes **70.3** at some confidence; by tag, dot-matrix 19 of
20, clean 19 of 31, small print 5.3 of 15, sideways 2.7 of 15. A reader
can only read what it is handed: behind M2's threshold even a perfect
reader gets about 57%, under B's 69%. M2b is lifting this (store photos in
training, gen-v2, a store-chosen threshold). M3 takes M2b's detector as it
comes, and gives itself the best chance of using it:

- **M3 chooses its own detector threshold, end to end** (R6): low enough to
  pass the doubted boxes, with the reader and the parser rejecting what
  isn't a date. M2b's F2 threshold was chosen for finding boxes, not for
  reading them.
- **The reader is scored twice:** alone, on the hand-drawn boxes, and end
  to end, on the detector's boxes, so a miss is put on the right stage.

### Why off-the-shelf OCR failed on crops

CLAUDE.md's lesson: RapidOCR read **25 of 44** dates from hand-drawn
crops against **31 of 44** from the whole photo. Scanify's own test
(`backend/tests/ocr_expiry/test_03_crop_strategies.py`, read 2026-11-11)
shows what was measured:

- the crops were **loose date panels** (the whole batch, MFG and expiry
  block), fed to **RapidOCR's full pipeline**: its own text detector, its
  angle classifier, then its recogniser;
- Scanify's notes give two causes: the **text detector collapsed** on the
  crops (one photo: 56 text boxes on the whole image, 7 on the crop, the
  MFG and EXP lines missed though fully inside), and **over-magnified
  text was misread** (`:23/07/2027` from the photo, `X : 23.0712027` from
  the crop).

RapidOCR 1.2.3's own config (in the laptop's venv, read 2026-11-11) shows
why:

- its models are **PP-OCRv3** detection and recognition, with the
  mobile v2.0 angle classifier;
- the detector resizes with `limit_type: min`, `limit_side_len: 736`:
  **any image whose short side is under 736 px is enlarged until it is
  736 px**. A 400 × 150 px panel is blown up about five times, far beyond
  the text sizes the detector was trained on;
- an image **30 px high or less, or wider than 8:1, skips the detector**
  and is read whole (`min_height: 30`, `width_height_ratio: 8`);
- the recogniser resizes every line to **48 px high**, keeping its aspect,
  normalises to `(x/255 − 0.5)/0.5`, and sorts a batch by aspect first.

So the lesson holds for the full pipeline on loose panels. **Nobody has
tried a tight date box given straight to the recogniser.** S2 does, before
anything is trained, because the answer decides how every crop in M3 is
cut and gives the untrained readers' honest numbers.

### What the store photos hold

Measured 2026-11-11 on the frozen labels with M2b's splits (scratch
scripts; S8 makes the role numbers reproducible). Design facts come from
the training parts only; validation is counted, not looked into.

| | Photos | Date boxes | expiry | mfg | other |
|---|---:|---:|---:|---:|---:|
| v1-train | 70 | 146 | 56 | 65 | 25 |
| v3-train | 10 | 21 | 8 | 10 | 3 |
| **training** | **80** | **167** | 64 | 75 | 28 |
| v1-val | 29 | 53 | 25 | 26 | 2 |
| v3-val | 5 | 9 | 4 | 5 | 0 |
| **validation** (`store-val`) | **34** | **62** | 29 | 31 | 2 |

The 4 dates M2b's keyword review found unboxed (`added_dates`, labelled
`other`) come on top. Keyword boxes: 340 in `v1_keywords.json` (M2b's plan
says 341; not looked into) and 61 in v3's. In the training part: 92 `due`,
137 `prod`, 82 `code`.

**Expiry formats** in training (72 photos with an expiry): `DD/MM/YYYY`
27, `MON YYYY` 13, `Month YYYY` 9, relative 8 (24 months from mfg 4, one
year 2, 18 months 2), `DDMMYYYY` 4, `DD/MM/YY` 3, `YYYYMMDD` 2, and one
each of `MM/YYYY-MM/YYYY`, `DD-MM-YYYY`, `DD Month YYYY`, `YYYY.MON.DD`,
`MON-YYYY`, `MM/YYYY`. Expiry years: 2025 2, 2026 12, 2027 43, 2028 12,
2029 2, 2031 1 (2025-06-18 to 2031-04); mfg dates 2023-06-19 to
2026-08-04. A derived expiry is the mfg date plus the shelf life, the day
kept when printed (`2025-01-10` + 24 months = `2027-01-10`; `2024-12` + 18
months = `2026-06`).

**Which dates a training photo shows:**

| Shows | Photos |
|---|---:|
| the expiry and the mfg date (some with other dates) | 62 |
| the expiry only | 2 |
| an mfg date (some with other dates), no expiry: 8 derived, 4 `other-face` | 12 |
| no date (`other-face`) | 4 |

So **"the latest date is the expiry" is right on the 62 and wrong on the
12**: on 15% of photos it would report a manufacture date as the expiry.
Something must veto it there.

**A keyword-role rule, as a first measure** (each upright expiry or mfg
box takes the class of the nearest `due` or `prod` box to its left on its
line, or else just above it):

| True label | Rule says right | wrong | nothing found |
|---|---:|---:|---:|
| expiry (50 upright boxes) | 39 | 3 | 8 |
| mfg (58 upright boxes) | 40 | 8 | 10 |

31 tall boxes (14 expiry, 17 mfg) were left out: a sideways date's line
runs up or down the photo, and which way is only known once S4 records
each box's turn. A keyword role decides badly (8 mfg dates sit beside a
`due` word), but **vetoes well**: the expiry is always the latest printed
date, so roles are only needed to say "this latest date is a manufacture
date" (decision 11).

### What ExpDate and the generator hold

- **ExpDate's Products sets carry the text of every date**: in real-train
  without real-val, 1,057 dates on 937 photos (819 with one date, 116 with
  two, 2 with three); in Products-Synth, 16,674 dates on 11,858 photos.
  **ASCII only** in both (0 of 17,731 texts has another character). Real
  texts are 8 to 11 characters (median 10); their boxes are 20, 29 and
  43 px high (10th, 50th, 90th percentile), about 5.5 times as wide. 28
  real texts are a month and day with no year (`02.18`); 4,880 synthetic
  ones fall outside `dates.YEARS` (2010–2014) or are month and day.
- **The test split marks the expiry** (`exp`, with its day, month and year
  parts); train and synth don't. Training targets that need a role (the
  VLM's) get it from the keyword rule, and ExpDate test, once, shows how
  far that rule agrees with ExpDate's own marks.
- **real-val**: the 165 validation photos (187 dates), taken out of
  real-train by the YOLO conversion.
- **gen-v2-sample** (500 photos, set seed 6): 1,183 dates (2.4 a photo:
  623 expiry, 498 mfg, 62 import), 580 `due`, 526 `prod`, 265 `code` and
  360 class-less tokens (shelf-life phrases, prices, unit prices, phone
  numbers, barcode digits, separators), each with its text and pixel box;
  39 date formats (the commonest `Month YYYY` 114, `MM/YY` 72, `MM/YYYY`
  66, `DDth Month YYYY` 59); blocks upright 312, sideways 201, tilted 105,
  upside down 69, each with its quarter turn. Its characters: space,
  `& ( ) , - . / :`, digits, letters, `|`. At 200,000 photos that is about
  473,000 date crops and 144,000 class-less ones, every one with exact
  text.

### Checked and refuted (don't re-learn them)

- *"ExpDate's transcriptions hold Korean."* No: every one is ASCII.
- *"The latest date is the expiry."* Not on the store: 12 of 80 training
  photos show dates but no expiry.
- *"Crops hurt any reader."* Only measured for RapidOCR's full pipeline on
  loose panels (above); S2 measured the recogniser alone: on tight boxes of
  v1's 81 expiries it reads 70 (RapidOCR) and 78 (PP-OCRv6), none wrong,
  against A's 49 and B's 57 from the whole photo.

---

## How M3 reads a photo

```
photo, upright, at most 1,600 px on its long side
  → detector (M2b's, at M3's own threshold): date boxes + due/prod/code boxes
  → each date box cut with S2's padding
  → turned upright: a wide box read at 0° and 180°, a tall box at 90° and 270°;
    the read that holds a date wins, then the more confident
  → reader: characters + a confidence
  → parser (reader/parse.py, on dates.py): the dates in it, or none
  → roles: a date beside a due word is an expiry, beside a prod word an mfg date
  → the expiry: the latest date, unless its role says it is a manufacture date
  → no expiry, but an mfg date and a shelf-life phrase after the due word:
    the expiry worked out (by R5)
  → a plausibility check against the photo's month
  → the expiry, or none
```

The VLM takes the same photo and answers `{expiry, mfg}` itself: no
detector, no parser, no rules. That is what it is there to test.

---

## What we expect (written before any run)

| Reader | real-val crops | store-val crops | store sideways | store dot-matrix | derived expiries | `other-face` wrong reads | v2 end to end against B |
|---|---|---|---|---|---|---|---|
| RapidOCR full pipeline on crops | middling | low | low | low | — | — | — |
| RapidOCR recogniser alone (PP-OCRv3) | high | middling | up with turning | low | — | — | — |
| PP-OCRv6 small alone, untrained | high | above v3 | up with turning | low | — | — | C: below B |
| CRNN `exp` | high | low (formats, month words) | middling | middling | none read | few | below B |
| CRNN `exp+gen` | high | clearly up | up | up | strip reads some | few | about B |
| CRNN `all` | high | a little up (within the band, likely) | up | up | | few | about B |
| PP-OCRv6 small, fine-tuned | the highest | the highest | up | up | | few | above B in right reads, if M2b's detector finds most expiries |
| Qwen3-VL-2B, untrained | high on clean print | middling | middling | low | some right | **more** (calls an mfg the expiry) | below B |
| Qwen3-VL-2B, fine-tuned | high | high | up | middling | the most right | few | about B or above, slow |

If fine-tuning lifts store-val but not v2's unseen designs, the readers
learned packs rather than print: a finding, written up as one. If no
reader beats B end to end, the table of where photos were lost says
whether the detector or the reader held it back.

---

## What we start from

| Fact | Where it comes from | What it means for M3 |
|---|---|---|
| Baselines on v1, and by v1's split (tables above) | M0's run records, split 2026-11-11 | B's wrong-read rate is the bar; v1-val alone can't separate readers |
| M2's detector keeps 46 of 81 store expiries, boxes 70.3 | M2 | the reader must cope with the detector's doubted boxes |
| RapidOCR 1.2.3: PP-OCRv3; its detector enlarges any image with a short side under 736 px; 48 px recogniser input | its `config.yaml` | why crops failed; S2 tests the recogniser alone |
| Scanify's crop test: 31 of 44 whole, 25 of 44 on loose panels | Scanify `test_03_crop_strategies.py` | the lesson covers the full pipeline only |
| ExpDate's real and synthetic dates all carry text; ASCII only | checked 2026-11-11 | 17,731 date crops with text, for free |
| Store: 167 training date boxes on 80 photos, 62 validation boxes on 34 | frozen labels, M2b's splits | store crops are few: weighted, design-balanced |
| Store labels hold each expiry's and mfg's **value** (`2027-03`), not its printed text; other dates hold nothing | `store.py` | the owner types the text (S4) |
| 12 of 80 training photos show dates but no expiry | frozen labels | roles must veto the latest date |
| gen-v2's sidecar: every token's text, class, pixel box, role and date parts; every block's quarter turn | `gen_v2_sample.jsonl` | generated crops with exact text, turned upright exactly |
| gen-v2 isn't made on the cluster yet | M2b's status (S9's job next) | S3 waits for it, or runs the other piles first |
| The store export on the cluster: v1 and v3 upright at ≤1,600 px, each photo's box labels in label-file order | M2b S6, `store_train.write_export` | store crops come from it; a small `text.json` joins it |
| PP-OCRv6 small recogniser: about 5.2M parameters, 20.4 MB, input 3 × 48 × 320, CTC head (an NRTR head guides training only), ONNX and safetensors on Hugging Face, pretrained `.pdparams`; training config `configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml` (Adam, cosine, lr 5e-4, batch 128, max 25 characters) | PaddleOCR docs and repo, HF cards | S6 checks the route |
| `paddlepaddle-gpu==3.2.0` from the `cu126` index needs driver ≥ 550.54.14 (ours: 560.35.05); PP-OCRv6 in `transformers` ≥ 5.8 is documented for inference only | PaddleOCR install docs | Paddle training is possible; a PyTorch fine-tune must be proved first |
| Qwen3-VL-2B: `transformers` ≥ 4.57, one visual token per 32 × 32 px, an official GGUF (Q8_0 1.83 GB, mmproj F16 819 MB) for llama.cpp; converting a fine-tuned copy isn't documented | Qwen's model cards and repo | S7 proves the GGUF path on the untrained model first |
| The cluster couldn't be reached from off campus on 2026-11-11 (ssh timed out) | | S1 reads the queue before submitting |
| The laptop (8 GB) heats up on long CPU runs | owner, 2026-11-11 | CPU work runs on the cluster's `short` partition; only v2's one-off scoring runs on the laptop |
| QOS: 2 running, 5 submitted per user, CPU jobs included; M2b holds 2 GPU jobs and will submit S9's CPU job | `CLAUDE.md` | M3 holds 1 or 2 GPU jobs (S1's rule) |

---

## Decisions

Each comes with its reason. **(new)** marks a choice not written down
before; **(changes X)** a departure from an earlier decision.

1. **(changes `CLAUDE.md`'s M3 row: "a recogniser … or a small VLM")
   Three readers, both kinds.** CRNN + CTC (from scratch, its decoder
   walked through with the owner), PP-OCRv6 small (fine-tuned, the phone candidate),
   Qwen3-VL-2B (the heavy comparison). Dropped, agreed with the owner:
   **TrOCR** (the same job as the CTC readers at 61M to 334M parameters,
   slow and hard to run in a browser, and a decoder writing one character
   after another can fill in one it never saw); **PP-OCRv5** (v6 replaces
   it; comparing two versions of one model teaches nothing); **SmolVLM**
   (weak at reading, OCRBench 52.6), **Florence-2** (older),
   **PaddleOCR-VL** (built for documents). SVTR isn't a separate arm:
   PP-OCRv6's recogniser is a PPLCNetV4 backbone, a light SVTR neck and a
   CTC head; CRNN is that family's classic ancestor.
2. **(new) The detector proposes, the reader and parser filter.** The
   pipeline above; M3 chooses its own detector threshold end to end (R6).
3. **(new) Every photo is read at most 1,600 px on its long side**, shrunk
   as the store export shrinks (`store_train.export_image`), on both
   machines. The cluster's store copies are already that size, the laptop
   makes v2's the same way, and a phone's preview frame is about that
   size. Crops are cut from that image, not from the detector's 1,280 px
   input.
4. **(new) Crops are cut once, with a margin, and stored upright.** Each
   crop keeps half its box's short side of margin on every side, is turned
   upright by its known turn (a generated block's quarter, a store box's
   typed turn; ExpDate's are upright), is shrunk so its box is at most
   64 px across its short side (never enlarged), saved as JPEG 95, and
   goes into an LMDB file per pile with its text and box. The padding the
   reader sees is cut from the margin at load time, so S2's choice of
   padding never needs new crops. Why LMDB: the cluster opens a file in
   about 25 ms; 700,000 crop files would cost hours a pass, one LMDB file
   nothing.
5. **(new) The crop experiment comes first** (S2, a CPU job): the untrained
   readers on hand-drawn date boxes, the full pipeline against the
   recogniser alone, three paddings, with and without turning tall crops.
   Its question and its rule (R1) are written before it runs, and the
   owner's guess goes into its README first.
6. **(new) The owner types the store dates' text** (S4): every date box of
   v1 and v3, training and validation, gets its printed text and its turn
   (0°, 90°, 180°, 270°), and on photos that print a shelf life the phrase
   gets a box and its text. Proposals come from S2's reads. Checked
   against the frozen values, frozen with a hash. v2 gets nothing new: its
   frozen expiries score it.
7. **(new) Three piles, three arms, each trained from the same start.**
   Piles: **ExpDate** (real-train without real-val, and Products-Synth),
   **gen** (gen-v2's crops), **store** (v1-train and v3-train). Arms:
   `exp`, `exp+gen`, `all`. Separate runs, not a chain: a model trained on
   one pile and then another forgets part of the first, and the order
   changes the result, so a gain couldn't be put on the data. `all` minus
   `exp+gen` is what store crops add; `exp+gen` minus `exp` what the
   generator adds. Runs: CRNN all three arms, 3 seeds each; PP-OCRv6 off
   the shelf, and `all` with 3 seeds; Qwen untrained, and fine-tuned once
   on its own set from the same three sources (whole photos and panels,
   decision 18), one seed, as each run costs hours.
8. **(new) gen-v2 only, not M2's gen50k.** The same generator with more
   kinds of print (worn, missing dot rows, laser windows, embossed) and
   the store's sizes. For a reader, gen50k adds nothing gen-v2 lacks.
9. **(new) Generated crops include non-dates.** Codes and class-less
   tokens of at least 3 characters (prices, unit prices, phone numbers,
   barcode digits, shelf-life phrases) go in with their text. Run at a low
   threshold, the detector hands the reader barcode digits and prices; a
   reader that has only seen dates would bend them into a date. One that
   reads them as they are leaves the parser to reject them.
10. **(new) One character set.** Digits, the 26 letters, space and
    `. / - : , ( ) & |`; anything else becomes `?`. The CRNN folds case
    (`Jan` and `JAN` read alike, 26 fewer classes); PP-OCRv6 keeps its own
    dictionary and case. Every score folds case and runs of spaces.
11. **(new) Roles veto; the latest date decides.** The expiry is the latest
    printed date, since mfg, packing and import dates come before it. So
    the rule is: take the latest valid date; if a `prod` word sits beside
    it (or `MFG`, `PKD`, `IMPORT` is inside its own read), it is a
    manufacture date and the answer is none; if it is the photo's only date,
    has no role, and falls before the photo's month, none too. Then the
    shelf-life strip (12) and the plausibility check (13). On the training
    photos with true reads, this answers right on the 62 that show both
    dates, unless an expiry wrongly takes a `prod` role (3 of 50 upright
    expiries: none, not wrong). On the 12 without an expiry it answers
    none wherever the latest date carries `prod` or is a lone date before
    the photo's month; 4 of them also show an "other" date, whose value
    S4's text gives, so S8 measures those. A role-deciding rule could turn
    the 8 mfg dates beside a `due` word into wrong reads. A learned role
    model goes to Later.
12. **(new) The shelf-life strip.** When no expiry comes out but an mfg
    date was read and a `due` word has no date after it, the strip after
    that word (on its line, 12 times its height long) is read; a phrase
    like `24 months from Mfg.` or `1 Year from Mfg. Dt.` plus the mfg date
    gives the expiry. Kept only by R5. 8 of v2's 74 expiries (and 8 of the
    72 in training) are derived: without the strip the CTC pipeline
    answers none on all of them, honestly but without credit.
13. **(new) A plausibility check, fixed now, not tuned:** an expiry more
    than 24 months before the photo's month, or more than 120 months after
    it, is dropped as a candidate. The photo's month comes from its EXIF
    (store photos; `text.json` carries it for the cluster), and the live
    app knows today's date. ExpDate photos have no date, so the check is
    off for them. It catches `08/2027` misread as `08/2021`; v1-train's
    earliest expiry (2025-06) sits inside it.
14. **(new) Orientation by reading.** A wide crop is read at 0° and 180°,
    a tall one at 90° and 270°. The read that holds a date wins; if both
    do, the more confident. Upside-down digits rarely make a date
    (`2026` turned is `9202`), so the parser is the judge. S2 checks it on
    the tall crops; a learned orientation classifier goes to Later.
15. **(new) One confidence per read, one operating point per reader.** For
    a CTC reader, a read's confidence is its **least sure character** (the
    lowest of its characters' probabilities), because one wrong digit is
    the danger; the mean is recorded beside it. For the VLM, the lowest
    token probability inside the expiry's value. R4 sets each reader's
    threshold on validation.
16. **(new) Score what ships.** Every CTC reader is scored as an ONNX file
    through ONNX Runtime, with one shared pre- and post-processing (the
    shared decoder), on both machines; the VLM as one GGUF file (Q8_0)
    through llama.cpp. PyTorch and Paddle train; they never score a
    reported number. Parity checks (S5, S6, S7) show the exported file
    reads as its training framework does.
17. **(new) Checkpoints by the public suites, runs by the store.** A run
    keeps its checkpoint with the best exact-text accuracy on real-val and
    synth-val-v2's crops pooled (scored every 5% of its budget). store-val
    is never seen by a run; it decides between runs (R2, R3).
18. **(changes CLAUDE.md's `{expiry, mfg, format}`) The VLM answers
    `{expiry, mfg}`.** `format` is never scored and costs tokens. It sees
    whole photos (ExpDate's and the store's), and for generated photos a
    panel cut round one generated block, because a whole generated photo
    also shows its background's own dates and two products' expiries
    can't both be "the" expiry. LoRA on the language model only, the vision
    tower frozen; our own training loop on `transformers` and `peft`, not
    the Qwen repo's script (pinned to torch 2.6) or TRL (no Qwen3-VL LoRA
    example), so the owner sees what supervised fine-tuning is: the loss
    on the answer's tokens only.
19. **(new) Baselines on v2, once:** A and B (never run on v2), and **C**:
    M2b's detector, PP-OCRv6 small untrained (recogniser alone, S2's
    recipe), M3's parser and rules. C against the fine-tuned reader is
    exactly what training the reader adds.
20. **(changes M2b's final test, which scored ExpDate test on the laptop)
    ExpDate test is scored once on the cluster's CPUs**: 665 photos
    through the detector and a 2B VLM would heat the laptop for an hour.
    v2 stays laptop-only, scored once at the end, about 15 minutes.
21. **(new) M2b's detector is an input, not a choice.** M3 uses what M2b's
    R3 and R4 pick. Until then M2's `gen-rot-1280` seed 0 stands in for
    building and checking the pipeline, and **R6 is run again** with
    M2b's detector before the final test. While designing M3, nobody opens
    M2b's per-photo v2 results: only its totals.
22. **(new) GPU work first.** S1 holds the GPU job(s) before any code; S9
    releases them once the first runs' data and code exist.

Left out, with reasons, in [Later](#later): a date grammar in the decoder,
distilling the VLM into the small reader, pseudo-labels, an orientation
classifier, a learned role model, bigger VLMs, Date-Synth and Components.

---

## The data

### Piles and roles

| Pile / set | What | Role | Where |
|---|---|---|---|
| `expdate-real` | real-train without real-val: 1,057 date crops | training | cluster |
| `expdate-synth` | Products-Synth: 16,674 date crops | training | cluster |
| `gen` | gen-v2: about 473,000 date crops, plus codes and class-less tokens | training | cluster |
| `store` | v1-train and v3-train: 167 date crops, plus shelf-life phrases | training, design-balanced | cluster (numbered) |
| `real-val` | 187 date crops | checkpoints (17), R2–R4 guard | both |
| `real-val-rot`, `real-val-rot180` | the same, cut from M2's and M2b's turned copies, not turned back | orientation | cluster |
| `synth-val`, `synth-val-v2` | M2's 2,100 and M2b's 2,400 target crops, one difficulty per stratum | checkpoints (synth-val-v2), diagnosis by stratum | cluster |
| `store-val` | v1-val and v3-val: 62 date crops; 34 photos end to end | R2–R6 | cluster (numbered), laptop |
| `store-train` | the training crops, scored as they are | memorisation check only | cluster |
| ExpDate test | 665 photos; their date crops | test, once (S13) | cluster, cut at S13 only |
| store set v2 | 99 photos | test, once (S13) | laptop only |

**Not used:** M2's gen50k (decision 8); Date-Synth and Components (their
labels are per part, day, month or year, without the separators, so they
give no whole printed line; Components are single parts while our readers
read whole dates; and Date-Real and Components-Real are cut from ExpDate's
real photos, test photos possibly among them: training on them could leak
into ExpDate test. Later, if the CRNN runs short of variety).

### A crop record

`reader/shards.py`, one LMDB per pile under
`$GENERATED_DIR/m3/crops/<pile>/`:

```python
@dataclass(frozen=True)
class Crop:
    key: str                  # "gen/000123456"
    jpeg: bytes               # upright, with its margin, box short side at most 64 px
    box: tuple[int, int, int, int]   # the box inside the crop, pixels
    text: str                 # as printed, case kept ("" never: such tokens are skipped)
    cls: str | None           # date, code, None (class-less)
    source: str               # photo and token index; store: the numbered photo only
    role: str | None          # expiry, mfg, import, other (dates)
    parts: tuple | None       # (year, month, day or None), dates only
    turn: int                 # the quarter it was turned to stand upright
    tags: dict                # style, kind, orientation, stratum, tag, design, part
```

`make_crops.py --pile <name>` writes a pile and a manifest (counts by
class, style and tag; settings; the hashes of its sources; the code's
SHA-256), then `--check` reads every record back: it decodes, its box lies
inside it, its text is non-empty, and the counts match the source.

### Store text (S4)

`data/store/text.json` (not committed; names photos), frozen with
`TEXT_SHA256` in `store.py`:

```
photo → {"month": "2026-08",                  # from EXIF, for decision 13
         "boxes": [{"label": "expiry", "xyxy": [...], "text": "04/12/2026", "turn": 0}, ...],
         "life":  [{"xyxy": [...], "text": "Best Before 24 months from Mfg.", "turn": 0}]}
```

Every date box of the frozen labels in their order (and the 4 dates M2b's
keyword review found unboxed, `added_dates`), each with text and turn. The import refuses
a file where an expiry's or mfg's text doesn't parse to its frozen value
(a typo, or a label to look at again: listed for the owner, never fixed
silently). `export_store_text.py` writes the numbered version into
`$STORE_TRAIN_DIR/text.json` for the cluster, with no photo name.

### The mix

```python
PILE_WEIGHT = {"expdate-real": 10.0, "expdate-synth": 20.0, "gen": 69.5, "store": 0.5}
```

Each sample picks a pile by weight (normalised over the arm's piles),
then a record uniformly; the store pile picks a design first, then one of
its crops, so a design seen 10 times weighs the same as one seen once (as
M2b decision 7). Seeded per run and per loader worker.

- **ExpDate real at 10%**, about 6 times its share by count: real print,
  the only real text in the `exp` arm besides synth.
- **store at 0.5%** of the CRNN's 10 million samples: about 300 views of
  each store crop, twice M2b's about 156 views of each store photo, with
  augmentation; more would memorise 167 crops (the store-train score
  shows it).
- In `exp`, real and synth are a third and two thirds.
- Training crops whose print can't be seen (under `m3.VISIBLE_MIN`, S5)
  are never drawn; suites keep every crop.

### Augmentation (fixed, not tuned)

On every pile, at load time: each box edge moved by up to ±10% of the box
height around S2's padding; a turn of up to ±4° (p 0.3); shrink to 35–100%
and back (p 0.4, small print); Gaussian blur σ 0.3–1.5 (p 0.3); JPEG
quality 25–90 (p 0.4); brightness and contrast ±30% and a colour cast
(p 0.5); colours inverted (p 0.05, light print on dark). No flips: text.

---

## The parser

`reader/parse.py`, on top of `dates.py`, which stays frozen as baseline
A's. Written in S2 from training data only (ExpDate train, v1-train and
v3-train formats, gen-v2's format list), frozen with `PARSE_SHA256` in
`m3.py` before S2's job runs; any later change is a new version, recorded
and re-scored.

```python
def read_dates(text: str, whole_crop: bool = True) -> list[Found]:
    """Every date in one crop's read, left to right: dates.find_dates, plus
    what a crop of one date allows and free OCR text doesn't:
    - ordinals: '19th December 2026', '1st Jan 2027';
    - DDMMYYYY with no separators ('04122026'), unless YYYYMMDD makes a date
      (both never can: a YYYYMMDD year starts 20, a DDMMYYYY month can't be 15-39);
    - a month and a 2-digit year when the date is the whole read ('08/27',
      'MAR 27', '03.27'): in a box the detector calls a date this is the
      natural reading, where in free text dates.py rightly skips it."""

def roles_in(text: str) -> str | None:
    """'expiry' if the read holds EXP, BB, BEST BEFORE, USE BY; 'mfg' if
    MFG, MFD, MANUF, PKD, PACKED, IMPORT; else None (a glued keyword)."""

def shelf_life(text: str) -> int | None:
    """Months from '24 months from Mfg.', '1 Year from Mfg. Dt.',
    '18 MONTHS FROM MANUFACTURE'; None for anything else."""

def add_months(d: Expiry, months: int) -> Expiry:
    """The day kept when printed, clipped to the month's end (31 Jan + 1 month = 28/29 Feb)."""
```

## Roles and the expiry

`reader/roles.py`:

```python
def beside(date: Box, keywords: Sequence[Box], turn: int) -> str | None:
    """The role of the nearest due or prod box in the date's own upright
    frame (turned by `turn`): on its line and to its left (vertical overlap
    at least half the smaller height, the gap at most 4 date-heights,
    allowing 0.3 height of overlap for a glued word), else just above it
    (horizontal overlap, gap at most 2 heights). S8 sets the numbers on
    v1-train and v3-train only."""

def choose_expiry(dates: Sequence[Read], month: str | None) -> tuple[Expiry | None, str]:
    """Decision 11. The latest plausible date (decision 13 when `month` is
    known); none when its role is mfg, or when it is the only date, has no
    role and falls before `month`. Returns the expiry and why (for the
    table of where photos were lost)."""

def life_strip(due: Box, image) -> Box:
    """The strip after a due word on its line: 12 heights long, 1.4 high, clipped."""
```

---

## The readers

### Shared

- **Input:** RGB, resized to 48 px high keeping the aspect, padded on the
  right to the batch's widest (a multiple of 8, at most 480 px; wider is
  squeezed), normalised to `(x/255 − 0.5)/0.5`, as PP-OCR does. One
  function, `reader/onnx_run.py`'s, for training's evaluation, ONNX
  scoring and the phone later.
- **Batches by aspect:** crops sorted into width buckets, so a batch pads
  little (RapidOCR sorts its batches for the same reason).
- **Decoding:** greedy CTC, `reader/ctc.py`, written against tests
  written first and walked through with the owner (S5): the best class per frame, repeats merged,
  blanks dropped, each kept character's probability returned.
- **Metrics:** exact text (case and spaces folded), character error rate
  (`reader/metrics.py`'s `cer`, walked through with the owner: edit distance
  over the truth's length), and date-level right / wrong / none
  (`scoring.score` on the parser's dates against the box's truth).

### CRNN + CTC

About 4.5M parameters, near PP-OCRv6 small's 5.2M, so the comparison is
pretraining and design, not size:

| Block | Layers | Output (H × W) |
|---|---|---|
| 1 | conv 3×3, 64, BN, ReLU; max-pool 2×2 | 24 × W/2 |
| 2 | conv 3×3, 128, BN, ReLU; max-pool 2×2 | 12 × W/4 |
| 3 | conv 3×3, 256 ×2, BN, ReLU; max-pool (2,1) | 6 × W/4 |
| 4 | conv 3×3, 384 ×2, BN, ReLU; max-pool (2,1) | 3 × W/4 |
| 5 | conv (3,1), 384, no padding, BN, ReLU | 1 × W/4 |
| sequence | 2-layer bidirectional LSTM, 128 a direction | W/4 frames × 256 |
| head | linear to 48 classes (blank, 46 characters, `?`) | W/4 × 48 |

A 320 px wide crop gives 80 frames, enough for the 25 characters CTC must
place with blanks between repeats. Training: `torch.nn.CTCLoss` (blank 0,
`zero_infinity`), log-softmax in fp32; AdamW, lr 1e-3, weight decay 1e-4,
cosine with 2% warmup, gradient clipping at 5; batch 256; bf16 on the
convolutions only, the LSTM and the loss in fp32; **budget 10 million
crops** (about 39,000 steps); 16 loader workers, pinned memory,
persistent workers. Seeds 0, 1, 2.

### PP-OCRv6 small

- **Off the shelf:** `PP-OCRv6_small_rec`'s ONNX file through our
  pre- and post-processing and its dictionary, checked against
  `paddleocr`'s own reads of the same crops (S2's) on real-val: at least
  99.5% identical, or we have the processing wrong.
- **Fine-tuning route (S6 decides, by a check, not a preference):**
  - **PyTorch**, if PP-OCRv6's recogniser loads in `transformers` (≥ 5.8)
    as a module whose CTC logits take gradients, reads real-val as the ONNX
    file does (99.5%), and 200 steps of our loop lower its CTC loss on
    2,000 crops. Then it trains in the CRNN's loop: one loader, one
    augmentation, one loss, only the model differs.
  - **Paddle**, otherwise: `~/venvs/paddle-gpu` (`paddlepaddle-gpu==3.2.0`
    from the `cu126` index, PaddleOCR at tag `v3.7.0`), the official
    `PP-OCRv6_small_rec.yml` with our overrides (pretrained weights, lr,
    budget, our crops written out by `mix()` as an LMDB in PaddleOCR's own
    layout, or as label files after a 32-thread cache warm-up), exported
    with `paddlex --paddle2onnx`. Its own augmentation, a difference
    recorded.
- **Recipe:** from the pretrained weights, the whole network, lr 5e-5 (a
  tenth of the config's 5e-4: fine-tuning moves a trained model a little),
  cosine with 3% warmup, batch 128, **budget 2 million crops**, the `all`
  mix, its dictionary and case kept (its head keeps its trained weights).
  Seeds 0, 1, 2.

### Qwen3-VL-2B

- **Prompt (fixed, the same untrained and fine-tuned):** *Read the dates
  printed on this product. Answer with JSON only: {"expiry": date or null,
  "mfg": date or null}. Write a date as YYYY-MM-DD, or YYYY-MM when no day
  is printed. If the expiry is printed as a shelf life from the
  manufacture date, work it out. If no expiry is visible, write null.*
- **Answers:** the first `{...}` in the output, parsed as JSON; a date
  that isn't a real `Expiry` counts as none.
- **Image size:** at most 1,600 × 1,200 px (the processor's pixel cap set
  to 1,920,000, about 1,875 visual tokens); ExpDate's photos stay at their
  own size (about 730 tokens).
- **Training set** (`make_vlm_set.py`, about 25,000 examples, one epoch):

  | Source | Examples | Target |
  |---|---:|---|
  | store, v1-train and v3-train, 4 turns, listed 3 times | 960 | the frozen labels (derived expiries included; `other-face` photos mostly null) |
  | ExpDate real-train without real-val, upright and turned 90°/270° (M2's `real_rot`) | 1,874 | the keyword rule: the date beside a `due` mark, else the only date, else the latest; mfg beside `prod` |
  | ExpDate synth, sampled | 5,000 | the same rule |
  | gen-v2, one panel cut round one block, sampled | 15,000 | the block's expiry and mfg from the sidecar; a `relative` block's expiry worked out |
  | negatives: ExpDate regions with no date, and regions with the mfg date only | 2,000 | `null`, or mfg only |

- **Recipe:** `Qwen3VLForConditionalGeneration` in bf16; LoRA r 16, α 32,
  dropout 0.05 on the language model's attention and MLP projections; the
  vision tower and its merger frozen; loss on the answer tokens only;
  AdamW lr 1e-4, cosine with 3% warmup; micro-batch 4 × accumulation 4;
  gradient checkpointing. Validated every 20% on real-val and store-val
  photos with the HF model on the GPU.
- **Scoring:** the LoRA merged, converted to GGUF (`convert_hf_to_gguf.py`,
  with `--mmproj`), the language model at Q8_0, the projector at F16;
  run by `llama-server` on 127.0.0.1 through `/completion` with `n_probs`,
  for the token probabilities behind decision 15. The same two files on
  the cluster's CPUs and the laptop (Metal).
- **The GGUF path is proved first, on the untrained model (S7):** our own
  conversion against the official GGUF, and both against the HF model, on
  50 real-val photos. If converting the fine-tuned model fails, the owner
  chooses, before any v2 scoring, between the HF model in bf16 on the
  laptop (validation scored the same way there) and a written, one-off
  exception for v2 on the cluster.

---

## Scoring

### The crop suites (reader alone)

`scripts/score_reader.py --reader <run> --suite m3`: every suite above,
through ONNX Runtime, with the orientation step. Per suite: exact text,
CER, date-level right / wrong / none at no threshold and at the reader's
R4 threshold, by stratum (synth-val, synth-val-v2), by tag (store), by
wide or tall; every read's confidence (for the curves); milliseconds a
crop on the machine's CPU. Per-crop reads of store crops stay in
`data/store/runs/` on the laptop or the job's run folder on the cluster,
never in `results/`.

### End to end (photos)

`scripts/score_pipeline.py --set <set> --reader <config>`: the detector
run **once per photo at confidence 0.001**, every box kept with its score;
every date box read once; the threshold sweep (R6) then only filters, so
it costs nothing. Per photo: the expiry or none, its score against the
truth, and its trace:

| Lost where | Meaning |
|---|---|
| detector | no date box at IoU ≥ 0.5 with the expiry box above the threshold |
| reader | the expiry box found, its read holds no date or the wrong one |
| role | the expiry read right, another date chosen (or a veto) |
| derived | no expiry printed; the strip found no phrase, or misread it |
| right | |

Per set: right / wrong / none by tag; `other-face` photos apart (a date
claimed there is wrong); derived apart; on v2, by design seen in training
(v1-train, v3-train), seen only in validation, or unseen.

### Operating points

The detector threshold (R6), each reader's confidence threshold (R4), the
strip (R5): chosen on validation only. The VLM gets R4 on its photo
answers.

---

## Decision rules

Written before any M3 run, as code in `m3.py`, applied by the chain and
the CPU jobs, re-checked by each experiment's `summarise.py`, as in M2.

```python
SEEDS = (0, 1, 2)
PILE_WEIGHT = {"expdate-real": 10.0, "expdate-synth": 20.0, "gen": 69.5, "store": 0.5}
ARMS = {"exp": ("expdate-real", "expdate-synth"),
        "exp+gen": ("expdate-real", "expdate-synth", "gen"),
        "all": ("expdate-real", "expdate-synth", "gen", "store")}
BUDGET = {"crnn": 10_000_000, "ppocrv6": 2_000_000}   # crops seen per run
EVALS = 20                     # checkpoints scored per run (every 5%), decision 17
PADS = (0.0, 0.15, 0.4)        # S2: padding, as a share of the box's short side
NOISE_STORE_MIN = 3            # store-val: one crop of 62 is 1.6 points
WRONG_CAP = 0.02               # wrong among kept reads: B's rate on v1, 2 of 87
VISIBLE_MIN = None             # S5: training crops under it are never drawn; set by eye on sheets, before any training

def crop_recipe(results: dict) -> dict:
    """R1 (S2). pad: the most right reads, summed over both untrained
    readers alone with turning on, over real-val and every store crop with
    a value (none of them test); a tie goes to the smaller pad. turn: kept
    if on tall crops it adds right reads and adds no wrong read."""

def store_band(crnn_all: Sequence[dict]) -> int:
    """The band on store-val: the spread of right reads over CRNN `all`'s
    3 seeds, at least NOISE_STORE_MIN. Fixed once those are scored, before
    R2 reads anything else."""

def guard(run: dict, ref: dict) -> bool:
    """No forgetting ExpDate: exact text on real-val at most 1 point under ref's."""

def decide_crnn(arms: dict[str, list[dict]], band: int) -> str:
    """R2. Eligible: the guard against `exp`'s mean. The highest mean right
    reads on store-val; within the band, `all` (it has seen store print),
    then `exp+gen`."""

def decide_reader(crnn: list[dict], ppocr: list[dict], band: int, ms: dict) -> str:
    """R3, the reader for the phone. Eligible: the guard. The highest mean
    right reads on store-val; within the band, the fewer milliseconds a crop
    on the laptop's CPU."""

def threshold(reads: dict[str, list[dict]], cap: float = WRONG_CAP) -> float:
    """R4, per reader configuration. The lowest confidence at which, on
    real-val and on store-val each, wrong reads among those kept are at most
    max(1, cap x the set's dates)."""

def adopt_strip(before: list[dict], after: list[dict]) -> bool:
    """R5, end to end on store-val photos. The strip only acts where the
    answer would be none, so it can only add right or wrong reads: it stays
    if it adds more right than wrong, and at most one wrong."""

def detector_threshold(rows: dict[float, list[dict]], b_wrong: int, cap: float = WRONG_CAP) -> float:
    """R6, end to end on store-val and real-val photos: the threshold with
    the most right reads whose wrong reads are at most B's on store-val
    (M0's reads on v1-val, S12's on v3-val) and at most cap x photos on
    real-val; a tie goes to the higher threshold (fewer boxes to read).
    Run again with M2b's detector before the final test."""
```

---

## The chain

`scripts/m3_chain.py` on `bestbefore/chain.py` (claims, watchdog, time
left, failures, replacements, dry run), as M2b's:

```
ready once S3/S4's crops and S5-S7's code are in (seed 0 of each first)
  m3-crnn-smoke                   1% of the budget, `all`                      ~5 min
  m3-crnn-all-s0, m3-ppocrv6-ft-s0, m3-vlm-smoke, m3-vlm-ft-s0
  m3-crnn-all-s1, -s2; m3-crnn-exp+gen-s0..s2; m3-crnn-exp-s0..s2              ~0.6 h each
  m3-ppocrv6-ft-s1, -s2                                                        ~0.5 h each
  m3-ppocrv6-off                  a record, no training: the pretrained ONNX file
scoring (first, as in M2)
  crop suites                     every CTC run, ONNX Runtime on the job's CPUs  ~3 min
  VLM validation                  HF model on the GPU                           ~10 min
decisions -> results/runs/m3-decisions.json
  band -> R2 -> R3 (R4-R6 come from S12's CPU job)
```

Every training run records its config, commit, seed, the crop manifests'
hashes, `PARSE_SHA256`, and images a second. A CTC run ends by exporting
its kept checkpoint to ONNX and checking it reads real-val as PyTorch did
(99.5%).

---

## The final test

Once, after R2–R6 are recorded and M2b's detector is chosen (M2b's R3 and
R4), with a guard that refuses a second run unless given a written reason:

- **Store set v2, on the laptop** (`score_pipeline.py --set store-v2`):
  A and B (first time on v2), C, the M3 reader (R3's, its R4 and R6
  thresholds, R5's strip), every CRNN arm and PP-OCRv6 run in the same
  pipeline, the VLM untrained and fine-tuned (GGUF, Metal); the reader
  alone on v2's expiry boxes (crops of the true boxes). About 15
  minutes. By tag, `other-face` and derived apart, by design seen or
  unseen, with the table of where photos were lost.
- **ExpDate test, on the cluster's CPUs**: the M3 pipeline (guard off: no
  photo dates) and the fine-tuned VLM end to end; the readers alone on
  the `exp` boxes (truth from `dmy_ann`) and the other date boxes (their
  text). A's 502 of 655 from M0 isn't re-run. The VLM's agreement with
  ExpDate's own `exp` marks is reported beside it (decision 18's rule).

## Report

- `results/m3_reader.md` from `scripts/m3_report.py`, from the records
  alone, as M2's: the headline (the M3 reader on v2 next to A, B, C and the
  VLM; ExpDate test), "What it shows" with every number read from the
  records, the table written before any run against what happened, the
  crop experiment, the crop suites by arm, stratum and tag, size and
  speed, the VLM, the operating points, v2 by tag, cause and design.
- Experiments, each with a README and a `summarise.py`: `m3-crop-reading`
  (S2), `m3-crnn-data` (the three arms), `m3-ppocrv6` (off the shelf
  against fine-tuned, and the route), `m3-vlm` (untrained against
  fine-tuned, GGUF against HF), `m3-pipeline` (roles, the strip,
  operating points).
- `README.md`: M3's section and headline, including where it doesn't beat
  B. `CLAUDE.md`: the milestone row and any lesson that changed (S2's
  above all), with the owner's OK.

---

## Code layout

```
src/bestbefore/
  m3.py               piles, arms, weights, budgets, recipes, PARSE_SHA256, rules R1-R6
  m3_results.py       the report's tables
  reader/
    charset.py        the character set, case folding, '?'
    crops.py          cut with a margin and padding, quarter turns, jitter
    shards.py         the LMDB record, writer, reader, checks
    piles.py          what each pile holds, where its crops come from (S3's)
    data.py           the training stream: mix(), design balance, augmentation, width buckets,
                      the visibility measure (S5)
    ctc.py            greedy_decode, confidences
    metrics.py        cer, exact text, date-level scores
    crnn.py           the CRNN
    train.py          one training loop for CTC readers (CRNN; PP-OCRv6 on the PyTorch route)
    ppocr.py          PP-OCRv6: dictionary, loading, export, the route check
    onnx_run.py       any CTC reader as ONNX: the one pre- and post-processing, batching by width
    parse.py          M3's parser on dates.py, roles inside a read, shelf life
    roles.py          beside(), choose_expiry(), the strip
    pipeline.py       photo -> expiry, with its trace
    vlm.py            prompt, targets, examples, answers, the llama.cpp client
  store.py            TEXT_SHA256; store_train.py: text.json's export
scripts/
  export_store_text.py, crop_experiment.py, make_crops.py, crop_sheet.py,
  propose_store_text.py, import_store_text.py, train_reader.py, export_reader.py,
  score_reader.py, make_vlm_set.py, train_vlm.py, vlm_gguf.sh, score_pipeline.py,
  m3_chain.py, m3_report.py
  changed: run_baseline.py (store-v2, v3-val), sets.py (store-v2, store-val, crop suites),
  pull_cluster_results.sh (m3 runs, never crops or weights)
slurm/
  m3_gpu.sbatch, m3_chain.sh, m3_cpu.sbatch (one CPU script, a task per argument:
  crop experiment, crops, visibility, smoke, route checks, GGUF, end to end, ExpDate test)
tests/
  test_reader_charset.py, test_reader_crops.py, test_reader_shards.py, test_reader_data.py,
  test_ctc.py (the decoder's spec), test_reader_metrics.py, test_crnn.py,
  test_reader_parse.py, test_roles.py, test_pipeline.py, test_vlm.py, test_m3_rules.py,
  test_m3_results.py
```

`pyproject.toml` gains `read = ["lmdb", "onnx", "onnxruntime"]` and
`vlm = ["transformers", "peft", "accelerate"]`, pinned in S1 to what the
cluster installs. `$GENERATED_DIR/m3/` holds crops and the VLM set;
checkpoints go to `$CHECKPOINT_DIR/m3/`.

---

## Slices

Numbered in the order to do them; a slice that ends in a cluster job comes
as early as its inputs allow. Every slice records its findings here when
done (cluster facts in `CLAUDE.md`), says what wasn't checked, and ends
with its commits.

### S1. Queue the GPU job and set up the environments · cluster · *submits 1 or 2 GPU jobs, held*

**Goal:** M3's place in the GPU queue, and everything the cluster must
download while it has internet.

- **First, the queue** (`squeue --me`): M2b's jobs A and B, and S9's CPU
  job if submitted. **Rule:** submit two held M3 GPU jobs if at most two
  of the owner's jobs are in the queue (one slot always stays free for CPU
  jobs), else one.
- `slurm/m3_gpu.sbatch`: `gpu`, 1 GPU, 24 CPUs, 96 GB, 3 days; it runs
  `slurm/m3_chain.sh` when it starts, and exits at once if that isn't
  there yet. `sbatch --hold`.
- **Environments, on the login node:**
  - the main venv: `pip install -e .[read,vlm]` with versions pinned
    (`lmdb`, `onnx`, `onnxruntime`, `transformers`, `peft`, `accelerate`);
    `transformers` at the newest release that loads both Qwen3-VL (≥ 4.57)
    and, if it can, PP-OCRv6 (≥ 5.8);
  - `~/venvs/ocr-cpu`: `module load Python/3.11.11`, `paddlepaddle==3.2.0`
    (CPU), `paddleocr==3.7.0`, `rapidocr-onnxruntime==1.2.3`, `onnxruntime`:
    for S2 and the parity checks on CPU nodes;
  - `llama.cpp` built from a pinned tag (cmake, CPU) in `~/opt/llama.cpp`.
- **Downloads** (compute nodes have no internet), each with its SHA-256
  into `docs/datasets.md`: PP-OCRv6 small recogniser (ONNX, safetensors,
  pretrained `.pdparams`, dictionary) and the small detector (S2's full
  pipeline; names checked here); Qwen3-VL-2B-Instruct (HF snapshot); the
  official GGUF (Q8_0, mmproj F16).

**Check:** `squeue` shows the job(s) `JobHeldUser`; each venv imports its
packages; `paddleocr` reads one real-val crop on a CPU node with both
models; files hashed. **Commits:** `docs: m3 plan` (this file, first),
`feat: m3 gpu job script and the read and vlm extras`

**Done 2026-11-12.** A first session stopped at a usage limit after the
main venv; a second finished the rest, and submitted the CPU check jobs at
the owner's word (`73948`, `73949`, `73950`). Departures: `docs: m3 plan` was already
committed (`c8b431f`), so S1's first commit is the feat; S1 also wrote
`scripts/fetch_m3_models.py` and `scripts/check_m3_env.py`, which the
plan's layout didn't name, and `slurm/m3_cpu.sbatch` (planned for S2) with
a first task, `env`.

- **The queue.** At 17:33 (`squeue --me`): M2b's `73832` and `73906`, both
  held, nothing running; S9's job `73944` `COMPLETED` (14:34 to 15:00). Two
  queued, so the rule gave **two** held M3 jobs. `slurm/m3_gpu.sbatch`
  (`421f2c6`) checks for `m3_chain.sh` before `nvidia-smi` (tested on the
  laptop: exit 1 without it, the chain run with it). The owner submitted
  **`73946` and `73947`** at 17:52:44; at 18:50 both `PENDING`,
  `JobHeldUser`, `gpu`, 24 CPUs, 96 GB, 3 days: 4 of 5 submitted, one slot
  for CPU jobs.
- **Quota:** 182.1 GB of 950 before, 191.1 GB after the downloads (models
  7.1 GB, `ocr-cpu` 1.4 GB, llama.cpp's build).
- **The main venv**, with no M2b job running: `pip install -e
  ".[read,vlm]"` with a constraints file of all 68 installed packages
  (`pip freeze --exclude-editable`), so nothing installed could move. The
  freeze before and after differ by **28 added packages and nothing else**
  (torch `2.11.0+cu128`, ultralytics 8.4.177, numpy 2.4.6, OpenCV 5.0.0,
  Pillow 12.3.0 kept); `pip check` clean. transformers 5.22.2 (the newest)
  needs no other torch, so no separate venv. Added: transformers 5.22.2,
  peft 0.21.2, accelerate 1.15.0, lmdb 3.0.0, onnx 1.23.2, onnxruntime
  1.31.1 (CPU and Azure providers), huggingface_hub 2.4.0, tokenizers
  0.23.3, safetensors 0.8.0, protobuf 7.36.2 and their own dependencies.
  Records in `~/bestbefore-data/env/m3-s1/` (freezes, constraints, dry
  runs, test output, the build log).
- **M2b unchanged:** `m2b_chain.py --dry-run` prints the same 36 lines
  before and after; the whole suite gives 737 passed, 2 skipped before and
  after (the same two skips, laptop-only data and pixels), output identical
  but for timing (4 min 45 s and 2 min 57 s on the login node, niced).
- **transformers 5.22.2 holds** `Qwen3VLForConditionalGeneration`, and
  PP-OCRv6 as `pp_ocrv6_small_rec`, `pp_ocrv6_tiny_rec`,
  `pp_ocrv6_small_det` and `pp_ocrv6_medium_det` (also PP-OCRv5's four,
  `pp_lcnet_v4`): S6's PyTorch route has a model to check.
- **`~/venvs/ocr-cpu`** (Python 3.11.11, pip 24.0 as a fresh venv brings):
  paddlepaddle 3.2.0 (CPU, no CUDA), paddleocr 3.7.0, paddlex 3.7.2,
  rapidocr-onnxruntime 1.2.3, onnxruntime 1.31.1, numpy 2.3.5 (paddlex
  caps it below 2.4), Pillow 12.3.0 (pinned to the main venv's, so both
  decode a crop alike); 73 packages, 1.4 GB. **One OpenCV:** paddlex pins
  `opencv-contrib-python==4.10.0.84` and RapidOCR asks for
  `opencv-python` (the resolver chose 5.0.0.93), two packages writing one
  `cv2/`; so RapidOCR went in with `--no-deps` (all its other needs come
  with paddlex, plus `six`). `pip check`'s one complaint is that, as
  expected; `cv2` is 4.10.0, contrib modules present. Note for S2:
  RapidOCR here runs on OpenCV 4.10, baselines A and B ran on the laptop's.
- **PaddleOCR's models.** PaddleX 3.7.2 knows PP-OCRv6 `tiny`, `small` and
  `medium`, detector and recogniser, each with `_pretrained` training
  weights. Its OCR pipeline defaults to the medium pair, plus
  `PP-LCNet_x1_0_textline_ori` (the angle classifier), and page
  orientation and unwarping models (for scanned pages: off on crops, not
  fetched). Its detector enlarges an image only when its short side is
  under **64 px** (`limit_type: min`), against RapidOCR 1.2.3's 736:
  Scanify's crop failure shouldn't repeat there as it did, which S2 will
  show. PaddleX fetches `PaddlePaddle/<model>` from Hugging Face into
  `$PADDLE_PDX_CACHE_HOME/official_models/`, and finds them there offline
  with `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`; fonts are fetched only
  to draw results.
- **The downloads** (`scripts/fetch_m3_models.py`, into
  `$CHECKPOINT_DIR/pretrained/` beside the YOLO weights): PP-OCRv6 small
  recogniser (Paddle, ONNX, safetensors), small detector, the angle
  classifier, the recogniser's `.pdparams` training weights, Qwen3-VL-2B-
  Instruct and the official GGUF (Q8_0, F16 projector). 43 files, 7.1 GB,
  each repo pinned to its commit, every LFS file matching Hugging Face's
  SHA-256; all in `docs/datasets.md`. From the login node: Qwen's 4.27 GB
  in 40 s, the GGUF pair's 2.65 GB in 30 s, the `.pdparams` (124.9 MB)
  from Baidu's BOS in 20 s. The recogniser's dictionary is inside its
  `inference.yml` (no separate file).
- **Licences:** all **Apache-2.0** (model cards), none gated; the
  `.pdparams` states none itself and is PaddleOCR's (Apache-2.0); the
  GGUF repo has no LICENSE file, only its card's tag. Nothing stops this
  use.
- **llama.cpp** at tag **`b12017`** (commit `5047748f`, 2026-10-30, about
  two weeks old; every build tag is published as a prerelease), in
  `~/opt/llama.cpp`, built on the login node with `module load
  cmake/3.31.4` and **gcc-toolset-12** (`source
  /opt/rh/gcc-toolset-12/enable`: gcc 12.2.1, binutils 2.38), 4 threads,
  niced: 269 s, 124 MB. Options `GGML_NATIVE=OFF GGML_BACKEND_DL=ON
  GGML_CPU_ALL_VARIANTS=ON LLAMA_CURL=OFF LLAMA_USE_PREBUILT_UI=OFF`, no
  tests or examples: 14 CPU variants (x64 to Sapphire Rapids, Zen 4), the
  best picked when a binary starts, so one build serves every node.
  Binaries include `llama-server`, `llama-mtmd-cli`, `llama-completion`,
  `llama-cli`, `llama-quantize`. Found on the way:
  - **the first build failed**, with the default compiler (OpenHPC's gcc
    12.3.0), on the Sapphire Rapids variant only: the system assembler,
    binutils 2.30 (RHEL 8.8), doesn't know AMX instructions (`tileloadd`,
    `tdpbssd`). gcc-toolset-12's assembles them;
  - **llama.cpp's configure downloads a web UI** from a moving `latest`
    tag by default (`LLAMA_USE_PREBUILT_UI=ON`): turned off, so the build
    fetches nothing unpinned. `llama-server` runs without its web page; S7
    needs only its HTTP API;
  - `llama-server --version`: `0.8.0-dev (build 1, commit 5047748)` (build
    1: a shallow clone); `llama-cli --list-devices` lists none (the CPU
    isn't a "device"), so the CPU check generates 8 tokens, whose log names
    the variant loaded;
  - the login node `mn1` is a Xeon Gold 6442Y too (2 × 24 cores, 96
    threads) with AVX-512 BF16 and AMX;
  - **its AMX variant dies on the cluster** (below), so
    `libggml-cpu-sapphirerapids.so` is kept in `build/bin/disabled/` (with
    a README saying why): ggml scans `build/bin` only, not its folders, and
    picks the best of the 13 left, `zen4` (AVX-512 VBMI, VNNI, BF16).
- **Found in passing (M2b's):** the cluster's `.env` sets only
  `EXPDATE_DIR`, `CHECKPOINT_DIR`, `GENERATED_DIR`; M2b's dry run says
  "`$STORE_TRAIN_DIR` unset". The export is there (`~/bestbefore-data/store`,
  mode 700). M2b S10 and M3 S2 both need the line added; not changed here.
- **The CPU check** (`sbatch --cpus-per-task=4 --mem=16G --time=00:30:00
  slurm/m3_cpu.sbatch env`, on `short`; `cn1` each time, kernel
  `4.18.0-477.10.1.el8_8`, no internet):
  - **job `73948` (`3414dea`) failed, in llama.cpp only.** Everything
    before it passed, as in `73950` below; then `llama-completion` died
    with **`Illegal instruction`** at its first computation (the warm-up),
    after loading the `sapphirerapids` variant;
  - **job `73949`, a one-off diagnosis** (in `~/bestbefore-data/env/
    m3-s1/diag/`, not in the repo): a small C test runs AMX tile
    instructions on `cn1` after the same permission request ggml makes
    (`arch_prctl`, XTILEDATA granted), so AMX itself works there; the
    `sapphirerapids` variant, with Q8_0 weights repacked into its AMX
    buffer, dies with SIGILL again; and each other variant, forced alone
    beside a copy of the binary, runs: `zen4`, `cooperlake`, `icelake`,
    `cascadelake`, `skylakex`, `haswell` at **22.4 to 23.0 tokens a
    second** on 4 CPUs, `x64` at 10.5. So the fault is in `b12017`'s AMX
    path, not the node; why wasn't looked into;
  - **job `73950` (`9e14f5e`) passed**, in 32 s: the main venv's packages
    at their versions; Qwen3-VL-2B's config and processor offline in 0.3 s
    (`Qwen3VLProcessor`, patch 16, merge 2: a visual token per 32 × 32 px;
    pixels between 65,536 and 16,777,216 by default, so S7 sets its own
    cap); the safetensors config is `pp_ocrv6_small_rec`; the ONNX
    recogniser takes `x [N, 3, 48, W]` and gives **18,710 classes** a frame;
    the crop, real-val's first date (`img_00003.jpg`, `21.05.26`, 240 ×
    23 px with a 0.15 margin). In `ocr-cpu`: PaddleOCR loads all three
    models from the local cache; **the recogniser alone reads
    `21.05.26`, score 0.996** (0.09 s); **the full pipeline finds no text
    line** on that crop (0 lines), nor does **RapidOCR's** (0 lines): one
    crop, no evidence yet, but S2's question in small. llama.cpp loads
    `zen4` and runs 84 tokens a second on the prompt, 23 generating;
  - **ONNX Runtime pins a thread per core of the whole node** unless told
    how many: inside a 4-CPU job each pin fails (`pthread_setaffinity_np
    failed ... Specify the number of threads explicitly`). Harmless here,
    but 48 threads on 4 CPUs would crawl, so every session of ours sets
    `intra_op_num_threads` to the job's CPUs (the check does since
    `9e14f5e`); RapidOCR's own sessions still print it (129 lines);
  - `sacct`'s `MaxRSS` for `73950` reads **27.5 GB under a 16 GB limit**,
    and the job wasn't killed: the sample overcounts here (shared or mapped
    pages counted more than once, likely; not looked into).
- **Not checked:** llama.cpp loading the projector or reading a photo
  (S7); `convert_hf_to_gguf.py`'s Python needs (S7: they
  may want their own venv); the PyTorch route for PP-OCRv6 (S6); Paddle's
  GPU build (`~/venvs/paddle-gpu`, only if S6 picks the Paddle route);
  whether the GPU nodes' CPUs match the CPU nodes' (llama.cpp picks its
  variant at run time either way).

### S2. The crop experiment · laptop, then a CPU job · *submits 1 CPU job*

**Question:** fed a tight date box, how well does an untrained reader
read it, and does skipping its own text detector fix what Scanify saw?

**Before the job, in `experiments/m3-crop-reading/README.md`:** the
question, the method, R1, and **the owner's guess**: which tags get better
when the recogniser reads the crop alone, and which don't.

- `reader/parse.py` (frozen here: `PARSE_SHA256` in `m3.py`, begun),
  `reader/crops.py`.
- `scripts/export_store_text.py --values`: `$STORE_TRAIN_DIR/text.json`
  with each numbered photo's box values (expiry, mfg) and month; copied to
  the cluster by CLAUDE.md's owner-only rules.
- `scripts/crop_experiment.py`, in `~/venvs/ocr-cpu`:

  | Factor | Levels |
  |---|---|
  | reader | RapidOCR 1.2.3 (PP-OCRv3: baselines A and B's), PP-OCRv6 small (`paddleocr` 3.7) |
  | what runs | **full**: its own detector, angle classifier and recogniser on the crop; **alone**: the recogniser only (RapidOCR's `text_recognizer`, `paddleocr`'s `TextRecognition`) |
  | padding | 0, 0.15, 0.4 of the box's short side |
  | turning | as cut; turned (decision 14), alone only |

  18 configurations × about 386 crops: real-val's 187 dates (text), and
  every store expiry and mfg box with a value, training and validation
  (199; none of them test). Each crop read once per configuration; about
  7,000 reads, minutes on 24 CPUs.
- **Skipped, with reasons:** enlarging crops first (the recogniser resizes
  every crop to 48 px high anyway); sharpening and thresholding (measured
  by Scanify: thresholding cost 12 of 44, mild sharpening gained 2);
  grayscale against colour and resize filters (small effects; with a few
  hundred crops a 1 or 2 date difference is noise, and many knobs find a
  lucky one).
- `slurm/m3_cpu.sbatch crop-experiment`: `short`, 24 CPUs, 32 GB, 2 h.
- **Read the same session** if it ends (CPU nodes are rarely full):
  right / wrong / none and exact text per configuration, by tag and wide
  or tall; R1's padding and turning into `m3-decisions.json`.

**What it decides:** the padding and turning every M3 crop uses; C's
recipe; the "untrained" rows of the report. **If it contradicts
CLAUDE.md's lesson** (the recogniser alone on tight crops doing as well
as RapidOCR on the whole photo), the lesson is corrected there, with the
owner's OK. **Commits:** `feat: m3 date parser for one crop's read, frozen`,
`feat: the crop experiment, untrained readers on date boxes`,
`docs: m3 plan, S2's result`

**Done 2026-11-13.** Job `73956`, read the same session. The README
(`experiments/m3-crop-reading/`) holds the question, the method, R1, the
owner's guess (written in before any code for the job), every table and
the decision; `summarise.py` rebuilds them from the reads. Departures: the
README went in first as its own commit (`d0ff195`); the summary script,
`crop_scores.py` and `store_text.py` aren't in the plan's layout; R1's
turning was overruled by the owner (below).

- **The owner's guess:** only `clean` better read alone; PP-OCRv6 the
  better reader; R1's pad 0.4. Measured: alone (turned) beats the full
  pipeline on every tag for both readers but small print for RapidOCR
  (the same); PP-OCRv6 reads more (181 store dates to 165) but 4 wrong to
  0; the pad is 0 (0.4 the worst).
- **The parser** (`reader/parse.py`, frozen `026755390a03…` in
  `m3.PARSE_SHA256`; `score` and `summarise.py` refuse any other). Written
  from real-train's dates without real-val (26 of its 1,057 are two parts
  with no year, `02.18`; 2 odd ones, `2021.19.05`), synth's (whose misses
  are years outside `dates.YEARS`), v1-train's and v3-train's expiry
  formats (all 11 read: tested) and gen-v2's 39 formats plus its 2 ranges.
  On every ExpDate real-train and synth date text (17,918, real-val's
  included, as a check of equality only) its race over `dates.py`'s own
  patterns gives exactly `dates.find_dates`. Every gen-v2 format reads to
  its printed date in title, upper and lower case **but `DD.MM.YY`**: in
  591 of 1,200 draws both readings make a date and `dates.py`'s year-first
  rule (ExpDate's) wins. Scoring generated crops by date (S10) must use
  the sidecar's parts and text, not parse the truth. The whole-read rule
  reads ExpDate's month-day texts as month and year; on real-val 1 truth
  depends on it, and 4 of 187 transcriptions give no date (left out of
  date scores).
- **`text.json`** (`store_text.py`, `scripts/export_store_text.py
  --values`; kept out of `store.py` and `store_train.py`, which M2b's
  running job imports: S4's `TEXT_SHA256` should follow suit while M2b
  runs). 114 photos (80 training, 34 validation), every date box by its
  label-file line: 198 expiry and mfg boxes, every one with a value (64 and
  74 training, 29 and 31 validation), 34 other dates without; no expiry box
  sits on a derived photo. **51 photos carry no EXIF capture month** (32
  training, 19 validation: all v1's); v1's dated ones are 2026-08, v3's
  2026-10. Its check ties it to the export's manifest (`842395a5…`) and
  each line to the manifest's labels. Written at `4d3166e`, SHA-256
  `168667105890…`; on the cluster `~/bestbefore-data/store/text.json`, 600,
  `find -perm /077` over the export: 0.
- **The laptop got its own `~/venvs/ocr-cpu`** with the cluster's versions
  (paddlepaddle 3.2.0 has a macOS arm64 wheel; paddleocr 3.7.0, paddlex
  3.7.2, numpy 2.3.5, opencv-contrib 4.10.0.84, onnxruntime 1.31.1,
  RapidOCR 1.2.3 with `--no-deps` and `six`), and the three PaddleX models
  (37 MB) copied from the cluster into `data/checkpoints/pretrained/
  paddlex/`. Smoke: 4 crops of each set, 156 reads in 8 s. **The laptop and
  `cn1` read those 156 alike:** every text identical, scores within 0.0002.
- **Libraries, as met:** RapidOCR 1.2.3 has no thread option; its three
  sessions take theirs from `rapidocr_onnxruntime.utils.SessionOptions`,
  replaced in the read step (they sit at `text_detector.infer.session`,
  `text_cls.infer.session`, `text_recognizer.session.session`). The read
  step records every ONNX Runtime session made and refuses one without the
  job's threads: 3 were made, all RapidOCR's, so **PaddleOCR 3.7 runs on
  Paddle, not ONNX Runtime**; it takes `cpu_threads`. RapidOCR returns
  `('', 1e-50)` for an empty read; its full pipeline drops lines under 0.5.
- **The cluster.** The three commits added files and changed only
  `slurm/m3_cpu.sbatch`; M2b's chain has no git or source-tree check, so
  the pull (`4d3166e`) changed nothing it runs while `73832` ran. The
  owner gave the word to submit; I ran `sbatch --cpus-per-task=8
  --mem=16G --time=01:00:00 slurm/m3_cpu.sbatch crop-experiment` (8 CPUs,
  not the plan's 24: the work was sized at minutes). It started at once
  on `cn1`, 18:55 to 19:34: cut seconds, models loaded in 56 s, 7,176 reads
  in 2,240 s, then scoring. The run folder (`~/bestbefore-data/generated/
  m3/crop-experiment/20261113-185541-m3-crop-experiment/`, 700, crops and
  per-crop reads) had 0 entries open; its reads were copied, without the
  crops, to `data/store/runs/` (700).
- **Slow on `cn1`:** the laptop's smoke promised about 6 minutes; the
  reads took 37. Per crop, the recognisers alone 27 to 40 ms (RapidOCR)
  and 87 to 102 ms (PP-OCRv6), the full pipelines 266 to 406 ms and 1.2 to
  1.5 s. TotalCPU 4 h 58 min in 38:55 on 8 CPUs; `sacct` MaxRSS 1.6 GB.
  Paddle printed `ReduceMeanCheckIfOneDNNSupport` 15 times and warned that
  `OMP_NUM_THREADS` isn't 1. Why it is slow wasn't looked into (Paddle's
  oneDNN meeting crops of every size, likely).
- **The result** (README, every table): on v1's 81 expiries in frame,
  baseline A on the whole photo 49 / 7 / 25, B 57 / 2 / 22; on the tight
  box, RapidOCR's full pipeline 34 / 2 / 45, its recogniser alone (turned,
  pad 0) **70 / 0 / 11**, PP-OCRv6's **78 / 0 / 3**. On all 198 store crops
  at pad 0, turned: PP-OCRv6 181 / 4 / 13, RapidOCR 165 / 0 / 33; on
  real-val 172 / 5 / 6 and 147 / 7 / 29. Every wrong store read alone at
  pad 0 is an mfg date (two one print's `FEB` read as `FE8`, parsed as
  August; a year digit; a day). The full pipeline's loss fits Scanify's
  cause: on real-val RapidOCR's pipeline reads 96 at pad 0 and 41 at 0.15,
  as the crops 30 px high or less, which skip its detector, fall from 112
  to 46. **CLAUDE.md's crop lesson was corrected, with the owner's OK.**
- **R1** (`results/runs/m3-decisions.json`): pad **0** (665 right at 0 and
  at 0.15, 564 at 0.4; the tie to the smaller), turning **not kept** (tall
  crops 0 / 0 / 82 as cut, 71 / 2 / 9 turned). **The owner kept turning
  (2026-11-13), a departure:** both wrong reads are the `FE8` misread in
  the right orientation (turning picked the right way up on every tall
  crop), and without turning no tall crop is read. `m3.CROP_DEPARTURES`
  ties it to this run; `m3.crop_recipe_used()` gives pad 0 with turning to
  S3 and baseline C, and a rerun of R1 doesn't inherit it.
- **For later** (the parser is frozen): a month number with a letter glued
  before it (`FE8 2025`) made a wrong date; a new version could refuse it.
- **Not checked:** exact text on store crops (S4 types it); the character
  error rate (the owner's, S5); confidences as decision 15 defines them
  (the libraries give a mean); why `cn1` is slow; the full set's speed on
  the laptop.

### S3. Crop shards · laptop, then a CPU job · *submits 1 CPU job*

**Needs gen-v2 on the cluster** (M2b S9's job). If it isn't there, the job
makes every other pile now and gen-v2's in S9's session.

- `reader/shards.py`, `scripts/make_crops.py`, `scripts/crop_sheet.py`.
- Piles: `expdate-real`, `expdate-synth`, `gen` (dates, codes, class-less
  tokens of 3 characters or more; turned upright by their block's
  quarter), and the suites `real-val`, `real-val-rot`, `real-val-rot180`
  (as cut), `synth-val`, `synth-val-v2` (targets as cut, so the orientation
  step is tested). Never ExpDate test.
- **Laptop trial** on gen-v2-sample and real-val: the records read back,
  and a sheet of 64 crops a pile for the owner (generated and ExpDate
  only).
- `slurm/m3_cpu.sbatch crops`: `short`, 24 CPUs, 48 GB, 4 h; one worker
  pool a pile; then `--check`. Estimate: gen-v2's 200,000 photos opened
  once (about 3.5 minutes of opening on 24 threads, plus decoding), about
  700,000 crops, about 3.5 GB.

**Check:** the manifests' counts against the sidecars and annotations;
every record decodes; a crop sheet per pile looked at. **Commits:**
`feat: m3 crop shards, lmdb records with text and turn`,
`docs: m3 plan, S3's crops`

**Done 2026-11-13.** Job `73958`, read the same session. Departures:

- `reader/piles.py`, not in the plan's layout: what each pile holds and
  where its crops come from; `reader/shards.py` holds the record and the
  LMDB; `crops.cut` cuts a stored crop. A trial pile, `gen-sample`
  (gen-v2-sample, the laptop's), and a smoke task, `crops-smoke`, which
  sized the job at the owner's ask.
- **`expdate-synth` leaves out the 14 synthetic copies** of validation
  photo `img_00277.jpg` (`m2.REUSED_VAL_COPIES`, M2's decision 18): 16,652
  dates, not the 16,674 of "What ExpDate and the generator hold", which
  counts all 11,858 photos. real-val decides checkpoints (decision 17).
- **The record:** `box` in floats rounded to 0.01 px, not whole pixels;
  `turn` is the turn a crop was given when cut (0 in a suite), and
  `tags["upright"]` the turn still needed to stand it upright: 0 in every
  training pile, a suite's truth for decision 14. JPEG 95 at 4:4:4 (no
  colour halved round small print).
- **An ExpDate date's parts come from the frozen parser** on its
  transcription (`parts_from: parser`): train and synth mark none. A
  generated date's come from its sidecar (`parts_from: sidecar`), never
  from its text, as S2 found they must.
- `m3.EDGE_JITTER` (0.10, the augmentation's edge jitter) is written now,
  so `make_crops.py` refuses to cut when the recipe's padding plus the
  jitter won't fit in the margin.

What was found:

- **The recipe** (`m3.crop_recipe_used`): pad 0 with turning kept, R1's
  run with the owner's departure; 0 + 0.10 fits the margin of 0.5. Every
  manifest keeps it. Nothing here depends on the pad: crops keep half
  their box's short side of margin.
- **ExpDate's own tall dates:** 78 synthetic date boxes are taller than
  wide (none in real-train or real-val). Of 25 looked at, 23 are upright
  text drawn tall and 2 sideways dot-matrix, so ExpDate's crops stay as
  the photo shows them (`shape` counted in the manifest).
- **The laptop trial** (real-val and gen-sample, 1,888 crops) took 1.6 s
  with 4 workers; `--check` passed; 1 worker and 4 wrote identical
  records. **The laptop's real-val and the cluster's are identical byte
  for byte** (187 of 187 JPEGs, boxes and texts).
- **The smoke** (job `73957`, the first 2,000 photos of each pile, 24
  CPUs): 26 s on `cn1`, every check passed. gen ran at 2.8 ms a photo
  (wall), opening a photo 52 ms per worker against the laptop's 4.4 ms;
  7.3 KB a crop in LMDB. The owner gave the word for 24 CPUs, 32 GB, 1 h.
- **The job** (`73958`, `short`, `cn1`, started at once): 21:38 to 21:44,
  5 min 45 s; TotalCPU 31 min; `sacct` MaxRSS 1.7 GB (sampled). The
  plan asked 4 h, far more than it took: opening gen-v2's 200,000 photos
  (37 GB) on 24 workers took 310 s, 1.56 ms a photo, 33.7 ms per worker
  (the smoke's cold first 2,000: 52 ms).

  | Pile | Crops | Photos | Seconds | `data.mdb` |
  |---|---:|---:|---:|---:|
  | expdate-real | 1,057 | 937 | 0.7 | 10 MiB |
  | expdate-synth | 16,652 | 11,745 | 6.6 | 188 MiB |
  | gen | 676,218 | 198,552 | 310 | 4,571 MiB |
  | real-val, real-val-rot, real-val-rot180 | 187 each | 165 each | 0.2 to 0.3 | 1.8 MiB each |
  | synth-val | 2,100 | 2,100 | 1.0 | 21 MiB |
  | synth-val-v2 | 2,400 | 2,400 | 1.2 | 26 MiB |

- **`--check` passed on all 8:** every record decodes, lies inside its
  crop, has text and sound class, role, turns and parts; the counts match
  the manifest; recounted from the sources, records + skipped = planned
  (nothing was skipped, and every wanted generated token had a box); the
  sources' hashes are unchanged; and the planned dates equal what the
  sources' own manifests hold (1,057 and 16,652 by M2's turned copies,
  483,566 by gen-v2's, 187 by M2's and M2b's turned copies, 2,100 and
  2,400 targets). Reading back gen took 5.4 s on 24 workers.
- **gen:** 483,566 dates (249,771 expiry, 206,780 mfg, 27,015 import),
  89,314 codes and 103,338 class-less tokens; 404,520 crops needed no
  turn, 102,330 and 101,129 a quarter either way, 68,239 a half; 228 are
  taller than wide once upright (steeply tilted blocks). 1,448 photos
  gave no crop: each has no block (gen-v2 skipped them, as its manifest
  counts). Its
  characters: space, `( ) , - . / :`, digits, every capital, and lower
  case but `q w z`; no `&` or `|`, which only keyword and separator
  tokens carry.
- **Suites:** real-val-rot's 187 crops are all tall, 103 to turn a
  quarter anticlockwise and 84 clockwise, as their photos turned;
  real-val-rot180's all need a half turn; synth-val keeps its 398
  sideways targets as cut, synth-val-v2 its 300 upside-down ones.
- **Parts:** 4 of real-val's 187 dates give the parser no date (as in
  S2), 12 of expdate-real's 1,057 and 4,034 of expdate-synth's 16,652
  (2,409 with a year from 2010 to 2014, outside `dates.YEARS`; 1,527
  without a 4-digit year; 98 other). Parts are for scoring; training
  reads the text.
- **Sheets looked at** (64 crops each; all 8 piles and both trial piles,
  in `$GENERATED_DIR/m3/crops/sheets/` and the laptop's
  `data/generated/m3/`): boxes sit on the print, texts match it, and
  every turned crop stands upright. **About 4 or 5 of 64 generated crops
  show no print a reader could read**, even enlarged (gen-sample's and
  gen's sheets alike): ink nearly the colour of its background, dots laid
  over the background photo's own print, or defocus. Their text is
  exact, but the photo doesn't show it. synth-val-v2's worn and
  low-contrast strata show the same. Asked by the owner afterwards, the
  worst were stretched to full contrast: `#318046` and `#66043` are pure
  noise. The generator's ink check passed them (0.094 and 0.045 against
  its bar of 0.04) before its camera added noise of about 5.5 grey
  levels. S5 leaves such crops out of training (the owner's go-ahead).
- **ExpDate has black photos:** 36 expdate-synth crops on 30 photos span
  under 16 grey levels inside their box; `img_06565.jpg`, all black (mean
  0.1), has two dates annotated. None in expdate-real or real-val; one in
  synth-val and one in synth-val-v2. They are kept: 0.2% of the pile.
- **Size:** 4.71 GiB of LMDB for 3.39 GiB of JPEGs (5.2 KB a crop): each
  record takes whole 4 KiB pages, 39% more; on the laptop's 16 KiB pages
  60% more. Quota 196.8 GB of 950 after the job.
- The whole test suite on the login node: 921 passed, 2 skipped (the same
  two), in 4 min 32 s.

**Not checked:** how readable generated crops are, beyond the sheets (no
measure over the pile); reading the piles at a training loader's pace,
from a GPU node (S5); every one of the 78 tall synthetic boxes (25 looked
at); the store pile (S4). The crops-smoke folder (`$GENERATED_DIR/m3/crops-smoke/`, 150 MB)
was left; the task remakes it.

### S4. Store text · owner in Label Studio, then laptop

- `scripts/propose_store_text.py`: one Label Studio task per v1 and v3
  photo, training and validation: each date box with S2's best read (the
  recogniser alone, R1's recipe) as a proposal, a turn choice, and a
  `life` box tool for shelf-life phrases.
- **The owner** checks every text, sets each sideways box's turn, and
  boxes the shelf-life phrase on each derived photo. About 230 boxes, an
  hour or so.
- `scripts/import_store_text.py`: the checks of [Store
  text](#store-text-s4), then `text.json` frozen (`TEXT_SHA256`); its
  numbered copy to the cluster; `make_crops.py --pile store` there (a
  minute in a CPU job), with `store-val` and `store-train` suites.
- **Then S8's role measurement** can turn tall boxes upright.

**Check:** every box has text and a turn; every expiry and mfg text parses
to its frozen value, or is listed and settled with the owner. **Commits:**
`feat: store date text, typed and frozen`, `docs: m3 plan, S4's text`

### S5. The CRNN · laptop, then a CPU smoke

- `reader/charset.py`, `reader/crnn.py`, `reader/data.py`,
  `reader/train.py`, `reader/onnx_run.py`, `scripts/train_reader.py
  --reader crnn --arm all --seed 0`, `scripts/export_reader.py`.
- **`ctc.greedy_decode` and `metrics.cer`** are written by Claude, not
  the owner (the owner's choice, 2026-11-13: a change from this plan's
  first version), against tests written first, then **walked through
  with the owner line by line** (`tests/test_ctc.py`, `tests/test_reader_metrics.py`):
  repeats merged, blanks dropped, a repeated character kept when a blank
  separates it (`2 0 0 -> "20"`, `2 0 - 0 -> "200"`); `cer("08/2027",
  "08/2021") = 1/7`.
- **Tests:** the mix's shares and seeding; design balance; padding and
  jitter stay inside the margin; width buckets pad less than random
  batches; the visibility measure on made-up crops (print at falling
  contrast under fixed noise falls with it; print visible in colour only
  still scores; a crop under the cut-off is never drawn); the CRNN's output frames for a 320 px crop; a tiny overfit (50
  crops, loss near zero); ONNX reads equal PyTorch's.
- **Empty crops left out of training** (moved from Later, the owner's
  go-ahead 2026-11-13, after S3's sheets). About 4 or 5 of every 64
  generated crops show no print at all, even with their contrast
  stretched to full range (the generator checks its ink before the camera
  adds noise, blur and glare), and ExpDate's black synthetic photos give
  36 blank crops; each still carries its text. A reader trained on them
  learns to answer a date from nothing, the wrong read M3 most wants to
  avoid. So:
  - `reader/data.py` gets a **visibility measure** of the stored crop:
    how far the print inside the box stands out from the crop's own
    noise, in colour (S3's `#441001` shows its print in colour only).
    It is computed once per record by a pass over each training pile (a
    CPU job: `m3_cpu.sbatch visibility`) and saved beside its LMDB.
  - **The cut-off is set by eye, before any training:** sheets of
    training crops sorted by the measure, from piles `gen`, `expdate-real`
    and `expdate-synth`, looked at with the owner; the cut-off is where
    the print stops being readable to a person looking hard. Faint,
    blurred but readable print stays: it is what the store's dot-matrix
    and small print look like. Written into `m3.VISIBLE_MIN` with the
    sheets it was chosen on.
  - `mix()` skips training crops under the cut-off. **Suites are never
    filtered** (a reader scored on them is scored on every crop); their
    measure is recorded so a suite can be split by it.
  - **Check:** how many each pile loses (about 5% expected for `gen`; over
    10% of any pile means look again before training), a sheet of the
    dropped crops (nearly all empty) and one of the kept crops just above
    the cut-off (all readable).
- **Smoke:** `m3_cpu.sbatch smoke-crnn` (`short`): 2,000 steps at batch 64
  on `all`; the loss falls; ONNX parity; images a second (the GPU
  estimate's base).

**Commits:** `feat: crnn reader with ctc, the training stream`,
`feat: ctc greedy decoder and character error rate`,
`feat: ctc readers exported to onnx, one pre and post`

### S6. PP-OCRv6: the route and the fine-tune · laptop, then CPU checks

- `reader/ppocr.py`; the route check in `m3_cpu.sbatch ppocr-route` (the
  three conditions under [PP-OCRv6 small](#pp-ocrv6-small)); our ONNX path
  against `paddleocr`'s reads from S2 (99.5%).
- PyTorch route: `train_reader.py --reader ppocrv6`; Paddle route:
  `~/venvs/paddle-gpu`, the config with overrides, the LMDB or label
  files, the ONNX export. Either way a CPU smoke (200 steps) and parity.

**Check:** the route recorded with its evidence. **Commits:**
`feat: pp-ocrv6 small, its onnx path and the fine-tune route`,
`docs: m3 plan, S6's route`

### S7. The VLM: data, training loop, GGUF path · laptop, then CPU checks

- `reader/vlm.py`, `scripts/make_vlm_set.py` (a CPU job:
  `m3_cpu.sbatch vlm-set`), `scripts/train_vlm.py`,
  `scripts/vlm_gguf.sh`.
- **The GGUF path on the untrained model** (`m3_cpu.sbatch vlm-gguf`): our
  conversion against the official GGUF and the HF model, on 50 real-val
  photos: answers and seconds a photo on 24 CPUs. On the laptop, one
  real-val photo through llama.cpp (Metal): seconds and memory.
- **Untrained, on validation** (CPU job): real-val and store-val photos;
  the untrained rows.
- **Tests:** targets from labels, rules and sidecars; panel cuts keep the
  background's date boxes out; JSON answers parsed leniently, a bad date
  is none; the loss mask covers only the answer.

**Commits:** `feat: the vlm set, answers as json`,
`feat: vlm lora training loop`, `feat: vlm scored as gguf through llama.cpp`,
`docs: m3 plan, S7's gguf check`

### S8. Roles, the strip and the pipeline · laptop

- `reader/roles.py`, `reader/pipeline.py`, `scripts/score_pipeline.py`.
- `beside()`'s numbers set on v1-train and v3-train only (the role
  measure above, now with tall boxes turned by S4's turns), recorded; then
  `choose_expiry` on those photos with true reads: the upper bound of the
  rules.
- **Tests:** made-up layouts (glued words, two dates on a line, a range,
  sideways panels, an mfg-only photo, a derived photo); the plausibility
  window; the trace.

**Commits:** `feat: roles that veto the latest date, the shelf-life strip`,
`feat: the m3 pipeline, photo to expiry with its trace`

### S9. The chain and the rules · laptop · *releases the GPU job*

- `m3.py`'s rules, `scripts/m3_chain.py`, `slurm/m3_chain.sh`,
  `pull_cluster_results.sh` (m3 runs: JSON and logs, never crops or
  weights). gen-v2's pile, if S3 left it.
- **Tests:** `tests/test_m3_rules.py` on made-up records (a tie goes to
  `all`, then the faster reader; the band at least 3; R4 per set; R5 and
  R6 on made-up rows); readiness waits for each crop pile, the store text
  and the code.
- Dry runs on both machines; owner: `scontrol release <job>`.

**Commits:** `feat: m3 chain and its decision rules`

### S10. CRNN and PP-OCRv6 results · cluster runs, then laptop

Pull; the band, R2, R3; the crop suites by arm, stratum and tag; the
store-train gap (memorisation); size and milliseconds a crop on the
laptop's CPU (from ONNX, a few hundred real-val crops: light). READMEs of
`m3-crnn-data` and `m3-ppocrv6`. **Commits:** `docs: m3 plan, S10's readers`

### S11. VLM results · cluster, then a CPU job

Merge, convert (GGUF), score validation on CPUs; HF against GGUF; R4 for
the VLM. `m3-vlm`'s README. **Commits:** `docs: m3 plan, S11's vlm`

### S12. End to end on validation · CPU job, then laptop

- `m3_cpu.sbatch e2e`: real-val and store-val photos through the
  pipeline with M2's stand-in detector: R4 per reader, R5, R6; the table
  of where photos were lost; the VLM's photo answers beside it.
- B on v3-val's 5 photos (laptop, Scanify's venv: seconds), for R6's cap.
- **Again once M2b's detector is chosen**: R6 recorded for it.

**Commits:** `feat: m3 operating points`, `docs: m3 plan, S12's validation`

### S13. The final test · laptop (v2), CPU job (ExpDate test) · once

As [The final test](#the-final-test). **Commits:** `docs: m3 plan, S13's test`

### S14. Report · laptop

`m3_report.py`, `results/m3_reader.md`, the experiments' READMEs, the
README section, CLAUDE.md (owner's OK). **Commits:**
`feat: m3 report from the run records`, `docs: m3 results in the readme`

---

## Order and sessions

```
S1 ─► GPU job(s) held, environments
S2 ─► [CPU: crop experiment] ─► R1: padding, turning
S3 ─► [CPU: crop shards] ◄── gen-v2 (M2b S9's job)
S4 (owner) ─► store text frozen ─► [CPU: store pile]
S5 CRNN ─┐
S6 PP-OCRv6 ─┼─► S9 chain ─► release ─► [GPU: CRNN x9, PP-OCRv6 x3, VLM] ─► S10, S11
S7 VLM ──┘
S8 pipeline (after S4)                              S12 [CPU: end to end] ◄── M2b's detector
                                                    S13 [laptop v2, CPU ExpDate test] ─► S14
```

| Session | Slices | Ends with |
|---|---|---|
| 1 | S1 | the GPU job held, the environments ready |
| 2 | S2 | the crop experiment read, R1 recorded |
| 3 | S3 | the crop shards checked |
| 4 | S4 (with the owner) | store text frozen, the store pile |
| 5 | S5 | the CRNN smoke passed |
| 6 | S6 | PP-OCRv6's route proved |
| 7 | S7 | the GGUF path proved, the untrained VLM scored |
| 8 | S8 | the pipeline on training photos |
| 9 | S9 | the GPU job released |
| 10 | S10, S11 | R2, R3, the VLM |
| 11 | S12 | operating points |
| 12 | S13, S14 | M3's results |

S2 to S8 run while the GPU job waits. S8 needs only S4. S12's second pass
and S13 wait for M2b's detector.

## Compute budget

| GPU work | Runs | Each | GPU hours |
|---|---:|---|---:|
| CRNN smoke | 1 | 5 min | 0.1 |
| CRNN, 3 arms × 3 seeds | 9 | about 0.6 h (10M crops; the loader the limit) | 5.4 |
| PP-OCRv6 fine-tune | 3 | about 0.5 h | 1.5 |
| VLM smoke and fine-tune | 2 | 0.2 h; about 3 h (25,000 examples, about 700 tokens each) | 3.2 |
| scoring (crop suites on the job's CPUs, VLM validation on the GPU) | 14 | 3-10 min | 1 |
| **total** | | | **about 11, under one 3-day job** |

| CPU work (`short`) | Estimate |
|---|---|
| S2 crop experiment, about 7,000 reads | minutes |
| S3 crops, about 700,000 | under an hour (took 6 minutes) |
| S5 visibility measure, about 694,000 training crops | minutes |
| S7 VLM set (25,000 images) and GGUF checks | under an hour |
| S11 VLM validation through GGUF, 199 photos | about 30 minutes |
| S12 end to end, 199 photos, every reader | under an hour |
| S13 ExpDate test, 665 photos (pipeline and VLM) | about 2 hours |

**Laptop:** S13's v2 pass, about 15 minutes; S10's milliseconds a crop,
seconds. Nothing else heavy.

**Storage** on the cluster: crops about 4 GB, the VLM set about 3 GB, the
merged VLM 4.5 GB and its GGUF 2.6 GB, CTC checkpoints under 0.3 GB: about
15 GB of 950.

**Optimisations, fixed now:**

- Crops cut once into LMDB, padding taken from the margin at load time:
  no file per crop, no re-cut for a new padding.
- Batches by width; bf16 on the CRNN's convolutions; pinned, persistent
  loader workers.
- The detector run once per photo at 0.001, so R6's threshold sweep only
  filters; every box read once per reader.
- Two reads per crop for orientation, not four (wide or tall decides).
- The VLM: visual tokens capped at about 1,875, LoRA with gradient
  checkpointing, ExpDate photos at their own size.
- Scoring through ONNX Runtime and llama.cpp on CPUs; GPU time only for
  training and the VLM's HF validation.
- B's v1-val numbers from M0's records, not re-run.

## Files at the end of M3

```
docs/              m3-date-reader.md; datasets.md gains the PP-OCRv6 and Qwen files
src/bestbefore/    m3.py, m3_results.py, reader/ (charset, crops, shards, piles, data, ctc,
                   metrics, crnn, train, ppocr, onnx_run, parse, roles, pipeline, vlm);
                   changed: store.py, store_train.py, sets.py
scripts/           export_store_text.py, crop_experiment.py, make_crops.py, crop_sheet.py,
                   propose_store_text.py, import_store_text.py, train_reader.py,
                   export_reader.py, score_reader.py, make_vlm_set.py, train_vlm.py,
                   vlm_gguf.sh, score_pipeline.py, m3_chain.py, m3_report.py;
                   changed: run_baseline.py, pull_cluster_results.sh
slurm/             m3_gpu.sbatch, m3_chain.sh, m3_cpu.sbatch
experiments/       m3-crop-reading/, m3-crnn-data/, m3-ppocrv6/, m3-vlm/, m3-pipeline/
tests/             the files under Code layout
results/           m3_reader.md (committed; runs/ is not)
```

## Risks

- **The detector holds the reader back.** If M2b doesn't lift store
  recall, no reader beats B end to end. The reader-alone tables and the
  table of where photos were lost show it plainly.
- **Store crops are few** (167 to train on, 62 to choose with). The band,
  the public suites for checkpoints, seeds, and choices only where the
  numbers can carry them.
- **Generated print looks pasted** (M2's lesson: synth-val high, the store
  flat). store-val judges; crop sheets are looked at.
- **PP-OCRv6's fine-tuning route.** Proved in S6 before any GPU time.
- **The fine-tuned VLM may not convert to GGUF.** Proved on the untrained
  model in S7; the fallback is the owner's choice, written before v2.
- **The VLM invents dates** on photos without one. Negatives in training;
  `other-face` reported apart.
- **Misread digits** (`8`/`3`/`6`/`0`, `1`/`7`) make wrong dates. The
  least-sure-character confidence and R4's cap; a date grammar in the
  decoder is in Later.
- **ExpDate targets for the VLM come from a rule**, which may disagree with
  ExpDate's convention. Measured once on ExpDate test's marks.
- **Typos in the store text.** The import checks every expiry and mfg text
  against its frozen value.
- **A photo without EXIF** has no month: the plausibility check is off for
  it, counted.
- **Leaks.** v2 never leaves the laptop; ExpDate test is cut only at S13;
  Date-Real and Components unused; v1-val and v3-val never trained on;
  v2 split by design seen or unseen.
- **The queue and the 5-job cap.** S1's rule leaves a slot for CPU jobs,
  which are short.
- **Laptop heat.** Only v2's 15-minute pass runs there.
- **One VLM seed.** Its spread is unknown; said beside every VLM number.

## Later

- **A date grammar in decoding**: beam search over CTC's outputs kept to
  date-shaped strings, if wrong reads are mostly impossible dates.
- **Distillation**: the fine-tuned VLM labels unlabelled store photos
  (pseudo-labels) to train the small reader; never v2.
- **A 4-way orientation classifier**, if reading both ways costs too much
  on the phone or misjudges.
- **A learned role model** (positions of dates and keywords), if the veto
  rule loses expiries.
- **Oriented boxes from the detector**, and tiled reading of small print.
- **The VLM on detector crops** (detector-assisted), and **Qwen3-VL-4B or
  8B**, if 2B shows promise.
- **PP-OCRv6 tiny and medium**, a size sweep for M5.
- **The store share swept** (0.5% fixed now), and gen50k against gen-v2 for
  reading.
- **Date-Synth and Components**, if the CRNN lacks variety and the leak can
  be ruled out.
- **TrOCR**, only to answer a question the CTC readers can't.
- **Frame voting** (M6): a date accepted once several frames agree.
- **int8 quantisation** of the reader (M5).
- **The generator's ink check after its camera** (gen-v3, if ever made):
  S3 found blocks that pass the check and are then buried by camera noise
  and blur. S5's visibility filter handles it for M3's crops.
- **Piles copied to a node's own disk** for training, if reading LMDB on
  Lustre holds S5's loader back (S5 measures the loader's pace first).
