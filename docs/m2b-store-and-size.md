# M2b: Store photos, the generator's gaps and a bigger detector

Planned 2026-10-31. Status: **planned**; S1 done 2026-10-31 (GPU jobs A
`73831` and B `73832` held, before any code). Next: S2.

**Goal.** Find more of the store's expiry dates than M2 does, without
forgetting ExpDate's, in four ways at once: train on the store photos
themselves (store set v1, now allowed), draw the layouts the generator has
never drawn, try a bigger model, and choose the threshold on photos that
look like the store's.

**Done when.** `results/m2b_detector.md` reports, from the run records
alone:

- **model size**: `n`, `s` and an `m` probe at 1,280 px, on public data
  (step 1) and after store fine-tuning (step 2);
- **the fine-tuning recipe**: replay against two-stage, next to the
  no-training control (M2's detector with a store-chosen threshold);
- **what the generator's new layouts add**, on their own synthetic
  strata and on real print turned 180°;
- **inference add-ons** (seed ensemble, test-time rotation, 1,536 px),
  each kept or dropped by a rule;
- **the M2b detector, scored once on store set v2**, by failure tag, by
  cause and by whether its pack design was in training, next to M1 and M2
  scored on v2 the same way, and once on ExpDate test.

M2b still judges **finding** dates only. Reading them is M3's job, and the
M2b detector is what M3 reads from.

---

## What M2 and the store gallery tell us

### The gap after M2

On store set v1 (81 expiry dates in frame), each run at its own threshold:

| Expiry dates found | ExpDate test | Store v1 | Boxed at any confidence |
|---|---:|---:|---:|
| M1 detector (`n640`, seed 0) | 637 / 665 | 18 / 81 | 43 |
| M2 detector (`gen-rot-1280`, seed 0) | 662 / 665 | 39 / 81 | 69 |
| `gen-rot-1280`, mean of 3 seeds (range) | 662.7 | 46 (39–53) | 70.3 (69–72) |

By tag (`gen-rot-1280`, mean of 3 seeds): dot-matrix 19 of 20, clean 19 of
31, small print 5.3 of 15, **sideways 2.7 of 15**. Precision 0.910 (M1
0.939). ExpDate is solved; the store isn't.

### Cause 1: seen but doubted

For the M2 detector, of the 81 expiries: **39 kept, 30 boxed below its
threshold (0.342), 12 never boxed** at any confidence (`store_gallery.py`,
the owner's look and Claude's, 2026-10-28). The threshold was chosen on
ExpDate's validation sets, where nearly every date is found with high
confidence; store dates are less sure.

- **Month words are most of the doubted.** 20 of the 30 below the
  threshold have the month as a word. Over all 36 month-word expiries: 9
  kept, 20 below, 7 never boxed.
- **The same print swings between photos.** One pack's `Best before:`
  date gets 0.84 in one photo and 0.010, 0.002 and nothing in three
  others; another pack's `January 2028` gets 0.75 and 0.084. The model
  sits near its decision boundary on this kind of print.
- **Seeds disagree a lot.** 55 of the 81 are found by at least one seed,
  37 by all three, 7 boxed by none. Seed 0, the M2 detector, is the
  weakest seed on full month names (0.02 to 0.08 where seeds 1 and 2 give
  0.6 to 0.85).

### Cause 2: layouts the generator never draws

Seen in the gallery, and absent from every training photo:

- **upside-down print (180°)**: 2 of the 15 sideways-tagged expiries are
  turned a half turn (their boxes are wide, not tall). One is large, crisp
  and clean, and no seed gives it more than 0.005. The generator turns
  blocks 0°, 90° or 270° only, and `rot`'s copies 90° or 270°;
- **a date glued to its keyword**, no space (`Best before:03-03-2027`,
  `EXP.FEB 2027`);
- **two dates on one line** (`MFG:FEB 2025 | IMPORT: MAR 2025`), where the
  detector draws one box across both;
- **dense stacked panels**: price, unit price, mfg, import, expiry and
  batch lines with tight spacing;
- **light laser marks on a dark printed window**;
- **barcode digits and unit prices** next to dates, the commonest false
  boxes, which no generated distractor imitates;
- **missing dot rows** (a clogged print nozzle) in dot-matrix print.

The import date's keyword is missing too: no generated word mentions
import, while imported packs print one.

### Cause 3: few designs, and the look of real print

- **A few pack designs dominate v1.** By eye, four designs give about 30
  of the 81 expiries; one pouch design alone gives about 10, and 6 of the
  12 never-boxed. v1 holds far fewer independent cases than 81. This
  matters for splitting v1 and for reading v2 (S4 counts it).
- **Store print isn't fainter; it's smaller and softer.** Measured inside
  each expiry box (2026-10-28, a scratch script; S3 makes it an
  experiment): contrast (Otsu split of luminance, median) 104 for kept,
  106 for doubted, 81 for never boxed, against 122 for ExpDate validation
  and 69 for generated dates. Only 1 of 81 store expiries is under 40,
  against 21% of generated dates. The doubted are softer (sharpness 0.34
  against 0.42 for kept; ExpDate validation 0.44, generated 0.32) and
  smaller (31 against 68 px on their short side), and small text looks
  soft, so the two are tangled.
- **Sideways dates are soft too**: 12 of the 15 are softer than the kept
  median, none is faint.

### Checked and refuted (don't re-learn them)

- *"Manufacture dates are found, expiries doubted."* Over 3 seeds, labelled
  manufacture dates are found 57.4% of the time, expiries 56.8% (upright:
  64.7% and 65.7%). Looking only at misses had suggested otherwise.
- *"Later years are doubted."* Every manufacture date is 2026 or earlier
  and found no more often (57.4%) than expiries from 2027 on (50.2%).
  ExpDate's real training dates are mostly 2020 to 2023 (1,065 of 1,244)
  and only 21 are 2027 or later, but the store shows no year effect: the 12
  early expiries found 94% of the time are mostly numeric (8) and
  dot-matrix (5), the easy kinds.
- *"More of the same generated data."* M2's data-size arm: 5 times more
  distinct generated photos gave +1.6 points on the hard sets and no
  visible change on the store (38.3 against 37 of 81).
- *"More faded or blurred generated print."* Generated dates are already
  fainter and softer than the store's on both measures above.

### Model size: what M1 measured, and what it couldn't see

| M1 run (ExpDate data only) | Real-val mAP@0.5 date | Minutes | Epochs |
|---|---:|---:|---:|
| `n640` | 0.991 | 65.5 | 100 |
| `n1280` | 0.994 | 96.5 | 96 |
| `s640` | 0.995 | 61.6 | 100 |
| `s1280` | 0.995 | 102.3 | 87 |

ExpDate validation was too easy to separate them, so M1's rule took the
cheapest. That shows nothing about store print. `s` costs little here:
`s1280` took 1.18 minutes an epoch against `n1280`'s 1.01. On COCO, a hard
and varied set, `s` beats `n` by about 7.5 mAP. And nano already fits what
it's shown (real-val 0.995, synth-val 0.994, real-val-rot 0.988), so size
should help with the fragility above, not with layouts it never saw.

---

## How M2b closes the gap

| Cause | What M2b changes | Measured on |
|---|---|---|
| seen but doubted | threshold chosen on store validation photos; store photos in training | v1-val |
| upside-down, glued, two per line, dense panels, laser windows, nozzle rows | gen-v2 draws them; ExpDate's training photos also turned 180° | synth-val-v2 strata, real-val-rot180, v1-val |
| import keyword, barcode digits, unit prices | gen-v2's words and distractors | synth-val-v2, store precision |
| the look of real print, few designs | store set v1 in training, design-balanced, turned 4 ways | v1-val; v2 split by seen and unseen designs |
| fragile features | `s`, and an `m` probe | every set; seed spread |
| unstable seeds | seed ensemble (by rule) | v1-val |
| embossed print, never measured | an embossed stratum | synth-val-v2 |

## What we expect (written before any run)

| Run | v1-val mAP (dates) | Store sideways | Real-val-rot180 | Store month words | Real-val |
|---|---|---|---|---|---|
| `n1280` (M2's weights, no store data) | the baseline | low | about 0 | doubted | 0.995 |
| `n1280` at a v1-val threshold | the same (mAP ignores the threshold) | a little up | about 0 | more kept, more false boxes | the same |
| `s1280` (step 1) | a little up | a little up | about 0 | a little up | as `n` |
| `m1280` (step 1, probe) | about as `s` | | | | as `n` |
| `n1280-ft` (step 2) | clearly up | up | up a lot | up | as `n` |
| `n1280-ft2` (two-stage) | as `ft`, perhaps lower | | | | as `n`, perhaps higher |
| `n1280-ft-gen50k` (side arm) | as `ft` | | as `ft` | | as `n` |
| `s1280-ft` | the highest | | | | as `n` |

On v2: designs seen in training well above unseen ones, and unseen ones
still above M2's detector. If fine-tuning lifts v1-val but not v2's unseen
designs, the model learned packs rather than print: that is a finding, and
it gets written up as one.

---

## What we start from

| Fact | Where it comes from | What it means for M2b |
|---|---|---|
| Store set v1: 99 photos, 81 expiries in frame, frozen (`7b9a4043…`) | M0 B3 | becomes training and validation data |
| Store set v2: 99 photos in `testset2/`, photo list frozen (`8be1f9e9…`); labelled and imported 2026-10-31 (74 with an expiry, 8 of them derived; 24 `other-face`; 1 unreadable); label freeze (D6) waits for the owner | M0 Part D | the test set; frozen before any store training |
| Permission of 2026-10-28: store photos may be trained on and shown to AI tools; never committed or published | `CLAUDE.md` | v1 goes to the cluster for training; nothing about a photo is committed |
| M2's 18 runs and their `last.pt` on the cluster; `gen-rot-1280` s0, s1, s2 are `n1280`'s step 1 | M2 S8 | no `n` retraining |
| `gen-rot-1280` trained in 202–204 min at 211 images/s; 640 px arms at about 363 images/s | M2 S9 | step 1's estimates |
| M2's two GPU jobs never ran together: job A did all 40.5 GPU-hours alone | M2 S9 | plan for one GPU at a time |
| Generation: 52,100 photos in 328 s on 24 CPUs, 185 kB each | M2 S6 | 200,000 cost about 25 minutes and 37 GB |
| `derived.turn` turns by any quarter, 180° included, exactly | `derived.py` | the 180° copies need no new geometry |
| `detection_eval` has no AP function; Ultralytics' mAP needs a single model | `detection_eval.py` | ensembles need our own AP (S13) |
| RapidOCR 1.2.3 is installed; `ensemble-boxes` isn't | the laptop venv | OCR can propose keyword boxes; WBF needs a pinned dependency |
| `m2.WEIGHTS = "yolo11n"` is fixed for every M2 arm; `GenSpec`'s hash (`3fc14178…`) is in M2's manifests | `m2.py`, `synth/spec.py` | M2b adds sizes and spec fields without changing M2's records |
| GPU queue at 21:00 on 2026-10-31: all 4 H100s busy until about 2026-11-03 to 11-06; 16 GPU jobs from 5 users wait ahead of A and B, most for 7 days | `squeue` | at worst (every job to its limit) A starts about 2026-11-27; jobs often end early |
| The laptop and GitHub have diverged: GitHub has `8843e48` (the job script) and two README edits; the laptop has `dcfe1be` | `git log` | merge before S2's push |

---

## Decisions

Each comes with its reason. **(new)** marks a choice not written down
before; **(changes Mn k)** a departure from an earlier milestone's
decision k.

1. **(new) Two steps for every size.** Step 1 trains on public data with
   M2's recipe and `gen-rot-1280`'s mix; step 2 fine-tunes on the store
   photos with replay. `n`'s step 1 is M2's `gen-rot-1280` (s0, s1, s2),
   reused, not retrained. Each size gets the same two steps, so size is
   compared at both stages; step 1 uses no store photo, so it starts
   first.
2. **(changes M2's fixed `yolo11n`) Sizes: `n` as the comparison, `s` as
   the main model (3 seeds), `m` as one probe** that gets 3 seeds only by
   rule R1. Not `l` or `x`: on COCO, `l` adds under 2 points over `m` at
   more cost, and with little real store data very large models overfit.
   M2 kept `n` to isolate the data's effect; that question is answered.
3. **(changes M2 17; `CLAUDE.md` 2026-10-28) Where store photos live.**
   v1, both its training and validation parts, goes to the cluster as
   exported copies (upright, numbered, no original names), because
   training and the chain's decisions run there. v2 never leaves the
   laptop: the test set stays where no training job can reach it, and is
   scored on the laptop's CPU, as M2 scored v1.
4. **(new) Pack design ids for v1 and v2, frozen before any store
   training.** The owner types one per photo: same brand, product line and
   date-print layout is one design. They decide the v1 split and v2's
   seen-or-unseen report. They are frozen with a hash, as the labels are,
   and stay in `data/store/`.
5. **(new) v1 split by design, 70% training and 30% validation**,
   stratified by tag (rule in [The data](#the-data)). Splitting by photo
   would put the same pack on both sides, and validation would look far
   better than it is.
6. **(new; M0 Later) Keyword boxes on v1** (`due`, `prod`, `code`),
   proposed by RapidOCR and checked by the owner in Label Studio. Unboxed
   `EXP` and `MFG` words would be learned as background, and M3 needs
   them. Box rule: the generator's (the word or phrase as printed, with
   its colon).
7. **(new) Store training photos: upright, at most 1,600 px, turned 0°,
   90°, 180° and 270°, listed design-balanced at 20% of every epoch.**
   Upright: Ultralytics reads files without their EXIF turn. 1,600 px: no
   input is above 1,280, as M2's backgrounds. Turned: real sideways and
   upside-down store print, exactly (M2 decision on `rot`). Design-balanced:
   every design weighs the same, so the pouch seen 10 times isn't
   memorised. 20%: about three times M2's 7% share of real photos,
   because store photos are the target.
8. **(new) The replay mix.** The other 80% keeps `gen-rot-1280`'s
   proportions: ExpDate real and its turned copies 5.7%, ExpDate synthetic
   and its turned copies 23.9%, generated 50.4%. ExpDate's training photos
   also get 180° copies, inside the turned share. Each run draws its own
   seeded sample of 25,000 photos per epoch.
9. **(new) The fine-tuning recipe.** From each seed's own step-1 `last.pt`
   (s0 from s0, never all from the seed that scored best on v1); the whole
   network; M2's recipe with `lr0=0.001` (a tenth); 10% of M2's budget
   (256,000 images, about 10 epochs); warmup and mosaic-off at M2's shares
   (3%, 10%). Fine-tuning moves a trained model a little; a tenth of the
   rate and budget is the usual starting point, and costs about 20 minutes
   a run.
10. **(new) One comparison arm: two-stage** (backbone frozen, `freeze=11`,
    for half the budget; then everything at half the rate for the other
    half). It is "train the head, then fine-tune", known to keep a model
    robust on new kinds of photos; its first stage is the frozen-layers
    result. Run on `n` only; rule R2 carries the better recipe to `s`.
11. **(new) One side arm: `n1280-ft-gen50k`**, `ft` with M2's 50,000
    generated photos instead of gen-v2, to measure what the new layouts
    add. Not a candidate for the detector, as M2's `gen10k` wasn't.
12. **(new) The no-training control.** Every step-1 model is also scored
    at a threshold chosen on v1-val; M2's three `n1280` seeds at that
    threshold are the control. Without it we couldn't tell how much of a
    gain is the threshold alone.
13. **(changes M2 11) Threshold: best F2 over v1-val's labelled dates**
    (expiry, mfg and other), per run. Store photos set it because store
    photos are what it's for. M2's pooled real-set threshold is recorded
    beside it.
14. **(changes M2 12) Choosing: mean mAP@0.5 for dates on v1-val**, with a
    no-forgetting guard on real-val and synth-val, and a noise band
    **measured before any M2b run is scored** (the control's seed spread,
    at least 0.03). Within the band, the cheapest wins. Rules R1 to R4.
15. **(new) gen-v2: 200,000 photos with every gap filled**, shares fixed by
    rule: each new kind about 10%, taken from upright and plain blocks;
    the month-word, small-print and dot-matrix shares unchanged. M2's
    spec keeps its hash and M2's photos their bytes (tests). **(changes
    M2's rule against leaks)** M2 designed nothing from a store photo,
    because v1 was its test set. gen-v2's new kinds come from v1's
    failures, now training data, and never from v2.
16. **(new) synth-val-v2 and real-val-rot180** join the suite; M2's four
    sets stay as they are.
17. **(new) Inference add-ons decided on v1-val** by R4: the 3 seeds
    merged by weighted boxes fusion, test-time rotation (0°, 90°, 180°,
    270°), and a 1,536 px input for 1,280 px models. Each costs
    inference time, so each must earn its place.
18. **(new) v2 is scored once, at the end**, for M1's `n640`, M2's
    `gen-rot-1280` (at M2's and at v1-val thresholds) and every M2b run;
    by tag, by cause and by design seen or unseen. ExpDate test once, for
    the M2b detector.
19. **(new) GPU work first.** Job A is released at S2 for step 1; job B
    stays held until step 2's data exists, so step 2 never waits behind a
    new job at the back of the queue.

Left out of M2b, with reasons, in [Later](#later): knowledge
distillation, hard-example mining, pseudo-labels, a mask of the pack,
k-fold validation, `l` and `x`, a full retrain on gen-v2.

---

## The data

### Roles

| Data | Role | Where |
|---|---|---|
| v1, about 70% of its designs | **training**: turned 4 ways, design-balanced | laptop, cluster |
| v1, the other 30% of designs | **validation** (`v1-val`): thresholds and every choice; never augmented, never trained on | laptop, cluster |
| v2 (`testset2/`) | **test**, once, at the end | laptop only |
| ExpDate train (real, synthetic), turned 90°/270° and 180° | training (replay) | both |
| gen50k (M2), gen-v2 (new) | training (replay) | cluster (gen-v2's sample also on the laptop) |
| real-val, real-val-rot, real-val-small, synth-val (M2's suite) | validation: no forgetting | both |
| real-val-rot180, synth-val-v2 (new) | validation: the new layouts | both |
| ExpDate test | test, once, the M2b detector | laptop |

### Design ids

`data/store/designs.csv`, typed by the owner: one row per photo of v1 and
v2, `photo,design`. One design is one brand, product line and date-print
layout: two flavours of a product with the same print in the same place
are one design. The owner names designs freely and consistently across v1
and v2, looking at the photos, never at a model's output. Frozen into
`data/store/designs.json` with `DESIGNS_SHA256` in `store.py`, before any
store training. Committed files hold counts only.

### The split

```python
def split_v1(designs: dict[str, str], labels: dict[str, dict], seed: int = 0) -> dict[str, str]:
    """photo -> "train" or "val". Each design takes its photos' commonest
    tag (other-face only if every photo is). Within each tag, the designs,
    sorted by id and shuffled with `seed`, give round(0.3 x n) of themselves
    to validation (at least 1 where a tag has 2 or more designs)."""
```

Written before the split exists; its counts go into
`results/store_designs.md` (designs in v1, in v2 and in both; photos,
expiries and dates per part and tag). v2 plays no part in the split.

### Keyword boxes

Proposed by `scripts/propose_keywords.py`: RapidOCR's text boxes whose text
matches a keyword pattern (`EXP`, `Expiry`, `Best Before`, `Use by`, `MFG`,
`Mfd`, `Manufactur`, `PKD`, `Packed`, `Import`, `Batch`, `B.No`, `Lot`,
times), loaded into a new Label Studio project as predictions, beside v1's
date boxes for reference. The owner fixes and submits each of the 99.

| Class | Boxed | Examples |
|---|---|---|
| `due` | the expiry word or phrase as printed, with its colon | `EXP.`, `Best Before:`, `Expiry Date :`, `USE BY` |
| `prod` | the manufacture, packing or import word or phrase | `MFG:`, `Mfd(Date):`, `Month and year of manufacture:`, `IMPORT:` |
| `code` | a batch or lot code with its label, or a time | `Batch No.: K2311A`, `B.NO: 2307B`, `14:32` |
| no box | prices, quantities, phone numbers, barcode digits, addresses, `12 months from Mfg.` | `MRP: ₹450.00`, `USP Rs. 1.50/g` |

Imported into `data/store/v1_keywords.json` with checks: every box inside
its photo; no keyword box overlapping a date box at IoU 0.5 or more; every
photo reviewed. v1's frozen label file doesn't change.

### The export

`scripts/export_store_yolo.py`, into `$STORE_TRAIN_DIR` (laptop
`data/store/train/`, cluster `~/bestbefore-data/store/`):

- every v1 photo upright (EXIF applied), shrunk to at most 1,600 px with
  an area filter, JPEG quality 95, named `v1-0001.jpg` on; the map back to
  photo names stays in `data/store/train/names.json`, on the laptop only;
- label lines: expiry, mfg and other dates as `date` (class 0), keyword
  boxes as their class;
- v1-train also at 90°, 180° and 270° (`derived.turn`); v1-val as it is;
- lists of v1-train's exported photos (all 4 turns) and of v1-val's; each
  training run's design-balanced lines are built from the first by
  `m2b.mix()`; a manifest with the counts and the hashes of the design,
  keyword and label files;
- it refuses a v2 photo, and refuses to run while `TEST_SET_V2_SHA256` is
  unset.

On the cluster the folder is `chmod 700`. Ultralytics' training-batch
mosaics land beside the weights there and are never pulled.

---

## The generator, v2

M2's generator unchanged, plus one new spec field: `GenSpec.v2`, a
`V2Spec`, `None` for M2's spec and left out of its hash, so `GEN` keeps
`3fc14178…`, M2's strata keep `ba77f1bd…`, and M2's photos come out byte
for byte as before (tests). Every v2 feature draws its random numbers
only when it is on.

| Change | Rule | Share |
|---|---|---|
| upside-down blocks | 180° ±8°, as sideways | orientation: upright 45%, tilted 15%, sideways 30%, **upside-down 10%** |
| glued keyword | the word ends in `:` or `.` and the date follows with no space | 30% of lines with a word before a date |
| two dates on one line | `MFG:FEB 2025 \| IMPORT: MAR 2025`: two word-and-date pairs, separated by ` \| `, two spaces, `/` or `,` | kind **line-pair 10%** |
| dense panels | a pair plus 3 to 5 extra lines (codes, distractors, an import line half the time), line spacing 1.0 to 1.2 em | kind **dense-panel 10%** |
| import keyword | `IMPORT:`, `Imp.`, `Date of Import:`, `Imported on:`, `Month & Year of Import:`, boxed `prod`; its date's role `import` | in line-pairs (half) and dense panels (half) |
| new distractors | 13-digit barcode numbers, run together or spaced 1-6-6 (`8901234567890`, `8 901234 567890`), unit prices (`USP Rs. 1.50/g`, `Unit Sale Price: Rs. 1.5 per ml`) | among distractor lines |
| nozzle rows | 1 or 2 whole rows of dots left out, the same rows in every line of the block | 15% of dot-matrix blocks |
| laser window | a dark printed rectangle (near-black, dark blue, dark green, dark brown) with light grey to white strokes, slightly soft, a faint glow; 30% dotified | style **window 10%** |

Kinds: single 25%, pair 30%, panel 15%, dense-panel 10%, line-pair 10%,
range 5%, relative 5%. Styles: solid 30%, dot-matrix 35%, label 15%,
embossed 10%, window 10%. Everything else is M2's spec. `₹` is printed
only if every font has the glyph (S7 checks; otherwise `Rs.`).

Boxes still come from the ink drawn (M2 decision 4): a glued date's box
stops where its ink starts; a line-pair gives two date boxes and two
keyword boxes; a dropped dot row stays inside its token's box, which keeps
the token's remaining ink.

**synth-val-v2**: 2,100 photos, 300 per stratum, held-out fonts on
validation backgrounds, one target each, set seed 5. Each stratum changes
one thing from M2's `clean`: `upside-down`, `glued`, `line-pair` (the target
is one of the line's two dates), `dense-panel`, `window`, `embossed`, and
`nozzle-rows`, which changes one thing from M2's `dot-matrix` stratum.
**gen-v2**: 200,000 photos, set seed 4; **gen-v2-sample**: 500, set seed 6,
to look at.

**real-val-rot180**: the 165 validation photos turned 180° (187 dates).
**real_rot180** and **synth_rot180**: ExpDate's training photos turned
180°, for the replay.

---

## Training

### Step 1: public data

`s1280` and `m1280` are M2's `gen-rot-1280` arm with `yolo11s` and
`yolo11m`: the same lists, mix (79,310 photos listed), budget (2.56 million
images, 32 epochs), recipe and seeds. Only the weights change.

| Run | Weights | Seeds | Estimate per run |
|---|---|---:|---:|
| `n1280` | M2's `gen-rot-1280` (no training) | 0, 1, 2 | — |
| `s1280` | `yolo11s` (COCO) | 0, 1, 2 | 4 h (1.17 × `n1280`'s 203 min) |
| `m1280` | `yolo11m` (COCO) | 0; 1, 2 only by R1 | 7 h (measured by seed 0) |
| `m2b-smoke` | `yolo11s`, 1 epoch, 2% of the mix | — | 5 min |

Each is scored on M2's suite at once (an early read), and on the M2b suite
once all its sets exist.

### Step 2: store fine-tuning

```python
STORE_SHARE = 0.20            # store photos' share of every epoch (decision 7)
EPOCH_PHOTOS = 25_000         # photos listed per epoch: the replay sample plus the store lines
BUDGET_FT = 256_000           # images per fine-tuning run: 10% of M2's BUDGET
FT_RECIPE = {**m2.RECIPE, "lr0": 0.001}

def mix(arm: StepTwo, pools: Mapping[str, list[Path]], designs: Mapping[str, str],
        seed: int) -> dict[str, list[Path]]:
    """One run's lists: the store lines, design-balanced over v1-train's
    designs and its 4 turns (STORE_SHARE of EPOCH_PHOTOS), and a sample of
    each replay pool in gen-rot-1280's proportions, drawn with `seed`
    (real 5.7%, synthetic 23.9%, generated 50.4% of the epoch)."""
```

| Arm | Starts from | Generated pool | Freezing | Runs |
|---|---|---|---|---|
| `n1280-ft` | `gen-rot-1280-s{k}` | gen-v2 | none | s0, s1, s2 |
| `n1280-ft2` | the same | gen-v2 | `freeze=11` for half the budget, then none at `lr0=0.0005` | s0, s1, s2 |
| `n1280-ft-gen50k` | the same | M2's gen50k | none | s0, s1, s2 |
| `s1280-<R2's recipe>` | `s1280-s{k}` | gen-v2 | as R2 chose | s0, s1, s2 |
| `m1280-<R2's recipe>` | `m1280-s{k}` | gen-v2 | as R2 chose | only if R1 passes |

Ultralytics' frozen layers still update their batch-norm statistics in
training mode. S10 measures how far they move during stage 1 and records
it; nothing is changed for it.

---

## The validation suite, M2b

Every run, M2's three `n1280` seeds included, is scored on the **M2b
suite** (`scripts/score_suite.py --suite m2b` → `suite-m2b.json`): M2's four
sets, real-val-rot180, synth-val-v2 (per stratum) and **v1-val**. Per set,
as in M2: found at IoU 0.5 and 0.3, boxed at any confidence, precision,
false boxes per photo, Ultralytics' mAP@0.5 per class. Then:

- `threshold_store`: best F2 over v1-val's dates (decision 13);
- `threshold_m2`: M2's rule, best F2 over the three real sets pooled.

Counts only: per-photo boxes on v1-val stay in `data/store/runs/` on the
laptop when the laptop scores them, and aren't kept on the cluster.

### Decision rules

Written before any M2b run, as code in `m2b.py`, applied by the chain and
re-checked by each experiment's `summarise.py`, as in M2.

```python
NOISE = 0.01               # real-val and synth-val, as M2
NOISE_STORE_MIN = 0.03     # v1-val: one date of about 60 is about 1.7 points

def noise_store(control: Sequence[dict]) -> float:
    """The noise band on v1-val: the spread (highest minus lowest) of
    mAP@0.5 for dates over the control's three seeds (M2's n1280, no
    store training), at least NOISE_STORE_MIN. For means of 3 seeds, the
    range is about a two-sigma band. Fixed once the control is scored,
    before any M2b run is."""

def guard(suites: Sequence[dict], control: Sequence[dict]) -> bool:
    """No forgetting: mean mAP@0.5 for dates on real-val and on synth-val,
    each no more than NOISE under the control's."""

def probe_m(m: dict, s: Sequence[dict], control: Sequence[dict]) -> bool:
    """R1. m1280 earns 2 more seeds and step 2 only if seed 0's v1-val
    mAP@0.5 beats every s1280 seed's, and it passes the guard."""

RECIPES = ("ft", "ft2")           # simplest first

def decide_recipe(n_runs: dict[str, list[dict]], band: float, control) -> str:
    """R2, on n. Eligible: the guard. Then the highest mean v1-val
    mAP@0.5; any recipe within `band` of it goes to the simplest."""

CANDIDATES = ("n1280", "n1280-ft", "s1280", "s1280-ft", "m1280", "m1280-ft")
#             cheapest first: inference cost, then training cost;
#             "-ft" stands for R2's recipe (ft or ft2)

def decide_detector(suites: dict[str, list[dict]], band: float, control) -> str:
    """R3. Only candidates with all 3 seeds scored take part (without R1,
    m1280 has one). Eligible: the guard. Then the highest mean v1-val
    mAP@0.5; any candidate within `band` of it goes to the cheapest. The
    M2b detector is that arm's seed 0, at its threshold_store, unless R4
    adopts the ensemble."""

ADDONS = ("ensemble", "rotation", "imgsz-1536")    # tried in this order

def adopt(before: dict, after: dict, band: float) -> bool:
    """R4, on the laptop. An add-on stays if it raises v1-val AP@0.5 for
    dates (our own, detection_eval.average_precision) by more than `band`
    over what is adopted so far, and real-val AP doesn't fall by more than
    NOISE. The threshold is chosen again (F2 on v1-val) for what is
    adopted."""
```

mAP is the measure because it doesn't depend on a threshold, as in M2.
With about 60 dates in v1-val, one date moves AP by about 1.7 points, so
the band can't be M2's 1 point. The no-training control competes too: if
fine-tuning doesn't beat M2's weights by more than the band, the rule
keeps M2's weights at a store-chosen threshold.

**Reading the side questions** (fixed now, no rule):

- **Model size:** `n1280` against `s1280` (against `m1280`) at step 1 and
  step 2, on every set, with the seed spread.
- **gen-v2:** `n1280-ft` against `n1280-ft-gen50k`, on synth-val-v2 per
  stratum first, then real-val-rot180 and v1-val.
- **Freezing:** `n1280-ft` against `n1280-ft2` (R2 decides; the table shows
  how far apart they are).
- **The threshold:** M2's detector at M2's threshold against at
  `threshold_store`, on v1-val and, once, on v2.

---

## The chain

`scripts/m2b_chain.py`, on the machinery M2's chain already has (claims,
watchdog, time left, failures, replacements, dry run), moved into
`bestbefore/chain.py` first with M2's behaviour unchanged.

```
stage 1: public data, ready at S2
  m2b-smoke                     1 epoch, s at 1280, 2% of the mix         ~5 min
  m2b-s1280-s0..s2              then m2b-m1280-s0 (longest last)          ~4 h, ~7 h
  m2b-n1280-s0..s2              control records: M2's weights, no training

stage 2: needs v2 frozen, v1 exported to the cluster, gen-v2, the 180° copies
  m2b-n1280-ft-s0..s2, -ft2-, -ft-gen50k-   (seed 0 of each first)          ~0.4 h each
  m2b-s1280-<recipe>-s0..s2     waits for R2                              ~0.5 h
  m2b-m1280-s1, -s2, then m2b-m1280-<recipe>-s0..s2   only if R1 passes      ~7 h, ~0.8 h

scoring (first, as in M2)
  suite.json (M2's suite)       every trained step-1 run, at once         ~10 min
  suite-m2b.json (M2b suite)    every run, controls first, once every M2b set exists

decisions → results/runs/m2b-decisions.json
  noise band (the control) → R1 (step 1) → R2 (step 2 on n) → R3 (every candidate)
```

**Jobs.** A is released at S2 and runs stage 1; when nothing is ready it
exits and, with tasks left, leaves a held replacement, as M2's did. B stays
held (keeping its place from 2026-10-31) until S10 finds stage 2's data
on the cluster. Both are on `gpu` with 3 days: step 1 is about 20 GPU-hours
and step 2 8 to 25, so each fits; FIFO order doesn't depend on the limit;
and a waiting job's limit can still be raised with `scontrol update` (M2
S1) if R1 adds `m`'s seeds.

---

## Inference add-ons

On the laptop, after R3, on v1-val only (`src/bestbefore/ensemble.py`):

- **ensemble**: the chosen arm's 3 seeds, boxes merged by weighted boxes
  fusion (`ensemble-boxes`, MIT, pinned);
- **rotation**: each photo also predicted at 90°, 180° and 270°, its boxes
  turned back (the inverse of `derived.turn_box`) and fused;
- **1,536 px**: a 1,280 px model fed 1,536 px (train-test resolution).

R4 adopts them in that order. `detection_eval.average_precision` (new) is
checked first against Ultralytics' mAP on real-val for single models: they
must agree within 0.01 (ours predicts with `rect=False`, Ultralytics'
validation with `rect=True`, as M2 S3 found). Inference time per photo on
the laptop's CPU is recorded for each configuration.

## The final test

Once, on the laptop's CPU, after R3 and R4:

- **Store set v2** (`eval_detector.py --set store-v2`, M0's Later item
  arriving with its scored-once guard; per-photo boxes in
  `data/store/v2/runs/`): M1's `n640` (3 seeds); M2's `gen-rot-1280` (3
  seeds) at M2's threshold and at `threshold_store`; every M2b run at its
  `threshold_store`; and the M2b detector with its add-ons. By tag, by
  cause (M2's split, plus upside-down) and by design: seen in v1-train,
  seen only in v1-val, unseen.
- **ExpDate test**: the M2b detector, its 3 seeds, once.

## Report

- `results/m2b_detector.md` from `scripts/m2b_report.py`, from the records
  alone, as M2's: the headline (the M2b detector on v2 and ExpDate test,
  next to M1 and M2), "What it shows" with every number read from the
  records, the table written before any run against what happened, the
  suite, step 1 and step 2 by size, the side questions, the add-ons, v2 by
  tag, cause and design, compute and images per second per run.
- Experiments, each with a README and a `summarise.py`:
  `m2b-store-look` (S3), `m2b-model-size`, `m2b-fine-tune`, `m2b-gen-v2`,
  `m2b-inference`.
- `README.md`: M2b's section and headline, including where it doesn't beat
  M2. `CLAUDE.md`'s milestone table gains M2b (with the owner's OK, S2).

---

## Code layout

```
src/bestbefore/
  chain.py          the GPU chain's machinery, from m2_chain.py (S2)
  m2b.py            layout, sizes, step-1 and step-2 arms, the mix, the recipe, rules R1–R4
  store_train.py    designs, the v1 split, keyword boxes, the YOLO export (S4–S6)
  ensemble.py       weighted boxes fusion, test-time rotation (S13)
  m2b_results.py    the tables (S11 on)
  synth/            spec.py (V2Spec, GEN_V2, SYNTH_VAL_V2), text.py, render.py, warp.py,
                    make.py, batch.py: v2's features, off for M2's spec (S7–S9)
  detection_eval.py average_precision (S13); store.py: DESIGNS_SHA256 (S4)
scripts/
  m2b_chain.py, m2b_report.py, export_store_yolo.py, propose_keywords.py,
  import_store_keywords.py, store_designs.py
  changed: train_detector.py (--milestone m2b), score_suite.py (--suite m2b),
  eval_detector.py (store-v2, m2b runs, add-ons), generate.py, derive_sets.py,
  fetch_pretrained.py (yolo11m), pull_cluster_results.sh (m2b runs)
slurm/
  m2b_gpu.sbatch (S1), m2b_chain.sh, m2b_generate.sbatch
tests/
  test_chain.py, test_m2b_rules.py, test_store_train.py, test_synth_v2.py,
  test_ensemble.py, test_m2b_results.py; changed: test_eval_detector.py,
  test_detection_eval.py, test_detector.py
```

`$STORE_TRAIN_DIR` joins `.env.example`. M2b's generated and derived data go
to `$GENERATED_DIR/m2b/` (`gen/`, `derived/`, `lists/`, `manifests/`), so
M2's folders don't change.

---

## Slices

Numbered in the order to do them. A slice that ends in an HPC job comes as
early as its inputs allow. Every slice records its findings here when done,
and says what wasn't checked.

### S1. Queue the GPU jobs · cluster · *submitted 2 GPU jobs, held*

**Done 2026-10-31**, before this plan. `slurm/m2b_gpu.sbatch` (`8843e48`,
on GitHub, not yet on the laptop): `gpu` partition, 1 GPU, 24 CPUs, 96 GB,
3 days, logs to `slurm/logs/m2b-<id>.out`; it runs `slurm/m2b_chain.sh` when
the job starts, and exits at once if that isn't there yet. Job A
**`73831`** and job B **`73832`**, submitted 20:37:55, both `PENDING`,
reason `JobHeldUser`.

The queue at 21:00 (read-only `squeue`): 4 H100s busy (the first frees
about 2026-11-03, the last about 11-06); 16 GPU jobs from 5 users ahead of
A and B, most asking for 7 days. Taken first come, first served with the
2-running-jobs cap per user, and every job running to its limit, A starts
about 2026-11-27. Earlier by however much those jobs end early.

**For the owner before S2's push:** the laptop is one commit ahead of
GitHub (`dcfe1be`) and three behind (`8843e48` and two README edits):
merge them.

### S2. Model size and step 1 · laptop · *releases job A*

**Goal:** stage 1 runnable on the cluster, then job A released.

**Code:**

- `bestbefore/chain.py`: M2's chain machinery (tasks and states, claims,
  the watchdog, `pick`, `replace_job`, the dry run), with the task list,
  readiness and runner passed in. `m2_chain.py` keeps its task list and
  rules on top; its dry run prints the same before and after.
- `bestbefore/m2b.py` (begun): `SIZES` (`n`, `s`, `m` → `yolo11n`,
  `yolo11s`, `yolo11m`), `STEP1` (`s1280`, `m1280` = `gen-rot-1280` with
  their weights), the layout, the control records (`m2b-n1280-s{k}`:
  `run.json` with `weights_from` M2's run and `status: finished`, no
  training).
- `train_detector.py --milestone m2b --arm s1280`: M2's path with the
  arm's weights; the recipe check is M2's, the record adds the size.
- `fetch_pretrained.py`: `yolo11m.pt` with its SHA-256.
- `scripts/m2b_chain.py`, `slurm/m2b_chain.sh` (as M2's): stage 1, the
  controls, M2-suite scoring of step-1 runs.
- `tests/test_chain.py` (the machinery, moved with M2's tests),
  `tests/test_m2b_rules.py` (begun): step 1 has `gen-rot-1280`'s lists,
  mix and schedule; only the weights differ; the control's weights are
  M2's `last.pt`.

**Steps:**

1. Laptop smoke: `train_detector.py --milestone m2b --arm s1280 --epochs 1
   --fraction 0.01 --device cpu --name smoke-laptop`.
2. `m2b_chain.py --dry-run` on the laptop.
3. Commit, push (after S1's merge). Cluster: `git pull`,
   `fetch_pretrained.py` (login node), `m2b_chain.py --dry-run`.
4. Owner: `scontrol release 73831`; `squeue -j 73831 --start`.

**Check:** the dry run on the cluster shows the smoke, `s1280` s0–s2 and
`m1280` s0 ready, the controls ready, stage 2 waiting for its data.
**Commits:** `refactor: the GPU chain's machinery in bestbefore.chain, m2's chain unchanged`,
`feat: m2b step 1, model size as an arm, s and m at 1280`,
`feat: m2b chain, step 1 and its scoring`,
`docs: m2b in CLAUDE.md's milestones` (owner's OK)

### S3. The store look, recorded · laptop

**Goal:** the gallery's numbers reproducible from a committed script, so
"What M2 and the store gallery tell us" rests on more than a scratch file.

- `experiments/m2b-store-look/measure.py` and README: per store expiry, the
  M2 detector's outcome (kept, doubted, never boxed, by
  `store_gallery.match`, the same matching as S8's counts); contrast,
  sharpness and size inside its box, by outcome, against real-val and
  gen-sample; month words by outcome; tall and wide sideways boxes; found
  rates by role and by year; how many expiries the 3 seeds find. Counts
  and medians only.
- After S4: the design counts (designs in v1; expiries per design).

**Check:** the README's numbers are the script's output; they match this
plan's within rounding (the scratch script and the committed one share
their matching). No photo or photo name in any committed file (searched).
**Commit:** `docs: m2b store look experiment`

### S4. Design ids and the v1 split · owner, then laptop

**Goal:** every v1 and v2 photo with a frozen design id, and v1 split.

1. Owner: M0's D6 (freeze v2's labels), if not done yet.
2. Owner: `data/store/designs.csv`, by the rule in [Design ids](#design-ids).
3. `scripts/store_designs.py --freeze`: every photo of v1 and v2 exactly
   once, ids not empty; writes `designs.json`; prints the hash for
   `DESIGNS_SHA256`.
4. `store_train.split_v1`; `results/store_designs.md` (counts only).

**Tests** (`tests/test_store_train.py`, made-up names): a photo missing
or listed twice fails the freeze; no design on both sides of the split;
the split is the same for the same input; each tag with 2 or more designs
has one in validation; a v2 photo can't enter the split.
**Check:** the counts; validation holds at least 20 expiries (if not, that
is a finding, recorded; the rule doesn't change).
**Commit:** `feat: store pack designs frozen, v1 split by design`

### S5. v1's keyword boxes · laptop, owner in Label Studio

**Goal:** `due`, `prod` and `code` boxes on all 99 v1 photos.

- `scripts/propose_keywords.py`: RapidOCR on each v1 photo (upright,
  1,000 px, M0's best size for it), text boxes matching the patterns in
  [Keyword boxes](#keyword-boxes), written as Label Studio predictions.
- A Label Studio project `Store v1 keywords`: labels `due`, `prod`, `code`;
  v1's date boxes shown for reference; v1's own project untouched.
- `scripts/import_store_keywords.py` → `data/store/v1_keywords.json`, with
  the checks.

**Tests:** the pattern matcher on made-up OCR lines (`EXP.FEB 2027` gives a
`due` box for `EXP.` only, its date stays out); the import checks catch a
box outside the photo, a keyword box over a date box, a photo not
reviewed.
**Check:** counts per class; how many proposals the owner kept, moved,
deleted and added (OCR's help, measured).
**Commit:** `feat: keyword boxes on store set v1, proposed by OCR and checked`

### S6. v1 as training and validation data · laptop, then the cluster

**Goal:** v1-train and v1-val exported, checked, and on the cluster.

- `scripts/export_store_yolo.py` as in [The export](#the-export);
  `m2b.store_lines()` (design-balanced) used by the mix.
- **Tests:** an EXIF-rotated photo's exported boxes sit on its date (a
  made-up photo with a flag and a coloured rectangle, as
  `test_derived.py`); the 4 turns' boxes; names are numbers; a v2 photo is
  refused; the export refuses to run while v2 isn't frozen; design-balanced
  lines give each design the same count.
- **Look:** a contact sheet of the export (`data/store/runs/`), every box
  on its date or keyword, turned copies right. Claude may look (allowed
  since 2026-10-28).
- **Cluster:** `rsync` the export to `~/bestbefore-data/store/`, `chmod
  700`; the manifest's hashes checked there.

**Check:** label files byte-identical on both machines; counts as the
manifest's.
**Commits:** `feat: store set v1 exported for training, upright, numbered and turned`

### S7. Generator v2, part 1: text and layout · laptop

**Goal:** glued keywords, line-pairs, dense panels, import words, the new
distractors.

- `synth/spec.py`: `V2Spec` (text part), `GenSpec.v2` left out of the hash
  when `None`.
- `synth/text.py`: the glued form (`Token.space_before=False` after a word
  ending in `:` or `.`), the `line-pair` and `dense-panel` kinds, import
  words (`prod`, role `import`), barcode digits and unit prices as
  distractors.
- `synth/render.py`: line spacing per kind.
- **Tests** (`tests/test_synth_v2.py`): `spec_hash(GEN)` is still
  `3fc14178…`; S5's 500-photo sample remade with `GEN` is byte-identical;
  each new form's text and boxes (a glued date's box holds only the date;
  a line-pair gives two date boxes); shares within 3 points over 10,000
  blocks; `₹` used only if every font has it.
- **Look:** the render sheet with the new kinds.

**Check:** 20 deliberate bugs, one at a time, all caught (as M2 S4).
**Commit:** `feat: generator v2 text: glued keywords, two dates on a line, dense panels, import words`

### S8. Generator v2, part 2: print and geometry · laptop

**Goal:** nozzle rows, the laser window, upside-down blocks.

- `synth/render.py`: nozzle rows in dotify; the `window` style.
- `synth/warp.py`: the upside-down orientation (`numpy.rot90`, k=2).
- `synth/spec.py`: `GEN_V2`.
- `scripts/generate.py --set gen-v2-sample --count 500`.
- **Tests:** a dropped dot row has no ink in any line of its block, and
  its token's box still holds the rest; a window's text is lighter than
  its rectangle; 180° maps boxes exactly; M2's photos unchanged.
- **Look:** contact sheet and date crops of gen-v2-sample, beside real
  ExpDate dates and, now allowed, v1-train's store crops (never v2's):
  does a new kind look pasted?

**Check:** every gen-v2 share within 3 points of its rule in the sample's
table; the M2 checks (boxes from ink, same bytes however split).
**Commit:** `feat: generator v2 print: nozzle rows, laser windows, upside-down blocks`

### S9. Generator v2, part 3: synth-val-v2, the 180° sets, generation at scale · laptop, then one CPU job

**Goal:** gen-v2, synth-val-v2, real-val-rot180, real_rot180 and
synth_rot180 on the cluster, checked.

- `synth/spec.py`: `SYNTH_VAL_V2`. `synth/batch.py`: the sets `gen-v2`
  (tasks 0–199, 1,000 photos each) and `synth-val-v2` (task 200).
- `scripts/derive_sets.py --only rot180`.
- `slurm/m2b_generate.sbatch`: `short`, 24 CPUs, 32 GB (M2 S6 measured
  900 MB a worker); the render sheet, the tasks, `--check`, the 180° sets.
  It releases nothing: job B waits for v1 too.
- Laptop trial first: `--tasks 0-4,200` timed, the check, synth-val-v2's
  contact sheet by stratum.

**Check:** both manifests `complete: true`, 200,000 and 2,100 photos; the
shares; real-val-rot180 165 photos and 187 dates; the render sheet's
pixels as the laptop's.
**Commits:** `feat: synth-val-v2 strata, embossed measured at last`,
`feat: generation of gen-v2 at scale, and the 180° sets`

### S10. Step 2 and the M2b suite · laptop · *releases job B*

**Goal:** everything stage 2 runs, then job B released.

- `m2b.py`: step-2 arms, `mix()`, `store_lines()`, `FT_RECIPE`,
  `BUDGET_FT`, the two-stage schedule, R1–R3, `noise_store`, `guard`.
- `train_detector.py --milestone m2b --arm n1280-ft --seed k`: starting
  weights from the step-1 run's record; the run's lists written by
  `mix()`; two-stage as one run with two stages recorded; frozen layers'
  weights compared before and after stage 1 (convolutions unchanged,
  batch-norm statistics measured).
- `score_suite.py --suite m2b`: `suite-m2b.json`, both thresholds.
- The chain's stage 2, its scoring and decisions; `pull_cluster_results.sh`
  pulls `-m2b-` runs' JSON and `results.csv`, never images.
- **Tests:** the mix's shares and seeding (the same seed, the same lists;
  another seed, another sample); store lines design-balanced; every arm's
  starting weights; the rules on made-up suites (a tie goes to the
  simplest and cheapest; a candidate failing the guard is out; the band
  is at least 0.03); readiness waits for each of v2 frozen, v1 on the
  cluster, gen-v2, the 180° sets and R2.

**Steps:** laptop smoke of `n1280-ft` (1 epoch, tiny fraction, CPU); dry
runs on both machines; owner: `scontrol release 73832`.
**Commits:** `feat: m2b step 2, fine-tuning with replay and the two-stage arm`,
`feat: the m2b validation suite with the store threshold`,
`feat: m2b chain, step 2 and the decision rules`

### S11. Step 2 on `n` · cluster, then laptop

**Goal:** the noise band, R2, and the side question on gen-v2.

Jobs A and B run what is ready. On the laptop:
`pull_cluster_results.sh`; `m2b_results` tables; `experiments/m2b-fine-tune`
and `m2b-gen-v2` `summarise.py` must agree with `m2b-decisions.json` (a
disagreement stops the slice).
**Commit:** `docs: m2b fine-tuning and gen-v2 results`

### S12. Step 2 on `s` (and `m` by R1) · cluster, then laptop

**Goal:** R3, the M2b detector's arm.

As S11, with `experiments/m2b-model-size`.
**Commit:** `docs: m2b model size results`

### S13. Inference add-ons · laptop

**Goal:** R4 applied, the M2b detector fixed.

- Copy the chosen arm's 3 `last.pt` (weights only).
- **Device check**: the chosen arm's seed 0 scored on v1-val on the
  laptop's CPU (`score_suite.py --out`, as M2 S8), against the GPU's
  `suite-m2b.json`: within a date or two, or stop.
- `detection_eval.average_precision`, checked against Ultralytics on
  real-val (within 0.01).
- `ensemble.py`: WBF and rotation; R4 in order; the threshold chosen again.
- **Tests** (`tests/test_ensemble.py`): a box turned and turned back is
  itself; WBF of identical boxes is that box; rotation with a fake model
  that only sees upright dates finds a turned one.

**Commits:** `feat: average precision in the evaluator, checked against ultralytics`,
`feat: seed ensemble and test-time rotation`,
`docs: m2b inference add-ons`

### S14. Store set v2 and ExpDate test, once · laptop

**Goal:** every planned run scored once on v2, and the M2b detector on
ExpDate test.

- `eval_detector.py`: `store-v2` in `SCORED_ONCE`, private boxes in
  `data/store/v2/runs/`; `m2b-` runs; a configuration with add-ons.
- Before the first v2 score: v2's labels frozen (`TEST_SET_V2_SHA256`
  set) and loading; designs frozen; every run's weights and
  `threshold_store` present (as M2 S8's pre-flight).
- The order of [The final test](#the-final-test); long steps in the
  background, each timed.

**Check:** every score recorded once; no `--again`; committed files hold
counts only (searched for photo names).
**Commits:** `feat: store set v2 as a scored-once set`,
`docs: m2b store v2 and expdate test results`

### S15. Report · laptop

**Goal:** `results/m2b_detector.md`, the README, this plan closed.

As in [Report](#report); the plan's status, measured compute and Later.
**Commits:** `feat: m2b report script`, `docs: m2b results report`,
`docs: readme for m2b`, `docs: m2b plan done`

---

## Order and sessions

```
S1 ─► jobs A, B held (done)
S2 ─► release A ─► [GPU: smoke, s1280 ×3, m1280 s0, M2-suite scores]
S3 (any time)
S4 ─► S5 ─► S6 ─► [rsync v1 to the cluster] ───────────────┐
S7 ─► S8 ─► S9 ─► [CPU job: gen-v2, synth-val-v2, 180° sets] ┤
                                          S10 ─► release B ─┴─► [GPU: controls and step-1 runs on the M2b suite,
                                                                  n step 2 ×9 → R2, s step 2 ×3, m by R1 → R3]
                                          S11 ─► S12 ─► S13 ─► S14 ─► S15
```

| Session | Slices | Ends with |
|---|---|---|
| 1 | S2, S3 | job A released |
| 2 | S4, S5 (with the owner) | designs frozen, keyword boxes in |
| 3 | S6, S7 | v1 on the cluster, gen-v2's text |
| 4 | S8, S9 | the generation job done |
| 5 | S10 | job B released |
| 6 | S11, S12 | R2 and R3 |
| 7 | S13, S14, S15 | M2b's results |

S4 to S9 run on the laptop while job A waits or trains; nothing there
needs a GPU. If A finishes stage 1 before S10, it exits; B keeps its place.

## Compute budget

| GPU work | Runs | Each | GPU hours |
|---|---:|---|---:|
| smoke | 1 | 5 min | 0.1 |
| `s1280` step 1 | 3 | 4 h | 12 |
| `m1280` step 1, probe | 1 | 7 h | 7 |
| M2-suite scores (step 1) | 4 | 10 min | 0.7 |
| M2b-suite scores (controls, step 1) | 7 | 15 min | 1.8 |
| `n` step 2 (3 arms) | 9 | 0.4 h + 15 min scoring | 6 |
| `s` step 2 | 3 | 0.5 h + 15 min | 2.3 |
| **without `m`'s extra seeds** | | | **about 30** |
| `m` by R1: 2 seeds, then step 2 ×3 | 5 | 7 h; 0.8 h + 15 min | 17 |
| **with them** | | | **about 47** |

| CPU work | Size | Estimate |
|---|---|---|
| gen-v2 + synth-val-v2 | 202,100 photos | about 25 min on 24 CPUs (M2 S6's 150 ms a photo per worker) |
| 180° sets | 13,000 photos written | minutes |
| store export | 99 photos, 70 × 4 turns | seconds, laptop |
| v2 scoring | about 25 configurations × 99 photos | under an hour on the laptop; with ensemble and rotation, 12 passes a photo |

**Storage** on the cluster: gen-v2 about 37 GB, synth-val-v2 0.4 GB, the
180° sets about 3 GB, the store export under 0.5 GB, checkpoints under 2 GB
(`m` is about 40 MB a run). About 43 GB, against 950 GB.

**Optimisations, fixed now:**

- **Step 1 starts first** and alone needs no new data. Its runs are
  scored on M2's suite at once, so `s` against `n` is visible days before
  the M2b suite exists.
- **One GPU at a time** is assumed (M2 S9); two at once is a bonus.
- **Batch 16, 8 loader workers, no RAM cache**, as M2, so recipes compare;
  images per second are recorded per run, and `m`'s seed 0 measures its
  cost before any more `m` is spent.
- **Fine-tuning runs are short** (about 20 minutes), so step 2's arms are
  cheap; the replay is a fixed sample per run, so an epoch is a fixed set
  of 25,000 photos.
- **The no-training control costs one scoring pass**, not a training run.
- **v2 is scored on the laptop**: about a minute a run at 1,280 px for
  `s`, so a full set of v2 scores fits in one sitting.

## Files at the end of M2b

```
docs/              m2b-store-and-size.md; datasets.md gains gen-v2 and the store training export
src/bestbefore/    chain.py, m2b.py, store_train.py, ensemble.py, m2b_results.py;
                   changed: synth/{spec,text,render,warp,make,batch}.py, detection_eval.py, store.py
scripts/           m2b_chain.py, m2b_report.py, export_store_yolo.py, propose_keywords.py,
                   import_store_keywords.py, store_designs.py; changed: train_detector.py,
                   score_suite.py, eval_detector.py, generate.py, derive_sets.py,
                   fetch_pretrained.py, pull_cluster_results.sh, m2_chain.py (on chain.py)
slurm/             m2b_gpu.sbatch, m2b_chain.sh, m2b_generate.sbatch
experiments/       m2b-store-look/, m2b-model-size/, m2b-fine-tune/, m2b-gen-v2/, m2b-inference/
tests/             test_chain.py, test_m2b_rules.py, test_store_train.py, test_synth_v2.py,
                   test_ensemble.py, test_m2b_results.py
results/           m2b_detector.md, store_designs.md (committed; runs/ is not)
```

## Risks

- **v1-val is small** (about 25 expiries, 60 dates). Few arms, a band
  measured from the control, mAP rather than counts, and the cheapest
  within the band all limit picking a lucky arm. k-fold is in Later.
- **Memorising v1's few designs.** Design-balanced lines, validation by
  design, and v2 reported as seen against unseen designs show it.
- **Forgetting ExpDate.** The guard on real-val and synth-val; replay at
  80%; ExpDate test once at the end.
- **The same pack in v1 and v2.** Design ids find it; v2's report splits
  by it, so it can't hide.
- **Store photos on the cluster.** Only v1 goes, as numbered copies in a
  `chmod 700` folder outside the checkout; never in `results/` or a log
  line by name; v2 stays on the laptop.
- **gen-v2 looks pasted.** The look check against real store crops (S8);
  synth-val-v2 high but real sets flat would say so, as in M2.
- **Two-stage's "frozen" batch-norm statistics still move** in Ultralytics.
  Measured in S10 and recorded.
- **`m` is slower than estimated.** Its seed 0 measures it; R1 decides
  whether any more is spent.
- **The queue is long** (S1). The jobs keep their places; S4 to S9 fill the
  wait.
- **Keyword OCR proposals are poor** on dot-matrix and sideways print. The
  owner checks every photo anyway; S5 measures how much OCR helped.
- **A wrong date is worse than none** (M3's metric). More boxes at a lower
  threshold mean more candidates for M3 to reject; precision is reported
  beside every count.

## Later

- **Knowledge distillation** (a big teacher, a small student): for when the
  detector must shrink, or the old data can't be kept. We keep it, and
  Ultralytics has no built-in distillation for detection.
- **Hard-example mining**: the detector already finds 95% or more of every
  synth-val stratum's targets, so mining generated photos would mostly
  repeat the faintest ones (S5 found some nearly invisible); the store's
  hard cases arrive directly with v1.
- **Pseudo-labels (Noisy Student)**, if the retailer has unlabelled store
  photos beyond v1 and v2; never v2's.
- **More labelled store photos, and internet photos** (Open Pet Food Facts,
  CC BY-SA, licences checked): through S4 to S6's tools, into training,
  never v2. Arrivals after S10 wait for a later round, so runs stay
  comparable.
- **A mask of the pack** for generated blocks (M2's Later).
- **k-fold validation by design**, if v1-val proves too noisy.
- **A full retrain** of `s` (or `m`) from COCO on gen-v2's whole mix with
  the store share, instead of two steps.
- **`l` or `x`**, only if `m` clearly beats `s`.
- **The pack's own orientation**: a whole photo turned (EXIF missing)
  against print turned on an upright pack; the gallery can't yet tell
  which the 15 sideways expiries are.
- From M2, still open: tiled inference; oriented boxes; a realism
  classifier for generated crops; superscript ordinals and Hindi words.
