# M2b: Store photos, the generator's gaps and a bigger detector

Planned 2026-10-31. Status: **in progress**; S1 done 2026-10-31 (GPU jobs
A `73831` and B `73832` held, before any code); changed 2026-10-31, before
any code: gen-v2 draws sizes like the store's (Cause 4, decision 15) and
worn patches (Cause 2); S2 done 2026-11-02 (job A released, B held for
S10); S3 done 2026-11-03; S4 done 2026-11-03 (v2's labels frozen, pack
designs frozen, v1 split: 25 expiries in validation); S5 done 2026-11-03
(341 keyword boxes on v1, every photo reviewed); S5b added 2026-11-04
(15 more store photos as store set v3); S5b done 2026-11-04 (v3 frozen:
10 photos train, 5 validate, 61 keyword boxes); S6 done 2026-11-04 (v1
and v3 exported, 80 photos train in 4 turns, 34 validate, on the cluster
and checked; 3 label problems found, then fixed at the owner's word, and
exported and copied again); S7 done 2026-11-05 (gen-v2's text and
line spacing, M2's generator unmoved); changed 2026-11-05, S8 (owner's
call): a block off its size is corrected again, then drawn afresh; S8 done
2026-11-05 (gen-v2's print, geometry and store-like sizes; gen-v2-sample
made and looked at, M2's generator unmoved); changed 2026-11-06 (owner's
call): tilted blocks sized by their digits, the sample remade; changed
2026-11-07, before S9's code (owner's calls): synth-val-v2's skipped
blocks drawn at a fresh size, and the generation job at 48 GB; S9's
laptop part done 2026-11-08 (the trial checked, the job written); S9
done 2026-11-12 (gen-v2, synth-val-v2 and the 180° sets made on the
cluster and checked; a crash met at scale fixed first); changed 2026-11-12,
before S10's code (owner's go-ahead): the smoke on a CPU node of the
cluster, the replay's 180° copies, ft2's stages, R2 without a recipe, the
band first; S10 done 2026-11-12 (step 2, the M2b suite and the rules R1 to
R3, smoked on a CPU node of the cluster; the frozen layers didn't move;
stage 2 ready). Next: job B's release, then S11.

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
- **missing dot rows** (a clogged print nozzle) in dot-matrix print;
- **worn print**: ink rubbed or smeared away over part of a date, faded
  to almost nothing there while the rest stays dark. The generator only
  makes patches up to 40% lighter. v1's doubted dates include a smeared
  pouch print. (First noticed on a v2 photo while labelling, 2026-10-31;
  recorded here so its source is plain. The change rests on v1's
  examples, and v2 shapes nothing.)

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

### Cause 4: generated dates are bigger than the store's

Measured 2026-10-31 (a scratch snippet; S3 makes it reproducible), each
date's box short side over its photo's long side: the store's own
small-print measure, made turn-neutral (for a sideways date the short side
is its print height).

| Dates | n | Under 1.5% | 1.5–2.5% | 2.5% and up | Median |
|---|---:|---:|---:|---:|---:|
| store v1, expiry dates | 81 | 31% | 47% | 22% | 1.83% |
| store v1, every date | 192 | 37% | 44% | 19% | 1.73% |
| generated (M2's spec, 1,000 photos) | 2,166 | 16% | 20% | 64% | 3.06% |

Half the store's dates are under 1.73%; the middle half lie between 1.36%
and 2.19%, and 90% are under 2.80%. Generated dates come out about 1.7
times bigger, for two reasons:

- **M2's size bands set the digits' height**, small (0.6–1.5%) on 30% of
  blocks and normal (1.5–4.5%, log-uniform) on 70%, and a date's box is
  taller than its digits (M2 S5: 1.3 times upright, more tilted).
- **The normal band itself sits above the store's print.** Even upright
  generated dates have a median box of 2.74% (21% under 1.5%); tilted ones
  5.23%, as an upright box round a tilted date is wide.

The store's middle band, 1.5–2.5%, where most of its dates and most of the
doubted ones lie, gets only 20% of generated dates. M2's plan set its bands
by rule and never against the store, on purpose (v1 was its test set).

### Checked and refuted (don't re-learn them)

- *"Manufacture dates are found, expiries doubted."* Over 3 seeds, labelled
  manufacture dates are found 57.4% of the time, expiries 56.8% (upright:
  64.7% and 65.7%). Looking only at misses had suggested otherwise.
- *"Later years are doubted."* Every manufacture date is 2026 or earlier
  and found no more often (57.4%) than expiries from 2027 on (50.2%).
  ExpDate's real training dates are mostly 2020 to 2023 (1,065 of 1,244)
  and only 21 are 2027 or later, but the store shows no year effect: the 12
  early expiries found 94% of the time are mostly printed as numbers
  only (10, 8 of them `DD/MM/YYYY`) and dot-matrix (5), the easy kinds.
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
| worn print | gen-v2's worn patches | synth-val-v2's `worn`, v1-val |
| generated dates too big | gen-v2 draws each block's size from v1-train's date sizes, measured as the store measures them | real-val-small, synth-val's `small`, v1-val |

## What we expect (written before any run)

| Run | v1-val mAP (dates) | Store sideways | Real-val-rot180 | Store month words | Small store dates | Real-val |
|---|---|---|---|---|---|---|
| `n1280` (M2's weights, no store data) | the baseline | low | about 0 | doubted | low | 0.995 |
| `n1280` at a v1-val threshold | the same (mAP ignores the threshold) | a little up | about 0 | more kept, more false boxes | more kept | the same |
| `s1280` (step 1) | a little up | a little up | about 0 | a little up | a little up | as `n` |
| `m1280` (step 1, probe) | about as `s` | | | | | as `n` |
| `n1280-ft` (step 2) | clearly up | up | up a lot | up | up | as `n` |
| `n1280-ft2` (two-stage) | as `ft`, perhaps lower | | | | | as `n`, perhaps higher |
| `n1280-ft-gen50k` (side arm) | a little below `ft` | | as `ft` | | below `ft` (M2's sizes) | as `n` |
| `s1280-ft` | the highest | | | | up the most | as `n` |

On v2: designs seen in training well above unseen ones, and unseen ones
still above M2's detector. If fine-tuning lifts v1-val but not v2's unseen
designs, the model learned packs rather than print: that is a finding, and
it gets written up as one.

---

## What we start from

| Fact | Where it comes from | What it means for M2b |
|---|---|---|
| Store set v1: 99 photos, 81 expiries in frame, frozen (`7b9a4043…`) | M0 B3 | becomes training and validation data |
| Store set v2: 99 photos in `testset2/`, photo list frozen (`8be1f9e9…`); labelled and imported 2026-10-31 (74 with an expiry, 8 of them derived; 24 `other-face`; 1 unreadable); labels frozen 2026-11-03 (`f1fe6b8d…`, D6) | M0 Part D | the test set; frozen before any store training |
| Permission of 2026-10-28: store photos may be trained on and shown to AI tools; never committed or published | `CLAUDE.md` | v1 goes to the cluster for training; nothing about a photo is committed |
| M2's 18 runs and their `last.pt` on the cluster; `gen-rot-1280` s0, s1, s2 are `n1280`'s step 1 | M2 S8 | no `n` retraining |
| `gen-rot-1280` trained in 202–204 min at 211 images/s; 640 px arms at about 363 images/s | M2 S9 | step 1's estimates |
| M2's two GPU jobs never ran together: job A did all 40.5 GPU-hours alone | M2 S9 | plan for one GPU at a time |
| Generation: 52,100 photos in 328 s on 24 CPUs, 185 kB each | M2 S6 | 200,000 cost about 25 minutes and 37 GB |
| Generated dates' boxes: median 3.06% of the long side, 16% under 1.5%; store v1's: median 1.73%, 37% under 1.5% | this plan, Cause 4 | gen-v2 matches v1-train's sizes |
| Backgrounds of at least 1,000 px, which small print needs: 711 real and 3,143 synthetic, of 8,185 | M2 S5 | smaller dates reuse those backgrounds more |
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
   and stay in `data/store/`. **(changed 2026-11-03, owner's request)**
   Claude drafted them from the photos and the owner approved them (S4).
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
    generated photos instead of gen-v2, to measure what gen-v2 adds: its
    new layouts and its store-like sizes together (separating the two is
    in Later). Not a candidate for the detector, as M2's `gen10k` wasn't.
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
    the month-word and dot-matrix shares unchanged. **(changed
    2026-10-31, before any code) Sizes match the store's:** each block is
    drawn so its first date's box, measured as the store measures it,
    follows v1-train's date sizes (Cause 4), instead of M2's two bands on
    the digits. **(changed 2026-11-06, S8, owner's call)** A tilted block's
    digits follow them, not its box ([Sizes like the
    store's](#the-generator-v2), below). M2's spec keeps its hash and M2's
    photos their bytes (tests). **(changes M2's rule against leaks)** M2 designed nothing from
    a store photo, because v1 was its test set. gen-v2's new kinds come
    from v1's failures (the gallery looked at all of v1, before the
    split), its sizes from v1-train's dates only, and nothing from v2.
    Worn patches were first noticed on a v2 photo, and are in because v1
    has the same kind; from here, generator ideas come from v1 only
    (looking at v2 to label it is fine).
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
| v3 (`trainset3/`, S5b), 15 photos | the same two roles: a design in v1 keeps its side, the rest split as v1 | laptop, cluster |
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
  photo names stays on the laptop only, **(changed 2026-11-04, S6, before
  the export ran)** in `data/store/train_names.json`, outside the export
  folder rather than in it, so copying the folder to the cluster can't take
  it along; designs go into the export as numbers (`d01` on), their names
  into the same map;
  **(changed 2026-11-04, S5b)** v3's photos the same way, `v3-0001.jpg` on,
  with their dates, keyword boxes and v3's split;
- **(added 2026-11-04, S6)** in sRGB and without metadata: v3's photos
  are Display P3, which a detector reading the numbers would see as
  washed-out sRGB; and 20 of v1's photos carry a GPS position in their
  EXIF, which the export drops with the rest of it;
- label lines: expiry, mfg and other dates as `date` (class 0), keyword
  boxes as their class; **(changed 2026-11-03, S5)** the 3 dates S5's review
  found unboxed in v1's frozen labels (`v1_keywords.json`'s `added_dates`)
  as `date` too, so they aren't learned as background **(4 since S6's
  look at the export found one more, added in the same review)**;
- **(added 2026-11-04, S6, after the export was looked at)** a box under
  2 px across on its exported photo is left out, and counted (each photo's
  `dropped` in the manifest): that small, it holds no print, so it's a slip
  of the pen, and Ultralytics drops boxes of 2 px or less after
  augmentation anyway. v1's frozen labels hold one (1.1 by 3.8 px, the
  only box under 7 px of 634); their file keeps it, as M0 to M2 were
  scored on it;
- v1-train also at 90°, 180° and 270° (`derived.turn`); v1-val as it is;
- lists of v1-train's exported photos (all 4 turns) and of v1-val's; each
  training run's design-balanced lines are built from the first by
  `m2b.mix()`; a manifest with the counts and the hashes of the design,
  keyword and label files (v1's and v3's); **(added 2026-11-04, S6)** and
  every exported file's SHA-256, so a copy is checked against it, and each
  photo's part, design number, tag and box labels (`expiry`, `mfg`, ...,
  in label-file order), so the cluster can tell an expiry from another
  date without the laptop;
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
| worn patch | a soft-edged region, 1 to 3 characters wide and up to two lines tall, where the ink fades by 60 to 95%; on solid, dot-matrix and window print | 10% of inked blocks |
| **sizes like the store's** | each block's first date's box short side, over the photo's long side, drawn from v1-train's date sizes (below) | every block (replaces M2's two bands); **a tilted block by its digits, not its box (changed 2026-11-06)** |

Kinds: single 25%, pair 30%, panel 15%, dense-panel 10%, line-pair 10%,
range 5%, relative 5%. Styles: solid 30%, dot-matrix 35%, label 15%,
embossed 10%, window 10%. Everything else is M2's spec. `₹` is printed
only if every font has the glyph (S7 checks; otherwise `Rs.`).

**Sizes like the store's** (Cause 4):

- **The target**: v1-train's dates (expiry, mfg and other; never v1-val's
  or v2's), each measured as its box's short side over its photo's long
  side. Their deciles, 11 edges, go into `V2Spec` (aggregate numbers,
  committed and hashed with the spec), with the two ends clipped to M2's
  floor and ceiling, 0.6% (8 px at 1,280 px input) and 4.5%. A block's
  target: a decile picked uniformly, then log-uniform inside it.
- **Hitting it**: the block is drawn once at a provisional size (digits at
  the target over 1.3, M2 S5's upright ratio), its first date's box is
  measured after the bending and turning, and the block is drawn again at
  the size that gives that box the target, with the same random draws
  (the generator's state saved before the first draw). A box varies with
  the font, the ordinal, the tilt and the dots, so measuring beats any
  fixed ratio; it costs one extra drawing per block, about a tenth more
  generation time.
  **(changed 2026-11-05, S8, owner's call, before the code was
  committed)** One correction left 8.2% of 2,000 blocks more than 1 px or
  5% from their target, dot-matrix and embossed print most, tilted above
  all (27% of tilted dot-matrix blocks): a box of dots or relief grows in
  steps, a row of dots or an edge of the relief appearing at once. So a
  block still off after its first correction is corrected once more from
  the same draws, and, still off, drawn again from fresh draws, as a block
  that lost its ink is (3 tries in all, `ComposeSpec.redraws`); it keeps
  its drawing closest to the target. On the same 2,000 blocks, 99.9% then
  land within 1 px or 5%, for 2.15 drawings a block against 2.00.
- **A tilted block is sized by its digits (changed 2026-11-06, S8, owner's
  call, before any gen-v2 photo trained anything).** v1-train has no
  tilted date: none of its 169 is more than about 8° off level or vertical
  (the box shape bounds it: a line tilted by θ has a box aspect under
  cot θ), so the store's box size is its print height. A tilted block's box
  mostly measures its tilt, 3.1 times its digits against 1.3 upright, so
  drawing that box to the store's size shrank its digits to a third: 44%
  of tilted blocks had digits under 6 px at 1,280 px input and 18% under 4
  px, smaller than any store date (about 6.5 px at the least), and in a
  blind read 1 of 14 dates under 4 px could be read exactly and 11 not at
  all. So a tilted block is drawn once, at the provisional size (digits at
  its target over 1.3, as an upright block's), with no correction, and its
  box is whatever its tilt makes it. Upright, sideways and upside-down
  blocks are drawn to their box as before. **The size check moves with it:**
  the sample's dates are compared with v1-train's over its untilted dates,
  as the store has no tilted ones; tilted boxes run above the store's range
  by design (over all dates the simulation gave 29.4 / 40.2 / 30.4%, median
  1.90%; over untilted dates nothing moved).
- A block that shrinks to find a free place keeps its shrunk size, and
  says so, as in M2. A block under 1.5% still gets a background of at
  least 1,000 px (M2's rule): 3,854 of the 8,185 backgrounds qualify, so
  they are reused more.
- synth-val-v2's strata keep M2's `clean` sizes, so each compares with M2's
  strata. Small print stays measured by M2's `small` stratum,
  real-val-small and v1-val.

Boxes still come from the ink drawn (M2 decision 4): a glued date's box
stops where its ink starts; a line-pair gives two date boxes and two
keyword boxes; a dropped dot row stays inside its token's box, which keeps
the token's remaining ink. **One exception:** a worn patch's boxes come
from the ink before the wear, so a half-worn date keeps its whole box,
as a person boxes it. The faint-ink floor still applies to what is left:
a date worn to nothing gets its block drawn again.

**synth-val-v2**: 2,400 photos, 300 per stratum, held-out fonts on
validation backgrounds, one target each, set seed 5. Each stratum changes
one thing from M2's `clean`: `upside-down`, `glued`, `line-pair` (the target
is one of the line's two dates), `dense-panel`, `window`, `embossed`,
`worn`, and `nozzle-rows`, which changes one thing from M2's `dot-matrix` stratum.
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

Ultralytics 8.4.177 keeps a frozen layer's batch-norm modules in eval
mode, so their statistics don't move: S10's smoke measured stage 1 and
found none of the frozen layers' weights or statistics changed. Every ft2
run measures it again and stops if a frozen weight moves. **(corrected
2026-11-12, S10: the plan first said the frozen layers' statistics still
update in training mode.)**

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
  stratum first (the layouts), then real-val-small and v1-val's dates by
  size band (the sizes), then real-val-rot180.
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
  store_train.py    designs, the v1 and v3 splits, keyword boxes, the YOLO export (S4–S6)
  ensemble.py       weighted boxes fusion, test-time rotation (S13)
  m2b_results.py    the tables (S11 on)
  synth/            spec.py (V2Spec, GEN_V2, SYNTH_VAL_V2), text.py, render.py, warp.py,
                    compose.py, make.py, batch.py: v2's features, off for M2's spec (S7–S9)
  detection_eval.py average_precision (S13); store.py: DESIGNS_SHA256 (S4), V3 (S5b)
scripts/
  m2b_chain.py, m2b_report.py, export_store_yolo.py, propose_keywords.py,
  draft_keywords.py, import_store_keywords.py, store_designs.py, store_heic.py,
  store_sizes.py
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
- Details fixed while writing it (2026-10-31):
  - **(changed, owner's OK) The laptop smoke trains on `base`'s lists.**
    The laptop holds only two of `gen-rot-1280`'s five lists, `train_real`
    and `train_synth_clean`: `real_rot`, `synth_rot` and `gen50k` were made
    on the cluster only, as M2 S3 found for `rot`. So the smoke as planned
    can't start there. It runs `s1280`'s weights, input size, recipe and
    schedule on `base`'s lists instead, through `--smoke-lists base`,
    which is refused unless `--fraction` is under 1, is recorded in
    `run.json`, and is never passed by the chain. The GPU job's
    `m2b-smoke` keeps `gen-rot-1280`'s lists at 2%. For the same reason
    the laptop's dry run shows `s1280` and `m1280` waiting for those three
    lists; only the cluster's can show them ready.
  - **(changed 2026-11-02, owner's OK) The smoke runs on a CPU node of
    the cluster**, `slurm/m2b_smoke_cpu.sbatch` (`short`, 24 CPUs, 64 GB):
    the laptop can't hold it. A batch of 16 photos at 1,280 px keeps about
    26 GiB for `s`'s backward pass, and even a 1-photo smoke froze the
    laptop in Ultralytics' validation at the end (the record below). On
    the cluster it trains on `gen-rot-1280`'s own lists, so it needs no
    `--smoke-lists`, which stays for a machine without them.
  - **`score_suite.py --milestone m2b`** (default `m2`, unchanged): stage
    1's scores on M2's suite need it to find `-m2b-` runs. The plan named
    only S10's `--suite m2b`.
  - **M2's tests stay as they are.** `m2_chain.py` keeps its names as thin
    calls into `bestbefore.chain`, so `tests/test_m2_rules.py` runs on it
    unchanged: that, and its dry run, are the proof that M2's chain is
    unchanged. `tests/test_chain.py` tests the machinery on its own, with a
    made-up milestone, rather than taking M2's tests over.
  - **The smoke is scored**, as M2's was: scoring an `m2b` run is tried on
    the GPU before the 4-hour runs, for about 3 minutes (5 scores on M2's
    suite, not the budget's 4).
  - **The controls come right after the smoke**: they take seconds and no
    GPU, and their records then exist from the first day.
  - **Stage 2 is listed**, every task waiting for its data (v2 frozen,
    v1's export, gen-v2, the 180° sets) and for S10's code, so none can be
    ready early. `s1280`'s and `m1280`'s step 2 are listed as
    `<recipe>` until R2 names one.
  - **The m2b run record** adds the size, the M2 arm whose lists it trains
    on, and the SHA-256 of the COCO weights it starts from. Its loader is
    M2's decision, 8 workers and no RAM cache, passed as `--workers 8`.

**Steps:**

1. Smoke on a CPU node of the cluster (changed twice, above): `sbatch
   slurm/m2b_smoke_cpu.sbatch`, which runs `train_detector.py --milestone
   m2b --arm s1280 --epochs 1 --fraction 0.01 --device cpu --name
   smoke-cpu` on `gen-rot-1280`'s own lists.
2. `m2b_chain.py --dry-run` on the laptop.
3. Commit, push (after S1's merge). Cluster: `git pull`,
   `fetch_pretrained.py` (login node), `m2b_chain.py --dry-run`.
4. Owner: `scontrol release 73831`; `squeue -j 73831 --start` (done by
   Claude, at the owner's request).

**Check:** the dry run on the cluster shows the smoke, `s1280` s0–s2 and
`m1280` s0 ready, the controls ready, stage 2 waiting for its data.
**Commits:** `refactor: the GPU chain's machinery in bestbefore.chain, m2's chain unchanged`,
`feat: m2b step 1, model size as an arm, s and m at 1280`,
`feat: m2b chain, step 1 and its scoring`,
`docs: m2b in CLAUDE.md's milestones` (owner's OK)

**Done 2026-11-02.** Code `fe8616f` to `12cf00f` (2026-10-31), 542 tests
pass; each commit also passes alone, in a worktree with the laptop's data
and nothing skipped (519, 528, 534, 542). Then this plan's change for the
smoke (`59f02aa`) and its job script (`ef1cde5`), 2026-11-02. Job A
(`73831`) was released at 15:45 on 2026-11-02, by Claude at the owner's
request (the slice had it as the owner's step); job B (`73832`) stays
held for S10. What was measured:

- **M2's chain is unchanged.** Its dry run prints the same 47 lines before
  and after the move, on the laptop and on the cluster (every task done).
  A scratch probe drove `m2_chain.py` through 7 made-up states (nothing
  made; stage 1's lists; the loader decided, with claims and failures; the
  generated data incomplete; the pilot failed; the pilot passed, with
  claims; every candidate scored), each with 4 settings of `squeue` and
  the job id: the dry run, `pick` at four amounts of time left,
  `replace_job`, and the job's loop with a failing task. It printed the
  same 1,601 lines before and after. `tests/test_m2_rules.py` is unchanged
  and passes. Nothing of M2's was written: its records, claims, decisions
  and weights were only read.
- **Step 1 is `gen-rot-1280` with other weights**, on the cluster's own
  lists: `train_real*3`, `real_rot*3`, `train_synth_clean`, `synth_rot`,
  `gen50k`; 79,310 photos listed, 32 epochs, warmup 0.96, mosaic off for
  the last 3, 2,537,920 images a run, 8 loader workers and no RAM cache.
  M2's `gen-rot-1280-s0` record has the same, and trained in 201.8
  minutes. The smoke trains on 1,586 photos (2%).
- **The control's weights** are M2's `last.pt`, the same bytes on both
  machines: s0 `8c9a03c7…`, s1 `09f49e9c…`, s2 `6de901f2…`. The GPU job
  writes the three records; none exists yet.
- **The COCO weights**, fetched on the login node with
  `fetch_pretrained.py` (2026-10-31): `yolo11m.pt`, 40.7 MB, SHA-256
  `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`
  (Ultralytics' assets, v8.4.0); `yolo11n` `0ebbc80d…`, `yolo11s`
  `85a76fe8…` and `yolo26n` `9b09cc8b…` as before. Each M2b run records
  its own weights' hash.
- **GPU memory at batch 16 and 1,280 px.** The GPU is an H100 NVL with
  95,830 MiB (`nvidia-smi` in M1's and M2's job logs). Ultralytics'
  `GPU_mem`, read from those logs: `s1280` 21.1 GB at most (M1); `n1280`
  10.1 GB (M2's three `gen-rot-1280` seeds) and 12.5 GB (M1's, which
  validated every epoch). `m` has no GPU measure, so the laptop's CPU
  measured what autograd keeps for the backward pass, per photo at 1,280
  px (a scratch script, each model built from its yaml): `n` 0.84 GiB,
  `s` 1.62, `m` 3.42. That ratio doesn't depend on the device. So `m`
  needs 2.10 times `s`'s memory, about 44 GB, under half the card; the
  same ratio puts `n` at 10.9 GB, against the 10.1 measured.
- **Host memory.** The job's 96 GB is host memory, not the GPU's. M2's job
  peaked at 30.1 GB (`sacct`'s sampled MaxRSS, which includes its speed
  test's RAM cache), M1's at 11.2 GB. Model size barely touches it (the
  loader holds the photos; `m`'s weights are 40 MB), and M2b uses no RAM
  cache.
- **Time.** M1's `s1280` trained 181 images a second (1.18 minutes an
  epoch of 12,795), so an `s1280` run takes about 3.9 hours, as planned.
  The laptop's CPU takes 2.26 times as long for `m`'s forward and backward
  pass as for `s`'s (5.7 against 2.5 s a photo at 1,280 px); at that ratio
  `m` trains about 80 images a second, about 9 hours a run (7.5 to 12.4,
  between that ratio and the ratio of their FLOPs, 3.2). The chain keeps
  the plan's 7 hours: it starts a task only with twice its estimate left,
  14 hours, which covers the range. Stage 1 is then about 21 hours of the
  job's 72. An epoch takes about 7 minutes for `s` and 16 for `m`, far
  inside the watchdog's 2 hours.
- **The dry runs.** On the cluster (2026-10-31 and 2026-11-02, at
  `12cf00f` and then `ef1cde5`, the last just before the release): the smoke, the three controls, `s1280` s0–s2 and `m1280` s0
  ready; the 15 step-2 runs waiting for v2 frozen, v1's export, gen-v2
  (all but the `gen50k` arm), the 180° sets and S10's code; `m1280` s1 and
  s2 waiting for R1; every score waiting for its run; no decisions. On the
  laptop: the controls ready; the smoke, `s1280` and `m1280` waiting for
  `real_rot`, `synth_rot` and `gen50k`, which only the cluster holds.
  Those reasons are M2's own text, so they name M2's slices ("S2's
  derive_sets.py", "S6's generation").
- **The cluster runs the code.** `train_detector.py` and `score_suite.py`
  import and parse there, and the commands the chain builds parse into the
  arms above.
- **The job script.** `slurm/m2b_gpu.sbatch` on the cluster was last
  changed at 20:37:55 on 2026-10-31, the jobs' submit time, so the copy
  Slurm keeps should be the committed one. `slurm/m2b_chain.sh` is there,
  and a test checks that the batch script runs it and that it runs
  `scripts/m2b_chain.py`.
- **The smoke: tried on the laptop, run on a CPU node of the cluster.**
  At `--fraction 0.01` a batch of 16 photos at 1,280 px keeps about 26 GiB
  for the backward pass (16 × 1.62), on an 8 GB laptop: swap reached 14.8
  of 15.4 GB, the process held 29 MB at 4% CPU, and it was stopped in
  batch 2 of 8. At `--fraction 0.0001` (1 photo, a batch of 1) the recipe
  check passed on a real trainer ("applied as asked: MuSGD, 1 training
  photos (1 distinct), RAM cache False") and the step took 25 s, but
  Ultralytics' two validations at the end, 165 photos in batches of 32
  (2.9 GiB at most, measured), take about 18 minutes each on that CPU.
  One try was stopped from outside during them (the session ended); the
  next wrote `last.pt` after the first and froze the laptop in the
  second, and was stopped by hand. All three records say failed. So the
  smoke ran as CPU job `73859` (`slurm/m2b_smoke_cpu.sbatch`, owner's
  go-ahead): on `gen-rot-1280`'s own lists, 79,310 listed and 793 trained
  on (265 distinct), it trained for 9.9 minutes (10 min 50 s in all, 30.5
  GB at most by `sacct`'s sample) and left `run.json` finished at
  `ef1cde5`, every applied argument as asked (MuSGD, `lr0` 0.01, momentum
  0.937 after warmup, batch 16, 1,280 px, `val` off), `yolo11s`'s hash,
  and a 19.3 MB `last.pt`. On a CPU Ultralytics uses no loader workers
  (0 applied, 8 recorded), as M2 S3 found. It is named `smoke-cpu`, so
  the chain's own `m2b-smoke` still runs first on the GPU.
- **S1's note on git** was already sorted: the merge `77f8d43` was on
  GitHub with `946b56f`, so the laptop and GitHub were level before S2's
  push, and the cluster fast-forwarded from `8843e48` to `12cf00f`.
- **The queue at the release** (2026-11-02 15:12, read-only `squeue` and
  `sinfo`): the four H100s run jobs ending 11-05 09:35, 11-07 23:14, 11-08
  16:31 and 11-09 03:56; six GPU jobs submitted before A wait ahead of it
  (apoorva 4, meera 2, one of them 12 hours); no reservation. The waiting
  jobs' `ReqNodeNotAvail, UnavailableNodes:cn[5-6]` only means every GPU
  is busy (cn5 mixed, cn6 allocated). First come, first served, with the
  2-running-jobs cap and every job running to its limit, A gets the fifth
  GPU to free up: about **2026-11-09 04:31**, when meera's 12-hour job
  ends, and earlier by however much those jobs end early. After the
  release A is `PENDING` with its submit time kept (2026-10-31 20:37:55);
  Slurm's own estimate, 2026-11-05 09:35, is only when the first H100
  frees up, as M2 S3 found. A's reason, "Nodes required for job are DOWN,
  DRAINED or reserved for jobs in higher priority partitions", is Slurm's
  word for busy GPU nodes here: every partition is tier 1 (`sinfo`) and
  every waiting job has priority 1.
- **Not checked:** an M2b run on a GPU (the job's smoke is the first), so
  `m`'s memory and speed stay estimates until `m1280-s0`; scoring an
  `m2b` run for real (tested with a stand-in; the job scores its smoke
  first); the copy of the batch script Slurm keeps (only its file time);
  the learnings guide (not written).
- **`CLAUDE.md`** gained M2b's milestone row and the cluster facts above
  (the GPU's model and memory, `short`'s nodes, what the waiting reason
  means, the laptop's limit and the CPU smoke's time), with the owner's
  OK.

### S3. The store look, recorded · laptop

**Goal:** the gallery's numbers reproducible from a committed script, so
"What M2 and the store gallery tell us" rests on more than a scratch file.

- `experiments/m2b-store-look/measure.py` and README: per store expiry, the
  M2 detector's outcome (kept, doubted, never boxed, by
  `store_gallery.match`, the same matching as S8's counts); contrast,
  sharpness and size inside its box, by outcome, against real-val and
  gen-sample; month words by outcome; tall and wide sideways boxes; found
  rates by role and by year; how many expiries the 3 seeds find; every
  store date's size against generated dates' (Cause 4: the bands, the
  median, by orientation). Counts, medians and percentiles only.
- After S4: the design counts (designs in v1; expiries per design).

**Check:** the README's numbers are the script's output; they match this
plan's within rounding (the scratch script and the committed one share
their matching). No photo or photo name in any committed file (searched).
**Commit:** `docs: m2b store look experiment`

**Done 2026-11-03.** `experiments/m2b-store-look/measure.py` ran on
2026-11-02 on the laptop's CPU in 7 s. Its README holds the output word
for word: the same output again from a clean start (no bytecode, an empty
environment). 542 tests pass. What was found:

- **The scratch code was gone from disk.** It was recovered from the
  session transcripts: the 2026-10-28 gallery script (the look by outcome,
  month words, the seeds) and its role and year snippet, and the
  2026-10-31 size snippets. `measure.py` ports their measures. It matches
  with `store_gallery.match`, as S8 does, and stops if a seed's counts
  differ from its `store.json`. All 3 agree: kept and boxed 39 and 69
  (s0), 53 and 70 (s1), 46 and 72 (s2). It reads store set v1 only, and
  stops on a photo from outside v1's folders.
- **Every number in "What M2 and the store gallery tell us" and Causes 1
  to 4 matched within rounding**: the outcomes (39 / 30 / 12), contrast
  and sharpness by outcome against real-val and gen-sample, month words
  (20 of the 30 doubted), the seeds (55, 37, 7), the found rates by role
  and year (57.4% and 56.8%; 50.2% and 94.4%), and the size bands
  (store expiries 30.9 / 46.9 / 22.2%, median 1.83%; generated
  15.9 / 19.8 / 64.3%, median 3.06%).
- **One count was wrong, and is corrected above** (owner's OK): of the 12
  early expiries, 10 are printed as numbers only, not 8. The 8 counted
  only `DD/MM/YYYY`; the others are 1 `MM/YYYY` and 1 `YYYYMMDD`. The
  claim (mostly numeric, the easy kinds) holds.
- **New: 8 of the 15 sideways expiries are small print** (under 1.5% of
  the photo's long side), measured by the box's short side as this plan
  defines size. The scratch script measured box height, which for
  sideways print is its length, so it found none. Sideways expiries are
  then small as well as turned, which gen-v2's sizes (Cause 4) should
  cover too.
- **The generated sizes hold over 5,000 photos**, not only task 0's 1,000:
  17.3 / 21.0 / 61.7% in the bands, median 3.04% (against 3.06%).
- **Photo names.** The committed files and the two new ones were searched
  for every v1 and v2 photo name (the name and its stem, 5 characters or
  more). The new files hold none. 9 lines in older files hit: generic
  example names (a folder and "sample" plus a number) in M0's
  `store_import.py` docstring, M0's plan and `tests/test_store_import.py`,
  committed since `cca856f` and `2b20d77`. They happen to match real v1
  names, and were left alone.
- **Not checked:** the design counts (designs in v1, expiries per design)
  and whether the same print swings between photos, both pending until
  S4; Cause 2's other layouts, only ever seen by eye; per-photo seed
  confidences (not printed, being per photo); the learnings guide (not
  written).

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

**Done 2026-11-03.** Step 1 is M0's D6 (`13eefd1`, recorded there): v2's
labels frozen, `f1fe6b8d…`, before anything trains on store photos. Then
the designs, frozen as `data/store/designs.json` (`DESIGNS_SHA256`
`fa888edb…`), and v1 split, `SPLIT_SHA256` `4418c65b…`;
`results/store_designs.md` holds the counts. 558 tests pass (542 before).
What was found:

- **(changed, owner's request) Claude drafted the design ids, the owner
  approved them.** Step 2 had the owner type them. Claude wrote
  `designs.csv` with every photo listed, made numbered contact sheets of
  all 198 (upright, 520 px, the hand-drawn date boxes on them, in
  `data/store/runs/designs/`), looked at every one and named each pack,
  using the owner's v2 notes for brands and barcodes. No model output was
  used. Then one sheet per design (`runs/designs/by-design/`) and a list
  of every photo with what Claude saw (`data/store/designs_review.csv`) for
  the owner's review; the owner approved the draft as it was. So **v2's
  photos were shown to Claude**, which D5 hadn't done: allowed by the
  2026-10-28 permission and by decision 15 (looking at v2 to label it).
  Nothing about the generator or training came from them, and decision 3
  holds: no v2 photo is on the cluster or in any training list.
- **The rule as applied**: one design is one brand and one back-panel
  template with the date printed the same way in the same place, whatever
  the flavour, size or life stage; another product line, template or print
  method is another design; an `other-face` photo takes its pack's design.
  The calls worth knowing: a treat jar's label and its base are one design
  (13 photos); every biscuit of one brand, jar or pouch, is one (12: the
  same batch sticker and best-before band); one brand's treat pouches from
  one maker are one (10); one chew brand splits by template and print
  (bags 8, coloured small packs 3, and three solid-printed packs of 1
  each); an importer's sticker is no design of its own, so one importer's
  sticker on two brands makes two designs; a box in v1 and its bars in v2
  are one (the same EXP and lot print).
- **Designs:** 85 in all, 40 in v1, 54 in v2, **9 in both**. Most are
  small: a median of 1 photo per design in each set; 23 of v1's 40 and 30
  of v2's 54 have one photo.
- **Cause 3 measured.** v1's largest design holds 10 of the 81 expiries in
  frame and its largest 4 hold 31, as the gallery's "about 30, one pouch
  about 10" had it. v2 is flatter: 5 and 15 of 68.
- **The same pack in both sets** (M0 Part D, Later): 30 of v2's 99 photos
  have a design v1 also has, but they hold only 11 of v2's 68 expiries;
  most are `other-face` or relative-expiry packs (one biscuit brand: 8
  photos, no expiry box).
- **The split** (seed 0): 27 designs, 70 photos and 56 expiries train;
  **13 designs, 29 photos and 25 expiries validation**, so the check (at
  least 20) holds. By photo tag, validation's expiries: clean 10,
  small-print 7, sideways 5, dot-matrix 3.
- **v1's four largest designs all fell in training** (a treat jar, a
  treat pouch, a dry food and a wet-food pouch): 30 of training's 56
  expiries. So v1-val is many small designs, and the pouch
  that gave 6 of M2's 12 never-boxed expiries is a training design, not a
  validation one. And **dot-matrix validation is thin: 3 expiries**. Both
  are the rule's outcome and stay; per-tag v1-val numbers for dot-matrix
  mean little.
- **v2 by design** (at the end, S14): seen in training (a design in
  v1-train) 16 photos with **4 expiries**; in v1-val only 14 photos with
  7; unseen 69 with 57. So "seen against unseen" on v2 rests on 4 and 11
  expiries: too few to compare on their own (S14 should report counts with
  every rate).
- **Details fixed while writing:**
  - **A tied tag** goes to the one first in `store.TAGS` (sideways before
    clean): the order photos take their tags in.
  - **"round(0.3 × n)" is rounded half up**, in integers: Python's
    `round` rounds half to even, which would give 4 of 15 designs, not 5.
    With half up, 2 or more designs always give at least 1, as the rule
    asks.
  - **One generator**, seeded with 0, shuffles the tags in `store.TAGS`
    order; every list is sorted first. The split hashes the same under 3
    hash seeds and in a fresh process.
  - **Added beyond the plan:** the freeze also refuses two ids that differ
    only in case or spaces (one design typed two ways would sit on both
    sides of the split) and a row with a stray comma; `designs.json`
    records the hashes of the label files it was named against;
    `SPLIT_SHA256` in `store_train.py`, checked by `store_train.v1_split()`
    (for S6's export), so the split can't change unseen; the script checks
    the file without writing (no flag) and writes the report
    (`--report`).
  - `tests/test_store_train.py`: 16 tests, made-up names, covering the
    plan's five and the above.
- **Photo names.** The committed files were searched for every v1 and v2
  photo name (and stem, 5 characters or more) and for every design name:
  none in this slice's changes, and no brand: the S4 record describes
  designs without naming them, as some brands are the retailer's own. `designs.csv`, `designs.json`,
  `designs_review.csv` and the sheets are under `data/`, gitignored.
- **Not checked:** whether every id is right, beyond the owner's approval
  (no second look, sheet by sheet, recorded); the design counts for S3's
  experiment and whether the same print swings between photos (S3's two
  items pending S4: not done here); the learnings guide (not written).

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

**Done 2026-11-03.** `data/store/v1_keywords.json` holds **341 keyword
boxes on 96 of v1's 99 photos**, every photo reviewed and every import
check passing; v1's frozen labels and its own project are unchanged. 602
tests pass (558 before). What was found:

- **OCR's proposals** (`scripts/propose_keywords.py`): RapidOCR 1.2.3 on
  each v1 photo, upright and shrunk to 1,000 px with Scanify's LANCZOS
  thumbnail, read 3,216 lines in 50 s. The matcher found **218 keywords on
  82 photos** (72 `due`, 95 `prod`, 51 `code`), none over a date box. The
  script runs RapidOCR's own steps to keep two flags `RapidOCR()` drops
  (a line read turned a quarter, a line its classifier turned 180°), which
  say which end of a line its keyword is at; its lines are those of
  `RapidOCR()` exactly (20 photos compared). RapidOCR boxes whole lines,
  so each keyword is cut from its line in proportion to its characters.
  Address lines (`Manufactured By:`, `Imported & Marketed By:`) and shelf
  lives (`12 months from Mfg.`) get no box.
- **The coordinates match v1's.** A sheet drew the proposals beside v1's
  frozen date boxes on every upright photo (`data/store/runs/`), and three
  tasks were opened in Label Studio (an EXIF quarter turn, an EXIF half
  turn with print reading down, an upside-down label): every keyword box
  sat just before its date. The tasks name each photo exactly as v1's
  project does, the boxes are drawn on the upright size (`original_width`,
  `original_height`), and keyword and date boxes go through one converter
  (`store_import.rect_box`, renamed from `_box`; v1's import unchanged).
- **The project, "Store v1 keywords"** (Label Studio 1.23.2, project 3, on
  127.0.0.1:8080 only: the laptop's network address gets no answer), made
  through the API from the browser pane, logged in as the owner. Labels
  `due`, `prod`, `code`; v1's date boxes as a second, read-only layer; the
  box rule in a header on every task and in the instructions. Its photos
  are served through v1's storages; it has none of its own. Database
  backed up to `data/store/backups/` before each write; v1's and v2's
  projects fingerprinted (rows, tasks, annotations, drafts, predictions,
  storages) before the first write and after the last: unchanged.
- **(changed, owner's request) Claude drafted the boxes**, between OCR and
  the owner, as S4's design ids were drafted. `scripts/draft_keywords.py`
  snaps every box to its ink (the line's padding trimmed; an end facing a
  date box stops at the last ink before it, so the colon stays and the date
  is out) and applies Claude's edits from looking at all 99 photos at full
  size (`data/store/keywords/edits.json`): **165 edits on 59 photos**, 123
  boxes added (77 drawn by eye, 46 from a second OCR pass: RapidOCR again
  on each photo's date area, enlarged, with lower detection thresholds, a
  scratch script whose output is in `edits.json`), 36 redrawn, 3 extended
  (a code's label over its value) and 3 deleted. The draft, **338 boxes on
  95 photos**, went to Label Studio as a second prediction
  (`claude-draft-1`), OCR's kept beside it, and became what each task opens
  with. It rebuilds byte for byte from `edits.json`.
- **The owner's review.** The owner changed **39 photos** (3 on OCR's
  proposals before the draft existed, 36 on the draft) and left 60 as
  drafted, then asked Claude to submit: Claude submitted all 99 through the
  API, each changed photo's draft as the owner left it and each other one
  the draft unchanged, which is what Submit on an untouched photo does
  (as in M0's D5). 99 annotations, one a task, none skipped, no draft left.
- **v1's labels miss 3 dates.** The owner boxed 3 dates in the date layer
  that v1's frozen labels have no box for (two import dates, and one on a
  line whose word OCR read as `MOI:`). v1's labels can't change, so check 4 now fails only
  when a frozen date box is moved or deleted; an added one is checked like
  a keyword box and kept in `v1_keywords.json`'s `added_dates` (all 3
  `other_date`). S6's export takes them as dates ([The export](#the-export),
  changed). A revision of v1's labels (M0 Part D, Later) would carry them.
- **The keyword boxes**: `due` 106 on 93 photos, `prod` 147 on 92, `code`
  88 on 76. 3 photos have none (no keyword in frame).
  **(changed 2026-11-04, S6)** S6's look at the export fixed this review
  in its project: a `prod` box on a shelf life deleted, a fourth missed
  date added and the keyword box beside it trimmed. Now 340 keyword boxes
  (`prod` 146) and 4 added dates; OCR's help below, added 125.
- **OCR's help**, by region id (Label Studio keeps a predicted region's id
  in the annotation made from it), OCR's 218 proposals against the final
  boxes: **kept 1, moved 214, deleted 3, added 126**, no label changed.
  "Moved" includes the draft's snapping, which moves every box a little,
  so it hides how good the proposals were: of the 215 that survived, 168
  overlap their final box at IoU 0.5 or more (71 at 0.7 or more), so OCR
  found the keyword and the box only needed fixing; 47 changed more,
  mostly a code's box extended over its value. Of the 126 added, 77 were
  drawn by eye, 45 came from the second OCR pass and 4 from the owner. So
  OCR found 63% of the final keywords (215 of 341), and missed mostly
  times, dot-matrix and sideways stamps, small print and misspelled packs
  (`BARTCH NO`).
- **The export**: `label-studio export 3 JSON` with the server's settings,
  as D5's; the full JSON export, not v1's JSON-MIN, as only it keeps
  region ids (a departure, written down before the code). The import twice
  gives the same `v1_keywords.json`, byte for byte.
- **Details fixed while writing:** a prod word after "from date of" is a
  shelf life's too (`24 MONTHS FROM DATE OF MANUFACTURE`; one first-pass
  proposal dropped in the draft, `proposals.json` kept as made);
  `propose_keywords.py` also saves OCR's lines (`lines.json`), which the
  draft cuts boxes from; measuring the draft's help as well as OCR's was
  dropped (owner's call).
- **Only v1.** Every script lists v1's photos from its folders and checks
  them against v1's frozen labels; nothing read `testset2/`. The committed
  files were searched for every v1 and v2 photo name (and stem, 5
  characters or more) and every design name: none in this slice's changes.
- **Not checked:** whether the owner looked at the 60 photos left as
  drafted (Label Studio keeps no record of a view; the owner said all 99
  were done); how far the owner's boxes moved from Claude's draft (not
  measured, owner's call); a sheet of the final boxes (only the draft's and
  OCR's were looked at); the second OCR pass as a committed script; S6's
  use of `added_dates`; the learnings guide (not written).

### S5b. Store set v3: 15 more store photos · laptop, owner in Label Studio

Added 2026-11-04, before any code (owner's OK): 15 store photos arrived on
2026-11-03, HEIC, 4,032 × 3,024, under `data/store/NewExtraStoreData/`.
None has a v1 or v2 photo's name; none reports an EXIF turn (`sips`), but
a HEIC can keep its turn in its own container, so S5b checks uprightness
after converting.

**Goal:** the 15 as **store set v3**, training data like v1: labelled,
design ids, split by design and keyword boxes, every file frozen, before
S6 exports them with v1.

**Decisions** (all new; v1's and v2's files don't change; 6 widens what
v1-train and v1-val mean):

1. **Their own set, not v1.** v1's labels (`7b9a4043…`), designs
   (`fa888edb…`) and split (`4418c65b…`) are frozen, and M0 to M2's tables
   rest on v1's. v3 is a third `StoreSet`: folder `trainset3/` under
   `$STORE_PHOTOS_DIR`, files in `data/store/v3/`, its own hashes. Its name
   says its role, as `testset2/` does.
2. **Converted once, to JPEG at quality 95, with `sips`** (macOS's own, no
   new dependency); the JPEGs are v3's photos from then on, never
   re-encoded. `data/store/v3/heic.json` maps each HEIC's SHA-256 to its
   JPEG's, so the conversion stays traceable. `trainset3/` goes into
   Scanify's `.git/info/exclude`, as `testset2/` did (M0 D1).
3. **Frozen twice, as v2:** the photo list before labelling, the labels
   after (`freeze_store_test_set.py --version v3`; its summary
   `results/store_set_v3.md`, as v3 is no test set).
4. **Labelled by M0's B1**, in a Label Studio project `Store set v3` with
   v1's interface and settings. B2's checks 1 to 3, 5 and 6 (4 is v1's
   only), and the small-print rule.
5. **Design ids in a file of their own**, `data/store/designs_v3.csv` →
   `designs_v3.json` (`DESIGNS_V3_SHA256`), named by S4's rule and, for a
   pack already in v1 or v2, with exactly that id; `designs.json` stays as
   frozen. Claude drafts them from the photos, the owner approves (as S4).
   The freeze refuses an id that differs from a v1 or v2 id only in case
   or spaces.
6. **The split: a known design keeps its side.** A v3 photo whose design
   is in v1 goes where that design is (train or val); every other v3
   design is split by S4's rule among v3's own (30% to validation by tag,
   seed 0). v1's split is unchanged (`SPLIT_SHA256`); v3's gets
   `SPLIT_V3_SHA256`. **From here, v1-train and v1-val, as this plan
   names them, hold v3's parts too**: the store lines, the thresholds,
   v1-val's scores and S8's sizes. No M2b run has been scored on v1-val
   yet, so the larger validation set costs no comparison.
7. **A design v2 also has** becomes "seen in training" in S14's report.
   Allowed (v2 shaped nothing); S5b counts it in `results/store_designs.md`.
8. **Keyword boxes by S5's tools**, which take the set (`--version
   v1|v3`): OCR's proposals, Claude's draft, the owner's review, the same
   checks, `data/store/v3/keywords.json`. A second project, `Store v3
   keywords`, as the dates must be frozen first (they're the reference
   layer and the snapping stops at them).

**Steps:**

1. `scripts/store_heic.py`: the 15 converted into `trainset3/`, `heic.json`
   written; `store.V3`; `freeze_store_test_set.py --version v3 --photos`.
2. Look: a contact sheet of the 15 upright (`load_upright`), to see each
   is the right way up; Label Studio's drawn size against `upright_size`
   (B2 check 5) confirms it once labelled.
3. Label Studio: project `Store set v3`, storage `trainset3`, 15 tasks.
   Owner labels by B1 (about 30 minutes), exports JSON-MIN to
   `data/store/v3/label_studio_export.json`.
4. `import_store_labels.py --version v3`, then
   `freeze_store_test_set.py --version v3`.
5. Design ids: Claude drafts `designs_v3.csv`, owner approves;
   `store_designs.py --version v3 --freeze`; the split and its counts into
   `results/store_designs.md` (v1's rows unchanged).
6. Keyword boxes: `propose_keywords.py`, `draft_keywords.py`,
   `import_store_keywords.py`, each `--version v3`; owner reviews 15 in
   `Store v3 keywords`.

**Tests** (made-up names): `V3` lists only `trainset3/`, and none of v1's
or v2's sets lists it; a v3 photo can't enter v1's split, nor a v2 photo
v3's; v1's split hashes as before; a known design takes its v1 side; the
v3 design freeze refuses a missing or doubled photo and an id typed two
ways across files; the keyword tools refuse a v2 photo; v1's keyword file
rebuilds byte for byte with `--version v1`.
**Check:** 15 photos, every file frozen and loading with its hash; v1's
`labels.json`, `designs.json`, split and `v1_keywords.json` byte-identical
before and after; Label Studio's v1, v2 and `Store v1 keywords` projects
unchanged (fingerprints, as S5); no photo or design name in a committed
file (searched).
**Commits:** `feat: store set v3, 15 more photos converted and listed`,
`feat: store set v3 labelled, its designs and split frozen`,
`feat: keyword boxes on store set v3`

**Done 2026-11-04.** Store set v3 is 15 photos, every file frozen and
loading with its hash: photo list `8b279416…`, labels `6b607800…`
(`data/store/v3/train_set.json`), designs `03647906…`, split `edd2c25c…`,
and 61 keyword boxes in `data/store/v3/keywords.json`. v1's `labels.json`,
`designs.json`, split and `v1_keywords.json` are byte-identical to before,
and v1's keyword file rebuilds byte for byte from its export with
`--version v1`. Label Studio's projects 1 to 3 were fingerprinted before
the first write and after the last, and are unchanged; project 4 too, from
its labels' freeze on. 611 tests pass (602
before). The first two commits each pass the suite on their own (604,
608). What was found:

- **The photos.** 15 HEIC, 4,032 × 3,024; no name matches a v1 or v2
  photo, and no JPEG is byte-identical to one. Converted once by
  `scripts/store_heic.py` (sips-316, macOS 26.6.2, quality 95) into
  `trainset3/`; `data/store/v3/heic.json` keeps each HEIC's hash beside its
  JPEG's. The script refuses to run while Scanify's git doesn't ignore
  `trainset3/`: added to its `.git/info/exclude` first, confirmed with
  `git check-ignore`, no tracked file of Scanify's changed. **14 of the 15
  carry an EXIF turn** (13 a quarter, 1 a half), though `sips` reported
  none on the HEICs: a contact sheet through `load_upright` showed all 15
  the right way up, and every box was drawn on the upright frame (B2's
  check 5).
- **`store.V3`** is a `StoreSet` with `training=True`: its frozen labels
  are `train_set.json` and their constant `LABELS_V3_SHA256`, so no file
  calls it a test set; errors say "store set". The freeze script writes
  `results/store_set_v3.md` (counts and hashes only).
- **(changed, owner's request) Claude drafted the label fields**, as S4's
  design ids. The owner drew every date box in project 4, `Store set v3`
  (v1's interface and storage settings, 15 tasks: the frozen list) and
  typed no field; Claude read every date from the owner's boxes at full
  size and wrote the tag, expiry, mfg, format, derived and notes into the
  owner's drafts through the API; the owner checked them in Label Studio,
  and Claude submitted. **A misread was frozen, then fixed:** one lid's
  `DOP: 10/23` was drafted as `10/25` and frozen (`bce9c59d…`); found at
  full size while naming the designs, before anything used it. Its file
  went to `data/store/backups/`, every other field was read again against
  its full-size print (all held), and the labels were frozen again with a
  history line saying why.
- **(departure from B1, owner's call) A packing date as mfg.** One pack
  (3 photos) prints "Packed on" and "Expire Date", no manufacture date.
  B1 makes a packing date `other_date` with the mfg field blank; the owner
  kept the box as `mfg` and had its field filled with the packed date,
  2026-02-05. v1 and v2 stay as labelled.
- **The labels:** 15 photos, every one with an expiry (7 to the day, 8 to
  the month, 3 derived from a shelf life); sideways 8, clean 6,
  small-print 1 (the small-print rule moved 1 from clean); 30 date boxes
  (15 mfg, 12 expiry, 3 other).
- **The designs** (Claude drafted, the owner approved, as S4): **9**, 2
  of them v1's (4 photos, both in v1-train) and 7 new; none is in v2, so
  v2's seen and unseen designs don't change. The freeze refuses an id
  that differs from a v1 or v2 id only in case or spaces.
- **The split:** the 4 photos of v1-train's designs train; v3's 7 new
  designs split by S4's rule, 2 of 5 sideways and 1 of 2 clean to
  validation. **10 photos train (8 expiries), 5 validate (4).** With v1's,
  v1-train holds 80 photos and 64 expiries, v1-val 34 photos and 29
  expiries (25 before).
- **Keyword boxes.** The keyword tools take `--version v1|v3` and refuse
  v2; v3's files are in `data/store/v3/keywords/`. OCR read 480 lines and
  proposed **32 boxes on 12 photos**, none over a date. Claude looked at
  all 15 at full size: 33 edits on 12 photos (19 boxes added by eye or
  from an OCR line, 10 from the second OCR pass), a draft of **61 boxes on
  14 photos**, loaded as project 5, `Store v3 keywords`, beside OCR's. The
  owner changed 10 photos; Claude submitted all 15. **One check caught a
  late edit:** the owner's open editor saved a draft on one photo 6 s
  after the submit; check 1 refused the export, and at the owner's word
  Claude copied that final version into the annotation and deleted the
  draft. All four checks pass, and the import twice gives the same file
  (now dated by the review's last change, not the day it runs).
- **The keyword boxes:** `due` 17 on 14 photos, `prod` 25 on 14, `code`
  19 on 12; 61 on 14 photos (the cap has no keyword).
- **OCR's help**, by region id: of its 32 proposals, **kept 0, moved 32,
  deleted 0; 29 added.** 16 of the 32 overlap their final box at IoU 0.5
  or more. OCR found 32 of the 61 final keywords (52%; v1's 63%): v3 is
  mostly stamped and sideways print, which it reads least.
- **The keyword patterns miss three store words**: `Expire Date:`, `BBD:`
  and `DOP:` (boxed by hand here). The patterns weren't changed, so v1's
  proposals stay as made; adding them is in [Later](#later).
- **Photo names.** The committed files were searched for every v1, v2 and
  v3 photo name (and stem, 5 characters or more), the HEIC names and every
  design name: none.
- **Not checked:** whether the owner looked at the 5 keyword photos left
  as drafted (Label Studio keeps no record of a view); the second OCR pass
  as a committed script (its output is in `edits.json`); S6's use of v3 (it
  isn't written yet); the learnings guide (not written).

### S6. v1 as training and validation data · laptop, then the cluster

**Goal:** v1-train and v1-val exported, checked, and on the cluster
(v3's parts with them, from S5b).

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

**Done 2026-11-04.** `scripts/export_store_yolo.py` wrote store sets v1
and v3 into `$STORE_TRAIN_DIR` in 17 s on the laptop's CPU, from code
`9b42b28` with a clean tree: 80 photos train in 4 turns (320 images), 34
validate, 710 files and the manifest, 160 MB. The copy in
`~/bestbefore-data/store/` (folders 700, files 600) is byte-identical,
file by file. 627 tests pass (611 before). What was found:

- **S4 to S5b were done**, checked first: v1's labels (`7b9a4043…`,
  every photo hashed), v3's photo list and labels, `designs.json` (typed
  against the frozen v1 and v2 labels) and `designs_v3.json`, both splits
  (`4418c65b…`, `edd2c25c…`) and both keyword files (341 boxes and 3 added
  dates; 61 boxes) load with their hashes. v2 was checked frozen by
  hashing its two JSON files only; no v2 photo was opened, here or by the
  export.
- **The export** (`store_train.export_plan` and `write_export`; [The
  export](#the-export), changed before it ran): each photo decoded from the
  bytes its frozen hash names, upright, in sRGB, shrunk with PIL's BOX
  filter to at most 1,600 px (the 63 photos of 4,032 px: 48 of v1's, all
  of v3's; v1's other 51, 1,599 or 1,600 px, keep their size), saved at
  quality 95 without metadata and numbered by set in photo order (`v1-0001`
  to `v1-0099`, `v3-0001` to `v3-0015`). Every expiry, mfg and other date
  is a `date` box (S5's 3 added dates too), every keyword box keeps its
  class. Folders `images/{train_0,train_90,train_180,train_270,val}/` and
  their `labels/`; `train.txt` and `val.txt` hold `./images/…` lines, valid
  on both machines as they are (Ultralytics replaces every `./` in a line,
  so the lists sit at the export's root, where that prefix is the only
  one); designs are `d01` to `d47`.
- **Counts** (the manifest's): v1 70 photos train and 29 validate, v3 10
  and 5; 31 designs train and 16 validate, none on both sides. Per photo,
  training holds 169 dates (64 expiries, 75 mfg, 30 other, 2 of them
  added) and 92 `due`, 138 `prod` and 82 `code` boxes; validation 63 dates
  (29, 31, 3, 1 added) and 31, 34 and 25. These are the frozen files'
  counts (64 and 29 expiries, as S5b had them). After the fixes below,
  training holds 74 mfg, 31 other (3 added) and 137 `prod`.
- **Boxes stay on their print.** The tests use made-up phone photos (a red
  date and a blue keyword off-centre, stored with 7 of EXIF's 8 flags,
  mirrors included, bigger than 1,600 px): every box covers its colour
  within 2 px after the EXIF turn, the shrink and each turn. On the real
  export every box reads back within 0.0008 px of where it was meant; each
  turned copy, turned back, matches its upright copy within 1.19 grey
  levels (mean absolute difference) and its boxes within 0.0016 px; every
  photo's size is its upright size shrunk by the rule; and each v1 photo
  matches its source, upright and shrunk, within 0.10 to 1.65 grey levels
  (JPEG's re-encoding).
- **Uprightness: v3 as S5b checked it, and further.** B2's check 5 again:
  every date box of v1 and v3 (drawn on 92 and 15 photos) and every region
  of both keyword projects (543 and 91) was drawn on the upright frame.
  That check can't see a half turn, which keeps width and height, so v3's
  HEICs were read as well: each one's own rotation (its `irot` box, no
  `imir`) agrees with its JPEG's EXIF flag (13 quarter turns, 1 half, 1
  none), so `sips` turned nothing. Two v3 photos show their pack upside
  down: the packs were held that way, both tagged sideways, their boxes on
  their print (S5b's sheet found all 15 the right way up, true of the
  camera, not of these two packs).
- **Colour and metadata.** v3's 15 photos are Display P3 (v1's carry no
  profile). Put into sRGB (relative colorimetric), they change by 0.9 to
  8.1 grey levels on average and by up to 62 in saturated colours (99th
  percentile of the largest channel change). 20 of v1's photos carry a GPS
  position in their EXIF. Pillow 12.3 writes no EXIF or colour profile
  unless asked, and the export checks that every file it writes has no
  EXIF, so no copy holds a turn, a GPS position or a camera's name. OpenCV,
  which Ultralytics reads with, gets the same pixels as PIL (a test).
- **Ultralytics reads it** (8.4.177, on a scratch copy, as its label cache
  goes beside the labels): `train.txt` 320 images, 0 corrupt, 1,924 boxes
  (4 times the per-photo counts); `val.txt` 34 images, 0 corrupt, 153
  boxes, 2 photos without one (bottle fronts). On the cluster that cache
  will land in `~/bestbefore-data/store/labels/`; the copy check ignores it.
- **The turn check's band** (`TURN_NOISE`): a right turn scores at most
  1.19; a wrong one (a 180° copy left unturned) 3.4 to 99, median 59, the
  least on a photo that is 97% white. The band was 3.0 for the first run;
  it is 2.0 from this measurement, and the export was run again: its 710
  files are byte-identical to the first run's, only the manifest's commit
  and code hash changed.
- **Every photo looked at, box by box** (29 sheets of photos with each box
  enlarged and 6 sheets of the 80 training photos in their 4 turns, in
  `data/store/runs/`, drawn from the export's own files; Claude, as allowed
  since 2026-10-28): every box on all 114 photos and 240 turned copies
  sits where its label puts it. **Three label problems** came up, all in
  files the export reads, none made by it; they are listed by export
  number in `data/store/export_review.md`, on the laptop only:
  - a dot-matrix import date on a jar base has no date box, neither in
    v1's frozen labels nor among S5's added dates, so it would be learned
    as background (the same design's 5 other jar bases have theirs boxed);
  - one `prod` box sits on a shelf life (`… from Mfg. Dt.`), against S5's
    box rule and the same design's other photos;
  - v1's frozen labels hold one stray `mfg` box of 1.1 by 3.8 px beside
    the real one: the only box under 7 px of 634. Ultralytics drops boxes
    under 2 px after augmentation, but S8's date sizes would count it.

  Minor: times, and batch values printed apart from their label, are boxed
  `code` on some photos and not on others. **None was fixed in the first
  export.** The first two are S5's review, fixed in its project and
  imported again; the third is in v1's frozen labels, which S5's checks
  keep as they are (a revision is M0 Part D's Later). The owner decides;
  any fix means exporting and copying again, seconds each.
- **The three fixed, at the owner's word** (the same day). The owner
  signed in to Label Studio in the browser pane; Claude made the edits
  through its API, as in S5, with the database backed up to
  `data/store/backups/` first and all 5 projects fingerprinted before the
  first write and after the last. Only the two photos' annotations in
  `Store v1 keywords` changed (and their tasks' times and the project's
  label summary); no draft is left.
  - The import date: an `other_date` box in the date layer, as S5's added
    dates, drawn on the full-size photo 3 px outside its ink (measured
    column by column). Its `IMPORT MONTH:` box, drawn by eye in Claude's
    S5 draft, ran over the date's first digit, as no date box stopped it;
    it now ends after its colon. A fourth date the frozen labels miss.
  - The shelf life's `prod` box, one the owner drew in S5's review, is
    deleted.
  - The stray box is left out by the export: any box under 2 px across on
    its exported photo, counted ([The export](#the-export), changed before
    the code). A revision of v1's labels would change the hash that
    `designs.json`, the keyword file and both splits are checked against,
    for one box of a pixel. v1's frozen file keeps it; it is the only box
    left out. A test covers the rule (a slip among the frozen dates and
    one among the added ones; 628 tests pass, 627 before).
  - **The import**: all four checks pass; `v1_keywords.json` differs from
    S5's in exactly those boxes, and the import twice gives the same bytes:
    340 keyword boxes (`prod` 146) and 4 added dates. OCR's help: added
    125, so OCR found 215 of the final 340 keywords (63%).
  - **The export again**, from `eae913c` with a clean tree: all 354
    images and both lists byte-identical to the first export; 12 label
    files changed, the three photos' in each of the 4 turns, as the edits
    say; the box and turn checks as before (worst 0.0008 px, 1.19 grey
    levels). The three photos and their turned copies were looked at:
    every box on its print, the import date's box tight on its digits.
  - **On the cluster**: the laptop's copy was made owner-only (folders
    700, files 600) before rsync, so `-a` carried that across: no entry was
    open to others at any time (0 after the sync, 0 after `chmod`).
    `check_export` found 0 problems over 710 files; all 711 files are
    byte-identical by SHA-256, the 354 label files among them. Jobs A and B
    untouched; nothing pulled back. Label Studio was stopped afterwards,
    its projects as after the edits. The first export's copy on the laptop
    was deleted, once compared.
  - Not fixed: the minor uneven keyword boxes above (the owner asked for
    the three).
- **The store lines** (`m2b.store_lines`, fed by `m2b.store_pool`): at the
  plan's 5,000 lines (20% of 25,000) each of the 31 training designs gets
  161, 4,991 in all, and a photo in one turn appears 3 to 41 times an
  epoch. 18 of the 31 designs have one photo, so 18 photos fill 2,898
  lines, 11.6% of every epoch, 161 lines each (about 1,650 over the
  fine-tuning budget, augmented); the largest design's 13 photos get 12 to
  16 each. That is decision 7 as written; S10 may cap it (nothing changed
  here).
- **On the cluster.** `~/bestbefore-data/store/` was made with `mkdir -m
  700`, then rsync sent 710 files, 166 MB, in 11 s, and the manifest last.
  **The laptop's rsync is macOS's openrsync** (protocol 29): it refuses
  `--chmod=D700,F600`, accepts `--chmod=u=rwX,go=` but applies nothing, and
  `-a` copied the laptop folder's 755 onto the folder made 700. So 724
  entries were readable by others for under two minutes, until `chmod`
  (folders 700, files 600; none with a group or other bit since). Nothing
  was reachable meanwhile: the home folder is 700. Then the manifest's
  SHA-256 was the same on both machines; `check_export`, its source run
  with the cluster's Python 3.11 as a test runs it, found 0 problems over
  710 files; all 711 files are byte-identical by SHA-256, the 354 label
  files among them; the image counts are the manifest's; no names map is
  there. M2's folders kept their times (checkpoints, ExpDate, generated);
  jobs A (`73831`, pending) and B (`73832`, held) were not touched;
  nothing was pulled back.
- **Details fixed while writing:** the names map outside the export, in
  `data/store/train_names.json` (a departure written down before the
  export ran, [The export](#the-export)); sRGB and no metadata (added
  there); the manifest also holds each photo's part, design number, tag
  and box labels in label-file order, and every file's SHA-256; the export
  writes into an empty folder only and refuses a photo changed since its
  freeze; it refuses to start while v2's photo list or labels hash is
  unset, and checks that `designs.json` was typed against v2's frozen
  labels, comparing hashes held in code and in that file, opening no v2
  file; `check_export` uses only the standard library, so its source runs
  where the package isn't installed (the cluster's checkout is still
  `2b68e2e`, not pulled, as job A waits on it).
- **Photo names.** The changes were searched for every v1, v2 and v3 photo
  name (and stem, 5 characters or more), the HEIC names, every design id
  and every word of one: none (5 common words, such as "pack", match).
- **Not checked:** training on the export (S10); `m2b.mix()`, and the
  cluster's `.env`, which has no `STORE_TRAIN_DIR` yet (S10);
  `scripts/export_store_yolo.py --check` on the cluster as a script (its
  function's source ran there); whether every keyword is boxed (only the
  uneven cases seen are noted); the learnings guide (not written).

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

**Done 2026-11-05.** Code `fe17238` (one commit); 663 tests pass (628 before:
35 new in `tests/test_synth_v2.py`, and one check added to
`tests/test_synth_batch.py`), and the commit passes alone, in a worktree with
the laptop's data and nothing skipped. What was found:

- **M2's generator didn't move.** `spec_hash(GEN)` is still `3fc14178…` and
  the strata's `ba77f1bd…`. S5's 500-photo sample, made again with `GEN` in
  memory, gives its 500 JPEGs, label files and sidecar lines byte for byte,
  both with M2's code before any change and with S7's (about 10 s on 4
  processes; now a test, which writes nothing). The render sheet (seed 0) is
  S4's, `e1aa1311…`, before and after. And M2's text draws exactly what it
  drew: a digest of 10,000 blocks and of the random generator's state after
  them, taken from M2's code before any change (`c74bec80…`), is pinned by a
  test, so one draw more or fewer fails it. Nothing of M2's was written: its
  photos, strata, lists and models were only read.
- **`V2Spec`** is `GenSpec.v2`, `None` in M2's specs. Its parts follow
  `GenSpec`'s: `text` (`V2Text`) and `style` (`V2Style`, the line spacing per
  kind), so S8 adds its print (nozzle rows, the window) to `style` and parts
  for the upside-down blocks, the sizes and the worn patches beside them.
  `spec.spec_dict` leaves `v2` out when it is `None`; the hash, the strata's
  hash and the manifests (`batch.check_set`, `generate.py --set`) all go
  through it, so a check run again on M2's sets would write M2's spec as
  before, without a `"v2": null` (a test). A spec with a `V2Spec` hashes
  differently, every number of it included. The new kinds' shares sit in
  `TextSpec.kinds`, as M2's do: `TEXT_V2` holds the plan's seven, and
  `text.random_block` refuses them without gen-v2's text. `GEN_V2` waits for
  S8, which has the sizes.
- **The rupee sign: `Rs.`** 4 of the 20 families have no glyph for it: Open
  Sans, Barlow Condensed, Roboto Mono and Share Tech Mono, all training
  families (the 4 held-out ones have it). So the new distractors print
  `Rs.`, as the plan's examples do and M2's `MRP Rs.` already did;
  `V2Text.currency` holds it, and a test reads every face's character map.
  The two new characters, `|` and `&`, are in all 24 files; `text.CHARSET`
  holds 69 characters (67 before), which M2's font check tests in every
  face.
- **Details fixed while writing** (by rule: no share of the plan changed,
  and none came from a store photo):
  - **Glued, per word and date.** First whether the date is glued (30%),
    then its word: one of the 15 of the 26 words that end in `:` or `.` if
    it is, any word if not. So the 30% holds over every word printed before
    a date: a line-pair's two pairs each (the plan's own example glues one
    and not the other), and ranges' and relative blocks' mfg lines too.
  - **A line-pair is one line**: the mfg date with the expiry or, half the
    time, an import date; the mfg pair first 80% (`TextSpec.mfg_first`, as
    a pair's lines); between them ` | `, two spaces, ` / ` or `, `, each a
    quarter. ` / ` is spaced, so a slash isn't printed into a numeric date,
    and `, ` stands against the first date. The mark is a token with no box,
    an empty one for two spaces. A line-pair with an import date prints no
    expiry.
  - **A dense panel's import line is one of its 3 to 5 extra lines** (the
    plan's brackets list it among them), placed anywhere, as a panel's extra
    lines are; the rest are codes and distractors, half each.
  - **An import date** is 1 to 6 whole months after mfg
    (`V2Text.import_months`; the plan's example is a month apart), always in
    a month before the expiry's, its day kept as `add_months` keeps it, in
    the block's format.
  - **Distractors: every template equally likely**, M2's six and the four
    new (barcode numbers run together or spaced 1-6-6, and two unit prices),
    so 40% of gen-v2's distractor lines are new; M2's text keeps its six.
    A barcode number is 12 random digits and their EAN-13 check digit; a
    unit price is Rs. 0.10 to 9.99 per g or ml (`USP Rs. 1.50/g`), the long
    form with one decimal (`Unit Sale Price: Rs. 1.5 per ml`), the shapes of
    the plan's examples.
  - **M2's draws keep their order.** Every new choice is drawn only with
    gen-v2's text. One rewrite would have moved M2's stream without changing
    a single token's meaning: a panel's extra line drawn inside the call
    that inserts it is drawn after its place, not before. It is drawn first,
    as M2 drew it; bug 5 below shows the digest catches that.
  - **`make.py` is unchanged**, so nothing draws gen-v2 photos yet. The plan
    gives `make.py` to S8 with the sizes; S8 passes `spec.v2`'s parts to
    `text.random_block` and `render.render`.
  - `scripts/render_sheet.py --v2` draws the new forms (run name
    `render-sheet-v2`); without the flag it draws M2's sheet, the same
    bytes.
- **Shares**, as the tests measure them (`tests/test_synth_v2.py`): over
  10,000 blocks of `TEXT_V2`, and the new kinds' own over 10,000 of each.
  Every one is within 1.4 points of its spec:

  | Share | Spec | Measured |
  |---|---:|---:|
  | kind: single / pair / panel / dense panel / line-pair / range / relative | 25 / 30 / 15 / 10 / 10 / 5 / 5% | 25.1 / 29.4 / 15.1 / 10.2 / 10.3 / 4.7 / 5.4% |
  | glued, of the 16,044 words printed before a date | 30% | 30.6% |
  | distractor lines (3,237): barcode / unit price | 20 / 20% | 20.4 / 19.0% |
  | a panel's or dense panel's other lines: codes | 50% | 50.6% |
  | month word / full name / a day on numbers / on words / an ordinal | 45 / 50 / 70 / 40 / 50% | 45.1 / 49.9 / 69.4 / 40.9 / 48.6% |
  | case of month words: upper / title / lower | 50 / 45 / 5% | 50.1 / 44.8 / 5.1% |
  | a single date with its word; the mfg line first | 70%; 80% | 69.3%; 80.0% |
  | line-pairs: an import date / the mfg pair first / glued | 50 / 80 / 30% | 50.1 / 80.1 / 30.2% |
  | line-pairs: ` \| `, two spaces, ` / `, `, ` | 25% each | 24.7 / 24.8 / 25.0 / 25.5% |
  | dense panels: an import line / 3, 4, 5 more lines / codes / mfg first | 50 / 33.3 each / 50 / 80% | 50.2 / 34.0, 32.5, 33.5 / 50.3 / 79.4% |

  M2's month-word and format shares hold in gen-v2's text, as the plan
  asks; the dot-matrix share is S8's (styles).
- **The new kinds' own shares over 10,000 of each**, as M2 S4 measured the
  ranges': at 10% of 10,000 blocks, 3 points is under 2 standard errors. A
  first run measured a dense panel's import share over its 1,000 dense
  panels at 46.4%; over 10,000 it is 50.3% (another seed: 50.2% in the
  table).
- **M2's text test counts every relative block as mfg first** (its one
  date is the mfg), so it holds pair, panel and relative blocks to 81.7%
  where the spec says 80%, inside its 3 points. S7's test compares the
  mfg line's place with the expiry's or the shelf life's instead; M2's test
  is left as it is.
- **Boxes come from the ink** (tests, every style, digits 6 to 24 px): each
  boxed token's box holds all of its own ink, found another way (its 4x
  sprite alone, shrunk by a plain mean), and touches it within 1 px; a
  glued date's box holds none of its word's ink but for one boundary column
  they may share, and the word's box ends where the date's starts; a
  line-pair gives two date boxes and two keyword boxes, left to right and
  apart, its separator unboxed; import words are `prod`, their dates role
  `import`.
- **Dense panels at 1.0 to 1.2 em**: in 11 of 600 (1.8%), two boxes on
  neighbouring lines overlap, by 3 px at most, 4 of the overlapping pairs
  with a date in them; M2's panels, at 1.2 to 1.6 em, 0 of 600 (a scratch
  measure, every style, digits 6 to 24 px). Each box is still its own
  token's ink (the tests).
- **By eye, the render sheet with the new kinds** (`render_sheet.py --v2`,
  `results/runs/20261105-144434-render-sheet-v2/`, from `fe17238` with a
  clean tree, SHA-256 `482eec78…`; the same bytes as the copy looked at
  before the commit). Rows are M2's print styles, columns a glued pair, a
  line-pair with the expiry, a line-pair with an import date, a dense panel,
  and a dense panel with its import line and distractors only, each normal
  size over small print. Every date, `due`, `prod` and `code` token has its
  box, tight on its ink at both sizes in all five rows, and no separator or
  distractor has one (barcode numbers, unit prices, phone numbers, weights,
  licence numbers). Glued words read as printed against their dates, each
  date's box starting where its word's ends. Dense panels are plainly
  tighter than M2's panels; at 1.0 em their lines nearly touch,
  condensed and embossed ones most. `&` and `|` draw in every font shown.
  As M2 S4 found for Oswald, 7-row dot-matrix in condensed fonts is hard to
  read, and there a colon and a full stop look alike (`Unit Sale Price:`).
  The sheet crops the longest line-pairs at its 900 px cells (the sheet's
  cell, not the drawing). M2's sheet, drawn again from the same commit, is
  S4's (`e1aa1311…`).
- **The tests catch what they should.** 20 deliberate bugs, one at a time,
  each caught by the synth tests: the glue share inverted; a glued word
  drawn from every word; a glued date keeping its space; M2's text drawing
  a glue decision too; M2's panel line placed before it is drawn; `v2:
  None` kept in M2's spec as data; the line-pair separator boxed as a code;
  a line-pair's import date in place of its mfg date; a line-pair's mfg
  pair always first; the word after `, ` losing its space; an import word
  boxed `due`; a dense panel's import date given the role `mfg`; an import
  date allowed in the expiry's month; a dense panel's extra lines as a
  panel's, 1 to 3; its import line on top of its 3 to 5; line spacing per
  kind ignored; a dense panel's spacing for every kind; the new distractors
  never drawn; a barcode's check digit weighted 3, 1, 3 …; and the rupee
  sign printed though fonts lack it. M2's two stream bugs fail the pinned
  digest and the 500-photo sample, the hash bug the hash test and M2's
  batch check. Two side catches were chance, and are recorded:
  - with bug 9's one draw fewer, a box test's 40 random blocks held no
    dense panel, which its own guard caught; that test now draws its dense
    panels and line-pairs on purpose;
  - with bug 4, M2's own check failed the batch tests' 6-photo set: its
    "month-word dates: a day printed" fell outside 3 standard errors. The
    dates of one block share a format, so they aren't independent draws,
    and on so few the bound is tight. True of M2's check before S7; it
    matters for S8's and S9's small gen-v2 test sets.

  Every patched file was put back from a copy and compared by SHA-256
  after each run; the tree is as before.
- **Searched**: S7's changes (every added line, the new test file and this
  record among them) for every v1, v2 and v3 photo name (and stem, 5
  characters or more), the HEIC names, every design id and every word of
  one (4 characters or more): no name and no id; 4 common English words of
  design ids match (such as `white`). The words, distractors and examples in
  the code and tests are made up.
- **Not checked:** gen-v2's text through warp, ink, camera and the sidecar
  (`make.py` is S8's), so neither a v2 block on a photo nor its sidecar
  record (whether a date is glued isn't kept anywhere yet: the sidecar
  keeps tokens, not their spaces); whether the new kinds look pasted (S8's
  look, beside store crops); the cluster's pixels for the new kinds; how
  much time the new kinds add (S8 times a photo); the learnings guide (not
  written).
- **For S8:** `make._block` passes `spec.v2.text` to `text.random_block` and
  `spec.v2.style` to `render.render`, and only when `spec.v2` is set, so
  M2's sidecars stay as they are; the sidecar marks each glued date (its
  `space_before`), so the sample's table can measure the glued share;
  `make.shares` adds gen-v2's rows (glued, import in line-pairs and dense
  panels, the separators, dense panels' lines, the new distractors).
  `GEN_V2 = replace(GEN, text=TEXT_V2, ..., v2=V2Spec(...))`.

### S8. Generator v2, part 2: print and geometry · laptop

**Goal:** nozzle rows, the laser window, upside-down blocks, worn patches
and sizes like the store's. Needs S4's and S5b's splits: the sizes come
from v1-train, v3's training part included.

- `synth/render.py`: nozzle rows in dotify; the `window` style.
- `synth/warp.py`: the upside-down orientation (`numpy.rot90`, k=2).
- `synth/compose.py`: worn patches among the ink defects, the boxes taken
  before them.
- `scripts/store_sizes.py`: v1-train's date sizes, measured as the store
  measures them, and their deciles, printed for `V2Spec` (aggregates only).
- `synth/make.py`: a block drawn at a provisional size, its first date's
  box measured, then drawn again to the target with the same random
  draws; the sidecar records the target, the box reached and whether it
  shrank.
- `synth/spec.py`: `GEN_V2`, with the deciles.
- `scripts/generate.py --set gen-v2-sample --count 500`.
- **Tests:** a dropped dot row has no ink in any line of its block, and
  its token's box still holds the rest; a window's text is lighter than
  its rectangle; 180° maps boxes exactly; a block drawn to a target lands
  within 1 px or 5% of it in every style and orientation **(changed
  2026-11-05, owner's call: at least 99% of blocks, and 95% in every style
  and orientation; the rest keep their closest drawing; changed 2026-11-06,
  owner's call: tilted blocks aren't drawn to a box, and their digits are
  their target over 1.3 exactly)**, and the second
  drawing uses the same draws as the first (the same font, ink and
  defects); a worn block's boxes equal the same block's unworn boxes, and
  its patch holds 60 to 95% less ink; a date worn to nothing is drawn
  again; M2's photos unchanged.
- **Look:** contact sheet and date crops of gen-v2-sample, beside real
  ExpDate dates and, now allowed, v1-train's store crops (never v2's):
  does a new kind look pasted, and do sizes look like the store's?

**Check:** every gen-v2 share within 3 points of its rule in the sample's
table; the sample's date sizes, measured as the store measures them
**(changed 2026-11-06, owner's call: its untilted dates)**, within 3 points
of v1-train's in each band (under 1.5%, 1.5–2.5%, 2.5% and up) and their
median within 0.1 points; the time per photo against M2's
(about a tenth more expected); the M2 checks (boxes from ink, same bytes
however split).
**Commits:** `feat: generator v2 print: nozzle rows, laser windows, upside-down blocks, worn patches`,
`feat: generator v2 sizes drawn to match store set v1's training dates`

**Done 2026-11-05; tilted blocks changed 2026-11-06 (below).** Code
`244349f` (print and geometry), the plan's change `c50dcf8` (owner's call),
`f1c39f5` (sizes), `4feeceb` (a fix the look found), `f8e34c5` (tests the
bug run asked for), then the record `5932286`, the plan's second change
`4c495e9` (owner's call) and `08db831` (tilted blocks): five code commits,
not the plan's two, one logical change each. 717 tests pass (663 before: 54
new, all in `tests/test_synth_v2.py`). Each code commit passes alone with
the laptop's data and nothing skipped: 699, 711 and 711 in worktrees;
`f8e34c5` 715 and `08db831` 717 in this checkout. gen-v2-sample, 500
photos, seed 6, from `08db831` with a clean tree (run
`20261106-205902-m2b-generate-gen-v2-sample`), is in
`$GENERATED_DIR/m2b/gen/` with `m2b/manifests/gen-v2-sample.json`.
`GEN_V2` hashes `2c8b24d5…`. What was found:

- **M2's generator didn't move.** `spec_hash(GEN)` is still `3fc14178…`
  and the strata's `ba77f1bd…`; S5's 500-photo sample is remade byte for
  byte (S7's test); M2's render sheet is S4's, `e1aa1311…`, before and
  after. New digests, taken from M2's code before any change, pin what
  M2's geometry (10,000 draws), inks (10,000) and print with its ink
  defects (120 blocks, pixels included, on the laptop's builds) draw, and
  the generator's state after them; and M2's sidecars keep their shape (no
  `lines`, no `space_before`). Every v2 path runs only with `spec.v2` set.
  Nothing of M2's was written: its sample, strata, lists and manifests were
  only read.
- **One of S7's checks changed:** "gen-v2's print changes nothing
  for M2's kinds" no longer holds for dot-matrix, whose gen-v2 blocks also
  draw their nozzle rows. For dot-matrix the test now holds only the draws
  before its first dot (the same font, size, spacing and dot grid), and a
  new test holds S7's own claim strictly: the line spacing per kind changes
  nothing for M2's kinds, in every style, dot-matrix and the window too.
  Solid, label and embossed print still draw exactly as M2's.
- **`V2Spec`** holds, beside S7's text: `style` (nozzle rows, the window),
  `geometry` (upside-down, ±8°), `compose` (worn patches, the window's ink)
  and `size` (below); the new shares sit in `STYLE_V2` (window 10%, taken
  from solid, dot-matrix kept at 35%) and `GEOMETRY_V2` (upside-down 10%,
  from upright), as S7's kinds in `TEXT_V2`. `GEN_V2` is `GEN` with those
  and `V2`; everything else is M2's. `SET_SEEDS` gains gen-v2-sample's 6
  (S9 adds 4 and 5). gen-v2's sets live under `$GENERATED_DIR/m2b/`
  (`m2b.GENERATED_SETS`, `batch.home`); `generate.py --set gen-v2-sample`
  draws with `GEN_V2` and records the dates' sizes.
- **v1-train's sizes** (`scripts/store_sizes.py`): read from the export's
  training photos, upright, checked against its manifest first (0
  problems), so v1-train with v3's training part and nothing else: **169
  dates** (64 expiry, 74 mfg, 31 other, 3 of them added by S5's review; the
  1.1 px slip left out, as the export leaves it out), other-face photos'
  6 included (Cause 4's 192 left every other-face photo out, and counted
  all of v1). Bands **35.5 / 42.0 / 22.5%**, median **1.736%**; deciles
  0.659, 1.118, 1.296, 1.451, 1.573, 1.736, 1.974, 2.181, 2.559, 3.247,
  5.704%; none under M2's floor, 3 over its ceiling, so the top edge is
  clipped to 4.5% (`V2Size.deciles`). The script says whether `V2Size`
  holds them (it does).
- **Drawing to a target: one correction fell short, and the owner chose
  the fix** (changed in the plan before the code, `c50dcf8`). Drawn once,
  measured and drawn again, 91.8% of 2,000 blocks landed within 1 px or 5%
  of their target; 27% of tilted dot-matrix blocks didn't, as a box of
  dots or relief grows in steps (one target of 11 px gave 9, 12, 9 and 13
  px). A second correction from the same draws gave 97.7%; the owner's
  choice, a second correction while off and then fresh draws (3 tries, as
  for lost ink), keeping the closest drawing, gave 99.9% for 2.15 drawings
  a block. Per style and orientation, 100 blocks each: every one 96% or
  more but tilted embossed print, 90 of 98 at first, then **95.2% of 290**
  (96.3% with 5 tries, 96.9% with 7), 1.5% of all blocks. So the test holds
  99% of 500 blocks drawn as gen-v2 draws them, and 90% in each style and
  orientation over 20 blocks each, a guard: **lower than the 95% per style
  and orientation written into the plan (Claude's proposal), because 20
  blocks can't tell 95% from 90%**; the 95% was this record's measurement.
  **Since 2026-11-06 tilted blocks aren't drawn to a box at all** (below),
  so those figures for them are history. Untilted, in the final sample:
  **576 of 576 unshrunk blocks** fit; over 5,000 photos 5,781 of 5,783; 1.98
  drawings a block over all blocks.
- **The same draws at another size** (a detail fixed while writing): the
  plan saves the generator's state before the first drawing, but the dots
  and the density field draw a number of values that depends on the size,
  so from one saved state a second drawing's defects would differ. So each
  try takes one number from the photo's generator, and the render, the ink
  defects, the density field and the wear each draw from their own
  generator of it: a drawing at any size makes the same choices (font,
  weight, spacing, dot grid and nozzle rows, window, relief, label, and the
  same defects record: density, spread, the scratched token, the worn
  patch in characters and lines; a test). The ink is chosen once, after
  the last drawing. Every v2 look is drawn before its first dot.
- **Details fixed while writing** (by rule; no share changed, none from a
  store photo):
  - **Nozzle rows** are rows inside a digit's height, never its top or
    bottom row, so a dropped row stays inside its token's box, as the plan
    says; they are left out after every dot is drawn and the sprite sized,
    so a block with dropped rows is the same block less those dots (a
    test), and the rows' numbers are drawn either way.
  - **The window**: near-black, dark blue, dark green and dark brown
    rectangles (M2's dark inks, with a green), 0.3 to 0.8 em of margin, a
    corner up to 0.1 em; strokes a grey of luminance 0.75 to 1.0 at M2's
    opacity, soft by 0.01 to 0.03 em (0.02 to 0.05 looked mushy by eye;
    M2's spread comes on top), with a glow of 0.06 to 0.15 over 0.10 to
    0.25 em, under the ink threshold so a glow alone is never ink.
    Dotified windows keep their dots' own blur (the softness on top halved
    their ink: mid-grey dots). The floor on faint ink measures a window's
    strokes against its rectangle. **(fixed after the look, `4feeceb`)** The
    rectangle multiplies the pack, as printed dark ink does: laid over like
    a sticker, it looked pasted on.
  - **Worn patches** centre on one of the block's dates, picked uniformly,
    anywhere along it; a character is the digits' height (as M2's density
    field), a line the block's line pitch; 1 to 2 lines tall; a soft edge
    over 15 to 35% of each half size, the fade (60 to 95%) full over the
    core. They are laid on in the print's own frame after M2's ink
    defects, and the boxes come from the unworn twin. **v2 lays M2's ink
    defects on before the warp** (M2 after it), so the twins share them
    exactly. A token worn to nothing, any boxed one (M2's rule on what is
    left), has its block drawn again.
  - **Upside-down** is a part of its own (`V2Geometry`); M2's specs refuse
    it, as S7's text refuses the new kinds.
- **The sample's shares**: 64 rows, every one within 3 points or 3
  standard errors (`share_ok`); **10 more than 3 points off**, at most 2.5
  SE: pairs 26.2% (30%), full month names 44.8% (50%), ordinals 53.5% (50%),
  colour cast 34.2% (30%), the line-pairs' ` / ` 12.3% and `, ` 35.6% (25%
  each, of 73), dense panels' import line 37.0% (50%, of 73), panels' codes
  54.2%, unit prices 23.2% (20%), nozzle rows 20.8% (15%, of 235). So the
  plan's check, every share within 3 points, doesn't hold on these 500
  photos. Over 5,000 made in memory from the same code and seed, only 2 of
  the 64 rows are more than 3 points off, dense panels' import line 45.9%
  and 4 more lines 29.2% (2.1 and 2.3 SE, of 684), and the others above come
  back (` / ` 22.6%, pairs 29.4%, full
  names 49.8%, nozzle rows 16.3%, upside-down 9.7%, tilted 14.8%, window
  10.2%). Noise in 500 photos, as M2 S5 found, and each draw's share is
  held within 3 points over 10,000 draws by the tests. The rest: 687 blocks
  placed, 9 skipped (6 too faint, 3 without a place), 7 shrunk, 17 kept a
  drawing from fresh draws; 1,183 generated dates; boxes date 1,821 (with
  the backgrounds'), `due` 918, `prod` 602, `code` 507.
- **The sample's sizes**, measured as the store measures its own, **over
  its 998 untilted dates (the owner's call, 2026-11-06): 33.2 / 42.0 /
  24.9%, median 1.812%**, against v1-train's 35.5 / 42.0 / 22.5% and
  1.736%: −2.3, 0.0, +2.4 points, median +0.08: **the check holds.** Its
  686 targets drew 24.2% in the top band, 1.7 SE over the deciles' own
  21.5%, and a block's other dates come out a little bigger than its
  fitted first date. Over 5,000 photos, 10,201 untilted dates: 34.0 / 44.6 /
  21.5%, median 1.750% (−1.6, +2.6, −1.0 points, +0.014). All dates, tilted
  ones included, as the plan first measured it and no longer does: 28.4 /
  38.4 / 33.2%, median 1.97%, tilted boxes running big by design. (Before
  the change, over all dates: 32.7 / 41.8 / 25.5%, median 1.812%, its top
  band 3.00002 points over, strictly just outside the check.)
- **Time: +50%, not "about a tenth".** On one CPU, the same 500 photo
  numbers in one session (OpenCV's threads off): M2 54.0 ms a photo
  (drawing 17.3, as S5's 54 to 57), gen-v2 81.0 ms (drawing 42.1). The
  rest is the same. A gen-v2 drawing takes about 15 ms, the first and the
  corrected alike, against about 12 ms for M2's one drawing of a block, and
  there are 2.0 a block: M2 S5 already measured drawing at a third of a
  photo, so one extra drawing alone was about +30%. Tilted blocks gave up
  their extra drawing but draw bigger, so the change saved about 1 ms. On 4
  workers, about 129 ms a photo each, 18 s for 500. At S9's scale, M2 S6's
  150 ms a photo a worker on the cluster becomes about 225: 200,000 photos
  on 24 CPUs in about 31 minutes.
- **Tilted blocks drew tiny, unreadable digits, and are now sized by their
  digits** (found in the first S8 sample, checked and changed 2026-11-06 at
  the owner's word, `4c495e9`, `08db831`). Drawn so an upright box round a
  tilted date had the store's size, a tilted block's digits were a third
  of the box (its box 3.1 times its digits, against 1.33 upright). Over
  5,000 photos, at 1,280 px input: 3.0% of all blocks had digits under 4 px
  and 9.7% under 6 px, down to 1.2 px; of tilted blocks 18% and 44%, a
  third of those tilted 25 to 35° under 4 px; M2's sample has none under
  6 px, its smallest 6.6.
  - **Unreadable**: a blind read (Claude, 53 first dates, shuffled, the
    text hidden until the reads were written; digits at 1,280 px input):
    under 4 px, 1 of 14 exact and 11 with nothing legible, all 14 tilted
    19 to 34°; 4 to 5 px, 2 of 13 exact; 5 to 6 px, 9 of 13; 6 to 8 px, 5 of
    13, where faint ink and bends cost more than size. One reader, few
    crops, but the fall between 6 and 4 px is plain.
  - **The store has none like them**: a line tilted by θ has a box aspect
    under cot θ, so any date tilted 20° or more has an aspect under 2.75
    and 11° or more under 5.14. Of v1-train's 169 only 2 are under 2.75 and
    114 under 5.2, and looked at, none of the 114 is more than about 8° off
    level or vertical; the two lowest are tall condensed print 5° off. Its
    smallest box, 0.66%, is about 6.5 px of digits at 1,280 px. v1-val and v2
    weren't looked at. So the store's box size is its print height; the
    plan's rule, written for boxes, measured the tilt on a tilted block.
  - **The change**: a tilted block is drawn once, at its target over 1.3,
    with no correction (`V2Size.by_digits`); its `fits` is null, and a
    drawing that lost its ink is drawn from fresh draws as before.
    Simulated on the 5,000 photos before choosing, against a floor on the
    digits at 0.5 or 0.6% of the long side, which left 24% or 10% of tilted
    blocks under 6 px and moved all-date sizes without curing them.
  - **After it**: 0 of the sample's 687 blocks have digits under 4 px
    (14 before), tilted blocks' median is 14.7 px, their smallest 6.1. Over
    5,000 photos, 0.23% under 4 px (16 blocks, none tilted, 9 of them
    bends) and 3.1% under 6 px; tilted blocks 0.7% under 6 px, their
    smallest 5.5. Read again, 17
    of 22 tilted dates exact, 2 close, 2 with nothing legible (not fully
    blind: some were in the first key); the misses are faint dot-matrix.
    401 of the sample's photos are byte-identical to the first sample's;
    of the 99 that changed, 98 held a tilted block and 1 a skipped one,
    whose orientation a skip doesn't record.
  Bends keep their box target, as the store's curved dates have the same
  box: 1.5% of untilted bends had digits under 4 px, 11.6% under 6 px.
- **The same bytes however split**: 1 worker and 4 write the same 1,001
  files (500 JPEGs, 500 label files, the sidecar), from `08db831` as from
  `4feeceb`.
- **The look** (Claude, as allowed since 2026-10-28; sheets in
  `data/store/runs/20261105-205647-gen-v2-look/` and, for the final sample,
  `20261106-210514-gen-v2-look/`, with the reads' and v1-train's tilt sheets
  beside them, never committed):
  gen-v2-sample's dates beside ExpDate's real-val dates and v1-train's store
  dates, every photo at 1,280 px, and each new kind zoomed with its
  surroundings. **The sizes look like the store's**: generated and store
  dates crowd the same 1 to 2.2% of the long side, many of both sideways;
  real-val's are bigger (median 2.5%) and crisper. Store print is crisper
  and higher in contrast than gen-v2's faint tail (low-contrast ink,
  embossed relief on flat colour), as Cause 3 measured for M2's dates: not
  S8's to change. **No new kind looks pasted on but the window**: nozzle
  rows read as a clogged head's print, worn patches as rubbed ink over
  part of a date, upside-down blocks as packs held the other way, glued
  words, line-pairs and dense panels as crowded label print, all in the
  pack's surface as M2's print is. The windows' flat rectangles looked laid
  on; printed instead (`4feeceb`), they carry the pack's light and texture,
  and its own print where they land on it, as overprinted dark ink shows.
  Tilted blocks, after the change, read as print and sit at the top of the
  sizes, their boxes big from the tilt.
- **The tests catch what they should**: 46 deliberate bugs, one at a time,
  every patched file put back and compared by SHA-256 after each run.
  The first part's 25, all caught: a top or bottom dot row dropped; the
  nozzle share inverted; rows never dropped; rows counted from each
  token's first cell, or other rows on odd lines; the dots drawn after the
  rows are dropped; a glow that counts as ink; the floor on the bare pack;
  no dark rectangle; strokes multiplied like dark ink; the dotified share
  inverted; a window's look drawn for every style; no half turn, or a
  quarter turn more, or a half turn mapped 1 px off; a worn block's boxes
  from its worn ink; its backing worn too; worn labels; half the fade;
  a token worn to nothing kept; a patch on any token; the worn share
  inverted; no glued mark in the sidecar; worn counted on labels in the
  shares; and M2's defects moved before its warp (the 500-photo sample).
  The sizes' 13: 11 caught at once (the density field from the defects'
  generator; every box said to fit; the same seed every try; no
  correction; small print on any background; the target recorded as
  reached; a decile drawn from nine; the last drawing returned with the
  closest's sizes; the wear placed in pixels; a worn patch recorded in
  pixels; and one draw more in M2's sizes, the 500-photo sample again).
  **Two got through**, no second correction and the last drawing kept
  rather than the closest, as fresh draws hide both in a share: a test now
  drives the steps with a stand-in drawing whose boxes it sets
  (`f8e34c5`), and catches both. It also found the size tests judging a
  fit with `make.fits` itself; they now work the tolerance out on their
  own. The tilted rule's 8, 7 caught at once (tilted corrected to a box;
  every block kept at its first drawing; digits at the target, not over
  1.3; recorded as fitting; tried 3 times though kept; a block worn to
  nothing kept; the size check over every date); the manifest's size check
  over every date got through, as a 3-photo sample held no tilted block, and
  is caught now the test makes 12 and checks one is there.
- **Searched**: S8's added lines for every v1, v2 and v3 photo name (and
  stem, 5 characters or more), the HEIC names, every design id and every
  word of one (4 characters or more): no name and no id; 8 common words of
  design ids match (colours, `pack`, `clear`, `fresh`, `training`).
- **Not checked:** gen-v2 on the cluster's pixels and at scale, and
  synth-val-v2 (S9's); whether any of it helps a model (S10 to S12); the
  look beyond the 50-photo sheets and about 300 crops; tilted dates on
  v1-val or v2 (not looked at, and v2 won't be); the learnings guide (not
  written).
- **For S9:** `batch.Layout.spec` gives `GEN_V2` to every set in
  `m2b.GENERATED_SETS` (add `gen-v2` and `synth-val-v2` there and their
  seeds, 4 and 5); `batch.home` puts their files under `m2b/`; a job's time
  is about 1.5 times M2's per photo; `generate.py --check` still checks
  M2's two sets only, and its gen-v2 check must compare sizes over untilted
  dates. Suggested, not in S9's slice yet: make gen-v2-sample on the
  cluster, compare it with the laptop's and run the share and size checks
  on it. gen-v2 may differ there more than M2 did: a box 1 px different can
  change a block's size and all that is drawn after it in that photo.

### S9. Generator v2, part 3: synth-val-v2, the 180° sets, generation at scale · laptop, then one CPU job

**Goal:** gen-v2, synth-val-v2, real-val-rot180, real_rot180 and
synth_rot180 on the cluster, checked.

- `synth/spec.py`: `SYNTH_VAL_V2`. `synth/batch.py`: the sets `gen-v2`
  (tasks 0–199, 1,000 photos each) and `synth-val-v2` (task 200).
- `scripts/derive_sets.py --only rot180`.
- `slurm/m2b_generate.sbatch`: `short`, 24 CPUs, 32 GB (M2 S6 measured
  900 MB a worker) **(changed 2026-11-07, owner's call: 48 GB, below)**;
  the render sheet, the tasks, `--check`, the 180° sets. It releases
  nothing: job B waits for v1 too.
- Laptop trial first: `--tasks 0-4,200` timed, the check, synth-val-v2's
  contact sheet by stratum.

**From S8 (added 2026-11-06), before S9's code:**

- **synth-val-v2 must keep M2's sizes, and the code can't do that yet.** Its
  strata keep M2's `clean` sizes ([The generator, v2](#the-generator-v2)),
  but `make.make_one` draws store sizes for every spec with `v2` set. S9
  adds a way for a v2 spec to keep M2's `SizeSpec` bands, with a test that
  a stratum's digits fall in `clean`'s band. Without it the strata get
  store sizes, with no error.
- **Tilted blocks are sized by their digits** ([The
  generator, v2](#the-generator-v2)); `GEN_V2` already does this. A size
  comparison with the store uses untilted dates
  (`make.date_sizes(sidecars, tilted=False)`), and code reading a sidecar
  allows for `fits` being null on tilted blocks.

**Added 2026-11-07, before S9's code** (the owner's points, and details
fixed while writing; no share of the plan changes, and nothing comes from
a store photo):

- **M2's sizes in a gen-v2 spec**: `V2Spec.size = None` keeps M2's
  `SizeSpec` bands on the digits; such a block is drawn once at its band's
  height through gen-v2's drawing, with no correction to a box, as a tilted
  block is. `GEN_V2` keeps its hash (`2c8b24d5…`). A test checks that every
  stratum's digits are the height M2's `print_height` drew, in clean's
  1.5–4.5% band, and that a stratum without it would draw the store's.
- **synth-val-v2's strata**: each is M2's `clean` (`nozzle-rows`: M2's
  `dot-matrix` stratum) drawn by gen-v2's code with every new thing off
  (no glue, nozzle rows or worn patches) and M2's sizes, then one thing
  on. Where that thing needs a second number, the second only makes it
  possible: `glued` also prints the word before every date (a single date
  has one 70% of the time), and `worn` is solid print only (clean's labels
  are never worn). So every target of those strata is glued, or worn.
  In a line-pair or a dense panel **the target is one of its dates, each
  equally likely** (the plan named only the line-pair's); the others stay
  plain dates. The other strata keep one date, the target. The share rows
  find an import date by its word, as its role may now say `target`.
- **(changed 2026-11-07, owner's call, after a scratch run of the trial
  and before any code was committed) A stratum's block that finds no place
  in its 10 fresh draws is drawn at a fresh size from the same band**, 10
  more times, up to 10 sizes (`make.FRESH_SIZES`). At clean's sizes (digits
  up to 4.5%) a whole line-pair can be wider than a narrow portrait photo
  (557 to 908 px wide) at any of its draws: M2's rule, 10 fresh blocks at
  one size, left 5 of the 300 line-pair photos without their target, which
  the check refuses. With it, those 5 are drawn at a smaller size, still in
  clean's band, and every photo has its target. M2's synth-val keeps a
  block's size through its 10 draws and gen-v2 draws one block, so neither
  draws differently (tests).
- **`generate.py --milestone m2b`** for the tasks and the check: M2b's
  tasks are numbered on their own (gen-v2 0–199, synth-val-v2 200), so
  `--milestone m2b --tasks 0-4,200` is the laptop trial. M2's commands,
  tasks and check are as they were, and M2b's check reads and writes
  nothing of M2's: checking M2's sets again would rewrite their manifests,
  whose SHA-256 M2's runs record. M2b's files go under `m2b/` (task records
  `m2b/gen/sidecars/tasks/NNN.json`).
- **The check, extended** (owner's point 3): gen-v2's shares against
  `GEN_V2` and its **untilted dates' sizes against v1-train's** (S8's
  35.5 / 42.0 / 22.5% and median 1.736%, the median V2Size's middle
  decile): each band as a share (within 3 points, or 3 standard errors on
  a small sample, as M2's shares), the median within 0.1 points (or, on a
  small sample, the share of dates under v1-train's median within 3
  standard errors of a half). At 200,000 photos that is S8's check exactly;
  tilted dates are recorded beside. synth-val-v2, stratum by stratum: its
  shares, one target per photo, and every block's digits in its band. A
  manifest says `complete: true` only with no problem, and only then are
  the lists written; a failed check stops the job.
- **The job also makes gen-v2-sample** (S8's suggestion, the owner's
  point 6), to compare with the laptop's, and draws gen-v2's render sheet
  (`render_sheet.py --v2`). It hashes M2's manifests and lists before and
  after. The 180° sets go under `m2b/` (`derived/`, `lists/`,
  `manifests/derived.json`), M2's derived files untouched (a test).
- **The check's memory** (owner's point 5): it reads the photos back on
  processes started before it parses the 200,000 sidecar lines at once. On
  Linux those processes are forked copies of it, and a parent holding
  every parsed line could have that memory copied into each of them as
  they run. M2's check gives the same results; its memory was never
  measured beyond Slurm's samples.
- **(changed 2026-11-07, owner's call) The job asks 48 GB**, as M2's did,
  not 32: in the scratch run a worker peaked at 885 MB on the laptop (M2's
  at 773 there and 900 on the cluster), so 24 workers come near 25 GB if
  they peak at once; the check's main process holds about 3.3 GB for
  200,000 sidecar lines (2.4 GB measured for them parsed); and the 42 GB
  the job writes fills page cache charged to its memory. A `short` node has
  510 GB, so the larger ask costs nothing in the queue. 4 hours of
  `short`'s 6.

**Check:** both manifests `complete: true`, 200,000 and 2,400 photos; the
shares; real-val-rot180 165 photos and 187 dates; the render sheet's
pixels as the laptop's.
**Commits:** `feat: synth-val-v2 strata, embossed measured at last`,
`feat: generation of gen-v2 at scale, and the 180° sets`

**The laptop's part, 2026-11-08.** The plan's additions `efd3d0f`, then
code `ce1b8c5` (the strata, M2's sizes in a gen-v2 spec) and `fd7abd4` (M2b's
layout and check, `generate.py --milestone m2b`, `derive_sets.py --only
rot180`, the job). 738 tests pass (717 before: 21 new, 9 in
`tests/test_synth_v2.py`, 11 in `tests/test_synth_batch.py`, 1 in the new
`tests/test_derive_sets.py`); each code commit passes alone, in a worktree
with the laptop's data and nothing skipped (726, 738). What was found:

- **M2 didn't move.** `spec_hash(GEN)` is still `3fc14178…` and the strata's
  `ba77f1bd…`; `GEN_V2` keeps S8's `2c8b24d5…`. S5's gen-sample and S8's
  gen-v2-sample are each remade byte for byte by a test (500 photos each),
  so gen-v2 draws as S8 drew it. 100 of M2's photos (gen 0 to 29, each
  synth-val stratum's first 10), made by `f1442ae`'s code and by the new
  code, give the same JPEGs and sidecar lines. All 15,638 of M2's files on
  the laptop hash the same before this slice and after everything in it
  (one SHA-256 over all, `2fb8b426…`): the trial, its check, both sheets
  and the 180° sets wrote nothing of M2's.
- **A scratch run of the trial, before any code was committed, found two
  problems.** (1) A bug of Claude's: line-pairs' and dense panels' import
  shares measured a quarter and a third, against a half. A target date
  loses its role, and half the line-pairs' targets (a third of the dense
  panels') were import dates. The rows now find an import date by its
  word; in gen-v2 the two counts agree on every block, so S8's shares
  don't move. (2) 5 of the 300 line-pair photos had no target. They are
  narrow portrait photos (557 to 908 px wide) at large heights (30 to 49
  px digits), where no line-pair fitted in any of M2's 10 fresh draws at
  one size. The owner chose a fresh size after 10 draws (above). Remade
  that way, those 5 photos and no other drew a fresh size. The scratch run
  also measured memory, and the owner chose 48 GB (above).
- **The trial** (`generate.py --milestone m2b --tasks 0-4,200 --workers 4`,
  run `20261108-003145-m2b-generate`, from `fd7abd4` with a clean tree):
  5,000 gen-v2 photos and all 2,400 of synth-val-v2 in 159 s. That is 84.9
  ms a photo per worker (gen-v2's tasks 85 to 99, synth-val-v2's 70). By
  stage: drawing 38.1, camera 17.6, background 8.6, JPEG 8.4, blending
  6.2, surface 5.1, writing 0.5, placing 0.3. M2's trial, on 4 workers
  too, took 79 ms a photo, drawing 24.2: here drawing takes 1.6 times as
  long and a gen-v2 photo 1.1 to 1.25 times, under S8's +50% on one CPU.
  Each worker peaked at 833 MB
  (885 in the scratch run; M2's 773), the main process at 111 MB. Its
  14,800 files are the scratch run's, byte for byte.
- **The slowest photo took 13.1 s**: gen-v2's 2,639th. 12 s of it went on
  drawing a large window block that then found no place, drawn at its size
  and again at 0.8. A window's glow is blurred at 4x, which costs seconds
  at large sizes; the next slowest photos, window panels, took 1.1 to 1.6
  s. It is rare, and making it faster would change gen-v2's pixels, so
  nothing was changed.
- **The check on the trial** (`--milestone m2b --check`, run
  `20261108-003532-m2b-generate-check`, 8.1 s on 4 workers, 229 MB):
  - gen-v2 incomplete with exactly one problem, "tasks 5-199 not made (no
    record)", so its manifest says `complete: false`, no list was written
    and the command exits 1;
  - synth-val-v2 complete with no problem: its list, sidecar and manifest
    written;
  - every image opens whole, and every label file equals the one rebuilt
    from its sidecar line.
- **gen-v2's shares over its first 5,000 photos**: all 64 rows pass, and
  none is off by 3 standard errors. The farthest is windows dotified,
  25.1% (spec 30%, −2.8 SE, of 696 windows), so 4,000 window blocks were
  drawn in memory as gen-v2 draws them: 29.2% of those kept are dotified,
  and only 61 were drawn again. So the trial's figure is noise, not a
  bias, and the row should pass at 200,000 photos, where the rule is 3
  points and 3 SE is 0.8.
- **gen-v2's sizes over those 5,000**, the store's way: 10,138 untilted
  dates, **34.4 / 43.4 / 22.2%, median 1.750%**. Against v1-train's 35.5 /
  42.0 / 22.5% and 1.736%, that is −1.1, +1.4 and −0.3 points and +0.014:
  the check passes. All 11,962 dates, tilted ones included: 29.8 / 39.2 /
  31.0%, median 1.917%.
- **synth-val-v2**: 2,400 photos, each with one target. Every block's
  digits lie in clean's band (medians 2.44 to 2.77% by stratum); 243
  blocks were skipped (240 found no place, 3 too faint), 222 of them
  line-pairs; 5 line-pair photos drew a fresh size; 316 share rows, none
  off. **By eye** (`contact_sheet.py --set synth-val-v2 --per-tag 8`, run
  `20261108-003549-contact-synth-val-v2`), each row changes its one thing.
  Targets turn upside down; glued words run into their dates
  (`EXP:12/08/2021`, `B.B.31102031`); line-pairs show their separators,
  with the target on either date; dense panels stack tightly with barcode
  digits and an `Imp.` line; windows show light strokes on dark
  rectangles, some dotified; embossed relief is soft; worn dates have part
  faded away; nozzle-row dot-matrix misses whole rows. Every box sits on
  its print. As in M2's synth-val, some targets are faint: low-contrast ink
  on busy backgrounds, and faint dot-matrix.
- **The 180° sets on the laptop** (`derive_sets.py --only rot180
  --workers 4`, 23 s): **real-val-rot180 165 photos and 187 dates**, as
  planned; real_rot180 937 photos (1,057 dates) and synth_rot180 11,844
  (16,652); worst box error 0.0 px. They are written under `m2b/` and listed
  in `m2b/manifests/derived.json`; M2's files are unchanged (above). A
  sheet of 12 turned validation dates shows every box on its print,
  upside down. 2.85 GB on the laptop, where M2 kept no turned training
  copies; real-val-rot180 is the M2b suite's.
- **The render sheet**, `render_sheet.py --v2` on the laptop (run
  `20261108-003625-render-sheet-v2`, `sheet.png` SHA-256 `a91a5045…`), to
  compare with the cluster's pixels.
- **The job's size** (owner's point 5):
  - **memory**: a worker peaks near 1 GB on the cluster (laptop 833 to
    885 MB, times M2's 900/773), so 24 come to about 25 GB if all peak at
    once. The check's main process held 2.4 GB for 200,000 gen-v2 sidecar
    lines (the trial's 5,000 parsed 40 times over), with their guards,
    shares, counts and sizes in 72 s; with the lines' text twice, about
    3.3 GB. 48 GB, the owner's call.
  - **time**: the trial's stages scaled by M2 S6's laptop-to-cluster
    ratios (drawing 1.57, camera 1.73, writing 41 ms) give about 161 ms a
    photo per worker: 202,400 photos in about 23 minutes on 24 CPUs (S8's
    225 ms: 31). Then the check (M2's took 25 s for 52,100 photos; here
    about 4 times the photos and its parsing), and the 180° sets (M2 S2's
    turned sets: 2.5 minutes on 16 CPUs). That is under an hour of the
    4 asked (`short` allows 6).
  - **disk**: gen-v2's photos average 186.8 kB, 37.4 GB for 200,000; with
    synth-val-v2 (0.43 GB), the label files, the sidecars twice (0.38 GB
    each), the 180° sets (2.85 GB) and the cluster's sample, about 42 GB.
    The quota is read on the cluster.
- **Searched**: S9's added lines for every v1, v2 and v3 photo name (and
  stem, 5 characters or more), the HEIC names, every design id and every
  word of one (4 characters or more): no name and no id; 3 common words of
  design ids match (`clear`, `fresh`, `training`).
- **Not checked yet**: everything on the cluster, which couldn't be reached
  from off campus on 2026-11-07 and 11-08: the fonts there, the job, the
  pixels across machines, the quota. Not run: a deliberate-bug run (S9's
  slice asks for none); the learnings guide (not written).

**On the cluster, done 2026-11-12.** Job **`73944`**, submitted by
Claude at the owner's word after a first job crashed (below): `short`,
`cn1`, 24 CPUs, 48 GB, from `01f4fb9` with a clean checkout. Before it,
on the login node: `git pull` to `c8b431f`, then `01f4fb9`; `fetch_fonts.py`
checked all 24 files (hashes, the 69 characters, axes, licences); the
same library builds as the laptop (Pillow 12.3.0, FreeType 2.14.3,
libjpeg-turbo 3.1.4.1, OpenCV 5.0.0, numpy 2.4.6; Python 3.11.11 against
3.11.13); our jobs 2 of 5 submitted (job B `73832` and `73906`, both
held), none running. What was found:

- **The first job crashed, at gen-v2's photo 17,868** (job `73943`, at
  `c8b431f`, submitted by Claude at the owner's word: `FAILED` after 4 min
  17 s, 17 tasks done). The cause is a bug in S8's worn patches.
  `compose.random_wear` chose among dates whose sprite exists, but small
  dot-matrix print (digits of 3 to 4 px) can lose every dot under the ink
  line to the density field laid on before the wear, and then
  `render.ink_box` gave nothing to unpack. The laptop draws photo 17,868
  without error: across machines a photo can take other draws after a
  1 px difference (below), so the cluster's copy met a block the laptop's
  never drew. On purpose, 6,400 small blocks on the laptop gave 6 of these
  crashes, all dot-matrix. **The fix** (`01f4fb9`): the patch picks among
  dates with ink. A drawing that lost a date's ink loses that date's box
  too and is drawn again, so the choice made there was never kept, and the
  fix changes no photo: the laptop's trial, remade, gives all 14,806 files
  byte for byte, and S8's sample too (a test). A new test reproduces the
  crash without the fix; 739 tests pass. That job also hashed M2's files
  and made the render sheet and the sample, which the second job made
  again, the same (below).
- **It ran 26 min 41 s** (14:34:10 to 15:00:51), 8 h 50 min of CPU time,
  and wrote 40.5 GB (`sacct`), in order:
  - the render sheet and gen-v2-sample: about 13 s;
  - the 202,400 photos: 24.1 min, the backgrounds loaded in 2.8 s
    (24.8 s cold in the first job, on the same node);
  - the check: 99.7 s;
  - the 180° sets: about 35 s, synth_rot180's 11,844 photos taking 33.4 s.
- **Speed: 170.6 ms a photo per worker** (gen-v2's 171.4; median 134.6,
  95th percentile 311.5), within the laptop's estimate of 161 to 225. By
  stage: drawing 64.5, writing 38.3, camera 34.9, background 14.9, JPEG
  9.9, blending 4.4, surface 3.3, placing 0.5. M2 S6's cluster job took
  150 ms, drawing 37.9: gen-v2 draws 1.7 times as long, and a photo takes
  1.14 times as long. The slowest photo took 33.4 s (on the laptop, the
  slowest drew a large window block for 12 s).
- **Memory: a worker peaked at 1,813 MB**, twice the laptop's 833 to 885
  MB, and the main process at 318 MB; Slurm's sampled `MaxRSS` says 5.2
  GB. Had all 24 workers peaked together, 32 GB would not have held, so
  the owner's 48 GB was needed. Which photo took a worker that high
  wasn't looked at (its peak isn't recorded per photo).
- **The check passed**: gen-v2 **200,000 photos and synth-val-v2 2,400,
  both `complete: true` with no problem**; every image opens whole, every
  label file equals its sidecar line's, every background and font is from
  its own set's lists and split, and the fonts match their pinned
  SHA-256s. The manifests name `GEN_V2`'s `2c8b24d5…` and the strata's
  `2295a4bd…`, made and checked at `01f4fb9`. The chain's own
  `m2b.list_missing` finds all five lists vouched for: `gen_v2` (200,000),
  `real_rot180`, `synth_rot180`, `synth_val_v2` (2,400) and
  `real_val_rot180`.
- **gen-v2's shares at 200,000 photos**: all 64 rows within 0.6 points of
  the spec, the farthest windows dotified 29.4% (−2.1 SE, of 27,462
  windows; the trial's 25.1% was noise, as measured then), then line-pairs'
  import dates 49.5% (−1.7 SE). Upside-down 10.05%, glued 30.08%, worn
  patches 9.98%, nozzle rows 14.8%, windows 9.9%, line-pairs 9.95%, dense
  panels 10.1% (all within 2 SE). synth-val-v2: all 316 rows of its 8
  strata pass, its counts are the laptop trial's exactly (2,400 blocks,
  243 skipped, the same boxes by class), and its 5 line-pair photos at a
  fresh size are the trial's.
- **gen-v2's sizes**, over its **411,587 untilted dates: 34.27 / 44.12 /
  21.61%, median 1.748%**. Against v1-train's 35.5 / 42.0 / 22.5% and
  1.736%, that is −1.2, +2.1 and −0.9 points and +0.012: the check holds.
  The middle band runs 2 points over, as the deciles' own log-uniform
  targets do (44.4%); the trial's 5,000 photos gave −1.1, +1.4 and −0.3.
  All 483,566 dates, tilted ones included: 29.73 / 39.88 / 30.39%,
  median 1.899%.
- **What was made**: 276,849 blocks placed of 280,308 drawn; 1,604 (0.6%)
  skipped for want of a place and 1,855 (0.7%) as too faint; 15,600 got a
  fresh ink, 5,620 were drawn again, 2,831 shrank. 483,566 new dates (2.42
  a photo); boxes with the backgrounds' own: date 741,826, due 366,932,
  prod 250,338, code 186,858. The photos take 37.30 GB (186.5 kB each)
  and synth-val-v2's 0.43 GB; the joined sidecars 386 MB and 3.2 MB.
  The quota read 142.0 GB used before and 182.1 GB after, of 950
  (`lfs quota`: home is Lustre, and `quota` prints nothing).
- **The 180° sets**: **real-val-rot180 165 photos and 187 dates**,
  real_rot180 937 photos (1,057 dates) and synth_rot180 11,844 (16,652),
  each worst box error 0.0 px. These counts are M2's `real_val_rot`,
  `real_rot` and `synth_rot`'s exactly, and the laptop's.
- **M2 didn't change**: its 3 manifests and 8 lists hash the same before
  the jobs and after (Claude's hashes, and the job's own `diff`, which
  printed "unchanged"); their files keep their October dates.
- **The cluster draws the same pixels, almost, and the same sizes**
  (owner's point 6):
  - **the render sheet**: the PNG file differs, but all 22,854,650 pixels
    are the laptop's (the PNG compressor differs, as M2 S6 found); both
    cluster jobs drew the same bytes;
  - **gen-v2-sample**, made on the cluster: all 500 label files and the
    whole sidecar are the laptop's byte for byte, every block, box, size
    and fit, so its shares and sizes are exactly S8's (untilted 33.17 /
    41.98 / 24.85%, median 1.812%), which pass; 440 of its 500 JPEGs are
    the laptop's, and in the other 60 the median photo has 0.0054% of its
    pixels changed, at most 0.057%, typically by 2 grey levels and at most
    12 (one 8×8 JPEG block in one photo);
  - **the trial's 7,400 photos** against the cluster's copies of them:
    4,505 of 5,000 gen-v2 JPEGs and 2,240 of 2,400 synth-val-v2 JPEGs
    identical, 4,996 of 5,000 and all 2,400 label files; 11 of 7,400
    sidecar lines differ. 5 differ only in the JPEG quality drawn after
    the camera's noise, as M2 S6 found. 1 block landed in another free
    place. 3 had a layer or a box 1 px off after the warp. 2 differ only
    in the 4th decimal of how much the ink shows. **No block's size,
    target, box reached, fit, drawings, style or font differs**: the 1 px
    differences moved boxes, never a size. In 80 sampled JPEGs that
    differ with the same sidecar line, the median photo has 0.0028%
    (gen-v2) and 0.004% (synth-val-v2) of its pixels changed, by at most
    4 and 7 grey levels;
  - so the laptop's look shows the pixels trained on for about 90% of
    photos, and within a few grey levels for nearly all the rest.
- **Job A had finished**: `73831` ran 2026-11-08 16:55 to 11-09 11:25 on
  `cn6` (18.5 hours) and left a held replacement, `73906`. Job B (`73832`)
  stays held for S10; nothing was released.
- **Not checked:** the cluster's photos by eye (the laptop's sheets show
  the laptop's copies, which differ in about 10% of photos by a few grey
  levels); the true peak memory of the 24 workers together, and the
  check's own peak there (only each worker's peak and Slurm's samples
  are recorded); which photo took a worker to 1.8 GB; why the cluster's
  photo 17,868 drifted from the laptop's; whether another crash only the
  cluster's drifts can reach remains (the job made all 202,400 photos);
  a deliberate-bug run (none asked); the learnings guide (not written).
- **`CLAUDE.md`** gained, with the owner's OK: the Lustre quota command,
  a CPU node's cache, fetching several files with `tar` over `ssh`, and
  gen-v2's match with the laptop, its crash met only on the cluster and
  its workers' 1.8 GB.

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

**Steps:** smoke of `n1280-ft` and `n1280-ft2` on a CPU node of the
cluster **(changed 2026-11-12, before the code: the laptop can't train at
1,280 px, S2)**; dry runs on both machines; owner: `scontrol release 73832`.
**Commits:** `feat: m2b step 2, fine-tuning with replay and the two-stage arm`,
`feat: the m2b validation suite with the store threshold`,
`feat: m2b chain, step 2 and the decision rules`

**Added 2026-11-12, before S10's code** (the owner's go-ahead; no rule,
band, budget or share of the plan changes):

- **The smoke runs on a CPU node of the cluster**, as S2's did:
  `slurm/m2b_smoke_cpu.sbatch step2` trains `n1280-ft` and `n1280-ft2`, one
  epoch a stage, on an epoch of 400 photos (`--epoch-photos`, smoke only,
  recorded, never passed by the chain), then scores the `n1280-ft` smoke on
  the M2b suite, so the suite's first real run is on a CPU node, not in the
  GPU job. `--fraction` won't do here: Ultralytics keeps the first photos in
  sorted path order, so 1% of a step-2 list is ExpDate photos alone, and no
  store or gen-v2 photo would be read.
- **The 180° copies in the replay** (decision 8): each ExpDate group keeps
  gen-rot-1280's half upright, half turned, and its 90°/270° and 180° copies
  form one turned pool, sampled together, so about half of its turned
  copies are upside down. Shares of the epoch: real upright 2.8%, real
  turned 2.8%, synthetic upright 11.9%, synthetic turned 11.9%, generated
  50.4%, each pool drawn without repeats with the run's seed; the store
  lines 20%.
- **ft2's stages**: each stage is M2's schedule over its half of the budget
  (5 epochs, warmup 3%, mosaic off for its last epoch), so ft2 has 2
  mosaic-off epochs of 10 where ft has 1. Stage 1's weights go to
  `<run>-stage1/` beside the run's own checkpoint folder, which holds the
  final weights, as every run's does.
- **R2 with no recipe passing the guard** (the plan didn't say): step 2 on
  `s` and `m` is blocked, and R3 waits, for the owner to look.
- **The band first**: the M2b suite scores the control's three seeds before
  any other run, so `noise_store` is fixed before any M2b run is scored, by
  construction.
- **The M2b suite's counts are at `threshold_store`**, with
  `threshold_m2`'s beside them, and v1-val's dates also by Cause 4's size
  bands. Ultralytics' validation folders go in the run's own results
  folder, so scoring the control writes nothing into M2's checkpoints.
- **The store lines stay uncapped** (S6 left it to S10): decision 7 as
  written. v1-val's designs are all unseen in training, so it shows any
  memorising.
- **Frozen batch-norm**: Ultralytics 8.4.177 puts the frozen layers'
  batch-norm modules in eval mode every epoch (`BaseTrainer._model_train`),
  and MuSGD skips parameters without a gradient. So the measurement in
  [Step 2](#step-2-store-fine-tuning) should find nothing moved; S10's
  record says what it finds, and that sentence is corrected then.
- **R1 isn't known in S10**: it compares v1-val mAP, which only the M2b
  suite holds; the GPU job scores the control and step 1 on it first, and
  R1 follows.
- **Jobs**: A's replacement, `73906`, is held too. Only `73832` is
  released: it is the oldest waiting GPU job, and its 3 days cover stage 2
  (S10's record has the estimate). `73906` waits behind other users' jobs,
  and two running jobs would fill the account's 2 running slots, holding
  back M3's CPU jobs.
- **The cluster's `.env`** gains `STORE_TRAIN_DIR=~/bestbefore-data/store`
  (missing, as M3 S1 found), added by the owner.

**Done 2026-11-12.** Code `3e1d020` (step 2), `93e9420` (the M2b suite) and
`6b43d15` (the chain and the rules), after the plan's change `3aa946e`.
760 tests pass (739 before: 21 new, in `tests/test_m2b_rules.py`,
`tests/test_score_suite.py` and `tests/test_chain.py`), and each code
commit passes alone, in a worktree with the laptop's data and nothing
skipped (747, 752, 760). On the cluster, at the owner's word: `git pull` to
`6b43d15`; `STORE_TRAIN_DIR` added to its `.env` by Claude (the owner's
request, not the owner's step as written above); the CPU smoke, job
**`73951`**, submitted by Claude. What was found:

- **Stage 1 is complete**, pulled and checked: the smoke, the three
  controls, `s1280` s0–s2 and `m1280-s0` all `finished`, the five trained
  runs scored on M2's suite; their 13 record files are byte-identical on
  both machines, every `last.pt` they name is on the cluster, and there were
  no failures or claims. Job A ran from 2026-11-08 16:55 to 11-09 11:25 and
  left `73906` held, with 19 tasks left.
  - `s1280` trained 230.3 to 239.4 minutes a run, 177 to 184 images a
    second (the plan's 4 hours), and scored on M2's suite in 1.6 minutes;
  - `m1280-s0` trained 395.6 minutes, **107 images a second**: 6.6 hours,
    under the plan's 7 and faster than S2's estimate (about 80 a second,
    7.5 to 12.4 hours);
  - GPU memory at batch 16 and 1,280 px (Ultralytics' `GPU_mem` in job A's
    log): `s1280` 16.3 GB and **`m1280` 30.8 GB**, where S2 estimated about
    44 GB from the laptop's backward pass (M1's `s1280`, which validated
    every epoch, peaked at 21.1 GB).
- **R1 isn't known yet** (above): no run has an M2b suite. The GPU job
  scores the control and then step 1 on it first.
- **The main venv still holds the pinned versions** after M3 S1 extended it:
  torch `2.11.0+cu128`, ultralytics 8.4.177 (numpy 2.4.6, OpenCV 5.0.0,
  Pillow 12.3.0), read on the login node; the smoke's records name the same.
- **M2's chain is unchanged**: its dry run prints the same 47 lines at
  `6ee18bf` and at `6b43d15` (the chain's task list may now be made from the
  decisions, `Chain.task_list`; M2's passes its fixed list), on the laptop's
  records.
- **Step 2's mix on the cluster's own lists** (seed 0, read-only on the
  login node): the five list sizes M2's runs recorded (937, 937, 11,844,
  11,844, 50,000) are the lists' sizes today. Pools: 320 store copies (80
  photos in 4 turns, 31 designs), real 937 upright and 1,874 turned,
  synthetic 11,844 and 23,688, generated 200,000 (gen-v2) or 50,000 (M2's
  gen50k). An epoch lists 4,991 store lines (161 a design, all 320 copies
  among them), real 709 + 709, synthetic 2,987 + 2,987 and generated
  12,609: **24,992 photos**, the store's 20.0%. ft trains 10 epochs (249,920
  images), warmup 0.3, mosaic off for the last; ft2 5 + 5, warmup 0.15 and
  mosaic off for the last of each. Of the turned copies drawn, 51.5% (real)
  and 50.7% (synthetic) are upside down.
- **The smoke** (job `73951`, `short`, `cn1`, 24 CPUs, 64 GB, started
  20:58:43; `slurm/m2b_smoke_cpu.sbatch step2`):
  - `smoke-n1280-ft-cpu` started from the control's seed 0, M2's last.pt
    (`8c9a03c7…`, as S2 recorded it), on 413 photos an epoch: 93 store
    lines, real 11 + 11, synthetic 48 + 48, generated 202, every pool read
    and none corrupt. Ultralytics trained as asked (MuSGD, `lr0` 0.001,
    `freeze` none, batch 16, 1,280 px, 1 epoch, warmup 0.03), with 0 loader
    workers on a CPU (as S2 found); 2.8 minutes, 3 min 15 s in all.
  - `smoke-n1280-ft2-cpu`: stage 1 with `freeze` 11 at `lr0` 0.001 (1.6
    minutes; layers 0 to 10 listed as frozen), into `<run>-stage1/`; stage
    2 from stage 1's `last.pt`, `freeze` none at `lr0` 0.0005 (2.2
    minutes), into the run's own folder; 3 min 50 s in all. Both stages
    applied as asked; the record's `applied` and `stage` show stage 2's, and
    `stages` both.
  - The `n1280-ft` smoke scored on the M2b suite on the CPU
    (`score_suite.py --suite m2b`): 12.1 minutes for all 7 sets (5,070
    photos, each predicted and then validated by Ultralytics, about 40 ms an
    image), `suite-m2b.json` written whole, with both thresholds, synth-val-v2
    by its 8 strata, v1-val by its 5 tags and its 63 dates by size band (23
    / 28 / 12). Its numbers are a one-epoch smoke's and mean nothing.
    Ultralytics' validation folders are empty, and no box of a photo was
    kept.
  - The job: `COMPLETED`, 19 min 15 s, `sacct`'s sampled `MaxRSS` 17.1 GB
    (of 64).
- **What the smoke left on the cluster** (the home folder is 700, so none
  of it is reachable by others):
  - v1-val's label cache, `store/labels/val.cache` (box coordinates of the
    34 numbered photos, no pixels), written with mode 664 where S6 made
    every store file 600: the owner chmods it (below). Ultralytics rewrites
    it only when its photos change, so once is enough;
  - every step-2 run's label cache (`$EXPDATE_DIR/yolo/labels/real-train.cache`,
    664) lists its store lines' boxes and numbered paths beside ExpDate's;
  - the training-batch mosaics (`train_batch*.jpg`) beside each run's
    weights show store photos, as S6 expected; never pulled.
- **The frozen layers didn't move at all** (ft2's stage 1, from M2's s0 to
  stage 1's `last.pt`, both as Ultralytics loads them): in layers 0 to 10,
  0 of 1,357,104 convolution weights, 0 of 8,368 batch-norm weights and
  biases, and 0 of 4,184 running means and 0 of 4,184 running variances
  changed; in the layers it trained, 1,085,421 of 1,247,436 values (87%).
  So Ultralytics 8.4.177 holds a frozen layer's batch-norm statistics, as
  its source said (`_model_train` puts those modules in eval mode), and
  [Step 2](#step-2-store-fine-tuning) and the risk on it are corrected
  (owner's go-ahead, before the code). The check stays in every ft2 run: a
  frozen weight that moves stops it.
- **Ultralytics' label cache** for a step-2 run lands beside the first label
  file in sorted path order, `$EXPDATE_DIR/yolo/labels/real-train.cache`,
  where every run of M1 to M2b's step 1 has written its own; each step-2 run
  rewrites it for its lists. v1-val's lands in the store export's
  `labels/val.cache` (S6 expected it; `check_export` ignores it).
- **The cluster's dry run before the release** (after the `.env` line and
  the smoke): stage 1 done; the 9 step-2 runs on `n` ready; step 2 on `s`
  waiting for R2, and `m`'s runs for R1; the control's three M2b-suite
  scores ready, every other run's waiting for the noise band. **Nothing
  waits on data**, and nothing is blocked, failed or claimed; no decision
  yet. The smoke's runs are named apart (`smoke-n1280-ft-cpu`, ...), so the
  chain counts none of them.
- **Stage 2's GPU time**, from step 1's measured speeds (`n` 208, `s` 181,
  `m` 107 images a second) and M2's suite's 1.5 to 2.5 minutes a score (the
  M2b suite holds 5,070 photos against 2,471: about twice as long):

  | GPU work | Runs | Each | GPU hours |
  |---|---:|---|---:|
  | the M2b suite: the control and step 1 | 7 | about 4 min | 0.5 |
  | `n` step 2 (3 arms) | 9 | 20 min of training, about 5 of setup, cold reads and validation, 4 of scoring | 4.4 |
  | `s` step 2 | 3 | 23 + 5 + 4 min | 1.6 |
  | **without `m`'s seeds** | | | **about 6.5** |
  | `m` by R1: seeds 1 and 2, both suites | 2 | 6.6 h + 7 min | 13.4 |
  | `m` step 2 | 3 | 39 + 5 + 5 min | 2.5 |
  | **with them** | | | **about 22.5** |

  Every run fits: the longest, an `m` seed, takes about 6.6 of the job's
  72 hours, and the chain starts it only with 14 hours left (twice the
  plan's 7). An epoch takes 2 to 4 minutes and a run's setup minutes (the
  smoke's 413 photos were read in under 2 s), far inside the watchdog's 2
  hours; a score about 5 of the timeout's 60 minutes.
- **Jobs**: our jobs were `73832` and `73906` (M2b) and `73946`, `73947`
  (M3), all held, and `73951`, the smoke. `73832`, submitted 2026-10-31, is
  the oldest waiting GPU job: it gets the first H100 to free up, at the
  latest about 2026-11-15 01:00 (a 3-day job on `cn6` ends then), earlier
  if any job ends early. `73906`, submitted 2026-11-09, waits behind four
  other users' jobs.
- **Job B released**: `73832`, at 21:37:13 on 2026-11-12, by Claude at the
  owner's word, after the smoke and the dry run above. It is `PENDING` with
  reason `Resources`, first of the waiting GPU jobs (its submit time kept,
  2026-10-31 20:37:55; the jobs behind it show `Priority`, `ReqNodeNotAvail`
  or their owners' caps). The four H100s run jobs that reach their limits at
  2026-11-15 01:00 and 11-16 11:25 (`cn6`), 11-18 02:35 and 11-18 10:47
  (`cn5`), so B starts at the latest about 11-15 01:00. `73906` stays held.
- **Details fixed while writing** (no rule, band, budget or share changed):
  - **`mix()` takes the pools**, not the arm (the arm only chooses them,
    `replay_lists`); each replay pool is drawn without repeats with its own
    seed, `"{seed}/{pool}"`, so the pools' order changes nothing.
  - **A run's lists** go to `results/runs/<id>/lists/<pool>.txt`, absolute
    paths, the store's numbered; their lines, distinct photos and SHA-256
    are in `run.json`, beside the pools' sizes, every manifest's hash and
    the store export's. It validates on real-val, as M2's runs.
  - **The store export is checked** against its manifest
    (`store_train.check_export`) at the start of every step-2 run.
  - **Starting weights**: the `checkpoint` of the step-1 run's record, its
    SHA-256 in the new run's `start`; the control's own recorded SHA-256 is
    compared first.
  - **ft2 as one run**: `applied` and `stage` in `run.json` follow the
    stage, so the watchdog sees stage 2 start; each stage's record, with
    its applied arguments, time and checkpoint, under `stages`. `freeze` is
    read back with what Ultralytics applied and checked against the stage's
    recipe.
  - **`train_detector.py` refuses** `--cache`, `--budget-share` and
    `--smoke-lists` for a step-2 arm, and `--epoch-photos` without
    `--epochs`.
  - **`suite-m2b.json`**: `threshold` is `threshold_store` (v1-val's),
    `threshold_m2` beside it; each set's counts at `threshold`, and under
    `at_threshold_m2`; by stratum or tag (v1-val by photo tag); v1-val's
    dates `by_size`. Every set is needed. v1-val is a set of its own
    (`sets.py`'s `v1-val`, `m2b.load_v1_val`), whose per-photo files go to
    `data/store/runs/`, as the store set's.
  - **`score_suite.py` finds the control's weights** on the laptop too (by
    `weights_from`, M2's copy), for S13's device check should R3 keep M2's
    weights.
  - **Chain task names**: `suite-m2b-<run>` for the M2b suite;
    `score-m2b-<run>` stays M2's suite. The dry run's column is 30 wide.
  - **The rules read the M2b suite** for the control too (its real-val and
    synth-val scored again there), not M2's `suite.json`. "Beats" (R1) is
    strictly greater; "within the band" includes its edge, as M2's.
  - `pull_cluster_results.sh` brings `results.csv` from M2b's checkpoint
    folders too; a two-stage run's first stage lands in `<id>-stage1/`.
- **Searched**: S10's 1,840 added lines (from `6ee18bf`: the code, this
  plan's change and this record) against all 213 photo names of v1, v2 and v3
  (and stems of 5 characters or more, 313 in all; v2's from S4's
  `designs.json`, no v2 file opened), the HEIC names, all 92 design ids and
  their 128 words of 4 characters or more: no name and no id; 2 common
  words of design ids match (`clear`, `training`).
- **Not checked:** a step-2 run on a GPU, so its speed, the cold first read
  of its photos and the M2b suite's time there stay estimates (the GPU job's
  first runs measure them); two jobs writing one label cache at once (only
  one job is released); R1 to R3 on real suites (none exists); whether the
  store lines' repeats are memorised (v1-val's and v2's unseen designs will
  show it, S11 and S14); the learnings guide (not written).
- **`CLAUDE.md`** gained, with the owner's OK: `m`'s measured GPU memory and
  every size's training speed at 1,280 px, `STORE_TRAIN_DIR` in the
  cluster's `.env` and v1-val's label cache to `chmod`, the step-2 smoke's
  CPU times, and what `Resources` and `Priority` mean in the queue.

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
**From S8 (added 2026-11-06):** the report, the README and `docs/datasets.md`'s
gen-v2 entry say that gen-v2's sizes follow the store's on untilted blocks
only: tilted blocks are sized by their digits and left out of the size
comparison, as the store has no tilted date (S8's record).
**Commits:** `feat: m2b report script`, `docs: m2b results report`,
`docs: readme for m2b`, `docs: m2b plan done`

---

## Order and sessions

```
S1 ─► jobs A, B held (done)
S2 ─► release A ─► [GPU: smoke, s1280 ×3, m1280 s0, M2-suite scores]
S3 (any time)
S4 ─┬─► S5 ─► S5b ─► S6 ─► [rsync v1 and v3 to the cluster] ────┐
    │                                                           │
S7 ─┴─► S8 ─► S9 ─► [CPU job: gen-v2, synth-val-v2, 180° sets] ─┤
        (S8's sizes need S4's and S5b's splits)                 │
                                           S10 ─► release B ─────┴─► [GPU: controls and step-1 runs on the M2b suite,
                                                                       n step 2 ×9 → R2, s step 2 ×3, m by R1 → R3]
                                           S11 ─► S12 ─► S13 ─► S14 ─► S15
```

| Session | Slices | Ends with |
|---|---|---|
| 1 | S2, S3 | job A released |
| 2 | S4, S5 (with the owner) | designs frozen, keyword boxes in |
| 2b | S5b (with the owner) | store set v3 frozen, its keyword boxes in |
| 3 | S6, S7 | v1 and v3 on the cluster, gen-v2's text |
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
| gen-v2 + synth-val-v2 | 202,400 photos | about 32 min on 24 CPUs (M2 S6's 150 ms a photo per worker, plus half: S8 measured +50%, not the tenth first planned) |
| 180° sets | 13,000 photos written | minutes |
| store export | 99 photos, 70 × 4 turns | seconds, laptop |
| v2 scoring | about 25 configurations × 99 photos | under an hour on the laptop; with ensemble and rotation, 12 passes a photo |

**Storage** on the cluster: gen-v2 about 37 GB, synth-val-v2 0.45 GB, the
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
                   changed: synth/{spec,text,render,warp,compose,make,batch}.py, detection_eval.py, store.py
scripts/           m2b_chain.py, m2b_report.py, export_store_yolo.py, propose_keywords.py,
                   draft_keywords.py, import_store_keywords.py, store_designs.py, store_heic.py,
                   store_sizes.py; changed: train_detector.py, import_store_labels.py,
                   freeze_store_test_set.py (v3),
                   score_suite.py, eval_detector.py, generate.py, derive_sets.py,
                   fetch_pretrained.py, pull_cluster_results.sh, m2_chain.py (on chain.py)
slurm/             m2b_gpu.sbatch, m2b_chain.sh, m2b_generate.sbatch
experiments/       m2b-store-look/, m2b-model-size/, m2b-fine-tune/, m2b-gen-v2/, m2b-inference/
tests/             test_chain.py, test_m2b_rules.py, test_store_train.py, test_synth_v2.py,
                   test_ensemble.py, test_m2b_results.py
results/           m2b_detector.md, store_designs.md, store_set_v3.md (committed; runs/ is not)
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
- **v1-val isn't blind to gen-v2's design.** The gaps gen-v2 fills were
  found by looking at all of v1, validation designs included; only the
  sizes are drawn from v1-train alone. So a gain from gen-v2 on v1-val may
  be a little optimistic; v2, which shaped nothing, is the blind test.
- **Store-like sizes copy how v1's photos were framed.** A date's size over
  the photo depends on how close the camera was. v2, taken in the same
  stores, shows whether that carries over; a new store's staff might
  frame differently.
- **gen-v2 looks pasted.** The look check against real store crops (S8);
  synth-val-v2 high but real sets flat would say so, as in M2.
- **Two-stage's frozen layers** could still change in training. They don't
  in Ultralytics 8.4.177, batch-norm statistics included (S10's record), and
  every ft2 run checks them. **(corrected 2026-11-12, S10: the plan first
  said the statistics still move.)**
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
  comparable. **15 more store photos arrived on 2026-11-03** (HEIC, under
  `data/store/`, not yet looked at): their own training-only set beside v1,
  not added to it, so v1's frozen labels, designs and split stay as they
  are (owner's OK); store set v3, done in S5b.
- **A dates-only arm**: the detector trained without the `due`, `prod` and
  `code` classes, to measure whether they help or cost the `date` class
  (asked by the owner in S5; every run so far has all four, M1 decision 2).
  Only if dates turn out to be the bottleneck.
- **Keyword boxes for new store photos** start from S5's tools: OCR's
  proposals snapped to their ink, then a second OCR pass on the date area.
  The patterns could add `Expire`, `BBD` and `DOP` (S5b found them on
  store packs), with v1's and v3's proposals kept as made.
- **A mask of the pack** for generated blocks (M2's Later).
- **k-fold validation by design**, if v1-val proves too noisy.
- **A full retrain** of `s` (or `m`) from COCO on gen-v2's whole mix with
  the store share, instead of two steps.
- **Sizes apart from layouts**: a gen-v2 with M2's sizes, to tell what the
  store-like sizes add on their own (the side arm measures both together).
- **`l` or `x`**, only if `m` clearly beats `s`.
- **The pack's own orientation**: a whole photo turned (EXIF missing)
  against print turned on an upright pack; the gallery can't yet tell
  which the 15 sideways expiries are.
- From M2, still open: tiled inference; oriented boxes; a realism
  classifier for generated crops; superscript ordinals and Hindi words.
