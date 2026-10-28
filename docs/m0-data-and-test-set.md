# M0: Data and test set

Planned 2026-10-13. Status: **M0 done** (2026-10-21). Part A on
2026-10-13; Part B (store test set v1 frozen) and Part C (baselines) on
2026-10-21. The table: [`results/m0_baselines.md`](../results/m0_baselines.md).
On the store photos, baseline B reads 60 of 87 expiries right (2 wrong)
and A 49 (12 wrong); on ExpDate test, A reads 502 of 655 and B 57.
The ideas behind each step, explained simply: [`learnings/m0.md`](learnings/m0.md).
**Store test set v2** (planned 2026-10-28), the test set once store
photos are trained on: [Part D](#store-test-set-v2-part-d), below.

**Goal.** Get the public data into a shape a detector can train on, turn the
store photos into a labelled test set that never changes again, and measure
the two baselines every later model has to beat.

**Done when.** `results/m0_baselines.md` holds the baseline table on the
ExpDate test split and on the store photos, broken down by failure tag.

Nothing in M0 needs a GPU, so it all runs on the laptop. Scripts read their
paths from `.env`, so they run unchanged on the cluster in M1.

---

## What we start from

**ExpDate.** One Google Drive folder
(<https://drive.google.com/drive/folders/1YuxWzVj6bT6gs6XlewEGetYrdgwZt7EH>)
with seven zips. M0 needs two of them:

| File | Size | Contents (per the paper) |
|---|---|---|
| `Products-Real.zip` | 630 MB | 1,767 photos: 1,102 train / 665 test |
| `Products-Synth.zip` | 1.08 GB | ~12,000 synthetic photos, train only |

The annotations are JSON: per image `height`, `width` and `ann`; each `ann`
holds `cls`, `bbox` as pixel `[x1, y1, x2, y2]`, `transcription` and
`dmy_ann` (the day / month / year boxes inside a date). In the files the
classes are `date`, `due`, `prod` and `code`, and the test split marks its
one expiry per photo as `exp`. Full notes: [`datasets.md`](datasets.md).

**Store photos** (`$STORE_PHOTOS_DIR`), 99 in total:

| Folder | Photos | Expiry labels |
|---|---|---|
| `samples1/` | 9 | none yet |
| `samples2/` | 49 | 44 known, 5 with no expiry from that photo alone |
| `samples3/` | 41 | none yet |

The 44 known expiries are in Scanify's `backend/tests/ground_truth.py`,
typed up from the docx that came with the photos. They're stored as
`DD/MM/YY`. A pack that prints only a month is resolved to the last day of
that month, so the file doesn't say whether the pack printed a day.

**Scanify's reader** is `scan()` in `$SCANIFY_DIR/backend/app/scanner.py`.
It runs in Scanify's own venv (Python 3.14, `rapidocr-onnxruntime==1.2.3`).
It returns `expiry` as an ISO date, again resolving month-only dates to the
last day of the month. Measured result on the 44: **32 right, 12 missed,
0 wrong.**

---

## Decisions

Each comes with the reason for it. The ones marked **(new)** add to or
pin down something in `CLAUDE.md`, so check them before slice B1.

1. **One `Sample` type for both datasets.** The ExpDate loader and the
   store loader both return the same thing: image path, size, boxes
   (normalised `xyxy` in the upright frame, with a label), the true expiry
   (year, month, day-or-none) and the failure tag. The contact sheet, the
   stats and the baselines then each get written once and run on either
   set with `--set expdate-test` or `--set store`.
2. **Every image is opened through one function** that applies the EXIF
   orientation (`bestbefore.images.load_upright`). Phone photos are often
   stored sideways with a "rotate me" flag. If one script respects the flag
   and another doesn't, boxes land in the wrong place without any error.
3. **(new) A validation split comes out of Products-Real train:** 15% of
   the images, seed 0. M1 needs something to choose epochs and settings on,
   and the rules forbid tuning on the test split. Products-Real test is
   touched only for final numbers.
4. **(new) What `other-face` means:** *the expiry isn't visible in this
   photo* (it's on another face of the pack). A photo of the expiry panel
   with no barcode in frame is **not** `other-face`. The date is right
   there, so it counts towards reading accuracy and gets tagged by what
   makes it hard to read. Without this rule, close to half the store photos
   would be dropped from reading accuracy.
5. **(new) One tag per photo, picked by a fixed order:** the first one that
   applies out of `other-face` → `sideways` → `curved` → `dot-matrix` →
   `small-print` → `clean`. Geometry comes before print style because a
   sideways or curved line breaks text *detection* before the font matters.
   A fixed order makes the tagging repeatable. *(Changed on 2026-10-21:
   `small-print` vs `clean` is now computed from the expiry box by the
   import, not taken from the hand tag. See B2, check 6.)*
6. **Expiry labels record what the pack prints:** `2026-11-20` if it prints
   a day, `2031-04` if it prints only a month. This is what lets the
   scoring follow the rule "exact day / month / year, or month and year
   only when the pack prints no day".
7. **(new) The store labels never enter git.** They're per-photo data about
   the retailer's products, and only aggregate numbers leave the machine.
   They live in `data/store/` (gitignored). The repo holds only their
   SHA-256 hash plus counts per tag, which is enough to prove the test set
   never changed.
8. **Baseline B runs in Scanify's own venv, as a subprocess.** That
   reproduces exactly the setup Scanify was measured with, keeps Scanify's
   dependencies out of this project, and copies no Scanify code. The run
   also records Scanify's git commit, so baseline B stays pinned.
9. **Baseline A uses the same RapidOCR version as Scanify** (`1.2.3`), with
   default settings on the full upright photo. A and B then differ only in
   what Scanify adds (resize, sharpening, rules), so the gap between them
   has a clear meaning.
10. **(new) This repo uses Python 3.11** (`requires-python >= 3.11`).
    The plan first said 3.12, but the laptop has 3.11, not 3.12, and 3.11
    works just as well with PyTorch, Ultralytics and the usual HPC modules.
    Scanify's 3.14 doesn't matter, because B runs in its own venv.
11. **Per-image outputs go in `results/runs/<run>/`, never in `results/`
    itself.** The gitignore lets `.md` and `.csv` files at the top of
    `results/` into git, so a per-photo CSV left there could be committed
    by accident.

---

## Slices

Each slice ends in a commit that runs on its own, except B1, which is
labelling by hand. The order and what blocks what are covered below the
slices.

### Part A: ExpDate

#### A1. Project setup

**Goal:** an installable package with config, image loading and run
records, so every later slice starts from the same base.

- `pyproject.toml`: package `bestbefore` from `src/`, `requires-python =
  ">=3.11"`. Dependencies: `numpy`, `pillow`, `python-dotenv`; each later
  slice adds what it needs (`rapidocr-onnxruntime==1.2.3` arrives with
  C1). Extras: `dev` (`pytest`) and `download` (`gdown`).
- `src/bestbefore/config.py`: reads `.env` and exposes `EXPDATE_DIR`,
  `STORE_PHOTOS_DIR` and `SCANIFY_DIR` as `Path`s. Fails with a clear
  message naming the missing variable.
- `src/bestbefore/images.py`: `load_upright(path) -> PIL.Image`, which
  applies the EXIF orientation. Every image is opened through this.
- `src/bestbefore/runs.py`: `start_run(name, config, seed) -> Path` creates
  `results/runs/<YYYYMMDD-HHMM>-<name>/run.json`, holding the config, the
  git commit (and whether the tree had uncommitted changes), the seed, the
  Python version and package versions.
- `tests/test_images.py`: builds a small image tagged "rotate 90°" on the
  fly and checks that it loads upright.

**Check:** a fresh venv, `pip install -e ".[dev]"`, then `pytest` passes.
**Commit:** `feat: package skeleton, config from .env, upright image loading and run records`

#### A2. Download ExpDate and write the dataset notes

**Goal:** the two zips unpacked under `$EXPDATE_DIR`, a loader that reads
them, and a written record of exactly what was downloaded.

- `scripts/download_expdate.py products-real products-synth` downloads
  each zip (or uses one already in `$EXPDATE_DIR/downloads/`), checks its
  SHA-256 and unzips it into its own empty folder:
  `$EXPDATE_DIR/products-real/` and `$EXPDATE_DIR/products-synth/`. On the
  cluster, run it on the login node.
- `docs/datasets.md`: the source URL, licence (CC BY 4.0) and citation,
  the Drive file IDs, the sizes and SHA-256 of both zips, the folder layout
  as unpacked, the number of images and annotations per split, the class
  names actually found, and anything surprising.
- `src/bestbefore/samples.py`: the shared `Sample`, `Box` and `Expiry`
  types. `src/bestbefore/expdate.py`: `load_split("real-train" |
  "real-test" | "synth") -> list[Sample]`, which checks that every image
  has an annotation and every annotation has an image.
- `scripts/check_expdate.py`: counts per split, plus the problems worth
  knowing about: a JSON size that doesn't match the upright image, boxes
  outside the image, and test expiries with no full date.
- `tests/test_expdate.py`: parses a small hand-written JSON in the
  documented format.

**Check:** the counts match the paper (1,102 / 665 / ~12,000). If they
don't, write down the real numbers. Don't "fix" them.
**Commit:** `feat: ExpDate loader and dataset notes`

#### A3. Look at the data: contact sheets and stats

**Goal:** see the labels with your own eyes before trusting them, and get
the numbers that set M1's input size.

- `scripts/contact_sheet.py --set <set> --n 50 --seed 0`: a grid of 50
  thumbnails with their boxes drawn, coloured by class, with the
  transcription written on each box. Run it on `real-train`, `real-test`
  and `synth`. The sheets go to `results/runs/` and aren't committed,
  because they're made from ExpDate images.
- `scripts/box_stats.py --set <set>` writes `results/m0_expdate_stats.md`
  (committed). For each split:
  - images, image size (median and range), and how many carry an EXIF
    orientation flag
  - box counts per class
  - dates per photo (0 / 1 / 2 / 3+)
  - date box size as a share of the photo's area (p10 / median / p90)
  - date box height in pixels when the photo is resized to 640, and the
    share under 8 px and under 16 px. This is what decides whether M1
    can use YOLO's default 640 input or needs a larger one.
  - share of tall boxes (height > width), which shows how many sideways
    dates ExpDate has
- Answer these four questions in `docs/datasets.md`:
  1. Do the boxes line up on the photos?
  2. **What does a `date` box cover:** just the digits, or the "EXP" word
     too? The store boxes in B1 must follow the same rule, or a detector
     will lose IoU only because the two sets were labelled differently.
  3. How does the test set mark which date is the expiry (`exp`)? Can
     expiry and manufacture dates be told apart in train?
  4. How many synthetic and real photos have dot-matrix, curved or
     sideways dates? A rough count is enough; it sets expectations for M2.

**Check:** rerunning both scripts gives the same numbers.
**Commit:** `feat: contact sheets and box stats; ExpDate findings`

#### A4. Convert to YOLO format

**Goal:** a dataset Ultralytics can train on directly in M1.

- `src/bestbefore/yolo.py`: normalised `xyxy` ⇄ YOLO `cx cy w h`.
  `src/bestbefore/sets.py`: `load_set(name)`, so scripts take `--set`.
- `scripts/expdate_to_yolo.py` writes to `$EXPDATE_DIR/yolo/`:
  - `images/<split>/` as symlinks (no second copy of 1.7 GB), and
    `labels/<split>/*.txt` with one `class cx cy w h` line per box
  - image lists `train_real.txt`, `train_synth.txt`, `val.txt` and
    `test.txt`
  - `expdate.yaml` with `train: [train_synth.txt, train_real.txt]`. Keeping
    real and synthetic in separate lists means M2's "with and without
    synthetic data" comparison is a one-line change.
- Classes: `0 date`, `1 due`, `2 prod`, `3 code`, named as in the files.
  The test set's `exp` boxes become `date` for detection; the loader still
  knows which one is the expiry, for reading.
- Boxes are clipped to the image. Zero-area boxes are dropped and counted
  in the log.
- `tests/test_yolo.py`: `xyxy → yolo → xyxy` round trip within 1 px.

**Check:** the converter reads every label file back and compares it with
the source boxes (worst error: 0.003 px). This is stronger than comparing
contact sheets by eye. The symlinks are absolute paths, so on the cluster
you rerun the converter instead of copying the folder.
**Commit:** `feat: ExpDate to YOLO conversion with a validation split`

#### What Part A found

The details are in [`datasets.md`](datasets.md). Four findings change
later slices:

- **Box rule for the store photos (B1):** a date box covers the date value
  only, not the "EXP" / "Best before" words and not a time or lot code
  printed after it.
- **ExpDate has almost no sideways dates** (0% of real date boxes are
  taller than wide), and few Indian formats: half the test set is Korean
  `YYYY.MM.DD`. Both are now firm requirements for the M2 generator.
- **Real dates are small:** about 16 px tall at YOLO's default 640 input,
  31 px at 1280. M1 compares the two on the validation split.
- **Synthetic photos pair PROD and EXP dates at random**, so they teach
  where dates are, not which one is the expiry.

### Part B: Store test set

#### B1. Label the store photos (by hand, no commit)

**Goal:** for all 99 photos: the expiry as printed, a failure tag, and a
tight box around every date. One pass per photo, in Label Studio, running
locally. Expect about 3 to 4 hours; split it over two sittings.

**Setup.** Install Label Studio outside the project env, and serve the
photos from where they already are, so no copies are made. *(Changed on
2026-10-20: the laptop has no pipx, so Label Studio got its own venv in
`~/.venvs/label-studio`, which does the same job. Its database lives in
`data/label-studio/` (gitignored), so the labels stay with the other
store data, and it listens on 127.0.0.1 only, so nothing else on the
network can reach the photos.)*

```bash
/opt/homebrew/bin/python3.11 -m venv ~/.venvs/label-studio && ~/.venvs/label-studio/bin/pip install label-studio
```

```bash
set -a && source .env && set +a && LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$STORE_PHOTOS_DIR" LABEL_STUDIO_BASE_DATA_DIR="$PWD/data/label-studio" ~/.venvs/label-studio/bin/label-studio start --internal-host 127.0.0.1 --port 8080
```

Run the second command from the repo folder. In the Claude app, the same
command is saved as `label-studio` in `.claude/launch.json`.

Create a project, paste the interface below into *Labeling Interface →
Code*, then add each `samples*/` folder under *Cloud Storage → Add Source
Storage → Local files*, with the file filter `.*\.jpe?g$`. Never use a
hosted Label Studio, because the photos must stay on this machine.

```xml
<View>
  <Image name="img" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="box" toName="img">
    <Label value="expiry" background="#d62728"/>
    <Label value="mfg" background="#1f77b4"/>
    <Label value="other_date" background="#7f7f7f"/>
  </RectangleLabels>
  <Header value="Failure tag: the first that applies"/>
  <Choices name="tag" toName="img" choice="single" required="true">
    <Choice value="other-face"/>
    <Choice value="sideways"/>
    <Choice value="curved"/>
    <Choice value="dot-matrix"/>
    <Choice value="small-print"/>
    <Choice value="clean"/>
  </Choices>
  <Header value="Expiry: YYYY-MM-DD, or YYYY-MM if the pack prints no day. Blank if not in this photo."/>
  <TextArea name="expiry" toName="img" maxSubmissions="1" editable="true"/>
  <Choices name="derived" toName="img" choice="single">
    <Choice value="derived from mfg + shelf life"/>
  </Choices>
  <Header value="Mfg date if printed (same format rules)"/>
  <TextArea name="mfg" toName="img" maxSubmissions="1" editable="true"/>
  <Header value="Expiry format as printed, e.g. DD/MM/YYYY, MON YYYY, YYYYMMDD"/>
  <TextArea name="format" toName="img" maxSubmissions="1" editable="true"/>
  <TextArea name="note" toName="img" maxSubmissions="1" editable="true"/>
</View>
```

**Labelling rules**

- **Boxes.** One box per date, tight around the date value itself, as
  ExpDate does it: digits or month name with their separators, not the
  "EXP" / "Best before" words and not a time or lot code printed after
  it. Zoom in for small print. A sideways date still gets an upright
  rectangle. A range like `05/2025-05/2027` is two dates: the first half
  is `mfg`, the second is `expiry`. Box every date you can see, with
  `mfg` and `other_date` (packed on, imported on) as well as `expiry`.
- **Expiry.** Type it at the level of detail the pack prints. For a pack
  that prints only a mfg date plus a shelf life ("Best Before 1 Year from
  Mfg. Dt."): work out the expiry, tick *derived*, and box only the mfg
  date. If you can't read the date even zoomed in, leave it blank and
  write `unreadable` in the note; the photo is then counted separately.
- **Tags**, taking the first that applies:
  - `other-face`: the expiry isn't visible in this photo. Leave expiry
    blank, but box any dates that are visible.
  - `sideways`: the expiry line runs vertically, or is tilted more than
    about 30°, in the upright photo.
  - `curved`: the expiry is on a curved surface (cap, jar, can) that
    visibly bends the line.
  - `dot-matrix`: printed as dots (dot-matrix or inkjet).
  - `small-print`: the characters are tiny compared to the frame (roughly,
    the box is shorter than 1.5% of the photo's long side). B2 checks this
    against the boxes.
  - `clean`: none of the above.
- **Retype the 44 known expiries too.** Don't copy them from Scanify. B2
  compares your answers with Scanify's labels, and any disagreement points
  to a mistake in one of the two.

When done, export as **JSON-MIN** to `data/store/label_studio_export.json`.

**Done 2026-10-21.** All 99 photos, 199 date boxes (81 `expiry`, 91 `mfg`,
27 `other_date`). The pre-filled text fields came with no boxes, so every
box was drawn by hand. No photo was tagged `curved`: no date on a cap or
jar visibly bends its line.

#### B2. Import and check the labels

**Goal:** turn the export into `data/store/labels.json` and catch labelling
mistakes before freezing.

- `src/bestbefore/store.py`: the `labels.json` schema and the store loader,
  returning `Sample`s. One entry per photo, keyed by its path relative to
  `$STORE_PHOTOS_DIR`:

  ```json
  {
    "samples2/sample 1.jpg": {
      "expiry": "2026-11-20",
      "derived": false,
      "mfg": "2025-05-21",
      "format": "DD/MM/YYYY",
      "tag": "dot-matrix",
      "boxes": [{"label": "expiry", "xyxy": [0.12, 0.80, 0.31, 0.84]}],
      "note": ""
    }
  }
  ```

  (The values are made up to show the shape.)
- `scripts/import_store_labels.py` converts the export and fails loudly,
  listing the photos (on screen only), if any check fails:
  1. All 99 photos are labelled once, with no extras.
  2. Every expiry parses as `YYYY-MM-DD` or `YYYY-MM`, with a year between
     2024 and 2035.
  3. A photo with a visible, non-derived expiry has at least one `expiry`
     box. An `other-face` photo has no expiry and no `expiry` box.
  4. **The 44 match Scanify's labels.** It reads `ground_truth.py` from
     `$SCANIFY_DIR` by path and compares year and month, plus the day when
     the pack prints one. On a disagreement, look at the photo again; if
     Scanify's label turns out to be wrong, that's a finding worth noting.
  5. **Orientation:** the image size Label Studio recorded must equal the
     size from `load_upright`. If width and height are swapped, Label
     Studio drew on the raw frame and the boxes have to be rotated.
  6. `clean` photos whose expiry box is under the `small-print` threshold
     get flagged for a second look. *(Changed on 2026-10-21: the check
     found 20 of the 37 hand-tagged `small-print` photos above the
     threshold (1.52% to 2.76%), and no `clean` one below it. So the
     rule is now applied, not flagged: a `clean` or `small-print` photo
     is `small-print` exactly when its expiry box, or the mfg box for a
     derived expiry, is under 1.5% of the photo's long side. That makes
     the tag repeatable from the boxes. The owner chose this over keeping
     the hand tags.)*
- Draw the store boxes with `contact_sheet.py --set store` and check every
  photo by eye. The sheet stays in `data/store/`, never in git.
- `box_stats.py --set store` writes `results/m0_store_stats.md`
  (committed, aggregate only): photos per tag, expiry formats with counts,
  month-only vs day, derived, and date box sizes next to ExpDate's. That
  last column puts a number on how different the store photos are from
  the training data.
- `tests/test_store_import.py` uses a small fake export with made-up file
  names, so no store data goes into the tests.

**Done 2026-10-21.** The conversion and the checks went into their own
module, `store_import.py`, so `store.py` holds only what later
milestones load. Checks 1 to 5 all pass on the first export:

- Check 4 compares all **49** entries in Scanify's ground truth (the 44
  known expiries and the 5 marked "none from this photo"). All 49 agree.
- Check 5 matters: **46 of the 99 photos** carry an EXIF rotation flag,
  and Label Studio drew every box on the upright frame.
- Check 6 moved 20 photos from `small-print` to `clean`.

The contact sheet is written to `data/store/runs/`, not `results/runs/`,
so no store photo ends up anywhere a results folder gets copied.

**Commit:** `feat: store label import with consistency checks; store stats`

#### B3. Freeze the test set

**Goal:** make any later change to the test set impossible to miss.

- `scripts/freeze_store_test_set.py` writes `data/store/test_set.json`: the
  labels plus each photo's SHA-256, a version (`v1`) and a date. It prints
  the file's own SHA-256.
- That hash goes into `src/bestbefore/store.py` as `TEST_SET_SHA256`.
  `load_test_set()` checks the file hash and every photo hash, and stops
  with an error if anything changed.
- `results/store_test_set.md` (committed): version, hash, and counts (by
  tag, with expiry, month-only, derived, unreadable).
- After this point, a label changes only as a new version (`v2`), with a
  line in `results/store_test_set.md` saying what changed and why, and
  every baseline is rerun. *(Changed on 2026-10-28: `v2` is now a
  different set of photos, the test set once store photos are trained on
  (Part D). A label fix to a frozen set becomes a revision of that set:
  see Part D's Later.)*

**Done 2026-10-21.** v1 is frozen, with hash `7b9a4043…`. The script
refuses to overwrite a frozen version, and a new one needs `--why`.
Loading the set checks the file and all 99 photo hashes in about 0.2 s.

**Commit:** `feat: freeze store test set v1`

#### What Part B found

Full tables: [`results/m0_store_stats.md`](../results/m0_store_stats.md)
and [`results/store_test_set.md`](../results/store_test_set.md).

| Tag | Photos | With an expiry |
|---|---:|---:|
| other-face | 12 | 0 |
| sideways | 16 | 16 |
| curved | 0 | 0 |
| dot-matrix | 21 | 21 |
| small-print | 17 | 17 |
| clean | 33 | 33 |
| **all** | **99** | **87** |

- **87 photos count towards reading accuracy**, more than the 60 to 70
  first expected. 51 expiries are printed to the day and 36 to the month
  only; 6 are derived from mfg + shelf life.
- **Store dates are smaller than ExpDate's.** At 640 px input, the
  median date box is 11.0 px tall against 15.5 px on ExpDate test. 17.6%
  of store date boxes are under 8 px, against none in ExpDate. At
  1280 px only 0.5% are. This is the risk M1 named ("Store dates may be
  much smaller"). M1 D3 will show what it costs the 640 px detector.
- **17.6% of store date boxes are tall** (sideways), against 0% in
  ExpDate. M1 isn't expected to find these; that's M2's job.
- **Most store photos show two dates** (64% vs 9% in ExpDate), usually
  mfg and expiry. Telling them apart matters for the reader (M3).
- **15 expiry formats**, led by `DD/MM/YYYY` (27), `Month YYYY` (19) and
  `MON YYYY` (11). The M2 generator has to cover these.

### Part C: Baselines

#### C1. Scoring and baseline A

**Goal:** one scoring rule used by every reader from now on, and the first
baseline.

- `src/bestbefore/scoring.py`: `score(truth, prediction) -> "right" |
  "wrong" | "none"`. A day-level truth needs the exact day, month and
  year. A month-level truth needs only month and year (so B's last-day
  answers score correctly). Any other date is `wrong`, including reading
  the mfg date as the expiry. No date at all is `none`.
- `src/bestbefore/dates.py`: a simple date regex covering `DD/MM/YYYY`
  and `DD/MM/YY` (with `/ . -` or a space between parts), `MM/YYYY`,
  `YYYYMMDD`, `YYYY.MM.DD`, `YYYY.MON.DD`, `DD MON YYYY` and `MON YYYY`.
  Day before month when both are possible. Write it from ExpDate **train**
  transcriptions and the format list in `CLAUDE.md`, and freeze it
  before running on any test set.
- Baseline A picks **the latest date found** as the expiry, since an
  expiry comes after the manufacture date. It's the simplest rule that
  needs no label words.
- `src/bestbefore/baselines/rapidocr_regex.py`, plus
  `scripts/run_baseline.py --reader a --set expdate-test|store`. Per-image
  results (OCR text, dates found, the one picked, the score) go to
  `results/runs/<run>/`; the summary goes to stdout and `run.json`.
- ExpDate truth: the date the test set marks as the expiry, with the day
  taken from `dmy_ann` (no day box means a month-level truth). Photos with
  no marked expiry are left out and counted.
- `tests/test_dates.py` and `tests/test_scoring.py`: every format, leap
  years, month-only truth against a last-day prediction, wrong vs none.

**Done 2026-10-21.** What the written plan didn't settle, decided from
real-train only, before any test photo was read:

- **Three 2-digit parts joined by dots are year first** (`21.08.15` is
  2021-08-15). 137 of the 147 such dates in real-train start with 19 to
  22 (Korean `YY.MM.DD`). With other separators, the rule stays day
  first.
- **Years must fall in 2015 to 2039**, so `21.08.03` can't become 2003.
- **Two-part dates with no 4-digit year (`08.27`) are skipped:** as
  likely a price or a weight.
- **A date-shaped match that isn't a real date blocks the shorter
  matches inside it**, so `29/02/2027` gives nothing, not "February 2027".
- The plan's `--set expdate-test` is `--set real-test`, the name used
  everywhere else. Store runs write to `data/store/runs/`, and every run
  records a SHA-256 of `dates.py` and the reader's module.
- `DDMMYYYY` and ordinals (`15th`) are left out on purpose: they're known
  only from the store photos.

The rules read 1,211 of the 1,244 real-train date transcriptions (97%)
whole; 31 of the 33 misses are two-part dates.

| Baseline A | n | right | wrong | none |
|---|---:|---:|---:|---:|
| ExpDate Products-Real test | 655 | 502 | 28 | 125 |
| Store, sideways | 16 | 6 | 5 | 5 |
| Store, dot-matrix | 21 | 7 | 0 | 14 |
| Store, small-print | 17 | 14 | 2 | 1 |
| Store, clean | 33 | 22 | 5 | 6 |
| **Store, all with an expiry** | **87** | **49** | **12** | **26** |

- 10 ExpDate test photos have no marked expiry and aren't scored. Of the
  12 store photos tagged `other-face`, A read a date from 4: necessarily
  not the expiry.
- **On the store photos, every error is a reading error.** All 26
  `none` photos had no date found at all. All 12 `wrong` reads are
  earlier than the truth, and in none of them was the true expiry among
  the dates found: OCR missed it, so the latest-date rule fell back to
  the mfg date. 5 of the 12 are packs that print only mfg + shelf life,
  which this rule can't get right. This matches Scanify's finding: the
  limit is reading the print, not parsing the text.
- On ExpDate, **16 of the 28 wrong reads are a month-only stub of a
  dated pack.** In 14 of them, OCR glued the day to the next printed
  token, usually a time or lot code (`2021.06.0717:57…`). The full date
  then failed the "no digit may touch a date" boundary, and the
  month-only pattern matched the stub (`2021.06`): a "none" became a
  "wrong". Of the 12 others, most are the year-first vs day-first
  ambiguity going the other way (`16.02.21` was 16 Feb 2021) or OCR
  digit errors (`2020.12.2` for `2020.12.21`). The true expiry was never
  among the dates found. The rules stay frozen: changing them now would
  be tuning on the test split. The lesson goes to Later.
- Dot-matrix is the hardest tag (7/21, with 14 `none`). Small print is
  not a problem for whole-photo OCR here (14/17).
- Both runs used code that wasn't committed yet, and the ExpDate run's
  `dates.py` hash predates a comment-only fix. **To do:** rerun A on
  ExpDate test once on the laptop from the committed code, so its record
  matches. The numbers can't change.

**Commit:** `feat: date scoring and baseline A (RapidOCR + regex)`

#### C2. Baseline B

**Goal:** Scanify's reader on the same sets, through the same scoring.

- `scripts/scanify_runner.py`: standalone, imports nothing from this
  package. It's run with `$SCANIFY_DIR/.venv/bin/python`, adds
  `$SCANIFY_DIR/backend` to `sys.path`, imports `app.scanner`, warms the
  OCR up once, then calls `scan()` on each image in a list and writes
  JSONL.
- `src/bestbefore/baselines/scanify.py` launches it as a subprocess,
  reads the JSONL back and scores it. `run.json` records
  `git -C $SCANIFY_DIR rev-parse HEAD`.
- `run_baseline.py --reader b --set expdate-test|store`.

**Check:** on Scanify's 44, B must reproduce **32 right, 12 none,
0 wrong**. If it doesn't, assume the bug is in this harness until shown
otherwise.

**Done 2026-10-21.** The check passes exactly: 32 right, 0 wrong, 12
none, scored Scanify's way (the ISO date from `scan()` equals its ground
truth). It runs automatically at the end of every B run on the store
photos and is saved in run.json. The runner sends its JSON lines to
stdout rather than a file, so results stream back as they come. Scanify
was at commit `9e05c00`, clean.

| Baseline B | n | right | wrong | none |
|---|---:|---:|---:|---:|
| ExpDate Products-Real test | 655 | 57 | 6 | 592 |
| Store, all with an expiry | 87 | 60 | 2 | 25 |

- **On ExpDate, B's rules don't fire.** Of the 448 photos A read right
  and B returned nothing for, 412 have the true date in B's own OCR text
  (`2021.08.03`). Scanify's rules were written for Indian packs and want
  an "EXP" label or an Indian format, so an unlabelled Korean date is
  "none". B is cautious, with only 6 wrong reads, but reads 9%. This is
  the reason a trained reader is needed: hand-written rules fit the
  photos they were written for.
- **B's 2 wrong store reads are a parsing failure, not a reading one.**
  Both packs print `19th December 2026`, and OCR read it perfectly; the
  rules skip the ordinal, read "December 2026" and answer the 31st. A
  got the same two wrong (it has no ordinal rule either, by design: see
  C1).
- On `other-face` photos B returned no date at all (0 of 12); A returned
  one for 4.

**Commit:** `feat: baseline B (Scanify's reader, run in its own venv)`

#### C3. The baseline table and README

**Goal:** the M0 deliverable.

- `scripts/baseline_table.py` reads the latest A and B runs and writes
  `results/m0_baselines.md`:

  | ExpDate Products-Real test | n | A right | A wrong | A none | B right | B wrong | B none |
  |---|---|---|---|---|---|---|---|

  | Store photos, by tag | n | A right | A wrong | A none | B right | B wrong | B none |
  |---|---|---|---|---|---|---|---|
  | dot-matrix · sideways · curved · small-print · clean | | | | | | | |
  | **all readable** | | | | | | | |
  | other-face: dates returned (should be 0) | | | | | | | |

  Report counts as `k/n`, not only percentages. With ~99 photos, some tags
  will have under 10, so a one-photo difference is noise.
- `README.md`: the problem, the baseline table, how to reproduce it, and
  the ExpDate credit (cite the paper and credit KIST, as CC BY 4.0
  requires).

**Done 2026-10-21.** `results/m0_baselines.md` is written from the
latest run of each baseline on each set, and lists those runs with their
commits. The README already had the problem and the ExpDate credit; it
gained the result, what it shows, and how to reproduce it. The markdown
table helper moved from `m1.py` to its own `tables.py`, so the M0 scripts
no longer import from M1, and text columns can be left-aligned.

All four runs were made before their code was committed, and the run list
says so. Each run also records a SHA-256 of the reader's code, so it
stays traceable. **Optional:** once this is committed, rerun all four
from the clean commit (about 15 minutes on the laptop) and rebuild the
table. The numbers can't change.

**Commit:** `docs: M0 baseline table and README`

---

## Order

```
A1 ─ A2 ─ A3 ─┬─ A4
              ├─ B1 ─ B2 ─ B3 ─┐
              └─ C1 (ExpDate) ─┴─ C1 (store) ─ C2 ─ C3
```

- **B1 is the long part, and it waits for A3**, because the box convention
  comes from ExpDate. Start labelling as soon as A3 answers question 2.
- A4 and baseline A on ExpDate can be written while the labelling is going
  on.

## Files at the end of M0

```
docs/        m0-data-and-test-set.md, datasets.md
src/bestbefore/
             config.py, images.py, runs.py, samples.py, sets.py, expdate.py,
             yolo.py, store.py, store_import.py, dates.py, scoring.py,
             tables.py, baselines/{rapidocr_regex,scanify}.py
scripts/     download_expdate.py, check_expdate.py, contact_sheet.py,
             box_stats.py, expdate_to_yolo.py, import_store_labels.py,
             freeze_store_test_set.py, run_baseline.py, scanify_runner.py,
             baseline_table.py
tests/       one file per module above
results/     m0_expdate_stats.md, m0_store_stats.md, store_test_set.md,
             m0_baselines.md            (committed; everything else ignored)
data/        expdate/, store/           (never committed)
```

## Risks

- **Label Studio might draw on the raw, unrotated frame.** Check 5 in B2
  catches it. *(Checked 2026-10-21: it didn't.)*
- **The store test set is small.** 87 photos with an expiry, spread over
  four tags (17 to 33 each, none `curved`). Per-tag results show
  direction, not precise rates.
- **Scanify keeps changing.** Its git commit is recorded in every B run;
  rerun B if it moves.

---

## Store test set v2 (Part D)

Planned 2026-10-28. Not a milestone: a second store test set, labelled
and frozen like v1 (B1 to B3), before anything trains on store photos.
Status: **D1 to D4 done** (2026-10-28): the photo list is frozen
(99 photos, `8be1f9e9…`) and Label Studio is ready. D5 and D6, labelling
and freezing the labels, are the owner's. The record is at the end of
this part.

**Why.** The retailer's owner has allowed training on the store photos
(`CLAUDE.md`, 2026-10-28). v1 becomes training data from the first
milestone that trains on it, so from then on it can't judge a model. v2,
new photos from the same stores, takes over as the test set. Results on
v1 from M0 to M2 stay as recorded.

**Why it lives here.** v2 follows B1 to B3 as written (the labelling
rules, the interface, the import checks, the freeze), so it sits next to
them instead of copying them into a new doc. And it isn't a milestone,
which is what a new sliced doc in `docs/` is for.

**Done when.** v2's photo list is frozen with its hash committed, the
code imports and freezes v2 next to v1, and Label Studio has a v2 project
ready to label. Labelling (D5) and freezing the labels (D6) follow, by
the owner.

### Decisions

1. **v1 doesn't change.** Its folders, its files in `data/store/`, its
   hash `TEST_SET_SHA256`, `read_test_set()` and the set name `store` stay
   as they are. M0, M1 and M2's tables must still rebuild byte for byte.
2. **v1 and v2 never mix.** Each set is a record in `store.py`
   (`store.V1`, `store.V2`): its folders under `$STORE_PHOTOS_DIR`, its
   files and its hashes. Everything that lists or loads store photos takes
   the record, so it says which set it means. v1's folders are named
   (`samples1` to `samples3`) instead of globbed as `samples*`.
3. **v2's photos are frozen twice.** First the photo list, each photo's
   SHA-256 (D3), before any labelling: that fixes which photos are the test
   set before anything trains on store photos. Then the labels (D6), as in
   B3. The import and the label freeze both check the list again, so a
   photo added, removed or changed in between is caught.
4. **(departure from B3)** B3 said a label change after freezing becomes
   "v2". v2 is now a different set of photos with its own files, and the
   freeze script's `--version` names the set. A label fix to a frozen set
   becomes a revision of it (Later).
5. **No pre-filled fields.** v1's text fields were pre-filled from
   Scanify's ground truth. v2 has no ground truth, and pre-filling from a
   model would mean running one on the test set, so every field is typed
   by hand. Check 4 (agreement with Scanify) is for v1 only.

### Slices

#### D1. The photos

- v2 lives in `$STORE_PHOTOS_DIR/testset2/`, where the owner put it (a
  second copy is on the owner's Desktop). It stays there: no third copy,
  and its name doesn't match `samples*`, so nothing written for v1 can
  pick it up.
- It holds both `.jpeg` and `.jpg` files. Check that they're all JPEGs
  from their headers, and convert nothing: the two endings are the same
  format, re-encoding would change the pixels and the hashes, and both the
  code and Label Studio's filter accept either ending.
- Check for byte-identical copies, within v2 and against v1.
- `$STORE_PHOTOS_DIR` is also the Scanify repo, which has a GitHub
  remote and ignores only `samples1/` to `samples3/`. Add `testset2/` to
  its `.git/info/exclude`: ignored on this machine, and no tracked file
  in Scanify changes.

#### D2. v1 and v2 side by side in the code

- `store.py`: a `StoreSet` record per set. `V1`: `samples1` to `samples3`,
  the files in `data/store/` exactly as today, `TEST_SET_SHA256`. `V2`:
  `testset2`, its files in `data/store/v2/`, `PHOTOS_V2_SHA256` and
  `TEST_SET_V2_SHA256`. `photos_on_disk`, `labels` and `load_samples`
  take the set; every M0 to M2 caller passes `store.V1`.
- v2's photo list: `freeze_photos()` writes it; `read_photo_list()`
  checks the list file against its hash, and that the folder holds
  exactly the listed photos, unchanged. `read_frozen(set)` loads a
  frozen set, and for v2 also checks that its photos are the list's.
- `import_store_labels.py --version v1|v2` (v1 by default, so B2's
  command is unchanged) reads and writes that set's files. Checks 1 to 3,
  5 and 6 are as for v1, and the small-print rule is unchanged. Check 4
  is for v1 only. For v2, check 1 compares with the frozen list, so a v1
  photo in a v2 export fails it.
- `freeze_store_test_set.py --version v1|v2 [--photos]`: `--photos`
  freezes the photo list (D3); without it, the labels (D6), only when they
  cover exactly the frozen list. v2's summary is
  `results/store_test_set_v2.md`; v1's is left as it is. A frozen file is
  never overwritten, as before.
- Tests, with made-up file names: two sets in one fake folder each list
  only their own photos; the list check catches a photo added, removed or
  changed and a changed list file; a v1 photo in a v2 export fails
  check 1; `V1` keeps today's folders, paths and hash.

**Check:** the tests pass; M0, M1 and M2's tables and v1's summary
rebuild byte for byte; rerunning v1's import into a scratch file
reproduces `labels.json` byte for byte. D2's code comes before D3
because D3 runs it; nothing trains on store photos in between.

**Commits:** `refactor: store photos listed and loaded by set, v1's
folders named`, then `feat: import and freeze store test set v2, its
photo list first`

#### D3. Freeze the photo list

```bash
python scripts/freeze_store_test_set.py --version v2 --photos
```

This writes `data/store/v2/photos.json` (never committed): every
photo's path and SHA-256. Its hash goes into `PHOTOS_V2_SHA256`, and
`results/store_test_set_v2.md` gets the count and the hash, nothing
else.

**Commit:** `feat: freeze store test set v2's photo list`

#### D4. The Label Studio project

The same install and the same local server as v1 (B1: 127.0.0.1 only,
`label-studio` in `.claude/launch.json`), with a new project that has
v1's interface and one source storage, `testset2`, with v1's settings.
The exact steps are in D's record.

#### D5. Label (by hand, no commit)

B1's rules, as written. There is no retyping step, as v2 has no Scanify
labels to compare with. Export as JSON-MIN to
`data/store/v2/label_studio_export.json`.

#### D6. Import, check and freeze the labels

```bash
python scripts/import_store_labels.py --version v2
```

```bash
python scripts/freeze_store_test_set.py --version v2
```

Paste the printed hash into `TEST_SET_V2_SHA256`, and commit it with
`results/store_test_set_v2.md`.

**Commit:** `feat: freeze store test set v2`

### What Part D found (2026-10-28)

No v2 photo was opened, viewed or decoded, and no model was run on one.
Only their bytes were read, for the hashes and the file-type header.

**D1, the photos.**

- 99 files, all JPEG by their header: 92 end in `.jpeg` and 7 in `.jpg`,
  all lower case. Nothing was converted or renamed.
- No two are byte-identical, and none is byte-identical to a v1 photo
  (0 of 99). The Desktop copy matches `testset2/` byte for byte, file for
  file.
- **Scanify's git didn't ignore `testset2/`**: it showed as untracked
  there, so a `git add -A` in Scanify would have staged all 99 photos,
  and Scanify has a GitHub remote. `testset2/` is now in Scanify's
  `.git/info/exclude`, and `git check-ignore` confirms it. This only holds
  on this laptop: another clone of Scanify wouldn't ignore it.

**D2, the code.**

- v1 is unchanged. The named folders `samples1` to `samples3` list the
  same 99 photos as the old `samples*` glob and as the frozen file.
  `labels(store.V1)` returns exactly what `labels()` returned.
  `data/store/test_set.json` still hashes to `7b9a4043…`.
- M0's baseline table and store stats, M1's and M2's reports rebuild byte
  for byte, before and after the change. So does v1's
  `results/store_test_set.md`, rebuilt from its frozen file by the new
  freeze script. `experiments/m1-store-misses/analyse.py` prints the same
  output before and after.
- v1's import, rerun into a scratch file, reproduces `labels.json` byte
  for byte, with all 7 checks passing. The freeze script refuses v1
  (already frozen, no photo list).
- Added beyond the plan: loading a frozen set also checks that the file
  names that set (`version`), so one set's file can't be read as
  another's.
- Tests: 500 before, 519 after, all passing.

**D3, the photo list.** Frozen 2026-10-28: 99 photos, all in
`testset2/`, SHA-256 of `data/store/v2/photos.json` `8be1f9e9cb25…`. Its
hashes match a `shasum` run made before any code was written.

**D4, Label Studio.** Label Studio 1.23.2, in `~/.venvs/label-studio`.
v1's project, read from Label Studio's database without changing it:

- its interface is B1's XML exactly, apart from whitespace
- it has three Local files storages, one per folder, with the filter
  `.*\.jpe?g$`, every file a task, and no recursive scan

The server listens on `127.0.0.1:8080` only: the laptop's network address
gets no answer.

To set up v2's project:

1. Start the server: `label-studio` in the Claude app's preview, or B1's
   command run from the repo folder. Open <http://localhost:8080> and log
   in with the account used for v1.
2. *Create Project*, named `Store test set v2`. Under *Labeling Setup →
   Custom template → Code*, paste B1's XML, then *Save*. Import no data.
3. *Settings → Cloud Storage → Add Source Storage*:
   - storage type *Local files*
   - title `testset2`
   - absolute local path: the value of `$STORE_PHOTOS_DIR` in `.env`,
     followed by `/testset2`
   - file filter regex `.*\.jpe?g$`
   - *Treat every bucket object as a source file* on, recursive scan off
   - *Check Connection*, *Add Storage*, then *Sync Storage*

   The project must show **99 tasks**. If it doesn't, stop.
4. Label every photo by B1's rules, then *Submit* each. Leave v1's
   project alone. A v1 photo in the v2 export fails check 1 anyway.
5. *Export → JSON-MIN*, then move the download to
   `data/store/v2/label_studio_export.json`. Then run D6.

Don't edit, rotate or re-save a photo in `testset2/`: macOS Preview's
rotate rewrites the file. Any change to a photo fails the photo-list
check at import and at freezing.

**Not checked:**

- Nothing about what the v2 photos show: their tags, whether a date is
  visible, their sizes, how many carry an EXIF rotation flag. Labelling
  and import check 5 will show these.
- Whether the same pack appears in both v1 and v2 (see Later).
- v2's Label Studio project isn't created yet, because that needs the
  owner's login. So the 99-task sync hasn't been seen, and the steps
  above haven't been run for v2.
- v2's import and label freeze have run only on made-up data in the
  tests, not on a real export.
- `results/m0_expdate_stats.md` wasn't rebuilt. It reads no store code.

### Later

- **`store-v2` as a `--set` name**, when M3 first scores a model on v2.
  It's left out until then on purpose: `eval_detector.py` keeps store
  predictions private and scores a test set once only for the sets it
  names (`SCORED_ONCE`), so the name has to arrive together with that
  guard. Its runs go to `data/store/v2/runs/`.
- B2's by-eye contact sheet for v2 needs that set name too. Until then,
  review the boxes in Label Studio.
- **A label fix to a frozen set as a revision** (e.g. `v2.1`), with its
  own file and a history line, keeping the original.
- **The same pack in both sets.** Only byte-identical copies are checked.
  A pack photographed for both v1 and v2 would let training on v1 leak
  into v2's results. Check this by eye while labelling, before v1 is
  trained on.
- **How v1 enters training** (M3): its photos and labels as a training
  list, with v2 never in one.

## Later

- Tag a sample of ExpDate test photos with the same failure tags, so
  ExpDate results can be split by tag too.
- A detection baseline with no training: does RapidOCR's own text box
  cover the expiry box? That would be M1's bar to clear.
- Allow several tags per photo if one tag turns out to hide too much (a
  curved, dot-matrix cap is both).
- Box the "EXP" / "MFG" words on store photos as well, to match ExpDate's
  `due mark` / `production mark` classes.
- Download Date-Synth, Date-Real and the Components sets (M3), and
  SmartDoc-QA and SPAQ (M4).
- **A date OCR glued to its neighbour should give "none", not a coarser
  date** (found in C1). 14 of baseline A's 28 wrong reads on ExpDate are
  a day glued to a time or lot code (`2021.06.0717:57`), which the regex
  then read as month only. Any rule-based reader has this trap. M3's
  recogniser reads a crop of the date alone, so it should avoid it;
  check that it does.
