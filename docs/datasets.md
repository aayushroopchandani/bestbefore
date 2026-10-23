# Dataset notes

What is actually in each dataset, as found by looking, not as described.
The numbers come from `scripts/check_expdate.py` and `scripts/box_stats.py`.

---

## ExpDate

Seker, A. C. & Ahn, S. C. (2022). *A generalized framework for recognition
of expiration dates on product packages using fully convolutional
networks.* Expert Systems with Applications, 203, 117310. Licence CC BY
4.0: cite the paper and credit KIST (Korea Institute of Science and
Technology). Page: <https://felizang.github.io/expdate/index_expdate.html>.

### Download

```bash
python scripts/download_expdate.py products-real products-synth
```

The script fetches the zips from the Drive folder linked on the page,
checks them against the hashes below and unpacks each into its own folder.
A zip fetched in the browser can be dropped into
`$EXPDATE_DIR/downloads/` first. On the cluster, run it on the login node.

| File | Drive ID | Bytes | SHA-256 |
|---|---|---|---|
| `Products-Real.zip` | `1ZyDjpA807OkYFN2ZKgkk9ps1s-UCELSn` | 660,701,583 | `31dff6e0045ab6f4683c9e20b5dd04fedd751c174b681193744a013637e38da6` |
| `Products-Synth.zip` | `1Ewnsqjm2SqdZvAN0EhVS6PulfoE59giX` | 1,156,418,424 | `b10d82a1af1268eeb58aa3831537b577fba867c02687ae2b7fa9c278e18977ec` |

The Date and Components zips (M3) are listed in the script too; their
hashes get recorded when they're first downloaded.

### Layout as unpacked

```
$EXPDATE_DIR/
├── downloads/                     the zips
├── products-real/Products-Real/
│   ├── train/                     annotations.json, images/ (1,102)
│   └── evaluation/                annotations.json, images/ (665)
└── products-synth/Products-Synth/ annotations.json, images/ (11,858)
```

`train/` also holds a stray Windows `desktop.ini`, which the loader ignores.

### Annotation format

One `annotations.json` per split, mapping each file name to its size and
boxes. Boxes are pixel `[x1, y1, x2, y2]` in the **upright** image. A test
image looks like this:

```json
"test_00001.jpg": {"height": 1092, "width": 1455, "ann": [
  {"cls": "exp", "bbox": [638, 576, 950, 623], "transcription": "2021.08.03",
   "dmy_ann": [{"cls": "month", "bbox": [793, 578, 854, 622], "transcription": "08"},
               {"cls": "year",  "bbox": [639, 577, 753, 621], "transcription": "2021"},
               {"cls": "day",   "bbox": [889, 579, 949, 623], "transcription": "03"}]}]}
```

The class names in the files differ from the project page:

| In the files | Meaning | Where |
|---|---|---|
| `date` | a printed date | train and synth: every date. Test: the dates that are **not** the expiry (68) |
| `exp` | the expiry date, with `dmy_ann` day / month / year parts | test only, exactly one per photo |
| `due` | the expiry mark: "EXP", "Best before", 까지 ("until") | all |
| `prod` | the production mark: "MFG", 제조 ("made") | all |
| `code` | a batch or lot code | all |

### What's in it

| | real-train | real-test | synth |
|---|---:|---:|---:|
| photos | 1,102 | 665 | 11,858 |
| `date` boxes | 1,244 | 68 | 16,674 |
| `exp` boxes | 0 | 665 | 0 |
| `due` / `prod` / `code` | 895 / 100 / 541 | 460 / 37 / 229 | 5,397 / 2,185 / 5,518 |

The photo counts match the paper (1,102 / 665, "about 12,000" synthetic).

**Problems found, and what is done about them**

- 3 test photos are stored sideways with an EXIF rotation flag. Their JSON
  size and boxes are in the upright frame, so `load_upright` puts them in
  the right place (checked by eye).
- 1 test `due` box lies wholly outside the image. Clipping leaves it
  empty, so the YOLO conversion drops it.
- 99 synthetic photos have no date at all. They stay as negative examples.
- 10 test expiries have no full date to score against: 9 print no year
  (e.g. `04.05`), and 1 has `30` as its month. They're left out of reading
  accuracy and counted. Another 34 print a month and year only.

### Answers from the contact sheets

Sheets of 50 photos per split, plus zoomed date crops, made with
`scripts/contact_sheet.py --seed 0`.

1. **Do the boxes line up?** Yes, on all three splits, including the 3
   EXIF-rotated test photos.
2. **What does a `date` box cover?** The date value only: digits or month
   name, with the separators between them. It leaves out the "EXP" / "Best
   before" / 까지 words (those are `due` boxes) and a time or lot code
   printed right after the date. The boxes are tight, with a pixel or two
   of margin. **The store boxes follow this same rule.**
3. **Which date is the expiry?** In the test split, the `exp` box, one per
   photo. In train and synth, every date is just `date`; its role can only
   be guessed from a nearby `due` or `prod` mark. The synthetic photos pair
   PROD and EXP values at random (some show an expiry before the production
   date), so synth teaches *where* dates are, not *which* is the expiry.
4. **Print styles.** Dot-matrix is common in real and synthetic photos.
   Curved surfaces (caps, can lids) appear in both, and synth adds tilted
   text. **Sideways dates are almost absent:** 0% of real date boxes and
   0.5% of synthetic ones are taller than wide. The M2 generator has to
   supply them.

### Date formats in the test split

The `exp` transcriptions, with digits as `9` and letters as `A`. There are
32 distinct shapes; the most common:

| Shape | Photos | Share |
|---|---:|---:|
| `9999.99.99` | 339 | 51% |
| `99.99.9999` | 60 | 9% |
| `99/99/9999` | 47 | 7% |
| `99.99.99` | 40 | 6% |
| `9999/99/99` | 28 | 4% |
| `99 99 9999` | 20 | 3% |
| `99/99/99` | 18 | 3% |
| `99/9999` | 16 | 2% |
| any shape with the month as a word | 29 | 4% |

Half the test set is Korean `YYYY.MM.DD`. The Indian store formats
(`April 2031`, `20260518`, `MM/YYYY-MM/YYYY` ranges) are rare or missing,
which is one more thing for M2 to cover.

### Box sizes

Full table: [`results/m0_expdate_stats.md`](../results/m0_expdate_stats.md).

- **Real dates are small.** The median date box covers about 0.45% of the
  photo. Resized to 640 px on the long side, its short side is about 16 px
  (p10: 12 px); at 1280 px it's about 31 px. Almost none drop under 8 px at 640,
  so YOLO's default 640 input is workable, and 1280 is worth comparing in
  M1 on the validation split.
- **Synthetic dates are bigger and easier:** median 1.3% of the photo,
  27 px at 640, on smaller photos (719 px long side vs ~1,200 real).
  That's a reason to keep the real train split in M1 and to measure what
  synth adds rather than assume it helps.
- Real photos usually hold one date (87 to 90%); synthetic ones hold two
  dates 41% of the time.

### Duplicates across the splits

Checked on 2026-10-20 in
[`experiments/expdate-near-duplicates`](../experiments/expdate-near-duplicates/README.md).

- **Test is clean:** no test photo is a copy of a train, validation or
  synthetic photo, so M1's test result stands.
- **One validation photo is reused in Products-Synth:** `img_00277.jpg`
  (a tin) is the background of 14 synthetic training photos, with its
  real date erased and fake dates printed on. That makes validation
  optimistic by at most one photo of 165, which changes none of M1's
  choices. Leave those 14 out of training from M2 on.
- **Products-Synth is built partly on real photos:** at least 8
  Products-Real *train* photos are also synthetic backgrounds. That's
  harmless (both are training data), but synthetic and real aren't fully
  separate.
- **Same product, separate photo:** 2 clear validation / train pairs
  (and 1 possible) show different packs of the same product. That's
  normal for photos taken in shops, and they're left in.

---

## Generated data (M2)

Photos made for M2 by drawing dates onto ExpDate photos
(`docs/m2-synthetic-generator.md`, "The generator"; code in
`src/bestbefore/synth/`). They are built on ExpDate's photos, so they
inherit its licence (CC BY 4.0: cite the paper, credit KIST), and like
ExpDate they are never committed. Nothing in them was designed from a
store photo.

### What is made

| Set | Photos | Seed | Backgrounds | Fonts | Used for |
|---|---:|---:|---|---|---|
| `gen` (list `gen50k`) | 50,000 | 1 | ExpDate training photos, half real (930), half synthetic (7,255), at least 640 px, never a validation or test photo nor the 14 copies | the 16 training families | training (`gen`, `gen-rot`, `gen-rot-1280`) |
| `gen10k` | 10,000 | | `gen`'s photos 0 to 9,999 (a list, not new photos) | | training (`gen10k`, listed 5 times) |
| `synth-val` | 2,100 | 2 | the validation photos (164 of 165: one is under 640 px) | the 4 held-out families | validation suite only |
| `gen-sample` | 500 | 3 | as `gen` | as `gen` | looking at (S5), never training |

Each generated photo is one background with 1 block (60%) or 2 (40%) of
printed text drawn on it, through one camera: dates in 39 formats and 2
range formats (a month word on 45% of them, full names and ordinals
included), the words and codes around them, and distractors with numbers
that are not dates. Every share and range is in `GenSpec`
(`synth/spec.py`); its SHA-256 goes into every manifest and training
record.

**synth-val** has 7 strata of 300 photos, each one block of one line on a
validation photo, its date the **target**, loaded as `exp` (so the
evaluator's expiry recall is the target recall), its stratum as the
photo's tag. `clean` is upright, numeric, normal size, solid or label
print, flat; `sideways`, `month-word`, `small` (on validation photos of at
least 1,000 px), `dot-matrix` and `curved` each change one thing from it;
`mixed` is drawn with the training spec.

### What was made (job `72995`, 2026-10-23)

Checked by `generate.py --check` on the cluster: every image opens whole,
every label file equals the one rebuilt from its sidecar line, and every
share the spec sets is within 3 points at 50,000 photos (in fact within
0.73), or within 3 standard errors in synth-val's 300-photo strata, with
the shares a stratum fixes exact.

| | `gen` | `synth-val` |
|---|---:|---:|
| photos | 50,000 | 2,100 |
| new dates | 109,628 (2.19 a photo) | 2,100 targets |
| boxes (with the backgrounds' own): date / due / prod / code | 174,042 / 92,530 / 48,574 / 38,159 | 4,494 / 3,011 / 223 / 1,001 |
| blocks placed / skipped (no free place, too faint) | 68,480 / 1,219, 344 | 2,100 / 4, 0 |
| JPEGs | 9.24 GB (185 kB each) | 0.39 GB |
| sidecar | 82 MB | 2.4 MB |
| spec SHA-256 | `3fc14178…` (`GenSpec`) | `ba77f1bd…` (the 7 strata) |

The whole run took 6 minutes on 24 CPUs (150 ms a photo per CPU). The
cluster (x86-64) and the laptop (arm64) draw the same photos up to the
last bits of floating-point work: 85% of training photos are
byte-identical, almost all the rest differ by a grey level or two in a
few pixels, and 0.17% place a block differently. Every box comes from the
machine that drew it, so the labels are right on each.

### Labels and files

- **Boxes** come from the ink drawn, after every bend and turn: the tight
  upright rectangle round each token's ink. Classes as ExpDate's YOLO
  labels: `0 date`, `1 due`, `2 prod`, `3 code`; the background's own boxes
  follow the generated ones. A date box covers the date with its
  separators and ordinal, not the `EXP` word (`due`); a range is two dates.
- **The sidecar**, one JSON line per photo in photo order, records
  everything drawn: the background, each block's text, style, font, size,
  geometry, ink and place, each token's text, class and box, and for
  dates their role (`expiry`, `mfg`, or synth-val's `target`), format and
  year, month and day (M3 can reuse them).

```
$GENERATED_DIR/m2/
├── gen/images/{gen50k,synth_val,gen_sample}/NNNNNN.jpg
├── gen/labels/{gen50k,synth_val,gen_sample}/NNNNNN.txt
├── gen/sidecars/{gen50k,synth_val,gen_sample}.jsonl     one line per photo
├── gen/sidecars/tasks/NN.jsonl, NN.json                 each task's lines and record
├── lists/{gen50k,gen10k,synth_val}.txt                  absolute paths
└── manifests/{gen,synth-val}.json                       complete: true once checked
```

### Fonts

20 families from the Google Fonts repository at commit `e7db6ca`, all
under the SIL Open Font License 1.1 (Roboto Condensed's own name table
still says Apache 2.0; either allows this use). 24 files, 15.6 MB, each
checked against its SHA-256 in `synth/fonts.py`; fetched by
`scripts/fetch_fonts.py` into `$GENERATED_DIR/fonts/`, never committed.

| Split | Families |
|---|---|
| training (16) | Roboto, Open Sans, Lato, Montserrat, Source Sans 3, Noto Sans, Work Sans, Oswald, Roboto Condensed, Barlow Condensed, Noto Serif, Merriweather, Roboto Mono, Source Code Pro, Share Tech Mono, VT323 |
| synth-val only (4) | IBM Plex Sans, Archivo Narrow, PT Serif, IBM Plex Mono |

### Reproduce

```bash
python scripts/fetch_fonts.py                          # on the cluster, on the login node
sbatch slurm/m2_generate.sbatch                        # or, anywhere:
python scripts/generate.py --tasks 0-50 --workers 24
python scripts/generate.py --check --workers 24
```

Photo `i` comes from `numpy.random.default_rng([seed, i])` alone, so any
split into tasks or processes writes the same files.
