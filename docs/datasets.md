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

---

## Generated data (M2b): gen-v2

M2's generator with its second version's features on
(`docs/m2b-store-and-size.md`, "The generator, v2", S7 to S9), drawn onto
ExpDate's training photos as M2's were, so the same licence holds (CC BY
4.0: cite the paper, credit KIST) and nothing is committed. Every new
feature draws its random numbers only when it is on: M2's spec keeps its
hash (`3fc14178…`) and M2's photos come out byte for byte as before
(tests). gen-v2's new layouts come from store set v1's failures, seen in
the store gallery, and its sizes from v1's training part only; nothing
comes from store set v2, the test set.

### What gen-v2 adds to M2's generator

| Change | Rule | Share |
|---|---|---|
| upside-down blocks | 180° ±8° | 10% of blocks (upright 45%, tilted 15%, sideways 30%) |
| a keyword glued to its date | `EXP.FEB 2027`: no space after a word ending in `:` or `.` | 30% of the words printed before a date |
| two dates on one line | `MFG:FEB 2025 \| IMPORT: MAR 2025`, separated by ` \| `, two spaces, ` / ` or `, ` | kind line-pair, 10% |
| dense panels | a pair plus 3 to 5 more lines at 1.0 to 1.2 em | kind dense-panel, 10% |
| import dates | `IMPORT:`, `Imp.`, `Date of Import:` and the like, boxed `prod` | half of line-pairs and dense panels |
| new distractors | 13-digit barcode numbers, unit prices (`USP Rs. 1.50/g`) | 40% of distractor lines |
| nozzle rows | 1 or 2 whole rows of dots left out, the same in every line | 15% of dot-matrix blocks |
| laser window | light strokes on a dark printed rectangle, slightly soft | style window, 10% |
| worn patch | ink faded by 60 to 95% over 1 to 3 characters | 10% of inked blocks |
| sizes like the store's | below | every untilted block |

**Sizes.** M2's generated dates came out about 1.7 times bigger than the
store's (median box 3.06% of the photo's long side, against 1.73%).
gen-v2 draws each block so its first date's box, measured as the store
measures it (the box's short side over the photo's long side), follows
the deciles of v1-train's 169 dates (store sets v1's and v3's training
parts; aggregate numbers, in `V2Spec`), clipped to M2's 0.6 to 4.5%: drawn
once, measured, and drawn again to its target from the same draws, up to
3 tries. **Only untilted blocks** (upright, sideways and upside-down) are
sized that way. **A tilted block is sized by its digits** (its target
over 1.3, the upright ratio) and drawn once: the store has no tilted date
(none of v1-train's 169 is more than about 8° off level or vertical), and
an upright box round tilted text measures the tilt, so drawing that box to
the store's size had shrunk tilted digits to a third (changed 2026-11-06,
S8). So gen-v2's sizes follow the store's on untilted dates, and are
compared with the store's over those only:

| Dates | n | Under 1.5% | 1.5–2.5% | 2.5% and up | Median |
|---|---:|---:|---:|---:|---:|
| v1-train (the target) | 169 | 35.5% | 42.0% | 22.5% | 1.736% |
| gen-v2, untilted | 411,587 | 34.27% | 44.12% | 21.61% | 1.748% |
| gen-v2, every date (tilted ones run big by design) | 483,566 | 29.73% | 39.88% | 30.39% | 1.899% |

### What was made (job `73944`, 2026-11-12)

| | gen-v2 | synth-val-v2 | gen-v2-sample |
|---|---:|---:|---:|
| photos | 200,000 | 2,400 (8 strata of 300) | 500 |
| set seed | 4 | 5 | 6 |
| backgrounds | as M2's `gen` | the validation photos | as M2's `gen` |
| fonts | the 16 training families | the 4 held-out families | the 16 training families |
| used for | training (M2b step 2's replay) | the M2b validation suite | looking at (S8), never training |
| new dates | 483,566 (2.42 a photo) | 2,400 targets | 1,183 |
| blocks placed | 276,849 | 2,400 | 687 |
| blocks skipped: no free place, too faint | 1,604, 1,855 | 240, 3 | 3, 6 |
| boxes, with the backgrounds' own: date / due / prod / code | 741,826 / 366,932 / 250,338 / 186,858 | | 1,821 / 918 / 602 / 507 |
| JPEGs | 37.30 GB (186.5 kB each) | 0.43 GB | |
| spec SHA-256 | `2c8b24d5…` (`GEN_V2`) | `2295a4bd…` (the strata) | `GEN_V2` |

**synth-val-v2's strata** each take M2's `clean` stratum (`nozzle-rows`:
M2's `dot-matrix`), drawn by gen-v2's code with every new feature off and
M2's sizes kept, and change one thing: `upside-down`, `glued`,
`line-pair`, `dense-panel`, `window`, `embossed`, `worn` and
`nozzle-rows`. Each photo has one target date (in a line-pair or a dense
panel, one of its dates, each equally likely), loaded as `exp`, as M2's
synth-val.

**The 180° sets** (`derive_sets.py --only rot180`): ExpDate's validation
photos turned 180° (`real-val-rot180`, 165 photos, 187 dates) for the
suite, and its training photos turned 180° for step 2's replay
(`real_rot180` 937 photos, 1,057 dates; `synth_rot180` 11,844 photos,
16,652 dates). Every box is turned exactly (worst error 0.0 px).

The check (`generate.py --milestone m2b --check`) passed with no problem:
every image opens whole, every label file equals the one rebuilt from its
sidecar line, every share is within 0.6 points of the spec at 200,000
photos, and the untilted sizes within the store's bands as above. The
whole job took 26 min 41 s on 24 CPUs (170.6 ms a photo per worker, a
worker peaking at 1.8 GB). The cluster and the laptop draw the same
photos up to the last bits of floating-point work: about 90% of the
photos are byte-identical, nearly all the rest differ by a few grey
levels, and no block's size differs.

```
$GENERATED_DIR/m2b/
├── gen/images/{gen_v2,synth_val_v2,gen_v2_sample}/NNNNNN.jpg
├── gen/labels/{gen_v2,synth_val_v2,gen_v2_sample}/NNNNNN.txt
├── gen/sidecars/                one JSON line per photo, and each task's record
├── derived/{images,labels}/     the 180° sets
├── lists/                       gen_v2, synth_val_v2, gen_v2_sample, real_val_rot180, real_rot180, synth_rot180
└── manifests/{gen-v2,synth-val-v2,gen-v2-sample,derived}.json
```

```bash
python scripts/fetch_fonts.py                          # on the cluster, on the login node
sbatch slurm/m2b_generate.sbatch                       # or, anywhere:
python scripts/generate.py --set gen-v2-sample --count 500 --workers 24
python scripts/generate.py --milestone m2b --tasks 0-200 --workers 24
python scripts/generate.py --milestone m2b --check --workers 24
python scripts/derive_sets.py --only rot180 --workers 24
```

---

## Store photos for training (M2b)

Store sets v1 and v3, exported for training by
`scripts/export_store_yolo.py` into `$STORE_TRAIN_DIR` (on the laptop
`data/store/train/`, on the cluster `~/bestbefore-data/store/`, folders
700 and files 600). They belong to the retailer: never committed, never
published, and only counts appear here. Store set v2, the test set, never
leaves the laptop: the export refuses a v2 photo, and refuses to run until
v2's labels are frozen.

- **Split by pack design** (`store_train.split_v1`, seed 0: about 30% of
  each tag's designs to validation, so a design's photos fall on one side; a
  v3 design already in v1 keeps its side): **80 photos train** (70 of v1,
  10 of v3) and **34 validate** (29 and 5, `v1-val`); 31 designs train and
  16 validate, none on both sides.
- **Each photo** upright (its EXIF turn applied), in sRGB, shrunk with
  PIL's BOX filter to at most 1,600 px, saved at JPEG quality 95 without
  metadata (no EXIF, GPS position or colour profile), and numbered by set
  (`v1-0001`, `v3-0001`, ...). The names map back only on the laptop
  (`data/store/train_names.json`); designs are numbers (`d01` to `d47`).
- **Training photos in 4 turns** (0°, 90°, 180°, 270°: 320 images, every
  box turned with them); validation photos as they are (34).
- **Labels**: every expiry, manufacture and other date as `date`, the
  keyword boxes as `due`, `prod` and `code`. Training holds 169 dates (64
  expiries), 92 `due`, 137 `prod` and 82 `code` boxes; validation 63 dates
  (29 expiries), 31, 34 and 25. 4 dates that v1's frozen labels miss were
  boxed in M2b's keyword review and are exported as dates; one box under
  2 px across is left out.
- **A manifest** holds the counts, each photo's part, design number, tag
  and box labels, and every file's SHA-256 (710 files, 160 MB);
  `export_store_yolo.py --check` checks a copy against it.

---

## M3's pretrained models

Fetched on the cluster's login node on 2026-11-12 (M3 S1) by
`scripts/fetch_m3_models.py`, into `$CHECKPOINT_DIR/pretrained/` beside the
YOLO weights (`~/bestbefore-data/checkpoints/pretrained/` there); never
committed. Each Hugging Face repo is pinned to a commit; each weights file
to its SHA-256, which matched Hugging Face's own LFS hash. 43 files,
7.1 GB. `python scripts/fetch_m3_models.py --check` hashes them again.

### Licences

| Model | Licence | Where it says so |
|---|---|---|
| PP-OCRv6 small recogniser and detector, PP-LCNet x1.0 textline orientation (Paddle, ONNX and safetensors copies) | Apache-2.0 | each repo's model card (`license: apache-2.0`) |
| `PP-OCRv6_small_rec_pretrained.pdparams` | Apache-2.0, as PaddleOCR | the file states none; it is the training weights PaddleX 3.7.2's own `PP-OCRv6_small_rec.yaml` links, from PaddleOCR (Apache-2.0) |
| Qwen3-VL-2B-Instruct | Apache-2.0 | the model card |
| Qwen3-VL-2B-Instruct-GGUF | Apache-2.0 | the model card; the repo holds no LICENSE file |

All allow fine-tuning, and publishing results and fine-tuned weights, with
the licence and a notice of changes kept.

### Files

| Repo (revision) | File | Bytes | SHA-256 |
|---|---|---:|---|
| `PaddlePaddle/PP-OCRv6_small_rec` (`bd619643`) | `inference.pdiparams` | 21,074,618 | `406e1e689c9a7fbb04178007a3fe10cf852afe7bf8bb3bc6dbb9d532b13bd907` |
| | `inference.json` | 208,004 | `f0bf53c853937a917affdd74467472167727f8ab0f0f7bded01c4a16c27e46e6` |
| | `inference.yml` (holds the dictionary) | 150,579 | `ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1` |
| `PaddlePaddle/PP-OCRv6_small_rec_onnx` (`b8f84f0b`) | `inference.onnx` | 21,159,378 | `5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634` |
| | `inference.json`, `inference.yml` | | the Paddle repo's, byte for byte |
| `PaddlePaddle/PP-OCRv6_small_rec_safetensors` (`fe049fb1`) | `model.safetensors` | 21,204,736 | `f65a332afe5aa663f0b9d5706f4ae8457b5b4058a842d5c1eb22df505c27d642` |
| | `config.json` | 1,083 | `8693fd8485e8543e13d0f6dde3891a3f4af9a47fc4e3a391fbed71f8015c899e` |
| | `preprocessor_config.json` | 206,177 | `2b24fa36f548893f26a931cf44f1a1a6b2b14ed7e2f8f8e28e6848801a8278db` |
| | `inference.yml` | | the Paddle repo's |
| `PaddlePaddle/PP-OCRv6_small_det` (`106c9759`) | `inference.pdiparams` | 9,817,888 | `5043d4ccc8d63402ccea8feefcee4db57077431a873e78d2191836a178a492da` |
| | `inference.json` | 229,307 | `89240f689a4a77aad75ef55a8df0a15c8e1d4980a327d17e58f24bbadde5aeab` |
| | `inference.yml` | 885 | `193f435274bf9f0b5f71a929bbfbcf148282df7e633b34e7c373e8f44741b516` |
| `PaddlePaddle/PP-LCNet_x1_0_textline_ori` (`cd237a44`) | `inference.pdiparams` | 6,743,918 | `0de2bcf996cf553e2b848dd7b1769dafffc6917b1ccdf55c1d8efe7909fbf743` |
| | `inference.json` | 104,399 | `b81929eeaff8e52db0fafb49c9fbfc5bc0572f81641fe0179cc52096323fb4d4` |
| | `inference.yml` | 735 | `8d5120d0e1a30a9df7ed46aa9119da3796ed066777089d1c1d705f132d5e90f9` |
| | `config.json` | 2,487 | `c1a2576f9713adabdc4a58d9d654fcc9ea93b6efd0e2d3581d5aab409a33f14f` |
| Baidu BOS, `paddlex/official_pretrained_model/` (Last-Modified 2026-06-10) | `PP-OCRv6_small_rec_pretrained.pdparams` | 124,912,348 | `25c9bd54b0e5900916e8bb6ada938abeffb1eac1baedac0ca54a45b1c9310825` |
| `Qwen/Qwen3-VL-2B-Instruct` (`89644892`) | `model.safetensors` (bf16) | 4,255,140,312 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| | `tokenizer.json` | 7,032,403 | `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7` |
| | `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| | `merges.txt` | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| | `tokenizer_config.json` | 10,868 | `c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399` |
| | `chat_template.json` | 5,502 | `6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4` |
| | `config.json` | 1,505 | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| | `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| | `video_preprocessor_config.json` | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| | `generation_config.json` | 269 | `1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4` |
| `Qwen/Qwen3-VL-2B-Instruct-GGUF` (`52d6c8ff`) | `Qwen3VL-2B-Instruct-Q8_0.gguf` | 1,834,427,424 | `1e8db19207c8ce0733ddd78c2eff8a9e22c27c82f4443df94c25792ed8fe04f2` |
| | `mmproj-Qwen3VL-2B-Instruct-F16.gguf` | 819,394,848 | `c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5` |

Each repo's `README.md` came too, and `.gitattributes` from all but the
GGUF repo (hashed by the script, not listed). The full revisions are in
the script.

### Notes

- **PaddleOCR finds its models offline** when `PADDLE_PDX_CACHE_HOME` is
  `$CHECKPOINT_DIR/pretrained/paddlex` and
  `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` (PaddleX 3.7.2's
  `utils/flags.py`); otherwise it fetches `PaddlePaddle/<model>` from
  Hugging Face on first use.
- **Not fetched:** PaddleOCR's default page models (`PP-LCNet_x1_0_doc_ori`,
  `UVDoc`; for scanned pages, switched off on crops), the medium and tiny
  PP-OCRv6 models, the GGUF's other quantisations (F16, Q4_K_M, the Q8_0
  projector), and the detector's ONNX and safetensors copies.
- **PaddleOCR 3.7.2's OCR pipeline** defaults to `PP-OCRv6_medium_det` and
  `_rec`; its detector enlarges an image only when its short side is under
  64 px (`limit_side_len: 64`, `limit_type: min`), where RapidOCR 1.2.3's
  does it under 736 px.
