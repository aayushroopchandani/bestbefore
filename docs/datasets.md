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
