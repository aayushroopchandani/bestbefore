# BestBefore

**Reading expiry dates on product packaging, and knowing before the photo
is taken whether the date will be readable.**

BestBefore reads expiry dates from photos of real product packaging, the
dot-matrix print on curved caps, sideways stamps and tiny lines that
off-the-shelf OCR misses. It grows out of Scanify, a stock app built for a
retailer whose inventory system has no expiry dates, and replaces that
app's hand-written rules with trained models.

The project covers the full pipeline, from data to deployment:

- **Date detector:** trained on the public ExpDate dataset, then on a custom
  synthetic generator that draws what public data misses, across 200,000+
  generated photos. Fine-tuning on real photos of Indian products is under
  way.
- **Date reader:** reads each date as day / month / year and tells
  the expiry from the manufacture date.
- **Readability checker:** small enough to run on a phone's live
  camera preview, warning staff before they take a photo that can't be read.

Models train on a university GPU cluster. Every result is measured against
baselines on real store photos held out from all training, and broken down
by failure type, not reported as one headline number.

![Twelve expiry dates cropped from product photos: dot-matrix print, a curved cap, faint ink, an out-of-focus date, an expiry printed above a production date, a month written as a word, a curved jar, a date under a lot code, low contrast, and three clean prints](docs/assets/expdate_dates.jpg)

<sub>Expiry dates from the ExpDate test set, each hard to read in a
different way. Photos: ExpDate (KIST, CC BY 4.0), cropped.</sub>

### What sets it apart

- **Rules before results.** Each milestone's plan writes its decision
  rules as code, and its predictions in plain words, before any GPU time
  is spent. Validation photos make every choice, the test sets are scored
  once, and predictions that failed are reported as failed.
- **Labels from the ink.** The synthetic generator knows every glyph it
  draws, so each box is the tight rectangle round that text's ink after
  bending, turning and shrinking. No box is estimated.
- **A fair ablation.** M2's six arms each change one thing, the training
  data or the input size, and each trains on the same budget of 2.56
  million images, under one recipe, with three seeds.
- **A GPU queue that runs itself.** Slurm jobs are queued and held on day
  one, so they keep their place in a queue that can take days while the
  code is still being written. Once running, they share one task list
  through atomic claims, stop stalled runs and submit their own
  replacements.
- **Every number traceable.** Each run records its settings, git commit,
  seed and library versions, and each training run the hashes of the data
  it trained on. Each generated photo comes from its own seeded random
  generator, and each report is written by a script from the records
  alone.
- **Leaks hunted down.** A perceptual-hash search over all 13,625 ExpDate
  photos, checked by eye, found one validation photo reused as the
  background of 14 synthetic training photos. They leave training from M2
  on.
- **Private photos stay private.** The retailer's photos are never
  committed or published, and the store test set never leaves the
  laptop. It was frozen by its hash before any training on store photos.

---

## Where this started

During an internship at a pet-supplies retailer, I built **Scanify**, a
stock-taking app. The retailer's inventory system has no expiry dates, so
staff photograph each pack and the app reads two things from the photo:
the barcode (`zxing-cpp`) and the printed expiry date (`RapidOCR`
followed by a hand-written, rule-based date parser). Later this model will be used.

Measured on the store photos, the rules got a long way, then hit a wall:

| | Photos |
|---|---:|
| Expiry read right | 32 of 44 |
| Expiry missed | 12 |
| Expiry read wrong | 0 |

Every miss came from **reading** the print, not from parsing the text:
the parser extracted every date the OCR returned. The misses were
dot-matrix print on curved caps and jar bottoms, dates stamped sideways,
small print, and a few large, clear "Best Before" lines the OCR skipped
entirely.

A blur score (variance of the Laplacian) was tried as an early warning
and **did not** predict failures: photos that were read and photos that
were missed scored the same. It measured how much fine detail was in the
frame, not whether the date could be read.

BestBefore replaces the hand-written rules with trained models, starting
exactly where the rules ran out.

---

## How it fits together

### In the store

The checker runs on every frame of the camera preview, so staff are told
to move closer or turn the pack **before** taking the photo. Once a photo
is taken, the detector finds the dates and the reader reads them.

```mermaid
flowchart TD
    CAM["Phone camera<br/>live preview"] --> CHK{"Readability<br/>checker"}
    CHK -- "unlikely to read" --> WARN["Warn: move closer,<br/>turn the pack"]
    WARN --> CAM
    CHK -- "looks readable" --> SHOT["Photo taken"]
    SHOT --> DET["Date detector<br/>finds every date"]
    SHOT --> BAR["Barcode decoder"]
    DET --> READ["Date reader<br/>expiry, mfg date and<br/>format, as JSON"]
    READ --> INV["Inventory record"]
    BAR --> INV
    class DET built
    class CHK,READ planned
    classDef built fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef planned stroke-dasharray:5 4
```

<sub>Green: built and measured. Dashed: still to build (M3 to M5).</sub>

### How the models are built

Each model starts from public data, adds what this project makes or
collects, and hands its output to the next one.

```mermaid
flowchart TD
    subgraph pub["Public: ExpDate"]
        PROD[("Products<br/>1,767 real +<br/>11,858 synthetic photos")]
        CROPS[("Date and Components<br/>578,000 synthetic crops")]
    end
    subgraph own["Made or collected here"]
        GEN["Synthetic generator<br/>M2: 52,100 photos<br/>M2b: gen-v2"]
        STORE[("Store sets v1 and v3<br/>private, split<br/>by pack design")]
    end
    PROD --> DET["Date detector, YOLO11<br/>M1: ExpDate<br/>M2: + generated dates<br/>at 1,280 px<br/>M2b: + store photos,<br/>larger sizes tried"]
    GEN --> DET
    STORE -- "M2b" --> DET
    DET --> READ["Date reader, M3<br/>day, month, year;<br/>expiry told from mfg"]
    CROPS --> READ
    GEN -.-> READ
    READ --> DMG["Damaged copies, M4<br/>each labelled: does<br/>the reader still<br/>read it right?"]
    DMG --> CHK["Readability checker, M4"]
    CHK --> PHONE["ONNX, quantised, M5<br/>runs in a phone browser"]
    class DET built
    class READ,DMG,CHK,PHONE planned
    class STORE private
    classDef built fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef planned stroke-dasharray:5 4
    classDef private fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
```

The checker learns from the reader itself. Each damaged copy is labelled
by whether the reader still gets the date right, so the checker learns
what actually makes reading fail, rather than a proxy like blur.

### The synthetic generator

ExpDate's dates are upright, mostly numeric and fairly large; the
store's are often sideways, spelt out (`December 2026`) and small. So the
generator draws those kinds of date onto real pack photos, one block of
text at a time:

```mermaid
flowchart TD
    subgraph blocks["Each of 1 or 2 blocks"]
        TXT["Text<br/>41 date formats,<br/>keywords, lot codes,<br/>prices as decoys"]
        STY["Print style<br/>solid, dot-matrix,<br/>label or embossed"]
        BIG["Render at 4×<br/>one ink mask per token"]
        WARP["Bend and turn<br/>tilt, sideways,<br/>can or cap"]
        SHR["Shrink to print size<br/>soft, as a camera<br/>sees it"]
        INK["Place and blend<br/>on a plain free spot;<br/>too faint: inked again"]
        TXT --> STY --> BIG --> WARP --> SHR --> INK
    end
    BG[("ExpDate training photo<br/>a real pack surface")] --> INK
    INK --> CAM["Camera effects<br/>on the whole photo<br/>defocus, motion, noise,<br/>glare, dim light, JPEG"]
    CAM --> JPG["the photo"]
    CAM --> LBL["YOLO labels<br/>boxes from the ink"]
    CAM --> SIDE["a JSON line<br/>every choice made,<br/>each date's role<br/>and value"]
```

- **Dot-matrix from any font.** The renderer samples a font on a grid 7 or
  9 dots tall and draws a dot wherever the text covers at least 40% of a
  cell. At 5 rows the digits garbled, so 5 was dropped
  ([`experiments/m2-dot-rows`](experiments/m2-dot-rows/README.md)).
- **Dates among look-alikes.** Panels print the date in the same font as
  prices, weights and batch codes, so the model has to learn what a date
  says rather than how it stands out.
- **Fonts held out.** 16 font families draw the training photos; 4 others
  draw only the validation photos, so a score there can't come from
  memorised fonts.
- **Same photo, however it is split.** Photo `i` of a set is drawn from
  its own generator, seeded `[set_seed, i]`, and every share and range
  lives in one frozen spec whose SHA-256 goes into every manifest. One
  24-CPU job made M2's 52,100 photos in 328 seconds.

### Laptop, cluster and GitHub

Code is written and tested on a laptop, training runs on the university
cluster's H100s, and the retailer's photos stay where the rules allow. A
milestone moves between them like this:

```mermaid
sequenceDiagram
    participant L as Laptop
    participant G as GitHub
    participant C as Cluster (Slurm)
    L->>C: day one: GPU jobs submitted, held
    L->>G: plan, code and tests
    G->>C: pulled on the login node
    Note over C: CPU job: generate and derive the photo sets
    L->>C: M2b only: store training copies,<br/>renamed, metadata stripped
    Note over C: GPU jobs released: the chain<br/>trains, scores and decides
    C->>L: run records and weights
    Note over L: the store test set scored once, on the CPU
    L->>G: report, aggregate numbers only
```

Store photos never went to the cluster until M2b, and the test set (v2)
never does: it is scored on the laptop's CPU. The training copies of v1
and v3 go upright and renamed (`v1-0001.jpg`), in sRGB with every bit of
metadata dropped (20 of v1's photos carried a GPS position), into a
folder only the owner can read. The map back to the photos' names stays
on the laptop.

### The GPU work queue

The queue for the cluster's 4 H100s is first come, first served, and
waits of days are normal. So each milestone submits its GPU jobs **held**
on day one, before the code exists: a released job keeps its place in
line, and the job script reads the chain's code only when it starts.
Every GPU job then runs the same loop over one shared task list:

```mermaid
flowchart TD
    START(["GPU job starts"]) --> PICK{"Next task?"}
    PICK -- "ready, not done,<br/>not claimed, and twice<br/>its estimate fits in<br/>the time left" --> CLAIM["Claim it with os.mkdir<br/>atomic: of two jobs,<br/>one wins"]
    CLAIM --> RUN["Train or score it<br/>no progress for 2 hours:<br/>stopped; failed twice:<br/>left for a person"]
    RUN --> DEC["Write each decision<br/>whose inputs are now<br/>complete: loader,<br/>pilot gate, detector"]
    DEC --> PICK
    PICK -- "none" --> LEFT{"Tasks left?"}
    LEFT -- "yes" --> REPL(["Submit its own<br/>replacement, held<br/>if nothing is ready"])
    LEFT -- "no" --> FREE(["Exit, freeing the GPU"])
```

A claim whose job Slurm no longer lists is taken over, even one left by
the same job before a requeue after a node failure. M2's chain took its
three decisions on its own as their inputs arrived: the data loader (8
workers, within 2% of the fastest), the pilot gate (a short run had to box
half of the generated validation dates before about 30 GPU-hours went to
the generated arms; it boxed 97%) and the M2 detector.

### How a result is earned

```mermaid
flowchart LR
    PLAN["Plan<br/>decision rules as code,<br/>predictions in writing"] --> TRAIN["Train<br/>every run records<br/>commit, seed, data"]
    TRAIN --> CHOOSE["Choose<br/>on validation<br/>photos only"]
    CHOOSE --> TEST["Test once<br/>ExpDate test and the<br/>frozen store set"]
    TEST --> REPORT["Report<br/>by a script, from the<br/>records, by failure tag"]
```

Baselines A and B are scored on the same test photos, and every number in
a report is read from the run records, so a result can't drift from the
runs behind it.

---

## Measuring success

Success is a results table **broken down by failure type**, not one
headline number. Each store photo carries one tag: `dot-matrix`,
`sideways`, `curved`, `small-print`, `other-face` (the date is on another
side of the pack) or `clean`.

| Model | Metric |
|---|---|
| Date detector | recall and mAP@0.5; on store photos, was the expiry found (IoU of at least 0.5)? |
| Date reader | exact day, month and year (month and year when the pack prints no day). **Wrong reads are counted separately from no reads.** |
| Readability checker | AUROC for "will be read right", plus agreement with OCR accuracy on SmartDoc-QA and with sharpness ratings on SPAQ |

A wrong date is worse than no date. Reading `08/2027` as `08/2021` would
mark a whole batch as expired, while a missed date only means typing it in.

### Baselines

Every model has to beat both of these on the same photos:

- **A. RapidOCR off the shelf, plus a simple date regex.** Public and
  reproducible by anyone.
- **B. Scanify's reader**, the rule-based system above, called from its
  own repository.

---

## Roadmap

```mermaid
flowchart LR
    M0["M0<br/>Data and<br/>test set"] --> M1["M1<br/>Date<br/>detector"]
    M1 --> M2["M2<br/>Synthetic<br/>generator"]
    M2 --> M2b["M2b<br/>Store photos,<br/>bigger detector"]
    M2b --> M3["M3<br/>Date<br/>reader"]
    M3 --> M4["M4<br/>Readability<br/>checker"]
    M4 --> M5["M5<br/>On the<br/>phone"]
    class M0,M1,M2 done
    class M2b now
    class M3,M4,M5 planned
    classDef done fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef now fill:#fef3c7,stroke:#d97706,stroke-width:3px,color:#78350f
    classDef planned stroke-dasharray:5 4
```

| | Milestone | Done when | Status |
|---|---|---|---|
| M0 | Data and test set | baseline table on ExpDate and on the store photos, by tag | **done** 2026-10-21 |
| M1 | Date detector | recall and mAP@0.5 on ExpDate test; expiry found on store photos | **done** 2026-10-21 |
| M2 | Synthetic generator | an ablation showing what synthetic data adds on the store photos | **done** 2026-10-28 |
| M2b | Store photos and a bigger detector | the M2b detector scored once on store set v2, by tag, cause and seen or unseen pack design, next to M1 and M2 | **in progress** |
| M3 | Date reader | end-to-end date accuracy vs baselines A and B, by tag | next |
| M4 | Readability checker | AUROC on held-out copies; SmartDoc-QA and SPAQ agreement; store photos | planned |
| M5 | On the phone | size, latency per frame, accuracy kept after quantising | planned |

### Progress so far

ExpDate is downloaded, checked, measured and converted to YOLO format.
Looking at the data turned up three things that shape the next
milestones:

- **The dates are small.** In real photos the median date box covers
  0.45% of the frame, about 16 px tall at YOLO's default 640 px input.
  M1 compares 640 and 1280 px on a validation split.
- **ExpDate has almost no sideways dates** (0% of real date boxes), and
  half its test set uses the Korean `YYYY.MM.DD` format. The M2 generator
  has to supply sideways dates and the Indian formats seen in stores
  (`April 2031`, `20260518`, `MM/YYYY-MM/YYYY` ranges).
- **The synthetic photos pair PROD and EXP dates at random,** so they
  teach where dates are, not which one is the expiry.

![Three handheld product photos with the expiry date boxed, each date zoomed in below; the dates cover 0.15%, 0.17% and 0.25% of the photo](docs/assets/expdate_small_dates.jpg)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/date_box_sizes_dark.png">
  <img alt="Histogram of date box heights at 640 px input. Real photos peak sharply around 14 to 16 px with a median of 16 px; synthetic photos spread much wider with a median of 27 px. Almost no boxes fall under 8 px." src="docs/assets/date_box_sizes_light.png">
</picture>

<sub>Synthetic dates are bigger, and so easier, than real ones. That is
why M1 keeps the real photos in training and measures what the
synthetic ones add. Photos: ExpDate (KIST, CC BY 4.0), boxes drawn and
cropped.</sub>

Details: [`docs/datasets.md`](docs/datasets.md).

### M0 result: the baselines to beat

Both readers were scored on the frozen store test set (87 photos with an
expiry in frame) and on ExpDate's test split, with one rule: **right** is
the exact date (month and year when the pack prints no day), **wrong** is
any other date, **none** is no date.

| Store photos | n | A right | A wrong | A none | B right | B wrong | B none |
|---|---:|---:|---:|---:|---:|---:|---:|
| sideways | 16 | 6 | 5 | 5 | 9 | 0 | 7 |
| dot-matrix | 21 | 7 | 0 | 14 | 7 | 0 | 14 |
| small-print | 17 | 14 | 2 | 1 | 16 | 0 | 1 |
| clean | 33 | 22 | 5 | 6 | 28 | 2 | 3 |
| **all** | **87** | **49** | **12** | **26** | **60** | **2** | **25** |
| **ExpDate test** | **655** | **502** | **28** | **125** | **57** | **6** | **592** |

A is RapidOCR plus a date regex, taking the latest date found. B is
Scanify's reader; this harness reproduces Scanify's own measurement on
its 44 photos exactly (32 right, 0 wrong, 12 none). Full table and run
records: [`results/m0_baselines.md`](results/m0_baselines.md).

What it shows:

- **On the store photos, Scanify's rules pay off.** B reads 60 of 87
  and gets only 2 wrong (both a `19th December 2026`: its rules skip the
  ordinal and answer the month's last day). A reads 49 and gets 12
  wrong, every one an earlier date taken for the expiry: OCR missed the
  real one, or the pack prints only a mfg date and a shelf life.
- **On ExpDate, the order flips: A reads 502 of 655, B only 57.** B's
  OCR does see the dates: of the 448 photos A read right and B returned
  nothing for, 412 have the true date in B's own OCR text. But B's rules were written for Indian
  packs and expect an "EXP" label or an Indian format, so an unlabelled
  Korean `2021.08.03` gets "none". Hand-written rules fit the photos they
  were written for.
- **Dot-matrix is the wall for both:** 7 of 21, and the 14 others give
  no date at all, because the OCR finds no date text. That is a reading
  problem, which is what M3's trained reader is for.
- **The bar for M3:** beat B's 60 of 87 on the store photos without
  more wrong reads, and beat A on ExpDate.

### M1 result: a detector that finds ExpDate's dates, not yet the stores'

A small YOLO (`yolo11n`, 640 px), fine-tuned on ExpDate on the university
cluster's H100s. The model size, input size, training data and confidence
threshold were all chosen on validation photos; each test set was scored
once, with three training seeds.

| Expiry dates found | ExpDate test | Store photos |
|---|---:|---:|
| the M1 detector (seed 0) | 637 of 665 | 18 of 81 |
| mean of 3 seeds | 97.1% | 26.7% |

By store failure tag (mean of 3 seeds): dot-matrix 73%, clean 23%,
sideways 0 of 15, small print 0 of 15. The boxes it does keep are
nearly all real dates: precision 0.97 to 0.99 on ExpDate, 0.93 to 0.95 in
the stores.

The gap is the finding. ExpDate's dates are numeric, upright and fairly
large, and the detector learned exactly that:

- **Sideways dates:** none in ExpDate, none found.
- **Month written as a word** (`December 2026`): 6% of ExpDate's real
  dates and 48% of its synthetic ones, all three-letter abbreviations
  (`DEC`), against 26 of the 66 upright store expiries, most of them full
  names. The detector finds 3% of the store's.
- **Print size:** numeric store dates are found 87% of the time when
  printed at ExpDate's size, 12% when under 1.5% of the photo.

That sets M2's job: synthetic dates that are sideways, month-named and
small, and a larger input (or tiles) for small print, judged on the store
photos by tag. Full report:
[`results/m1_detector.md`](results/m1_detector.md); the analysis of the
misses: [`experiments/m1-store-misses`](experiments/m1-store-misses/README.md).

### M2 result: generated dates double the store expiries found, but not the sideways ones

The same small YOLO, retrained on six kinds of training data, each with
the same budget of 2.56 million training images and three seeds: M1's
data (`base`); plus ExpDate's photos turned 90° or 270° (`rot`); plus
50,000 photos with generated dates drawn onto ExpDate's training photos
(`gen`); both (`gen-rot`); both at a 1,280 px input (`gen-rot-1280`); and
10,000 generated photos instead of 50,000 (`gen10k`). The generator draws
dates in solid, dot-matrix, label and embossed print, in Indian formats
with the month as a word or a number, small, sideways, tilted and on
curved surfaces, then adds blur, noise, glare and uneven light. Every box
comes from the ink it drew. A rule written before any run chose the M2
detector on validation photos: `gen-rot-1280`, seed 0. ExpDate test and
the store photos were each scored once, afterwards.

```mermaid
flowchart TD
    subgraph pools["Training pools"]
        REAL[("ExpDate real<br/>937 photos")]
        SYN[("ExpDate synthetic<br/>11,844 photos")]
        TURN[("Both, turned<br/>90° or 270°")]
        GEN[("Generated<br/>50,000 photos")]
    end
    pools -- "every arm: the same<br/>2.56 million images<br/>budget, 3 seeds" --> arms
    subgraph arms["6 arms, 18 runs"]
        BASE["base<br/>M1's data<br/>0.560"]
        ROT["rot<br/>+ turned<br/>0.881"]
        GENA["gen<br/>+ generated<br/>0.817"]
        GENROT["gen-rot<br/>+ both<br/>0.956"]
        WIN["gen-rot-1280<br/>+ both, 1,280 px<br/>0.989"]
        G10["gen10k<br/>10,000 generated × 5<br/>0.800, side question"]
        BASE ~~~ GENROT
        ROT ~~~ WIN
        GENA ~~~ G10
    end
    arms --> SUITE["Validation suite<br/>guard: ExpDate<br/>validation<br/>hard: its dates turned,<br/>its dates halved, and<br/>generated dates in<br/>held-out fonts"]
    SUITE --> RULE{{"Rule fixed before any run<br/>no loss on the guard, then<br/>the best mean score on the<br/>hard sets; within 0.01,<br/>the cheapest"}}
    RULE --> PICKED["The M2 detector<br/>gen-rot-1280, seed 0"]
    PICKED --> ONCE["Scored once<br/>ExpDate test: 662 of 665<br/>store photos: 39 of 81<br/>M1's detector: 637 and 18"]
    class WIN,PICKED win
    classDef win fill:#fef3c7,stroke:#d97706,stroke-width:3px,color:#78350f
```

<sub>Each arm's number is its score: mean mAP@0.5 for dates over the three
hard validation sets (3 seeds). `gen10k` answers a side question, whether
more distinct generated photos help (they do, by 1.6 points), and was
never a candidate.</sub>

| Expiry dates | ExpDate test, found | Store photos, found | Store, boxed at any confidence | Store precision |
|---|---:|---:|---:|---:|
| the M1 detector (seed 0) | 637 of 665 | 18 of 81 | 43 | 0.950 |
| M1, mean of 3 seeds | 646 | 21.7 | 45 | 0.939 |
| the M2 detector (seed 0) | 662 of 665 | 39 of 81 | 69 | 0.902 |
| M2 (`gen-rot-1280`), mean of 3 seeds | 662.7 | 46 (39 to 53) | 70.3 | 0.910 |

By store failure tag (mean of 3 seeds), M1 then M2: sideways 0 then 2.7
of 15, dot-matrix 14.7 then 19 of 20, small print 0 then 5.3 of 15, clean
7 then 19 of 31.

Every arm on the store photos, next to M1:

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#3b82f6, #d97706"}}}}%%
xychart-beta
    title "Store expiry dates, of 81 (mean of 3 seeds)"
    x-axis ["M1", "base", "rot", "gen", "gen-rot", "gen10k", "gen-rot-1280"]
    y-axis "Expiry dates" 0 --> 81
    bar [21.7, 27, 26.3, 38.3, 33.7, 37, 46]
    line [45, 43, 45.7, 52.7, 55, 49, 70.3]
```

<sub>Bars: found at each run's own threshold. Line: boxed at any
confidence. The rule, which never saw a store photo, picked the arm that
finds the most. The gap between line and bars is dates the model sees but
doubts; M2b goes after it.</sub>

What it shows:

- **The generated dates are what helps.** With them (`gen` against
  `base`), clean store expiries go from 9.3 to 18.3 of 31, and full month
  names (`December 2026`) from 1 to 6.7 of 19. ExpDate's photos turned
  add nothing on the store photos (`rot` 26.3 of 81, `base` 27).
- **Small print moved only at 1,280 px:** 5.3 of 15 small-print expiries,
  against 0 to 0.7 for every 640 px arm.
- **Nothing was lost on ExpDate:** the M2 detector finds 662 of 665 test
  expiries (M1's: 637), and every arm keeps a validation mAP@0.5 for
  dates of 0.990 to 0.995.
- **The M2 detector is its arm's weakest seed on the store photos:** 39
  of 81, where its arm's seeds range from 39 to 53. The seed was fixed
  before any run, so it stays; the spread says the store set is too small
  to tell seeds apart.

Where it doesn't beat M1:

- **Sideways dates are almost unsolved:** 2.7 of 15 store expiries,
  though ExpDate's validation photos turned sideways are found almost
  every time (`rot`: 183.7 of 187). Neither turned ExpDate photos nor
  generated sideways dates look like the store's sideways dates to the
  model; why is the next thing to find out.
- **More false boxes:** precision 0.910 against M1's 0.939, and 0.11
  false boxes a photo against 0.03. That is the price of a threshold
  chosen for recall.
- **The threshold still cuts many store dates:** 70.3 expiries get a box
  at some confidence, 46 are kept.
- **Predictions written before any run:** turned photos lifting sideways
  dates "a lot" (`rot`) and generated plus turned ones lifting them "the
  most" (`gen-rot`) both failed on the store photos. Month words "up a
  lot" held on generated validation photos (296.3 of 300 for `gen`) but
  only partly on the store (6.7 of 19 full names). Small print "up" with
  the generated photos held only at 1,280 px. "No loss on ExpDate" held
  for every arm.

The store set is small (15 sideways and 15 small-print expiries), and
M1's store result chose what M2 aimed at, so these store numbers are a
somewhat optimistic estimate for a new store. Full report, with every
arm by tag and by cause:
[`results/m2_detector.md`](results/m2_detector.md); the plan and every
slice's record:
[`docs/m2-synthetic-generator.md`](docs/m2-synthetic-generator.md).

### M2b, in progress: the store photos, gen-v2 and a bigger detector

The M2 detector finds 39 of the store's 81 expiry dates. Of the other 42,
30 get a box below its threshold and 12 none at all. The store gallery
showed why: a threshold chosen on ExpDate, layouts the generator never
drew, generated dates about 1.7 times bigger than the store's, and a few
pack designs behind many of the photos. Since the retailer's owner
allowed training on the store photos (2026-10-28), M2b goes after all of
it at once: it fine-tunes on the store photos themselves, draws the
missing layouts, tries bigger models and picks the threshold on store
photos.

```mermaid
flowchart TD
    subgraph step1["Step 1: public data"]
        N1["n1280<br/>M2's winning arm,<br/>reused"]
        S1["s1280<br/>yolo11s,<br/>3 seeds"]
        M1P["m1280<br/>yolo11m,<br/>1-seed probe"]
    end
    subgraph step2["Step 2: store photos"]
        MIX["Each epoch:<br/>20% store photos,<br/>design-balanced,<br/>turned 4 ways;<br/>80% replay: ExpDate,<br/>turned copies, gen-v2"]
        RECIPES["Recipes on n:<br/>ft, or ft2 in<br/>two stages;<br/>side arm: ft-gen50k"]
    end
    CTRL["Control: no training<br/>M2's weights, with a<br/>threshold set on<br/>store photos; its<br/>seeds set the band"]
    step1 --> step2
    step2 --> RULES{{"Rules R1 to R3<br/>best on v1-val,<br/>nothing forgotten;<br/>within the noise<br/>band, the cheapest"}}
    CTRL --> RULES
    RULES --> ADD["Rule R4: add-ons kept<br/>only if they earn it<br/>3-seed ensemble,<br/>test-time rotation,<br/>1,536 px input"]
    ADD --> FINAL["Scored once, on the<br/>laptop: store set v2<br/>by tag, cause and<br/>seen or unseen design;<br/>ExpDate test"]
    class CTRL control
    classDef control stroke-dasharray:5 4
```

The generator's second version draws what the store gallery showed and
M2 never drew, and sizes each block from the store's own date sizes.
Every new feature is off in M2's spec, which keeps its hash, so M2's
photos still come out byte for byte (tests check it).

```mermaid
mindmap
  root((gen-v2))
    Layouts
      Upside-down print
      A keyword glued to its date
      Two dates on one line
      Dense stacked panels
    Print
      Light laser marks on a dark window
      Missing rows of dots
      Worn, half-rubbed ink
    Words around dates
      Import dates and their keywords
      Barcode digits
      Unit prices
    Size
      Drawn from the store's own date sizes
```

Where it stands (2026-11-08): store set v2, 99 new photos, is labelled
and frozen as the new test set. Store sets v1 and v3 are split by pack
design (80 photos train, 34 validate), boxed for keywords and exported to
the cluster. gen-v2 is written and checked on a laptop trial. Next, one
CPU job makes its 200,000 photos, while step 1's GPU job waits in the
queue. Plan and every slice's record:
[`docs/m2b-store-and-size.md`](docs/m2b-store-and-size.md).

---

## Lessons carried over from Scanify

These were measured before this project started, and shape its design:

- **Cropping hurts off-the-shelf OCR.** RapidOCR read fewer dates from
  perfect hand-drawn crops (25 of 44) than from the whole photo (31 of
  44). A detector feeding crops to the same OCR won't beat Scanify; the
  gain needs a reader trained on crops.
- **Heavy preprocessing hurts.** Adaptive thresholding made OCR worse;
  mild sharpening helped slightly. Models trained on natural photos want
  natural photos.
- **Resolution has a sweet spot.** For RapidOCR, a longest side of
  1000 px beat 1600, 2200 and the original size.
- **A blur score over the whole frame measures detail, not focus.** If
  blur is scored again, it should cover only the region around the text.

---

## Data

### ExpDate (training)

| Subset | Size | Used for |
|---|---|---|
| Products-Real | 1,767 real photos (1,102 train / 665 test) | finding dates |
| Products-Synth | 11,858 synthetic photos | finding dates, training only |
| Date-Synth / Date-Real | 128,000 synthetic + a real test set | finding day / month / year inside a date |
| Components-Synth / -Real | 450,000 synthetic + a real test set | reading the characters |

ExpDate is not redistributed here; `scripts/download_expdate.py` fetches
it from the authors' release and checks it against recorded SHA-256
hashes.

### Store photos (private)

99 photos taken in the retailer's stores (87 with the expiry in frame),
labelled with the expiry, a tight box around every date, and a failure
tag. **They are never committed and never published:** only aggregate
numbers appear in this repository, never a photo, a photo's name or a
picture drawn from one.

From M0 to M2 they were for testing only, frozen by their hash so they
couldn't change between experiments, and every store number above was
measured before any training on them. On 2026-10-28 the retailer's owner
allowed them to be used for training and to be shown to AI tools for
analysis; publishing them is still not allowed. Once a milestone trains
on them, a new set of store photos, labelled and frozen before that
training, becomes the test set.

### For testing the readability checker

- **SmartDoc-QA:** smartphone photos of documents with defocus, motion
  blur and bad lighting, each labelled with its OCR accuracy.
- **SPAQ:** 11,125 photos from 66 phones with human sharpness ratings.

---

## Getting started

Requires Python 3.11 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,download]"
cp .env.example .env          # then set the data paths
```

Prepare ExpDate:

```bash
python scripts/download_expdate.py products-real products-synth
python scripts/check_expdate.py               # counts and problems per split
python scripts/contact_sheet.py --set real-test --n 50 --seed 0
python scripts/box_stats.py real-train real-test synth
python scripts/expdate_to_yolo.py             # YOLO dataset under $EXPDATE_DIR/yolo
python scripts/readme_figures.py              # the figures in this README
pytest
```

Reproduce the M0 baselines. Baseline A on ExpDate needs only the steps
above; baseline B needs a checkout of Scanify, and the store runs need the
store photos, both private:

```bash
pip install -e ".[baselines]"                 # RapidOCR 1.2.3, as Scanify uses
python scripts/run_baseline.py --reader a --set real-test
python scripts/run_baseline.py --reader b --set real-test
python scripts/run_baseline.py --reader a --set store
python scripts/run_baseline.py --reader b --set store
python scripts/baseline_table.py              # writes results/m0_baselines.md
```

Train and score the M1 detector. Training runs on a GPU cluster
(Slurm job scripts in `slurm/`; setup and steps in
[`docs/m1-date-detector.md`](docs/m1-date-detector.md)). Scoring runs on
the laptop's CPU, from the trained weights:

```bash
pip install -e ".[train]"
python scripts/eval_detector.py --set real-test --runs n640 n640-seed1 n640-seed2
python scripts/eval_detector.py --set store --runs n640 n640-seed1 n640-seed2
python scripts/m1_report.py                   # writes results/m1_detector.md
```

Generate, train and score M2. The derived sets, the 52,100 generated
photos and the 18 training runs are made on the cluster by Slurm jobs
(steps in
[`docs/m2-synthetic-generator.md`](docs/m2-synthetic-generator.md), S1 to
S6): `slurm/m2_derive.sbatch`, `slurm/m2_generate.sbatch`, then two
`slurm/m2_gpu.sbatch` jobs that share one work queue
(`scripts/m2_chain.py`), train every run and score each on the
validation suite. Back on the laptop, with each run's `last.pt` copied
into `$CHECKPOINT_DIR/m2/<run>/weights/`:

```bash
pip install -e ".[train,synth]"
python scripts/fetch_fonts.py                 # the generator's 20 fonts, checked by SHA-256
python scripts/render_sheet.py                # every print style, to look at
scripts/pull_cluster_results.sh               # run records, suites, the chain's decisions
python scripts/eval_detector.py --set store --runs m2-{base,rot,gen,gen-rot,gen10k,gen-rot-1280}-s{0,1,2}
python scripts/eval_detector.py --set real-test --runs m2-gen-rot-1280-s{0,1,2}
python experiments/m2-ablation/summarise.py   # and the other m2-* experiments
python scripts/m2_report.py                   # writes results/m2_detector.md
```

Every script run writes a record (settings, git commit, seed, library
versions, and for the baselines a hash of the reader's code) to
`results/runs/`, so any number can be traced and reproduced. Runs on the
store photos write theirs to `data/store/runs/`, which never enters git.

---

## Repository layout

```
bestbefore/
├── docs/            milestone plans, dataset notes, learning guides, figures
├── src/bestbefore/  datasets, format conversion, synthesis, models, evaluation
├── scripts/         command-line entry points
├── slurm/           cluster job scripts (from M1)
├── experiments/     one folder per question: script + README (from M1)
├── tests/
├── data/            not in git: ExpDate, converted labels, store photos
└── results/         summary tables in git; per-run outputs are not
```

How the code is layered: job scripts call thin entry points, the logic
lives in one library, and every run leaves a record that the reports are
built from.

```mermaid
flowchart TD
    SLURM["slurm/<br/>cluster job scripts"] --> CLI["scripts/<br/>command-line<br/>entry points"]
    CLI --> lib
    subgraph lib["src/bestbefore/"]
        DATA["Data<br/>expdate, yolo, derived,<br/>store, store_train"]
        SYNTH["synth/<br/>spec, text, render,<br/>warp, compose,<br/>camera, make"]
        MODEL["Model and scoring<br/>detector,<br/>detection_eval,<br/>baselines/"]
        RULES["Milestones<br/>m1, m2, m2b:<br/>arms, mixes,<br/>decision rules"]
        INFRA["Infrastructure<br/>chain, slurm,<br/>runs, config"]
        DATA ~~~ RULES
        SYNTH ~~~ INFRA
        MODEL ~~~ INFRA
    end
    lib --> REC[("results/runs/<br/>one record per run")]
    REC --> REP["results/<br/>reports, written<br/>from the records"]
    TESTS["tests/<br/>pytest"] -.-> lib
```

## Documentation

- [`docs/m0-data-and-test-set.md`](docs/m0-data-and-test-set.md): the M0
  plan slice by slice
- [`docs/datasets.md`](docs/datasets.md): what is actually in ExpDate
- [`results/m0_baselines.md`](results/m0_baselines.md): the baseline
  table, with the runs it came from
- [`results/m1_detector.md`](results/m1_detector.md): the M1 detector's
  results, from validation to the store photos
- [`docs/learnings/m0.md`](docs/learnings/m0.md): the ideas behind each
  step, explained simply
- [`docs/m1-date-detector.md`](docs/m1-date-detector.md): the M1 plan,
  from cluster setup to the final evaluation
- [`docs/learnings/m1.md`](docs/learnings/m1.md): the cluster, Slurm,
  fine-tuning, precision, recall and mAP, explained simply
- [`docs/m2-synthetic-generator.md`](docs/m2-synthetic-generator.md): the
  M2 plan: the generator, the ablation and every slice's record
- [`results/m2_detector.md`](results/m2_detector.md): the M2 ablation's
  results, from the validation suite to the store photos
- [`docs/m2b-store-and-size.md`](docs/m2b-store-and-size.md): the M2b
  plan: store fine-tuning, gen-v2, model size and every slice's record
- [`results/store_designs.md`](results/store_designs.md): store pack
  designs and the v1 split, as counts only

---

## Acknowledgements

This project uses the **ExpDate** dataset by the Korea Institute of
Science and Technology (KIST), released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

> Seker, A. C. & Ahn, S. C. (2022). *A generalized framework for
> recognition of expiration dates on product packages using fully
> convolutional networks.* Expert Systems with Applications, 203, 117310.

Dataset page: <https://felizang.github.io/expdate/index_expdate.html>

M2's generated photos are drawn onto ExpDate's training photos, so they
are ExpDate derivatives and are not redistributed either. The
generator's 20 font families are under the SIL Open Font License,
fetched from the Google Fonts repository at a pinned commit and checked
by SHA-256; they aren't redistributed here.
