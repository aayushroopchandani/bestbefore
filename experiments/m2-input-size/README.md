# M2 / decision 13: does a 1,280 px input find small print?

## Question

With the most varied training data (`gen-rot`), does a 1,280 px input find
the small dates that 640 px misses, on real print made small and on the
store photos?

## Why it matters

At 640 px input the median store date is 11 px tall, and 17.6% of store
date boxes are under 8 px; at 1,280 px only 0.5% are. M1 found 12% of
numeric store expiries printed under 1.5% of the photo's long side
([`m1-store-misses`](../m1-store-misses/README.md)). Small generated dates
are one fix (data); a larger input is the other (resolution). It costs
about 1.6 times the GPU time per run, and more on a phone.

## Method

| Arm | Training photos | Input | Epochs |
|---|---|---:|---:|
| `gen-rot` | ExpDate real train and its turned copies × 3, ExpDate synthetic and its turned copies, 50,000 generated photos | 640 | 32 |
| `gen-rot-1280` | the same | 1,280 | 32 |

Everything else as in [`m2-ablation`](../m2-ablation/README.md): the
recipe, 2.56 million training images, 3 seeds each, the validation suite
and each run's own F2 threshold. The arm was chosen before any run
(decision 13), so the 1,280 runs didn't wait for a result. Tiled
inference, the other way to see small print, stays in the plan's Later.

**Decision rule, written before any run**
([`docs/m2-synthetic-generator.md`](../../docs/m2-synthetic-generator.md),
"Reading the side questions"): compare `gen-rot-1280` against `gen-rot`,
on `real-val-small` first. The plan fixes no threshold for this reading.
Whether 1,280 px is used is the M2 detector rule's call: `gen-rot-1280`
is a candidate there, listed last as the most expensive, so it wins only
if it beats every cheaper arm by more than 1 point
([`m2-ablation`](../m2-ablation/README.md)).

## Result

Rebuild with:

```bash
scripts/pull_cluster_results.sh
python experiments/m2-input-size/summarise.py --write
```

<!-- results:start -->
**real-val-small first** (real print at half its size):

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| gen-rot | 3/3 | 0.911 (0.907–0.915) | 160.3/186 (157–162) | 162.3/186 (158–165) | 179.7/186 (179–181) | 0.884 (0.835–0.913) | 0.52 (0.37–0.78) |
| gen-rot-1280 | 3/3 | 0.986 (0.985–0.990) | 181.3/186 (181–182) | 181.3/186 (181–182) | 185.7/186 (185–186) | 0.954 (0.948–0.958) | 0.21 (0.20–0.24) |

mAP@0.5 for dates on every suite set:

| arm | seeds | real-val | real-val-rot | real-val-small | synth-val | hard sets, mean |
|---|---:|---:|---:|---:|---:|---:|
| gen-rot | 3/3 | 0.994 (0.994–0.995) | 0.971 (0.967–0.973) | 0.911 (0.907–0.915) | 0.986 (0.984–0.987) | 0.956 (0.953–0.957) |
| gen-rot-1280 | 3/3 | 0.995 (0.994–0.995) | 0.988 (0.984–0.991) | 0.986 (0.985–0.990) | 0.994 (0.994–0.995) | 0.989 (0.989–0.990) |

**The store photos** (S8), expiry dates found at the threshold, by failure tag:

| detector | seeds | sideways | dot-matrix | small-print | clean | all |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/15 | 14.7/20 (12–17) | 0/15 | 7/31 (6–8) | 21.7/81 (18–25) |
| gen-rot | 3/3 | 0.3/15 (0–1) | 17.3/20 (16–18) | 0/15 | 16/31 (15–17) | 33.7/81 (33–35) |
| gen-rot-1280 | 3/3 | 2.7/15 (2–3) | 19/20 | 5.3/15 (2–9) | 19/31 (15–23) | 46/81 (39–53) |
<!-- results:end -->

## What it means

Trained and scored on the validation suite by GPU job A (`72914`) on
`cn5` at `a53c1a9`, 2026-10-25 to 2026-10-26; scored once on the store
photos on the laptop's CPU on 2026-10-28 at `fb5aeaa` (clean tree).

- **On `real-val-small`, 1,280 px finds what 640 px misses:** 181.3
  against 160.3 of 186 dates at each run's threshold, mAP@0.5 0.986
  against 0.911, with fewer false boxes (0.21 against 0.52 a photo).
  The seeds' ranges are far apart (181–182 against 157–162).
- **It leads on every hard set**, so the M2 detector rule chose it
  ([`m2-ablation`](../m2-ablation/README.md)): mean mAP@0.5 0.989
  against 0.956, `real-val-rot` 0.988 against 0.971, `synth-val` 0.994
  against 0.986; `real-val` the same (0.995 against 0.994).
- **On the store photos it is the best arm:** 46 of 81 expiries against
  `gen-rot`'s 33.7 and M1's 21.7. Small-print expiries 5.3 of 15 against
  0 (boxed at any confidence 13 against 4.3), full month names 10 of 19
  against 5.7, sideways 2.7 of 15 against 0.3, clean 19 against 16 of 31.
- **Its seeds differ the most of any arm:** 39, 53 and 46 of 81, and 2,
  9 and 5 of 15 small-print expiries. At any confidence the three box
  69 to 72 of 81, so they see nearly the same dates and differ in how
  sure they are of them on store photos. Seed 0, the M2 detector by the
  rule declared in advance, has the lowest of the three thresholds
  (0.342, 0.424, 0.399; the arm's mean, 0.389, is the highest of any
  arm) and still keeps the fewest.
- **The cost:** 202 to 204 GPU minutes a run against 119 to 120 at 640
  px (1.7 times; 211 against 363 images a second), and four times the
  pixels per photo on a phone (M5).
