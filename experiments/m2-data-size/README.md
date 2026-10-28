# M2 / decision 14: do more distinct generated photos help?

## Question

At the same mix and the same training budget, does the detector learn
more from 50,000 distinct generated photos than from 10,000 seen five
times as often?

## Why it matters

Generating photos is cheap (minutes on the CPU nodes); training is the
cost. If 10,000 photos already give everything 50,000 do, the generator's
variety is what matters, not its count. If 50,000 clearly beat 10,000, a
larger set (200,000) is worth trying.

## Method

| Arm | Training photos | Listed | Epochs |
|---|---|---:|---:|
| `gen` | ExpDate real train × 5, ExpDate synthetic, the 50,000 generated photos once | 66,529 | 38 |
| `gen10k` | ExpDate real train × 5, ExpDate synthetic, the first 10,000 generated photos × 5 | 66,529 | 38 |

Everything else as in [`m2-ablation`](../m2-ablation/README.md): the
recipe, 2.56 million training images, 3 seeds each, the validation suite
and each run's own F2 threshold. Only the number of different generated
photos changes.

**Decision rule, written before any run**
([`docs/m2-synthetic-generator.md`](../../docs/m2-synthetic-generator.md),
"Reading the side questions"; `bestbefore.m2.more_photos_help`): if `gen`
beats `gen10k` by **more than 1 point** of mean mAP@0.5 for dates over the
three hard sets (`real-val-rot`, `real-val-small`, `synth-val`), more
distinct photos help and 200,000 becomes a Later candidate. Otherwise
50,000 is already enough. `gen10k` isn't a candidate for the M2 detector.

## Result

Rebuild with:

```bash
scripts/pull_cluster_results.sh
python experiments/m2-data-size/summarise.py --write
```

<!-- results:start -->
| arm | seeds | real-val | real-val-rot | real-val-small | synth-val | hard sets, mean |
|---|---:|---:|---:|---:|---:|---:|
| gen | 3/3 | 0.994 | 0.544 (0.515–0.582) | 0.921 (0.915–0.930) | 0.985 (0.985–0.986) | 0.817 (0.810–0.828) |
| gen10k | 3/3 | 0.992 (0.989–0.994) | 0.509 (0.481–0.527) | 0.913 (0.907–0.923) | 0.979 (0.978–0.980) | 0.800 (0.789–0.807) |

**By the rule: more distinct photos help: 200,000 becomes a Later candidate.** Mean mAP@0.5 for dates on the hard sets: `gen` 0.8167, `gen10k` 0.8004 (+0.0163; the rule needs more than +0.01).

**synth-val by stratum**, targets found at the threshold:

| arm | seeds | clean | sideways | month-word | small | dot-matrix | curved | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gen | 3/3 | 295.7/300 (295–296) | 296/300 (295–297) | 296.3/300 (296–297) | 273/300 (268–279) | 284.7/300 (284–285) | 293/300 (292–294) | 277.3/300 (275–281) |
| gen10k | 3/3 | 293/300 (292–294) | 291/300 (289–292) | 293.7/300 (293–294) | 262.7/300 (257–267) | 278.7/300 (275–281) | 285.7/300 (285–286) | 267.3/300 (264–270) |

**The store photos** (S8), expiry dates found at the threshold, by failure tag:

| detector | seeds | sideways | dot-matrix | small-print | clean | all |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/15 | 14.7/20 (12–17) | 0/15 | 7/31 (6–8) | 21.7/81 (18–25) |
| gen | 3/3 | 0.7/15 (0–1) | 18.7/20 (18–19) | 0.7/15 (0–1) | 18.3/31 (18–19) | 38.3/81 (37–40) |
| gen10k | 3/3 | 0.3/15 (0–1) | 18/20 (17–19) | 0.3/15 (0–1) | 18.3/31 (17–19) | 37/81 (34–39) |
<!-- results:end -->

## What it means

Trained and scored on the validation suite by GPU job A (`72914`) on
`cn5` at `a53c1a9`, 2026-10-25 to 2026-10-26; scored once on the store
photos on the laptop's CPU on 2026-10-28 at `fb5aeaa` (clean tree).

- **By the rule, more distinct photos help:** `gen` leads `gen10k` by
  1.63 points of mean mAP@0.5 on the hard sets (0.8167 against 0.8004),
  over the rule's 1 point. So 200,000 generated photos becomes a Later
  candidate, as the plan's Later item already says it would.
- **Most of the lead is on real turned dates:** `real-val-rot` 0.544
  against 0.509 (+3.5 points), `real-val-small` +0.8, `synth-val` +0.6.
  Every synth-val stratum is a little higher with `gen`, by 2.6 to 10.3
  targets of 300 (small print and `mixed` the most). Only on synth-val
  do the seeds' ranges not overlap; on both real hard sets they do
  (`real-val-rot` 0.515–0.582 against 0.481–0.527).
- **The store photos don't show it:** 38.3 against 37 of 81 expiries,
  ranges 37 to 40 and 34 to 39; by tag within a date of each other. A
  1.6-point mAP lead is too small for 81 dates to see.
- So the rule, fixed before the runs, says more photos help, and that
  stands; the lead itself is small, mostly on real turned dates, and
  within the seeds' spread on the real hard sets.
