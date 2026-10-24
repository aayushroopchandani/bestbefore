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
_Not run yet: the arms train in GPU jobs A (72914) and B (72915)._
<!-- results:end -->

## What it means

_Written once the six runs are scored._
