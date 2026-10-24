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
_Not run yet: the arms train in GPU jobs A (72914) and B (72915)._
<!-- results:end -->

## What it means

_Written once the six runs are scored._
