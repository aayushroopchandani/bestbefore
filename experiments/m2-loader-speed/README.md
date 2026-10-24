# M2 / decision 15: loader settings

## Question

Which data-loader settings train M2's arms fastest on the cluster's H100:
8 loader workers, 24 workers, or 24 workers with the photos cached in RAM?

## Why it matters

Every M2 arm trains on 2.56 million images, twice M1's `n640`, and there
are 18 runs. In M1, `s640` trained as fast as `n640` and Ultralytics warned
of slow image access, a hint that loading photos, not the GPU, set the
pace ([`docs/learnings/m1.md`](../../docs/learnings/m1.md), §2.9). If more
workers or a RAM cache were faster, every arm would finish sooner. If not,
the simplest setting is as good, and a RAM cache would only risk the job's
memory.

## Method

Three speed runs of `base` (12,781 photos listed), 3 epochs each, with
M2's recipe, run back to back by the chain in GPU job A (`72914`, one
H100 on `cn5`):

| Run | Loader workers | RAM cache |
|---|---:|---|
| `m2-speed-a` | 8 | no |
| `m2-speed-b` | 24 | no |
| `m2-speed-c` | 24 | yes, if our own check allows it |

- **Measured:** images per second in **epoch 2**, from the cumulative
  `time` column of Ultralytics' `results.csv`. Epoch 1 includes the
  workers' start-up and the first read of every photo; the last epoch
  includes the final validation.
- **The RAM check:** 30 sampled photos, scaled to the whole list, must need
  under 60% of the job's memory (`SLURM_MEM_PER_NODE`). Ultralytics' own
  check looks at the node's free memory instead.

**Decision rule, written before any run**
([`docs/m2-synthetic-generator.md`](../../docs/m2-synthetic-generator.md),
decision rule 1; `bestbefore.m2.loader_choice`): the fastest setting; any
setting within 10% of the fastest counts as just as fast, and then the
simplest wins (a, then b, then c). The chain applied it as soon as the
third run finished and wrote it to `results/runs/m2-decisions.json`;
`summarise.py` applies it again through the same function
(`m2.decisions()`) and checks the two agree.

## Result

Rebuild with:

```bash
scripts/pull_cluster_results.sh
python experiments/m2-loader-speed/summarise.py --write
```

<!-- results:start -->
_Not run yet: the speed runs are queued in GPU job A (72914)._
<!-- results:end -->

## What it means

_Written once the runs are in._
