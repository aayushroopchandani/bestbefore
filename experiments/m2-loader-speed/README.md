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
| run | workers | RAM cache | photos | epoch 1 s | epoch 2 s | epoch 3 s | images/s, epoch 2 | train minutes |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| m2-speed-a | 8 | no | 12781 | 55.6 | 35.1 | 50.1 | 364.0 | 4.3 |
| m2-speed-b | 24 | no | 12781 | 54.6 | 35.1 | 50.7 | 364.0 | 2.8 |
| m2-speed-c | 24 | filled (needs 11.2 of 57.6 GB) | 12781 | 55.1 | 34.5 | 51.3 | 370.3 | 3.0 |

**Decision, by the rule: `a` (8 workers, no cache)**, its 364.0 images/s are 98.3% of the fastest, `c`'s 370.3: within 10%, and it is the simplest such setting. The chain's own record (`results/runs/m2-decisions.json`) agrees.
<!-- results:end -->

## What it means

Run on 2026-10-24, 12:06 to 12:16, by GPU job A (`72914`) on `cn5`, at
`a53c1a9` (clean tree). Checked on the laptop after
`scripts/pull_cluster_results.sh`.

- **The rule was applied correctly.** `c` is the fastest, at 370.3 images
  per second. `a`'s 364.0 is 98.3% of that, inside the 10% band, and `a`
  is the simplest setting, so the rule gives `a`: 8 workers, no cache. The
  chain recorded exactly that, and `summarise.py` gets the same answer
  from the pulled `results.csv` files through the same function.
- **The loader isn't what limits a 640 px run on this node.** The three
  settings are within 1.7% of each other in epoch 2. 24 workers ran at
  exactly the 8-worker speed, and the RAM cache saved 0.6 s an epoch. So
  more workers and memory buy nothing here, and M1's hint that loading
  set the pace doesn't hold for `yolo11n` at 640 px. M1's `n640` also
  used 8 workers, at 354 images/s.
- **Only the first run read its photos cold.** For `m2-speed-a`
  Ultralytics warned of slow image access; for `-b` and `-c`, minutes later
  on the same node, it didn't, as the node's file cache by then held the
  photos (job log, `slurm/logs/m2-72914.out`). That cost `speed-a` about
  1.5 minutes before its first epoch (4.3 train minutes against 2.8 and
  3.0, for the same 141 s of epochs), not in its timed epoch: epoch 1 took
  54.6 to 55.6 s in all three runs.
- **The RAM cache fit.** Our check estimated 11.2 GB against a limit of
  57.6 GB (60% of the job's 96 GB), and Ultralytics filled it. With `a`
  chosen, no arm uses a RAM cache.
- **For the arms:** at 364 images per second, `base`'s 2.56 million images
  take about 1.95 hours at 640 px: the plan's estimate of 2 hours.
- **Not measured here:** whether 8 workers keep up with the generated
  arms, which list 66,529 to 79,310 photos instead of 12,781, or with
  `gen-rot-1280` at 1,280 px. Each run's own images per second are in the
  runs table of `results/m2_detector.md` (`scripts/m2_report.py`), so a
  slower arm will show there.
