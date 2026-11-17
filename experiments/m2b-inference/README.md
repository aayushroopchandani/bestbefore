# M2b / inference add-ons: do more passes find more store dates?

## Question

Does the M2b detector, `n1280-ft-s0`, find more of v1-val's dates when it
gets more than one look at each photo: its arm's three seeds merged (the
ensemble), the photo also turned 90°, 180° and 270° (test-time rotation),
or a 1,536 px input instead of 1,280? What does each cost in time on a
CPU? And does R4, written before any run, adopt any of them?

Beside it, the owner's question from S12: how much slower is `m1280-ft`,
which scored higher on v1-val but which R3 didn't choose, than `n` on the
same CPU?

## Why it matters

The M2b detector reads every photo M3 reads from, and a phone or a server
pays for every extra pass. The plan
([`docs/m2b-store-and-size.md`](../../docs/m2b-store-and-size.md),
decision 17) made each add-on earn its place by R4: it must raise v1-val
AP by more than the noise band. S12 found that band, 14.1 points, out of
reach: `n1280-ft-s0` scores 0.9514 on v1-val, so no add-on can gain more
than 4.9. R4 is applied as written all the same, and the add-ons are
measured as side readings, so S14 and M3 can see what they would have
given. A "what if" reading of R4 with the plan's own floor band, 0.03,
was fixed before any add-on was measured; it decides nothing.

## Method

| Configuration | Passes a photo |
|---|---:|
| none: `n1280-ft-s0` at 1,280 px | 1 |
| ensemble: the arm's 3 seeds | 3 |
| rotation: the photo at 0°, 90°, 180° and 270° | 4 |
| 1,536 px: the same weights fed 1,536 px | 1 |
| each pair, and all three | 3 to 12 |

- **Passes.** Every pass predicts one photo at a time with `rect=False`,
  as every M2b score does. A pass on a turned photo has its boxes turned
  back onto the upright photo (the inverse of `derived.turn_box`). Every
  pass of every configuration (3 seeds × 4 turns × 2 sizes = 24 a photo)
  was saved once by `scripts/m2b_inference.py predict` and each
  configuration built from them, so all 8 come from the same predictions.
- **Fusion.** `ensemble-boxes` 1.0.9's `weighted_boxes_fusion` with its
  defaults (IoU 0.55, the average confidence, every pass weighing the
  same), boxes under 0.001 skipped, one fusion over all of a
  configuration's passes, class by class. A box only some passes found
  keeps its mean confidence times the share that found it. A one-pass
  configuration is that pass's boxes, unfused.
- **Where.** v1-val's 34 store photos on the laptop (Apple M2), their
  per-photo boxes kept in `data/store/runs/`; real-val's 165 ExpDate
  photos in one CPU job on the cluster (`slurm/m2b_inference_cpu.sbatch`,
  job `73981`, `cn4`). No store photo was read on the cluster.
- **Scores.** Our own AP@0.5 for dates
  (`detection_eval.average_precision`, Ultralytics 8.4.177's curve with
  our matching), on v1-val and real-val. Counts on v1-val at each
  configuration's own F2 threshold (the plan's threshold rule); every
  configuration also boxes all 63 dates at some confidence, so they differ
  only in how sure they are.
- **R4** (`m2b.adopt`, written before any add-on was measured): the
  add-ons are tried in the order ensemble, rotation, 1,536 px, each on top
  of what is adopted so far. One stays if it raises v1-val AP by more than
  the band, and real-val AP falls by no more than 0.01. A gain of exactly
  the band is within it. With nothing adopted, the detector keeps its
  `threshold_store`, 0.1641.
- **The what-if**: R4's same steps with the band at 0.03, the plan's
  floor. Whatever it would keep goes to the plan's Later, never to S14.
- **Checks before any reading.** The device check: `n1280-ft-s0` scored
  again on v1-val on the laptop's CPU (`score_suite.py --sets v1-val
  --out`) against its GPU suite, within a date or two. The AP check: our
  AP for each seed's single pass against Ultralytics' mAP in its GPU
  suite, within 0.01 on real-val.
- **CPU time a photo**: the mean over v1-val's photos of each pass on the
  laptop (turning included) summed over a configuration's passes, plus the
  fusion. **The timing** (`scripts/m2b_inference.py time`): `n1280-ft-s0`
  at 1,280 and 1,536 px and `m1280-ft-s0` at 1,280 px, the same 34 photos
  loaded first, one photo a call, 2 warm-up calls each left out, then 3
  repeats taking the three in turn.

## Result

Rebuild with:

```bash
python experiments/m2b-inference/summarise.py --write
```

<!-- results:start -->
**The device check passes**: `n1280-ft-s0` on v1-val, on the laptop's CPU against its GPU suite, each at its own F2 threshold (0.1641 on the GPU, 0.1665 here; the GPU's is the run's): dates found 60/63 on the GPU, 60/63 here; expiries found 29/29 on the GPU, 29/29 here; boxed at any confidence 63/63 on the GPU, 63/63 here; mAP@0.5 for dates (Ultralytics') 0.9514 and 0.9514.

**The AP check passes**: on real-val ours is within 0.0005 of Ultralytics' for every seed (the plan asks 0.01).

| run | real-val ours | real-val Ultralytics | real-val difference | v1-val ours | v1-val Ultralytics | v1-val difference |
|---|---:|---:|---:|---:|---:|---:|
| n1280-ft-s0 | 0.9949 | 0.9950 | -0.0001 | 0.9596 | 0.9514 | +0.0082 |
| n1280-ft-s1 | 0.9949 | 0.9949 | -0.0000 | 0.9510 | 0.9369 | +0.0141 |
| n1280-ft-s2 | 0.9944 | 0.9939 | +0.0005 | 0.9513 | 0.9362 | +0.0151 |

**R4, worked out again** (the noise band 0.1414; R4's own AP, ours, before and after, like with like):

**R4, as written (band 0.1414, the noise band): adopts nothing**. Gains on v1-val over what it had kept, in R4's order: ensemble +0.0141, rotation +0.0143, imgsz-1536 +0.0256; the largest, imgsz-1536's +0.0256, is within the band. So the M2b detector is `n1280-ft-s0` alone at its threshold_store, 0.1641.

| add-on | against | tried | v1-val before | after | gain | band | real-val change | real-val guard | kept |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| ensemble | none | ensemble | 0.9596 | 0.9737 | +0.0141 | 0.1414 | +0.0001 | passes | no |
| rotation | none | rotation | 0.9596 | 0.9739 | +0.0143 | 0.1414 | -0.0021 | passes | no |
| imgsz-1536 | none | imgsz-1536 | 0.9596 | 0.9852 | +0.0256 | 0.1414 | -0.0021 | passes | no |

**The what-if (band 0.0300, the plan's floor; it decides nothing): adopts nothing**. Gains on v1-val over what it had kept, in R4's order: ensemble +0.0141, rotation +0.0143, imgsz-1536 +0.0256; the largest, imgsz-1536's +0.0256, is within the band. Nothing for the plan's Later.

| add-on | against | tried | v1-val before | after | gain | band | real-val change | real-val guard | kept |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| ensemble | none | ensemble | 0.9596 | 0.9737 | +0.0141 | 0.0300 | +0.0001 | passes | no |
| rotation | none | rotation | 0.9596 | 0.9739 | +0.0143 | 0.0300 | -0.0021 | passes | no |
| imgsz-1536 | none | imgsz-1536 | 0.9596 | 0.9852 | +0.0256 | 0.0300 | -0.0021 | passes | no |

Worked out again here with `m2b.decide_addons()` from the record's APs, R4 and the what-if are the record's, and its arm, run and threshold are R3's in the chain's record.

**Every configuration**, side readings, none adopted (counts on v1-val at each configuration's own F2 threshold; real-val's dates found at that threshold too; CPU time a photo is the laptop's, its passes' mean plus the fusion):

| configuration | passes | v1-val AP | real-val AP | threshold | dates found | expiry found | precision | false boxes | false / photo | boxed at any confidence | real-val found | CPU ms a photo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 1 | 0.9596 | 0.9949 | 0.1665 | 60/63 | 29/29 | 0.779 | 17 | 0.50 | 63/63 | 187/187 | 167 |
| ensemble | 3 | 0.9737 | 0.9950 | 0.2264 | 61/63 | 27/29 | 0.871 | 9 | 0.26 | 63/63 | 187/187 | 511 |
| rotation | 4 | 0.9739 | 0.9928 | 0.1772 | 63/63 | 29/29 | 0.818 | 14 | 0.41 | 63/63 | 187/187 | 681 |
| imgsz-1536 | 1 | 0.9852 | 0.9929 | 0.4860 | 62/63 | 29/29 | 0.925 | 5 | 0.15 | 63/63 | 187/187 | 270 |
| ensemble+rotation | 12 | 0.9841 | 0.9934 | 0.2284 | 62/63 | 28/29 | 0.912 | 6 | 0.18 | 63/63 | 187/187 | 2053 |
| ensemble+imgsz-1536 | 3 | 0.9790 | 0.9949 | 0.2131 | 63/63 | 29/29 | 0.840 | 12 | 0.35 | 63/63 | 187/187 | 817 |
| rotation+imgsz-1536 | 4 | 0.9856 | 0.9920 | 0.3075 | 62/63 | 29/29 | 0.899 | 7 | 0.21 | 63/63 | 185/187 | 1089 |
| ensemble+rotation+imgsz-1536 | 12 | 0.9841 | 0.9937 | 0.2199 | 63/63 | 29/29 | 0.887 | 8 | 0.24 | 63/63 | 187/187 | 3275 |

Passes: v1-val on Apple M2 (cpu, 7 torch threads, 34 photos, code `e38ae84`); real-val on Intel(R) Xeon(R) Gold 6442Y (cpu, torch's threads not recorded, 165 photos, code `e38ae84`).

**v1-val by failure tag**, expiry dates found at each run's threshold:

| arm | seeds | sideways | dot-matrix | small-print | clean | all expiry dates | all dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 60/63 |
| ensemble | 1/1 | 5/6 | 3/3 | 6/7 | 13/13 | 27/29 | 61/63 |
| rotation | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 62/63 |
| ensemble+rotation | 1/1 | 6/6 | 3/3 | 6/7 | 13/13 | 28/29 | 62/63 |
| ensemble+imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| rotation+imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 62/63 |
| ensemble+rotation+imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |

and boxed at any confidence:

| arm | seeds | sideways | dot-matrix | small-print | clean | all expiry dates | all dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| ensemble | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| rotation | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| ensemble+rotation | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| ensemble+imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| rotation+imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| ensemble+rotation+imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |

**v1-val's dates by size**, found at each run's threshold:

| arm | seeds | under 1.5% | 1.5-2.5% | 2.5% and up |
|---|---:|---:|---:|---:|
| none | 1/1 | 22/23 | 26/28 | 12/12 |
| ensemble | 1/1 | 22/23 | 27/28 | 12/12 |
| rotation | 1/1 | 23/23 | 28/28 | 12/12 |
| imgsz-1536 | 1/1 | 22/23 | 28/28 | 12/12 |
| ensemble+rotation | 1/1 | 22/23 | 28/28 | 12/12 |
| ensemble+imgsz-1536 | 1/1 | 23/23 | 28/28 | 12/12 |
| rotation+imgsz-1536 | 1/1 | 23/23 | 27/28 | 12/12 |
| ensemble+rotation+imgsz-1536 | 1/1 | 23/23 | 28/28 | 12/12 |

**The detector against `m` on the laptop's CPU**: Apple M2, 7 torch threads, 34 v1-val photos (long side 1600 px, median) loaded first, one photo a call, rect=False; 2 warm-up calls each left out, then 3 repeats.

| model | median ms | mean ms | each repeat's mean | against the first |
|---|---:|---:|---|---:|
| n1280-ft-s0@1280 | 175 | 178 | 170, 179, 185 | 1.00× |
| n1280-ft-s0@1536 | 279 | 312 | 283, 282, 372 | 1.59× |
| m1280-ft-s0@1280 | 819 | 905 | 890, 835, 989 | 4.68× |

Worked out again here with `m2b.decisions()`, the chain's whole record (`results/runs/m2b-decisions.json`: noise_store, R1, R2, R3) is the same.
<!-- results:end -->

## What it means

The passes were made on 2026-11-17 at `e38ae84`: v1-val's on the laptop
(Apple M2, 7 torch threads; the record's own count, 1, was read before
Ultralytics set the threads, fixed in `4f1e220` and noted in the record),
real-val's in job `73981` on `cn4` (Intel Xeon Gold 6442Y, 5.1 minutes for
3,960 passes; by Ultralytics' rule, `min(8, CPUs − 1)`, it predicted on
8 threads there though the job asked for 24, not measured there, and that
record holds the early count, 24). v1-val is small: one of its 63 dates is 1.6 points of recall, and
every gain below is one to three dates.

- **Both checks pass.** On the laptop's CPU `n1280-ft-s0` finds on v1-val
  exactly what it found on the GPU (60/63 dates, 29/29 expiries, 63/63
  boxed at any confidence, Ultralytics' mAP 0.9514 both); only its own F2
  threshold moves, 0.1641 to 0.1665, and the GPU's stays the run's. Our AP
  agrees with Ultralytics' on real-val within 0.0005 for every seed. On
  v1-val ours is 0.8 to 1.5 points higher (0.9596 against 0.9514 for seed
  0): our matching takes boxes by confidence, Ultralytics' by overlap, and
  its validation pads photos less. R4 compares ours with ours.
- **R4, as written, adopts nothing.** Measured on the detector alone, the
  ensemble gains 0.0141 on v1-val, rotation 0.0143 and 1,536 px 0.0256,
  each far inside the band, 0.1414. So the M2b detector stays
  `n1280-ft-s0` alone, at its `threshold_store`, 0.1641.
- **The what-if, with the plan's floor band 0.03, adopts nothing
  either.** 1,536 px came closest, +0.0256: under 0.03 by less than half
  a point, about a quarter of one date. So nothing goes to the plan's Later
  from it. Every add-on passes R4's real-val guard: rotation and 1,536 px
  each take 0.0021 off real-val AP, the ensemble adds 0.0001.
- **What they do on v1-val is make the detector surer, not find new
  dates.** All 8 configurations box all 63 dates at some confidence. At
  its own threshold the detector alone finds 60/63 with 17 false boxes
  (precision 0.779). 1,536 px finds 62/63 with 5 (0.925); the ensemble
  61/63 with 9 (0.871), but 27/29 expiries against the detector's 29/29,
  missing a sideways and a small-print one; rotation 63/63 with 14
  (0.818). The configurations that find all 63 are rotation,
  ensemble with 1,536 px, and all three. By size, the gains are in the
  smaller two bands: under 1.5% of the photo the detector alone finds
  22/23, and 1.5–2.5% 26/28.
- **Each costs time, most of it more than it gives.** On the laptop's
  CPU the detector alone takes 167 ms a photo. 1,536 px takes 270 (1.6
  times), the ensemble 511 (3.1), rotation 681 (4.1), ensemble with
  rotation 2,053 (12.3) and all three 3,275 (19.6). Fusion is under 10 ms
  a photo even for 12 passes. Rotation, at four times the cost, adds
  about what the ensemble does at three; and the combinations add nothing
  on AP over 1,536 px alone (rotation with 1,536 px 0.9856 against
  0.9852), at 3 to 12 times its cost. On real-val one of them,
  rotation with 1,536 px, finds 185 of 187 dates at its v1-val threshold,
  where every other finds all 187.
- **`n` is the fastest, by a wide margin.** On the laptop's CPU
  `n1280-ft-s0` takes 175 ms a photo (median, 3 repeats over the 34
  photos, 1,600 px on their long side), at 1,536 px 279 ms (1.6 times),
  and `m1280-ft-s0` 819 ms (4.7 times). That backs S12's reason for
  keeping R3's choice: `m`'s better store scores cost nearly five times
  `n`'s time a photo here. The third repeat ran slower for the two larger
  loads (372 and 989 ms against 283 and 835 to 890), likely the laptop
  heating; the medians stand on all three. These are a laptop CPU's
  times, not a phone browser's (M5's question) or a GPU server's.

## Decision

By R4, no add-on is adopted, and the what-if keeps none either. **The M2b
detector S14 scores once on store set v2 and ExpDate test is
`n1280-ft-s0` alone, at `threshold_store` 0.1641**, one pass a photo at
1,280 px. Nothing about the threshold or the rules changed. The
add-ons' readings and the times stay recorded here: where the detector
will run (live on a phone, or on a server per photo) is still open, and
M3 reads them when it decides that, and whether `m1280-ft` (the plan's
Later) is worth its 4.7 times the time.
