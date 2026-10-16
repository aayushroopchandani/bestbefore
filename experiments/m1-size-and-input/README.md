# M1 / C1: model size and input size

## Question

Which combination of model size (`yolo11n` or `yolo11s`) and input size
(640 or 1280 px) finds printed dates best on the validation photos?

## Why it matters

M0 measured how small the dates are: in real photos, a date box is about
**16 px** tall when the photo is shrunk to YOLO's default 640 px input,
and about **31 px** at 1280 ([`docs/datasets.md`](../../docs/datasets.md)).
A larger input should make small dates easier to find, but costs about 4×
the computation. The larger model can learn finer detail, but costs about
3×. Neither is worth paying for unless it finds clearly more dates.

| Model | Parameters | Computation per 640 px image |
|---|---:|---:|
| `yolo11n` | 2.6 M | 6.5 GFLOPs |
| `yolo11s` | 9.4 M | 21.5 GFLOPs |

## Method

Four runs, identical except for the two settings under test:

| Run | Model | Input |
|---|---|---:|
| `n640` | `yolo11n` | 640 |
| `n1280` | `yolo11n` | 1280 |
| `s640` | `yolo11s` | 640 |
| `s1280` | `yolo11s` | 1280 |

- **Training photos:** Products-Synth plus the Products-Real train split,
  12,795 photos (11,858 synthetic + 937 real).
- **Fixed settings:** COCO-pretrained weights, up to 100 epochs with early
  stopping (patience 30), batch 16, seed 0, YOLO's default augmentation
  with horizontal flips turned off (a flipped date is mirrored text).
- **Scored on:** the 165 validation photos (187 dates), taken out of the
  Products-Real train split with seed 0. The test split isn't touched.
- **Where:** one H100 on the university cluster, inside GPU job 72789,
  run back to back by [`scripts/m1_chain.py`](../../scripts/m1_chain.py).

**Decision rule, written before any run**
([`docs/m1-date-detector.md`](../../docs/m1-date-detector.md), C1): pick
the highest validation **mAP@0.5 for dates**. With only 165 validation
photos, differences under 1 point (0.01) are noise, so within 1 point of
the best, take the cheaper run: smaller model first, then smaller input
(`n640`, `n1280`, `s640`, `s1280`). The rule is code
(`bestbefore.m1.decide_c1`), so the chain and this summary apply it the
same way.

### Reading the table

| Column | Meaning |
|---|---|
| mAP@0.5 date | the area under the precision-recall curve for dates, IoU ≥ 0.5 (the deciding number) |
| mAP@0.5:0.95 date | the same, averaged over stricter IoU thresholds: how *tight* the boxes are |
| threshold | the confidence that gives the best F1 for dates on these photos |
| date recall | dates found at that threshold, out of 187 |
| precision | share of kept boxes that are real dates |
| false boxes / photo | kept boxes that match no date, per photo |
| recall at IoU 0.3 | dates "roughly found": enough for a reader that crops with a margin |
| epochs | epochs trained (under 100 means early stopping kicked in) |
| recall cross-check | `ok` when our evaluator's date recall is within 0.02 of Ultralytics' own at the same threshold |

## Result

Rebuild with:

```bash
scripts/pull_cluster_results.sh
python experiments/m1-size-and-input/summarise.py --write
```

<!-- results:start -->
| run | model | input | training photos | mAP@0.5 date | mAP@0.5:0.95 date | threshold | date recall | precision | false boxes / photo | recall at IoU 0.3 | epochs | minutes | recall cross-check |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n640 | yolo11n | 640 | synthetic + real | 0.991 | 0.895 | 0.728 | 185/187 | 0.974 | 0.03 | 185/187 | 100 | 65.5 | ok |
| n1280 | yolo11n | 1280 | synthetic + real | 0.994 | 0.912 | 0.553 | 187/187 | 0.979 | 0.02 | 187/187 | 96 | 96.5 | ok |
| s640 | yolo11s | 640 | synthetic + real | 0.995 | 0.912 | 0.787 | 186/187 | 0.995 | 0.01 | 186/187 | 100 | 61.6 | ok |
| s1280 | yolo11s | 1280 | synthetic + real | 0.995 | 0.922 | 0.633 | 186/187 | 1.000 | 0.00 | 186/187 | 87 | 102.3 | ok |

**Decision, by the rule above: `n640`**, within 0.004 of the best (s640, 0.995), which is under 1 point, and cheaper.
<!-- results:end -->

## What it means

Run on 2026-10-15/16 (GPU job 72789, one H100).

- **On ExpDate validation, all four settings are as good as each other.**
  Each finds 185 to 187 of the 187 dates, and mAP@0.5 spans 0.991 to
  0.995. That 0.4-point spread is under the 1-point noise band, so the
  rule picks the cheapest run: **`n640`, the smallest model at YOLO's
  default input size.**
- **The validation photos are close to their ceiling.** With every model
  missing at most 2 dates, this set can't tell the settings apart. It
  doesn't show that model or input size never matters. It shows that
  ExpDate's dates, at about 16 px tall at 640 px input, are big enough
  for the smallest setting.
- **The bigger settings buy tighter boxes, not more dates.** mAP@0.5:0.95,
  which rewards boxes that hug the date closely, rises from 0.895
  (`n640`) to 0.912–0.922. Finding a date only needs IoU 0.5, and the
  M3 reader can crop with a margin, so this doesn't change the decision.
- **Cost:** the 1280 px runs took about 50% longer (97 and 102 minutes
  against 62–66). `s640` trained as fast as `n640`, which suggests that
  loading photos, not the GPU, set the pace.
- **`n640` trained all 100 epochs without early stopping,** so it may
  still have been improving. A longer run is noted under Later in the
  M1 plan.
- **What this can't answer:** whether the store photos' small print needs
  1280 px. The store result (M1, D3) will show how `n640` copes, but it
  can't be used to change this choice, because the store photos are test
  data. If `n640` misses small print there, that's a finding for later
  milestones.
