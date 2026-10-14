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
_Not run yet: the runs are queued in GPU job 72789._
<!-- results:end -->

## What it means

_Written after the runs: what the numbers say about small dates, and
whether the extra cost of the larger model or input paid off._
