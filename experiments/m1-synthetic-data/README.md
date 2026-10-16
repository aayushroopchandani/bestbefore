# M1 / C2: does synthetic data help?

## Question

With the settings C1 picked, does training on the 11,858 synthetic
Products-Synth photos **plus** the 937 real ones find more dates than
training on the 937 real photos alone?

## Why it matters

Synthetic photos are free and plentiful, real ones are scarce. But M0
found two reasons the synthetic set might not help
([`docs/datasets.md`](../../docs/datasets.md)):

- **Its dates are bigger and easier.** The median synthetic date box is
  27 px tall at 640 px input, against 16 px in real photos, so a model
  can learn from them without learning to see small print.
- **It outnumbers the real photos 13 to 1.** A model may tune itself to
  how synthetic photos look rather than to real packaging.

M2 builds a better synthetic generator and asks the same question with
it, so this is also M2's baseline: how much does the *existing* synthetic
data add?

## Method

| Run | Training photos | Settings |
|---|---|---|
| the C1 winner (already trained in C1) | synthetic + real (12,795) | C1's winning model and input size, seed 0 |
| `<winner>-real` | real only (937) | the same |

Everything else is identical, and both are scored on the same 165
validation photos (187 dates). Both train for up to 100 epochs, so each
model sees every real photo the same number of times; the synthetic run
also sees the 11,858 synthetic photos in every epoch.

**Decision rule, written before any run**
([`docs/m1-date-detector.md`](../../docs/m1-date-detector.md), C2): keep
the synthetic photos **unless real-only is better by more than 1 point**
of validation mAP@0.5 for dates. The tie goes to keeping them, because the
plan says so (Products-Synth plus Products-Real is the M1 brief in
`CLAUDE.md`), not because more data is assumed to help. The rule is code
(`bestbefore.m1.keep_synthetic`), so the chain and this summary apply it
the same way. The columns are explained in
[`../m1-size-and-input/README.md`](../m1-size-and-input/README.md#reading-the-table).

## Result

Rebuild with:

```bash
scripts/pull_cluster_results.sh
python experiments/m1-synthetic-data/summarise.py --write
```

<!-- results:start -->
| run | model | input | training photos | mAP@0.5 date | mAP@0.5:0.95 date | threshold | date recall | precision | false boxes / photo | recall at IoU 0.3 | epochs | minutes | recall cross-check |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n640 | yolo11n | 640 | synthetic + real | 0.991 | 0.895 | 0.728 | 185/187 | 0.974 | 0.03 | 185/187 | 100 | 65.5 | ok |
| n640-real | yolo11n | 640 | real only | 0.981 | 0.865 | 0.682 | 179/187 | 0.984 | 0.02 | 180/187 | 100 | 6.3 | ok |

**Decision, by the rule above: keep the synthetic photos.** mAP@0.5 for dates: synthetic + real 0.991 vs real only 0.981 (+0.010).
<!-- results:end -->

## What it means

Run on 2026-10-16 (GPU job 72789, one H100).

- **The synthetic photos help a little.** With them, the model finds 185
  of 187 validation dates; without them, 179. mAP@0.5 for dates rises by
  1 point (0.991 against 0.981), and the boxes are tighter
  (mAP@0.5:0.95 0.895 against 0.865, +3 points).
- **The mAP@0.5 gain alone is about the size of the noise,** but every
  measure moves the same way, so keeping them is the right call by the
  rule and by the evidence.
- **The cost is ten times the training time** (66 minutes against 6),
  which is cheap on an H100.
- **For M2:** ExpDate's synthetic set adds about 1 point on ExpDate's own
  validation photos. M2's generator should be judged by what it adds on
  the store photos (sideways dates, Indian date formats, small print),
  which this synthetic set barely covers.
