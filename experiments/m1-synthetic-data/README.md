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
_Not run yet: C2 starts once all four C1 runs have finished (GPU job 72789)._
<!-- results:end -->

## What it means

_Written after the runs. Either answer is worth writing up: "the
synthetic photos don't help" would be believable after what M0 found,
and it would make M2's generator more important._
