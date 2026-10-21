# M1 on the store photos: which expiry dates does it miss?

## Question

M1's detector finds 97% of expiry dates on ExpDate's test split, but only
27% on the store photos (mean of 3 seeds, D3). Sideways dates are missed
entirely, as expected: ExpDate has none. Why are the others missed, the
`clean` ones included?

## Method

`analyse.py` takes the 66 non-sideways expiry dates in the frozen store
test set (v1) and the predictions of the three `n640` seeds, each at its
own validation threshold (D3's run, with `rect=False` padding). It splits
the dates two ways, both fixed before looking:

- **Format:** the month written as a word (`Month YYYY`, `MON YYYY`,
  `DD Month YYYY`, ...) or numbers only (`DD/MM/YYYY`, `MM/YYYY`, ...),
  from the format typed in each label.
- **Print height:** the expiry box's height as a share of the photo's
  long side, cut at 1.5% (the `small-print` rule) and 2.5% (about
  ExpDate's median date).

For each group: the share found at the threshold (IoU 0.5), and the share
that got a box at any confidence. Counts only; no photo is named.

```bash
python scripts/eval_detector.py --set store --runs n640 n640-seed1 n640-seed2
python experiments/m1-store-misses/analyse.py
```

## Result

Run on 2026-10-21. Percentages are means over the three seeds.

| Expiry dates | n | Found at threshold | Boxed at any confidence |
|---|---:|---:|---:|
| month as a word | 26 | **3%** | 44% |
| numbers only | 40 | **52%** | 84% |
| numbers only, print 2.5% and up | 13 | 87% | 92% |
| numbers only, print 1.5 to 2.5% | 19 | 46% | 91% |
| numbers only, print under 1.5% | 8 | 12% | 54% |
| month as a word, print 2.5% and up | 4 | 0% | 42% |
| month as a word, print 1.5 to 2.5% | 13 | 5% | 59% |
| month as a word, print under 1.5% | 9 | 0% | 22% |

- **A date with the month as a word is almost never found** (3%), even
  when it's large. ExpDate's dates are mostly numbers: only about 6% of
  real-train date transcriptions contain a month name. So the detector
  learned what *numeric* dates look like. 26 of the 66 store expiries
  are written this way.
- **For numeric dates, size decides.** 87% are found when the print is
  2.5% of the long side or more, about ExpDate's own size, close to
  ExpDate's 97%. Under 1.5% it's 12%.
- **Most misses are "seen but not sure", not "not seen".** For numeric
  dates from 1.5% up, about 90% get a box, but many fall under the
  threshold. The detector half-recognises them.

## Decision

Nothing in M1 changes: the model, input size and threshold were fixed on
validation, and the store photos only judge them. The two causes go to
M2, both to be measured on the store photos by tag:

- **Month-name formats** in the synthetic generator (`Month YYYY`,
  `MON YYYY`, `DD Month YYYY`, with ordinals like `19th`), since ExpDate
  barely has them.
- **Small print:** n1280 (or tiled inference) as an arm of the M2
  ablation, decided now, before any n1280 store result exists. Only 0.5%
  of store date boxes are under 8 px at 1280, against 17.6% at 640.
