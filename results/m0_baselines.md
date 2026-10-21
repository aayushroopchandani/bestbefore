# M0 baselines

The two readers every later model has to beat, scored the same way
(`bestbefore.scoring`): **right** is the exact day, month and year, or
month and year when the pack prints no day; **wrong** is any other date;
**none** is no date. Wrong is worse than none: a wrong date can mark a
good batch as expired.

- **A:** RapidOCR 1.2.3 on the whole upright photo, then a date regex,
  then the latest date found.
- **B:** Scanify's reader (the same RapidOCR plus resizing, sharpening
  and hand-written rules), run in its own environment.

## ExpDate Products-Real test

|  | n | A right | A wrong | A none | B right | B wrong | B none |
|---|---:|---:|---:|---:|---:|---:|---:|
| photos with a marked expiry | 655 | 502/655 | 28/655 | 125/655 | 57/655 | 6/655 | 592/655 |

Not scored: 10 photos with no marked expiry (A read a date from 2, B from 0).

## Store photos (test set v1), by failure tag

|  | n | A right | A wrong | A none | B right | B wrong | B none |
|---|---:|---:|---:|---:|---:|---:|---:|
| sideways | 16 | 6/16 | 5/16 | 5/16 | 9/16 | 0/16 | 7/16 |
| curved | 0 | - | - | - | - | - | - |
| dot-matrix | 21 | 7/21 | 0/21 | 14/21 | 7/21 | 0/21 | 14/21 |
| small-print | 17 | 14/17 | 2/17 | 1/17 | 16/17 | 0/17 | 1/17 |
| clean | 33 | 22/33 | 5/33 | 6/33 | 28/33 | 2/33 | 3/33 |
| **all readable** | 87 | 49/87 | 12/87 | 26/87 | 60/87 | 2/87 | 25/87 |

`other-face` photos (expiry not in the photo, so any date read is the wrong one): A returned a date for 4/12, B for 0/12.

No store photo is tagged `curved`: no date on a cap or jar visibly bends its
line. Tags and counts: [`store_test_set.md`](store_test_set.md).

## Runs

Made by `python scripts/baseline_table.py` from:

- A on real-test: run `20261021-210852-baseline-a-real-test`, this repo at `43a0f4c + uncommitted changes`.
- B on real-test: run `20261021-212745-baseline-b-real-test`, this repo at `5892900 + uncommitted changes`, Scanify at `9e05c00`.
- A on store: run `20261021-211420-baseline-a-store`, this repo at `43a0f4c + uncommitted changes`.
- B on store: run `20261021-212627-baseline-b-store`, this repo at `5892900 + uncommitted changes`, Scanify at `9e05c00`; on Scanify's own 44 photos: 32 right, 0 wrong, 12 none (reproduces Scanify's 32 / 0 / 12).
