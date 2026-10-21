# M1: the date detector

A small YOLO (`yolo11n`, 640 px input) fine-tuned to find every printed
date on a pack, trained on ExpDate's Products-Synth plus Products-Real
train. Everything was chosen on 165 validation photos; the ExpDate test
split and the store photos were each scored once, afterwards. M1 is judged
only on **finding** dates; reading them is M3's job. Plan and reasoning:
[`docs/m1-date-detector.md`](../docs/m1-date-detector.md).

## Headline

A date counts as found when a box at or above the run's threshold overlaps
it with IoU 0.5 or more. `n640` is the M1 detector (seed 0); the
means are over three training seeds, with their range.

|  | ExpDate test | store photos |
|---|---:|---:|
| expiry dates found, `n640` | 637/665 | 18/81 |
| expiry dates found, mean of 3 seeds | 0.971 (range 0.958 to 0.979) | 0.267 (range 0.222 to 0.309) |
| all dates found, `n640` | 697/733 | 38/199 |
| all dates found, mean of 3 seeds | 0.965 (range 0.951 to 0.973) | 0.230 (range 0.191 to 0.251) |
| precision, `n640` | 0.987 | 0.950 |
| mAP@0.5 for dates, mean of 3 seeds | 0.993 (range 0.992 to 0.993) | not computed (no YOLO labels for the store set) |

**The gap is the result.** On ExpDate, the kind of photo it trained on, the
detector finds almost every expiry date. On the retailer's store photos it
finds about a quarter. Sideways and small print are 0 of 15 each, and dates
with the month written as a word are almost never found
([`experiments/m1-store-misses`](../experiments/m1-store-misses/README.md)).
Closing that gap is M2's job.

## How the detector was chosen (validation, 187 dates)

**C1, model size and input size.** Rule fixed before the runs: the
highest mAP@0.5 for dates; within 1 point of the best, the cheapest run.
All four found 185 to 187 of 187 dates, so the rule took `n640`
([`experiments/m1-size-and-input`](../experiments/m1-size-and-input/README.md)).

| run | model | input | training photos | mAP@0.5 date | mAP@0.5:0.95 date | threshold | date recall | precision | false boxes / photo | recall at IoU 0.3 | epochs | minutes | recall cross-check |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n640 | yolo11n | 640 | synthetic + real | 0.991 | 0.895 | 0.728 | 185/187 | 0.974 | 0.03 | 185/187 | 100 | 65.5 | ok |
| n1280 | yolo11n | 1280 | synthetic + real | 0.994 | 0.912 | 0.553 | 187/187 | 0.979 | 0.02 | 187/187 | 96 | 96.5 | ok |
| s640 | yolo11s | 640 | synthetic + real | 0.995 | 0.912 | 0.787 | 186/187 | 0.995 | 0.01 | 186/187 | 100 | 61.6 | ok |
| s1280 | yolo11s | 1280 | synthetic + real | 0.995 | 0.922 | 0.633 | 186/187 | 1.000 | 0.00 | 186/187 | 87 | 102.3 | ok |

**C2, synthetic data.** Rule: keep the synthetic photos unless real only
is better by more than 1 point
([`experiments/m1-synthetic-data`](../experiments/m1-synthetic-data/README.md)).
The two runs also got different optimisers from Ultralytics'
`optimizer=auto`, so part of the gap may be the optimiser.

| run | model | input | training photos | mAP@0.5 date | mAP@0.5:0.95 date | threshold | date recall | precision | false boxes / photo | recall at IoU 0.3 | epochs | minutes | recall cross-check |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n640 | yolo11n | 640 | synthetic + real | 0.991 | 0.895 | 0.728 | 185/187 | 0.974 | 0.03 | 185/187 | 100 | 65.5 | ok |
| n640-real | yolo11n | 640 | real only | 0.981 | 0.865 | 0.682 | 179/187 | 0.984 | 0.02 | 180/187 | 100 | 6.3 | ok |

Decision: keep the synthetic photos.

**The threshold.** Each run keeps the confidence that gave the best F1 for
dates on validation: `n640` 0.728, `n640-seed1` 0.667, `n640-seed2` 0.664. F1 weighs a missed date and a false box
equally; for this project a miss costs more, so the next detector should
pick its threshold for recall, on validation (plan, Later).

## ExpDate Products-Real test (D2)

665 photos, 733 dates, 665 of them the expiry.

| run | threshold | mAP@0.5 date | dates found | precision | false boxes / photo | dates found, IoU 0.3 | expiry found | expiry found, IoU 0.3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n640 | 0.728 | 0.993 | 697/733 | 0.987 | 0.01 | 698/733 | 637/665 | 638/665 |
| n640-seed1 | 0.667 | 0.993 | 713/733 | 0.983 | 0.02 | 715/733 | 651/665 | 653/665 |
| n640-seed2 | 0.664 | 0.992 | 712/733 | 0.974 | 0.03 | 714/733 | 650/665 | 652/665 |

Mean over 3 runs:
- mAP@0.5 date: 0.993 (range 0.992 to 0.993)
- date recall: 0.965 (range 0.951 to 0.973)
- expiry recall: 0.971 (range 0.958 to 0.979)

Our date recall and Ultralytics' own, at the same threshold, differ by at most 0.003 over the 3 runs. In D2, `n640` put a box on every test expiry date
at some confidence: the ones it missed were under its threshold (plan, D2).

## Store photos, by failure tag (D3)

The frozen store test set v1 (`7b9a4043886c…`): 99 photos,
199 dates, 81 of them a printed expiry. Six more expiries are worked out
from a mfg date and shelf life, so they have no box to find.

| run | threshold | mAP@0.5 date | dates found | precision | false boxes / photo | dates found, IoU 0.3 | expiry found | expiry found, IoU 0.3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n640 | 0.728 | - | 38/199 | 0.950 | 0.02 | 39/199 | 18/81 | 19/81 |
| n640-seed1 | 0.667 | - | 50/199 | 0.926 | 0.04 | 51/199 | 22/81 | 23/81 |
| n640-seed2 | 0.664 | - | 49/199 | 0.942 | 0.03 | 50/199 | 25/81 | 26/81 |

Mean over 3 runs:
- date recall: 0.230 (range 0.191 to 0.251)
- expiry recall: 0.267 (range 0.222 to 0.309)

| tag | photos | expiry dates | n640 | n640-seed1 | n640-seed2 | expiry recall, mean | at IoU 0.3 | boxed at any confidence | all dates found | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| other-face | 12 | 0 | - | - | - | - | - | - | 0.143 (range 0.000 to 0.286) | 0.028 (range 0.000 to 0.083) |
| sideways | 16 | 15 | 0/15 | 0/15 | 0/15 | 0.000 (range 0.000 to 0.000) | 0.000 (range 0.000 to 0.000) | 0.000 (range 0.000 to 0.000) | 0.000 (range 0.000 to 0.000) | 0.000 (range 0.000 to 0.000) |
| dot-matrix | 21 | 20 | 12/20 | 15/20 | 17/20 | 0.733 (range 0.600 to 0.850) | 0.767 (range 0.650 to 0.850) | 0.933 (range 0.900 to 0.950) | 0.565 (range 0.413 to 0.652) | 0.032 (range 0.000 to 0.048) |
| small-print | 17 | 15 | 0/15 | 0/15 | 0/15 | 0.000 (range 0.000 to 0.000) | 0.000 (range 0.000 to 0.000) | 0.356 (range 0.267 to 0.533) | 0.000 (range 0.000 to 0.000) | 0.000 (range 0.000 to 0.000) |
| clean | 33 | 31 | 6/31 | 7/31 | 8/31 | 0.226 (range 0.194 to 0.258) | 0.237 (range 0.194 to 0.290) | 0.677 (range 0.613 to 0.742) | 0.271 (range 0.246 to 0.304) | 0.061 (range 0.030 to 0.091) |
| **all 99 photos** | 99 | 81 | 18/81 | 22/81 | 25/81 | 0.267 (range 0.222 to 0.309) | 0.280 (range 0.235 to 0.321) | 0.556 (range 0.518 to 0.617) | 0.230 (range 0.191 to 0.251) | 0.030 (range 0.020 to 0.040) |

"Boxed at any confidence" counts expiry dates that got any box at all,
however unsure: it separates "never seen" from "seen, but under the
threshold". It changes no headline number. Every photo is padded to the
same 640 px square (`rect=False`), as on validation, so the thresholds mean
the same thing here.

## Compared with the ExpDate paper

No comparable number. Seker & Ahn (2022) report about 97.74% accuracy for
recognising expiry dates: that is reading them end to end, not finding
them, so it belongs beside M3's reader, not this detector. No public
source gives their detection recall or mAP on Products-Real test.

## Runs

Made by `python scripts/m1_report.py` from:

- `n640` on ExpDate test: run `20261015-211752-m1-n640`, scored 2026-10-16 on the cpu at `b3c62ac`, threshold 0.7279.
- `n640-seed1` on ExpDate test: run `20261016-025220-m1-n640-seed1`, scored 2026-10-16 on the cpu at `b3c62ac`, threshold 0.6670.
- `n640-seed2` on ExpDate test: run `20261016-035339-m1-n640-seed2`, scored 2026-10-16 on the cpu at `b3c62ac`, threshold 0.6643.
- `n640` on the store photos: run `20261015-211752-m1-n640`, scored 2026-10-21 on the cpu at `d0746f4` + uncommitted changes, threshold 0.7279.
- `n640-seed1` on the store photos: run `20261016-025220-m1-n640-seed1`, scored 2026-10-21 on the cpu at `d0746f4` + uncommitted changes, threshold 0.6670.
- `n640-seed2` on the store photos: run `20261016-035339-m1-n640-seed2`, scored 2026-10-21 on the cpu at `d0746f4` + uncommitted changes, threshold 0.6643.
