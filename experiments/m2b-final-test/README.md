# M2b / the final test: store set v2, once, and ExpDate test

## Question

Does the M2b detector, `n1280-ft-s0`, find more of store set v2's expiry
dates than M1's and M2's detectors, on photos no model has trained or been
tuned on? Where does it, and where doesn't it: by failure tag, by cause,
and by whether a photo's pack design was trained on? Did fine-tuning on
store photos teach the model print, or only the packs it saw? And did it
keep ExpDate, which M1 and M2 had solved?

## Why it matters

Store set v1 became training and validation data in M2b, so it can no
longer judge a model. v2, 99 new photos from the same stores, was frozen
before any store training (labels 2026-11-03, designs the same day) and
kept off the cluster, so no training job could reach it (the plan,
decision 3). It is scored once, at the end, after every rule had chosen
(R1 to R4): nothing it shows can change a rule, a threshold, a run or the
detector. v1-val chose the detector, and v1-val's designs gave the gaps
the generator's new layouts fill, so v2 is the first blind test of M2b.

## Method

- **What is scored: 30 records from 27 prediction passes**, each at the
  threshold its own GPU record holds (`final_test.RUNS`, fixed in the
  plan's S14 note before any score):

  | Records | Weights | Threshold, from |
  |---|---|---|
  | M1's `n640` s0–s2 | `best.pt` | `val.json` (M1's best F1 on real-val) |
  | M2's `gen-rot-1280` s0–s2 | `last.pt` | `suite.json` (M2's best F2, the real sets pooled) |
  | the control, `n1280` s0–s2 | M2's `last.pt` again | `suite-m2b.json`'s threshold_store (best F2 on v1-val) |
  | `s1280`, `m1280` s0–s2 (step 1) | `last.pt` | the same |
  | `n1280-ft`, `n1280-ft2`, `n1280-ft-gen50k`, `s1280-ft`, `m1280-ft` s0–s2 | `last.pt` | the same |

  The M2b detector is `n1280-ft-s0` at 0.1641, alone: R4 adopted no
  add-on. M2's runs and the control share their weights, so each pair is
  predicted once and cut at both thresholds.
- **How**: `scripts/eval_detector.py --set store-v2` on the laptop's CPU
  (Apple M2), one photo a call (`final_test.BATCH`; 16 a call at 1,280 px
  overflowed the 8 GB laptop), `rect=False` as every M2b score, in 12
  parts with 4 minutes' rest after each. Every weights file is checked
  against its SHA-256 before it predicts (`m1.BEST_PT_SHA256`,
  `m2.LAST_PT_SHA256`, `m2b.LAST_PT_SHA256`). A run is scored once:
  `--again` is refused for v2, and so is a run with a record or kept boxes
  already. Each record holds counts only; the per-photo boxes stay in
  `data/store/v2/runs/`.
- **The device checks first**: `n1280-ft-s0`, `s1280-ft-s0` and
  `m1280-ft-s0` on v1-val on the laptop's CPU, at 16 photos a call and at
  one, against each run's GPU suite: within a date or two, or v2 isn't
  scored.
- **Counted as M2 counted v1**: an expiry in frame is an expiry box (68 on
  v2); found means a box of the run's at IoU 0.5 or more, at its
  threshold. An `other-face` photo (24) has no expiry; its dates (11)
  count among all dates (212), and every box on it toward precision and
  false boxes a photo. A derived expiry (8) has no box and isn't counted.
  The one unreadable expiry has its box and format, so it counts as one to
  find: finding it doesn't need reading it.
- **By failure tag** (M0's), **by cause** and **by pack design**:
  - causes are M2's (`m2_results.cause_of`) with two changes made before
    any score: its month-word groups are one, as v2's labels write every
    month word as `MON`; and a sideways-tagged expiry whose box isn't
    taller than wide is upside-down (S3's rule). Month word 25, numbers
    under 1.5% of the photo's long side 6, 1.5 to 2.5% 15, 2.5% and up 12,
    sideways 9, upside-down 1;
  - designs: **seen in training** (a design with photos in v1-train or
    v3-train), **seen only in validation**, **unseen**: 4, 7 and 57
    expiries, 21, 23 and 168 dates. Two groups are tiny, so counts sit
    beside every rate. For a run never trained on store photos the groups
    show how hard each is without store training.
- **Our AP@0.5 for dates** (`detection_eval.average_precision`, S13), as
  no threshold moves it: the thresholds run from 0.0025 to 0.69, so a
  count at the threshold is partly the threshold's.
- **"Learned packs, not print"**, read as fixed before any score: each
  fine-tuned arm against the control on v2's unseen designs, by AP. It
  lifts them if the seed ranges are apart, every fine-tuned seed above
  every control seed; not if they overlap.
- **ExpDate test**: the M2b detector's 3 seeds at their threshold_store,
  with the counts at M2's rule's threshold from the same pass, beside M1's
  and M2's own records (M2 S8).

## Result

Rebuild with:

```bash
python experiments/m2b-final-test/summarise.py --write
```

<!-- results:start -->
**The headline**: each milestone's detector, seed 0 at its own threshold, on store set v2 and on ExpDate test; its arm's three seeds beside it:

| detector | v2: expiry found | v2: its arm's 3 seeds | v2: precision | v2: false boxes / photo | v2: AP (ours) | ExpDate test: expiry found | ExpDate test: its arm's 3 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`), seed 0 | 20/68, 29% | 25/68, 37% (20–30) | 0.956 | 0.02 | 0.440 | 637/665, 96% | 646/665, 97% (637–651) |
| M2 (`gen-rot-1280-s0`) | 46/68, 68% | 48.7/68, 72% (46–51) | 0.871 | 0.17 | 0.703 | 662/665, 100% | 662.7/665, 100% (662–663) |
| M2b (`n1280-ft-s0`) | 65/68, 96% | 64/68, 94% (62–65) | 0.761 | 0.60 | 0.895 | not yet | not yet |

**The device checks** (v1-val on the laptop's CPU against each run's GPU suite, the GPU's count first; v2 is predicted on that CPU one photo a call):

| run | photos a call | dates found | expiry found | boxed at any confidence | mAP@0.5 date | moved | check |
|---|---:|---:|---:|---:|---:|---:|---|
| m2b-n1280-ft-s0 | 16 | 60/63 → 60/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9514 → 0.9514 | 0 | passes |
| m2b-n1280-ft-s0 | 1 | 60/63 → 60/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9514 → 0.9511 | 0 | passes |
| m2b-s1280-ft-s0 | 16 | 62/63 → 62/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9721 → 0.9709 | 0 | passes |
| m2b-s1280-ft-s0 | 1 | 62/63 → 62/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9721 → 0.9705 | 0 | passes |
| m2b-m1280-ft-s0 | 16 | 60/63 → 60/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9864 → 0.9866 | 0 | passes |
| m2b-m1280-ft-s0 | 1 | 60/63 → 60/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9864 → 0.9867 | 0 | passes |

**Store set v2, every record** (M1's and M2's detectors at their own thresholds; every M2b run at threshold_store, the control being M2's weights cut there):

| detector | seeds | threshold | expiry found | expiry at IoU 0.3 | expiry boxed at any confidence | all dates found | precision | false boxes / photo | AP@0.5 date (ours) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0.6864 (0.6643–0.7279) | 25/68, 37% (20–30) | 25/68, 37% (20–30) | 48/68, 71% (46–50) | 52.3/212, 25% (43–62) | 0.957 (0.946–0.969) | 0.02 (0.02–0.03) | 0.438 (0.417–0.457) |
| M2 (`gen-rot-1280`) at M2's threshold | 3/3 | 0.3885 (0.3424–0.4243) | 48.7/68, 72% (46–51) | 48.7/68, 72% (46–51) | 61.3/68, 90% (60–64) | 121.7/212, 57% (115–127) | 0.903 (0.871–0.934) | 0.13 (0.09–0.17) | 0.733 (0.703–0.769) |
| n1280 (control: M2's weights at threshold_store) | 3/3 | 0.0191 (0.0055–0.0387) | 58.3/68, 86% (58–59) | 58.3/68, 86% (58–59) | 61.3/68, 90% (60–64) | 161/212, 76% (159–163) | 0.550 (0.418–0.710) | 1.48 (0.66–2.29) | 0.733 (0.703–0.769) |
| s1280 | 3/3 | 0.0089 (0.0028–0.0121) | 57.7/68, 85% (55–59) | 57.7/68, 85% (55–59) | 59/68, 87% (56–61) | 170/212, 80% (165–175) | 0.672 (0.494–0.780) | 0.96 (0.48–1.81) | 0.794 (0.792–0.795) |
| m1280 | 3/3 | 0.0054 (0.0025–0.0073) | 59/68, 87% (58–61) | 59/68, 87% (58–61) | 59.7/68, 88% (58–61) | 174.7/212, 82% (172–178) | 0.721 (0.662–0.784) | 0.70 (0.48–0.92) | 0.820 (0.816–0.827) |
| n1280-ft (the M2b detector's arm) | 3/3 | 0.1711 (0.1280–0.2211) | 64/68, 94% (62–65) | 64/68, 94% (62–65) | 67/68, 99% | 186.3/212, 88% (180–191) | 0.774 (0.758–0.804) | 0.55 (0.44–0.62) | 0.897 (0.889–0.906) |
| n1280-ft2 | 3/3 | 0.2187 (0.2062–0.2403) | 63.7/68, 94% (63–64) | 63.7/68, 94% (63–64) | 67/68, 99% | 185.7/212, 88% (183–190) | 0.770 (0.748–0.807) | 0.56 (0.44–0.65) | 0.901 (0.898–0.904) |
| n1280-ft-gen50k | 3/3 | 0.3406 (0.1861–0.6168) | 60/68, 88% (57–62) | 60.3/68, 89% (57–63) | 66/68, 97% | 172.3/212, 81% (158–183) | 0.834 (0.786–0.913) | 0.36 (0.15–0.48) | 0.880 (0.875–0.883) |
| s1280-ft | 3/3 | 0.2960 (0.1276–0.5527) | 63/68, 93% (62–64) | 63/68, 93% (62–64) | 65.7/68, 97% (65–66) | 189.3/212, 89% (184–196) | 0.884 (0.827–0.939) | 0.26 (0.12–0.41) | 0.930 (0.921–0.938) |
| m1280-ft | 3/3 | 0.6514 (0.5841–0.6866) | 60.7/68, 89% (58–63) | 61/68, 90% (59–63) | 66.7/68, 98% (66–67) | 187/212, 88% (183–189) | 0.971 (0.958–0.984) | 0.06 (0.03–0.08) | 0.948 (0.945–0.950) |

**By failure tag**, expiry dates found at each run's threshold:

| detector | seeds | sideways | curved | dot-matrix | small-print | clean |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/10, 0% | 0/1, 0% | 16.7/28, 60% (14–19) | 0.7/5, 13% (0–1) | 7.7/24, 32% (6–10) |
| M2 (`gen-rot-1280`) at M2's threshold | 3/3 | 6.3/10, 63% (5–8) | 0/1, 0% | 26.7/28, 95% (26–27) | 3/5, 60% | 12.7/24, 53% (10–17) |
| n1280 (control: M2's weights at threshold_store) | 3/3 | 8.3/10, 83% (8–9) | 0/1, 0% | 27.3/28, 98% (27–28) | 4.3/5, 87% (4–5) | 18.3/24, 76% (17–20) |
| s1280 | 3/3 | 9.3/10, 93% (9–10) | 0/1, 0% | 27/28, 96% | 4/5, 80% | 17.3/24, 72% (15–19) |
| m1280 | 3/3 | 9/10, 90% | 0/1, 0% | 27/28, 96% | 3.7/5, 73% (3–4) | 19.3/24, 81% (18–21) |
| n1280-ft (the M2b detector's arm) | 3/3 | 9.3/10, 93% (9–10) | 1/1, 100% | 28/28, 100% | 5/5, 100% | 20.7/24, 86% (19–22) |
| n1280-ft2 | 3/3 | 8.7/10, 87% (8–10) | 1/1, 100% | 28/28, 100% | 5/5, 100% | 21/24, 88% (19–22) |
| n1280-ft-gen50k | 3/3 | 7.7/10, 77% (7–8) | 0.7/1, 67% (0–1) | 27.7/28, 99% (27–28) | 4.7/5, 93% (4–5) | 19.3/24, 81% (18–21) |
| s1280-ft | 3/3 | 9.7/10, 97% (9–10) | 1/1, 100% | 27.3/28, 98% (27–28) | 4.7/5, 93% (4–5) | 20.3/24, 85% (19–21) |
| m1280-ft | 3/3 | 10/10, 100% | 0.3/1, 33% (0–1) | 25.7/28, 92% (24–27) | 4/5, 80% | 20.7/24, 86% (20–22) |

boxed at any confidence:

| detector | seeds | sideways | curved | dot-matrix | small-print | clean |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/10, 0% | 0.3/1, 33% (0–1) | 27/28, 96% | 4/5, 80% | 16.7/24, 69% (15–18) |
| M2 (`gen-rot-1280`) at M2's threshold | 3/3 | 9/10, 90% | 0.3/1, 33% (0–1) | 28/28, 100% | 4.7/5, 93% (4–5) | 19.3/24, 81% (18–22) |
| n1280 (control: M2's weights at threshold_store) | 3/3 | 9/10, 90% | 0.3/1, 33% (0–1) | 28/28, 100% | 4.7/5, 93% (4–5) | 19.3/24, 81% (18–22) |
| s1280 | 3/3 | 9.3/10, 93% (9–10) | 0/1, 0% | 27/28, 96% | 4/5, 80% | 18.7/24, 78% (16–20) |
| m1280 | 3/3 | 9/10, 90% | 0/1, 0% | 27/28, 96% | 4/5, 80% | 19.7/24, 82% (18–21) |
| n1280-ft (the M2b detector's arm) | 3/3 | 10/10, 100% | 1/1, 100% | 28/28, 100% | 5/5, 100% | 23/24, 96% |
| n1280-ft2 | 3/3 | 10/10, 100% | 1/1, 100% | 28/28, 100% | 5/5, 100% | 23/24, 96% |
| n1280-ft-gen50k | 3/3 | 10/10, 100% | 1/1, 100% | 28/28, 100% | 5/5, 100% | 22/24, 92% |
| s1280-ft | 3/3 | 10/10, 100% | 1/1, 100% | 28/28, 100% | 5/5, 100% | 21.7/24, 90% (21–22) |
| m1280-ft | 3/3 | 10/10, 100% | 1/1, 100% | 27.7/28, 99% (27–28) | 5/5, 100% | 23/24, 96% |

precision (every box on the tag's photos against all their dates):

| detector | other-face | sideways | curved | dot-matrix | small-print | clean |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 0.667 (0.500–1.000) | – | – | 1.000 | 0.778 (0.667–1.000) | 0.935 (0.875–1.000) |
| M2 (`gen-rot-1280`) at M2's threshold | 0.568 (0.364–0.714) | 0.886 (0.875–0.895) | – | 0.934 (0.910–0.966) | 0.921 (0.875–1.000) | 0.950 (0.929–0.967) |
| n1280 (control: M2's weights at threshold_store) | 0.219 (0.123–0.333) | 0.587 (0.543–0.639) | – | 0.618 (0.476–0.759) | 0.403 (0.294–0.600) | 0.620 (0.480–0.809) |
| s1280 | 0.256 (0.119–0.333) | 0.767 (0.700–0.844) | – | 0.712 (0.566–0.793) | 0.671 (0.394–0.833) | 0.751 (0.581–0.857) |
| m1280 | 0.264 (0.233–0.294) | 0.792 (0.721–0.875) | 1.000 | 0.829 (0.780–0.861) | 0.631 (0.556–0.706) | 0.717 (0.589–0.803) |
| n1280-ft (the M2b detector's arm) | 0.583 (0.500–0.750) | 0.719 (0.674–0.767) | 0.833 (0.750–1.000) | 0.765 (0.748–0.789) | 0.793 (0.650–0.929) | 0.860 (0.824–0.887) |
| n1280-ft2 | 0.494 (0.467–0.545) | 0.712 (0.660–0.775) | 0.917 (0.750–1.000) | 0.764 (0.730–0.832) | 0.858 (0.765–0.933) | 0.851 (0.845–0.857) |
| n1280-ft-gen50k | 0.626 (0.615–0.636) | 0.753 (0.634–0.880) | 1.000 | 0.814 (0.755–0.909) | 0.970 (0.909–1.000) | 0.919 (0.873–0.980) |
| s1280-ft | 0.677 (0.615–0.778) | 0.796 (0.729–0.842) | 1.000 | 0.886 (0.826–0.962) | 0.922 (0.833–1.000) | 0.968 (0.923–1.000) |
| m1280-ft | 0.875 (0.750–1.000) | 0.955 (0.921–0.972) | 1.000 | 0.969 (0.947–0.986) | 1.000 | 0.989 (0.983–1.000) |

false boxes a photo:

| detector | other-face | sideways | curved | dot-matrix | small-print | clean |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 0.03 (0.00–0.04) | 0.00 | 0.00 | 0.00 | 0.13 (0.00–0.20) | 0.04 (0.00–0.07) |
| M2 (`gen-rot-1280`) at M2's threshold | 0.17 (0.08–0.29) | 0.18 (0.15–0.23) | 0.00 | 0.15 (0.07–0.21) | 0.13 (0.00–0.20) | 0.06 (0.04–0.07) |
| n1280 (control: M2's weights at threshold_store) | 1.19 (0.50–2.08) | 1.36 (1.00–1.62) | 0.00 | 1.62 (0.72–2.66) | 3.73 (1.20–5.20) | 1.27 (0.48–1.96) |
| s1280 | 1.10 (0.54–2.17) | 0.64 (0.38–0.92) | 0.00 | 1.10 (0.66–1.93) | 1.67 (0.40–4.00) | 0.73 (0.33–1.44) |
| m1280 | 0.68 (0.50–0.96) | 0.64 (0.31–0.92) | 0.00 | 0.52 (0.41–0.69) | 1.33 (1.00–1.60) | 0.84 (0.52–1.37) |
| n1280-ft (the M2b detector's arm) | 0.24 (0.08–0.38) | 0.92 (0.77–1.08) | 0.67 (0.00–1.00) | 0.79 (0.69–0.86) | 0.73 (0.20–1.40) | 0.36 (0.26–0.48) |
| n1280-ft2 | 0.31 (0.21–0.38) | 0.92 (0.69–1.23) | 0.33 (0.00–1.00) | 0.80 (0.52–0.97) | 0.47 (0.20–0.80) | 0.38 (0.37–0.41) |
| n1280-ft-gen50k | 0.17 (0.12–0.21) | 0.72 (0.23–1.15) | 0.00 | 0.60 (0.24–0.83) | 0.07 (0.00–0.20) | 0.19 (0.04–0.30) |
| s1280-ft | 0.15 (0.08–0.21) | 0.67 (0.46–1.00) | 0.00 | 0.34 (0.10–0.55) | 0.27 (0.00–0.60) | 0.07 (0.00–0.19) |
| m1280-ft | 0.04 (0.00–0.08) | 0.13 (0.08–0.23) | 0.00 | 0.08 (0.03–0.14) | 0.00 | 0.02 (0.00–0.04) |

our AP@0.5 for dates:

| detector | other-face | sideways | curved | dot-matrix | small-print | clean |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 0.141 (0.133–0.155) | 0.010 (0.000–0.025) | 0.211 (0.000–0.467) | 0.656 (0.644–0.663) | 0.273 (0.200–0.311) | 0.527 (0.479–0.571) |
| M2 (`gen-rot-1280`) at M2's threshold | 0.426 (0.341–0.469) | 0.636 (0.611–0.661) | 0.297 (0.000–0.555) | 0.863 (0.856–0.871) | 0.688 (0.673–0.716) | 0.750 (0.688–0.855) |
| n1280 (control: M2's weights at threshold_store) | 0.426 (0.341–0.469) | 0.636 (0.611–0.661) | 0.297 (0.000–0.555) | 0.863 (0.856–0.871) | 0.688 (0.673–0.716) | 0.750 (0.688–0.855) |
| s1280 | 0.467 (0.404–0.517) | 0.732 (0.716–0.754) | 0.000 | 0.912 (0.908–0.919) | 0.743 (0.659–0.816) | 0.809 (0.770–0.830) |
| m1280 | 0.390 (0.379–0.411) | 0.830 (0.795–0.862) | 0.223 (0.000–0.335) | 0.918 (0.904–0.938) | 0.782 (0.712–0.828) | 0.817 (0.766–0.873) |
| n1280-ft (the M2b detector's arm) | 0.621 (0.581–0.695) | 0.839 (0.826–0.852) | 0.995 | 0.940 (0.933–0.948) | 0.919 (0.891–0.939) | 0.920 (0.907–0.931) |
| n1280-ft2 | 0.586 (0.549–0.620) | 0.849 (0.844–0.856) | 0.940 (0.913–0.995) | 0.944 (0.930–0.955) | 0.922 (0.870–0.949) | 0.921 (0.913–0.925) |
| n1280-ft-gen50k | 0.615 (0.575–0.642) | 0.799 (0.785–0.819) | 0.995 | 0.930 (0.929–0.932) | 0.895 (0.827–0.948) | 0.907 (0.899–0.911) |
| s1280-ft | 0.694 (0.690–0.699) | 0.893 (0.869–0.910) | 0.995 | 0.970 (0.958–0.987) | 0.961 (0.930–0.988) | 0.930 (0.914–0.947) |
| m1280-ft | 0.680 (0.629–0.737) | 0.941 (0.937–0.945) | 0.995 | 0.979 (0.977–0.982) | 0.941 (0.904–0.995) | 0.952 (0.950–0.954) |

**By cause** (M2's causes, its month-word groups as one and upside-down print apart; S14's note), expiry dates found at each run's threshold:

| detector | seeds | month word | numbers: print under 1.5% | numbers: print 1.5 to 2.5% | numbers: print 2.5% and up | sideways | upside-down |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 7.3/25, 29% (5–11) | 0.7/6, 11% (0–1) | 8/15, 53% (6–10) | 9/12, 75% (8–10) | 0/9, 0% | 0/1, 0% |
| M2 (`gen-rot-1280`) at M2's threshold | 3/3 | 14/25, 56% (11–18) | 4.7/6, 78% (4–5) | 11.7/15, 78% (11–12) | 12/12, 100% | 6.3/9, 70% (5–8) | 0/1, 0% |
| n1280 (control: M2's weights at threshold_store) | 3/3 | 19/25, 76% (18–20) | 6/6, 100% | 13/15, 87% | 12/12, 100% | 8.3/9, 93% (8–9) | 0/1, 0% |
| s1280 | 3/3 | 17.7/25, 71% (15–19) | 6/6, 100% | 12.7/15, 84% (12–13) | 12/12, 100% | 9/9, 100% | 0.3/1, 33% (0–1) |
| m1280 | 3/3 | 18/25, 72% (17–20) | 6/6, 100% | 14/15, 93% | 12/12, 100% | 9/9, 100% | 0/1, 0% |
| n1280-ft (the M2b detector's arm) | 3/3 | 24/25, 96% (23–25) | 6/6, 100% | 13/15, 87% | 11.7/12, 97% (11–12) | 8.3/9, 93% (8–9) | 1/1, 100% |
| n1280-ft2 | 3/3 | 24.3/25, 97% (23–25) | 6/6, 100% | 12.7/15, 84% (12–13) | 12/12, 100% | 7.7/9, 85% (7–9) | 1/1, 100% |
| n1280-ft-gen50k | 3/3 | 22.7/25, 91% (21–24) | 5.3/6, 89% (4–6) | 12.7/15, 84% (12–13) | 11.7/12, 97% (11–12) | 7/9, 78% | 0.7/1, 67% (0–1) |
| s1280-ft | 3/3 | 22.7/25, 91% (22–23) | 6/6, 100% | 13/15, 87% | 11.7/12, 97% (11–12) | 8.7/9, 96% (8–9) | 1/1, 100% |
| m1280-ft | 3/3 | 20.3/25, 81% (19–21) | 5/6, 83% (4–6) | 14/15, 93% | 11.3/12, 94% (11–12) | 9/9, 100% | 1/1, 100% |

boxed at any confidence:

| detector | seeds | month word | numbers: print under 1.5% | numbers: print 1.5 to 2.5% | numbers: print 2.5% and up | sideways | upside-down |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 19.7/25, 79% (18–22) | 5/6, 83% | 12/15, 80% | 11.3/12, 94% (11–12) | 0/9, 0% | 0/1, 0% |
| M2 (`gen-rot-1280`) at M2's threshold | 3/3 | 21.3/25, 85% (20–24) | 6/6, 100% | 13/15, 87% | 12/12, 100% | 9/9, 100% | 0/1, 0% |
| n1280 (control: M2's weights at threshold_store) | 3/3 | 21.3/25, 85% (20–24) | 6/6, 100% | 13/15, 87% | 12/12, 100% | 9/9, 100% | 0/1, 0% |
| s1280 | 3/3 | 18/25, 72% (16–19) | 6/6, 100% | 13.7/15, 91% (13–14) | 12/12, 100% | 9/9, 100% | 0.3/1, 33% (0–1) |
| m1280 | 3/3 | 18.7/25, 75% (17–20) | 6/6, 100% | 14/15, 93% | 12/12, 100% | 9/9, 100% | 0/1, 0% |
| n1280-ft (the M2b detector's arm) | 3/3 | 25/25, 100% | 6/6, 100% | 14/15, 93% | 12/12, 100% | 9/9, 100% | 1/1, 100% |
| n1280-ft2 | 3/3 | 25/25, 100% | 6/6, 100% | 14/15, 93% | 12/12, 100% | 9/9, 100% | 1/1, 100% |
| n1280-ft-gen50k | 3/3 | 25/25, 100% | 6/6, 100% | 13/15, 87% | 12/12, 100% | 9/9, 100% | 1/1, 100% |
| s1280-ft | 3/3 | 24.3/25, 97% (24–25) | 6/6, 100% | 13.3/15, 89% (13–14) | 12/12, 100% | 9/9, 100% | 1/1, 100% |
| m1280-ft | 3/3 | 24.7/25, 99% (24–25) | 6/6, 100% | 14/15, 93% | 12/12, 100% | 9/9, 100% | 1/1, 100% |

**By pack design** (seen in training: a design in v1-train or v3-train; seen only in validation; unseen), expiry dates found at each run's threshold:

| detector | seeds | seen in training | seen only in validation | unseen |
|---|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 3/4, 75% | 2.7/7, 38% (1–5) | 19.3/57, 34% (16–22) |
| M2 (`gen-rot-1280`) at M2's threshold | 3/3 | 4/4, 100% | 6/7, 86% (5–7) | 38.7/57, 68% (37–40) |
| n1280 (control: M2's weights at threshold_store) | 3/3 | 4/4, 100% | 7/7, 100% | 47.3/57, 83% (47–48) |
| s1280 | 3/3 | 4/4, 100% | 7/7, 100% | 46.7/57, 82% (44–48) |
| m1280 | 3/3 | 4/4, 100% | 7/7, 100% | 48/57, 84% (47–50) |
| n1280-ft (the M2b detector's arm) | 3/3 | 4/4, 100% | 7/7, 100% | 53/57, 93% (51–54) |
| n1280-ft2 | 3/3 | 4/4, 100% | 7/7, 100% | 52.7/57, 92% (52–53) |
| n1280-ft-gen50k | 3/3 | 4/4, 100% | 7/7, 100% | 49/57, 86% (46–51) |
| s1280-ft | 3/3 | 4/4, 100% | 7/7, 100% | 52/57, 91% (51–53) |
| m1280-ft | 3/3 | 4/4, 100% | 7/7, 100% | 49.7/57, 87% (47–52) |

every date found:

| detector | seeds | seen in training | seen only in validation | unseen |
|---|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 5.3/21, 25% (5–6) | 6.7/23, 29% (3–11) | 40.3/168, 24% (35–45) |
| M2 (`gen-rot-1280`) at M2's threshold | 3/3 | 13/21, 62% (11–15) | 15/23, 65% (14–17) | 93.7/168, 56% (90–97) |
| n1280 (control: M2's weights at threshold_store) | 3/3 | 17.7/21, 84% (15–20) | 20.3/23, 88% (20–21) | 123/168, 73% (122–124) |
| s1280 | 3/3 | 18/21, 86% (17–19) | 21/23, 91% | 131/168, 78% (127–135) |
| m1280 | 3/3 | 18/21, 86% (16–20) | 21/23, 91% | 135.7/168, 81% (135–137) |
| n1280-ft (the M2b detector's arm) | 3/3 | 17/21, 81% (14–19) | 20.3/23, 88% (20–21) | 149/168, 89% (146–153) |
| n1280-ft2 | 3/3 | 17/21, 81% (16–18) | 20.7/23, 90% (20–21) | 148/168, 88% (144–153) |
| n1280-ft-gen50k | 3/3 | 16.3/21, 78% (15–18) | 20.7/23, 90% (20–21) | 135.3/168, 81% (122–145) |
| s1280-ft | 3/3 | 17/21, 81% (15–18) | 21/23, 91% | 151.3/168, 90% (148–157) |
| m1280-ft | 3/3 | 17.3/21, 83% (17–18) | 21/23, 91% | 148.7/168, 88% (145–151) |

precision:

| detector | seen in training | seen only in validation | unseen |
|---|---:|---:|---:|
| M1 (`n640`) | 1.000 | 0.952 (0.857–1.000) | 0.952 (0.946–0.957) |
| M2 (`gen-rot-1280`) at M2's threshold | 0.834 (0.789–0.867) | 0.818 (0.737–0.895) | 0.930 (0.900–0.951) |
| n1280 (control: M2's weights at threshold_store) | 0.547 (0.408–0.652) | 0.428 (0.227–0.690) | 0.587 (0.486–0.721) |
| s1280 | 0.599 (0.396–0.720) | 0.536 (0.368–0.656) | 0.715 (0.542–0.834) |
| m1280 | 0.653 (0.571–0.696) | 0.670 (0.583–0.750) | 0.741 (0.675–0.804) |
| n1280-ft (the M2b detector's arm) | 0.835 (0.818–0.864) | 0.637 (0.553–0.714) | 0.792 (0.769–0.816) |
| n1280-ft2 | 0.788 (0.739–0.842) | 0.648 (0.568–0.750) | 0.791 (0.769–0.812) |
| n1280-ft-gen50k | 0.902 (0.818–1.000) | 0.720 (0.606–0.875) | 0.848 (0.794–0.910) |
| s1280-ft | 0.939 (0.818–1.000) | 0.783 (0.724–0.875) | 0.895 (0.844–0.943) |
| m1280-ft | 0.981 (0.944–1.000) | 0.941 (0.913–0.955) | 0.974 (0.960–0.993) |

false boxes a photo:

| detector | seen in training | seen only in validation | unseen |
|---|---:|---:|---:|
| M1 (`n640`) | 0.00 | 0.02 (0.00–0.07) | 0.03 |
| M2 (`gen-rot-1280`) at M2's threshold | 0.17 (0.12–0.25) | 0.24 (0.14–0.36) | 0.10 (0.07–0.14) |
| n1280 (control: M2's weights at threshold_store) | 1.04 (0.50–1.81) | 2.69 (0.64–4.86) | 1.33 (0.70–1.88) |
| s1280 | 0.92 (0.44–1.81) | 1.48 (0.79–2.57) | 0.86 (0.38–1.65) |
| m1280 | 0.62 (0.44–0.94) | 0.76 (0.50–1.07) | 0.70 (0.48–0.96) |
| n1280-ft (the M2b detector's arm) | 0.21 (0.19–0.25) | 0.86 (0.57–1.21) | 0.57 (0.48–0.67) |
| n1280-ft2 | 0.29 (0.19–0.38) | 0.83 (0.50–1.14) | 0.57 (0.49–0.67) |
| n1280-ft-gen50k | 0.12 (0.00–0.25) | 0.62 (0.21–0.93) | 0.37 (0.17–0.52) |
| s1280-ft | 0.08 (0.00–0.25) | 0.43 (0.21–0.57) | 0.27 (0.13–0.42) |
| m1280-ft | 0.02 (0.00–0.06) | 0.10 (0.07–0.14) | 0.06 (0.01–0.09) |

our AP@0.5 for dates:

| detector | seen in training | seen only in validation | unseen |
|---|---:|---:|---:|
| M1 (`n640`) | 0.392 (0.375–0.403) | 0.603 (0.550–0.650) | 0.422 (0.402–0.436) |
| M2 (`gen-rot-1280`) at M2's threshold | 0.796 (0.765–0.820) | 0.783 (0.721–0.835) | 0.726 (0.695–0.767) |
| n1280 (control: M2's weights at threshold_store) | 0.796 (0.765–0.820) | 0.783 (0.721–0.835) | 0.726 (0.695–0.767) |
| s1280 | 0.805 (0.791–0.815) | 0.876 (0.853–0.899) | 0.783 (0.779–0.788) |
| m1280 | 0.831 (0.785–0.871) | 0.875 (0.866–0.893) | 0.812 (0.805–0.817) |
| n1280-ft (the M2b detector's arm) | 0.864 (0.780–0.925) | 0.865 (0.850–0.873) | 0.909 (0.905–0.915) |
| n1280-ft2 | 0.880 (0.860–0.920) | 0.884 (0.855–0.907) | 0.913 (0.909–0.920) |
| n1280-ft-gen50k | 0.866 (0.813–0.906) | 0.871 (0.825–0.901) | 0.890 (0.883–0.899) |
| s1280-ft | 0.909 (0.892–0.929) | 0.909 (0.907–0.911) | 0.940 (0.932–0.952) |
| m1280-ft | 0.912 (0.890–0.946) | 0.913 (0.911–0.915) | 0.960 (0.957–0.965) |

**Learned packs, not print?** Each fine-tuned arm against the control on v2's unseen designs, by our AP (read as fixed before any score: lifted only if the ranges are apart), beside v1-val:

| arm | v1-val mAP | v1-val, against the control | v2 unseen: AP (ours) | v2 unseen, against the control | v2 unseen: dates found | v2 unseen: expiry found |
|---|---:|---|---:|---|---:|---:|
| n1280 (control) | 0.720 (0.655–0.796) | – | 0.726 (0.695–0.767) | – | 123/168, 73% (122–124) | 47.3/57, 83% (47–48) |
| n1280-ft | 0.942 (0.936–0.951) | lifted: ranges apart | 0.909 (0.905–0.915) | lifted: ranges apart | 149/168, 89% (146–153) | 53/57, 93% (51–54) |
| n1280-ft2 | 0.937 (0.933–0.941) | lifted: ranges apart | 0.913 (0.909–0.920) | lifted: ranges apart | 148/168, 88% (144–153) | 52.7/57, 92% (52–53) |
| n1280-ft-gen50k | 0.926 (0.903–0.941) | lifted: ranges apart | 0.890 (0.883–0.899) | lifted: ranges apart | 135.3/168, 81% (122–145) | 49/57, 86% (46–51) |
| s1280-ft | 0.965 (0.961–0.972) | lifted: ranges apart | 0.940 (0.932–0.952) | lifted: ranges apart | 151.3/168, 90% (148–157) | 52/57, 91% (51–53) |
| m1280-ft | 0.987 (0.982–0.992) | lifted: ranges apart | 0.960 (0.957–0.965) | lifted: ranges apart | 148.7/168, 88% (145–151) | 49.7/57, 87% (47–52) |

**ExpDate test** (665 expiry dates, 733 dates; the M2b detector's arm once, M1's and M2's as their milestones scored them):

| detector | seeds | threshold | expiry found | expiry boxed at any confidence | all dates found | precision | false boxes / photo | mAP@0.5 date (Ultralytics) | AP@0.5 date (ours) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0.6864 (0.6643–0.7279) | 646/665, 97% (637–651) | – | 707.3/733, 96% (697–713) | 0.982 (0.974–0.987) | 0.02 (0.01–0.03) | 0.993 (0.992–0.993) | – |
| M2 (`gen-rot-1280`) at M2's threshold | 3/3 | 0.3885 (0.3424–0.4243) | 662.7/665, 100% (662–663) | 664.7/665, 100% (664–665) | 730.7/733, 100% (730–731) | 0.986 (0.981–0.989) | 0.02 (0.01–0.02) | 0.995 | – |
| M2b (`n1280-ft`) at threshold_store | 0/3 | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet |
| M2b (`n1280-ft`) at M2's rule's threshold | 0/3 | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet |

Every record in so far (30 of 30 on store set v2, 0 of 3 on ExpDate test) was scored on a CPU, at its own GPU record's threshold exactly, by the weights hashed for it; on v2, one photo a call, and its counts by cause and by design add up to its own.

For all 30 v2 records in, the per-photo boxes kept in `data/store/v2/runs/` give the record's counts again.

Worked out again here with `m2b.decisions()`, the chain's whole record (`results/runs/m2b-decisions.json`: noise_store, R1, R2, R3) is the same.
<!-- results:end -->

## What it means

Store set v2 was scored on 2026-11-19 at `5240800`, on the laptop's CPU
(Apple M2, 7 torch threads), one photo a call, in 12 parts. It holds 68
expiry dates in frame, 57 of them on packs of designs no run trained on,
and 212 dates in all; one expiry is about 1.5 points of a rate. The two
groups of designs v1 shares (4 and 7 expiries) are too small to read on
their own. ExpDate test is waiting for the cluster (below).

- **The checks pass.** On v1-val the laptop's CPU found, for `n`, `s` and
  `m`, at 16 photos a call and at one, exactly what each GPU suite found:
  no count moved, tag by tag. Every v2 record was cut at its own GPU
  record's threshold by the weights hashed for it; its counts by cause and
  by design add up; and the per-photo boxes kept give its counts again.
- **The M2b detector finds 65 of v2's 68 expiry dates**, M2's 46 and
  M1's 20; its arm 64 (62–65), M2's 48.7 (46–51), M1's 25 (20–30). It
  finds 88% of every date (186.3 of 212 for the arm), M2 57% and M1 25%.
  Its three misses are two clean numeric dates 1.5 to 2.5% of the photo
  high and one sideways date, all on unseen designs; it boxes 67 of the
  68 at some confidence.
- **It pays in precision.** At its threshold it keeps 0.60 false boxes a
  photo (precision 0.761), where M2's detector keeps 0.17 (0.871) and M1's
  0.02 (0.956). By our AP, which no threshold moves, it is still far ahead:
  0.895 against 0.703 and 0.440. A wrong date is worse than none, so M3's
  reader has to reject those boxes (M3's metric counts wrong reads apart).
- **Where it gains**: month words, 24 of 25 for its arm, where M2's
  weights find 19 at the store threshold and 14 at M2's (M1 7.3); clean
  print, 20.7 of 24 (M2 12.7); sideways, 9.3 of 10 (M2 6.3); and the one
  upside-down date, found by every seed of `n1280-ft`, `n1280-ft2`,
  `s1280-ft` and `m1280-ft`, by 2 of `n1280-ft-gen50k`'s 3, and by none of
  M1, M2, the control or `m1280` (`s1280` by 1 of 3). Dot-matrix goes
  from 26.7 of 28 to all 28. Numeric dates 1.5 to 2.5% high don't move:
  13 of 15 for the control and for the M2b arm, where its misses are.
- **The threshold alone** (M2's weights at the store threshold, the
  control): 48.7 to 58.3 of 68 expiries, but precision 0.903 to 0.550 and
  false boxes a photo 0.13 to 1.48. As on v1-val (S11), the threshold
  closes about half the gap to fine-tuning, at a precision cost
  fine-tuning doesn't pay (64 at 0.774 for the M2b arm).
- **Learned packs, not print? It learned print.** On v2's unseen designs
  every fine-tuned arm's AP lies above every control seed's, as fixed
  before any score: `n1280-ft` 0.909 (0.905–0.915) against 0.726
  (0.695–0.767), and so for `n1280-ft2`, `n1280-ft-gen50k`, `s1280-ft` and
  `m1280-ft`, as on v1-val. Unseen expiries found: 53 of 57 for the M2b
  arm, 47.3 for the control, 38.7 for M2 at its own threshold.
- **What we expected, against what happened.** "Designs seen in training
  well above unseen ones" doesn't hold. For `n1280-ft` our AP is 0.864 on
  designs seen in training, 0.865 on those seen only in validation and
  0.909 on unseen ones; it finds 81%, 88% and 89% of their dates. Every
  M2b run finds all 4 expiries of the trained designs and all 7 of the
  validation-only ones, but so does M2's detector the 4 (and 6 of 7) with
  no store training, so those few packs are easy ones, not learned ones.
  "Unseen ones still above M2's detector" holds: 53 against 38.7 of 57.
  Month words, which S11 left to v2: 24 of 25, up from the control's 19.
- **Model size** (a side question; v2 changes no rule): by our AP, step 1
  gives `n` 0.733, `s` 0.794 and `m` 0.820, and after fine-tuning `n`
  0.897, `s` 0.930 and `m` 0.948, every `m` seed above every `s` seed
  above every `n` seed, as on v1-val (S12). At its v1-val threshold, 0.65,
  `m1280-ft` keeps far fewer false boxes (0.06 a photo, precision 0.971)
  and finds fewer expiries (60.7 of 68, though it boxes 66.7).
- **gen-v2** (a side question): `n1280-ft` against `n1280-ft-gen50k`, our
  AP 0.897 (0.889–0.906) against 0.880 (0.875–0.883), ranges apart; 64
  against 60 expiries and 186.3 against 172.3 dates. On v1-val they
  overlapped (S11), so on v2 gen-v2 adds a little, measurably.
- **The recipe**: `n1280-ft` against `n1280-ft2`, AP 0.897 against 0.901,
  ranges overlapping, as R2 found on v1-val.
- **The unreadable expiry** (counted as one to find): M2's runs and the
  control find it at every seed, `n1280-ft` at 2 of 3 and `m1280-ft` at 1
  of 3, M1 at none; every M2b run boxes it at some confidence.
- **ExpDate test**: not yet. Its job (`slurm/m2b_test_cpu.sbatch`) is
  ready and dry-run on the cluster, and waits for a free slot: the
  account's 2 running jobs are another project's.

## Decision

Nothing changes: v2 decides nothing (S14's note). **The M2b detector stays
`n1280-ft-s0` at `threshold_store` 0.1641.** On store photos no model had
seen, it finds 65 of 68 expiry dates against M2's 46, on unseen pack
designs too, at a cost in false boxes that M3's reader has to absorb.
`m1280-ft` stays M3's fallback by the plan's own Later rule; v2's AP puts
it first, as v1-val did, and that is recorded for M3, not acted on here.
