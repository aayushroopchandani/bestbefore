# M2 / the ablation: turned copies, generated dates, or both?

## Question

What does each kind of new training data add to the date detector:
ExpDate's training photos turned 90° (`rot`), 50,000 photos with generated
dates (`gen`), or both (`gen-rot`), against M1's data with M2's recipe
(`base`)? Measured on the validation suite's hard sets and, once, on the
store photos, next to M1. And which arm becomes the M2 detector?

## Why it matters

M1's detector found 97% of ExpDate test's expiry dates but 27% of the
store photos' (mean of 3 seeds). Four causes explain the misses
([`docs/m2-synthetic-generator.md`](../../docs/m2-synthetic-generator.md),
"What M1 tells us"): sideways dates were never in training; ExpDate's month
words are all short and big, the store's often full names and small;
small print is seen but doubted; and the threshold was chosen where every
date is easy. Turning photos is the cheap remedy for the first cause; the
generator targets the first three. If the cheap one does as well on
sideways dates, the generator has to earn its place on the others.

## Method

| Arm | Training photos (real ones repeated to about 7%) | Listed | Epochs |
|---|---|---:|---:|
| `base` | ExpDate real train + synthetic (the 14 copies of a validation photo left out) | 12,781 | 200 |
| `rot` | the same, plus one copy of each turned 90° or 270° | 25,562 | 100 |
| `gen` | `base`'s, plus 50,000 generated photos | 66,529 | 38 |
| `gen-rot` | `rot`'s, plus 50,000 generated photos | 79,310 | 32 |

- **Fixed for every arm:** `yolo11n` from COCO weights, 640 px, 2.56
  million training images, MuSGD with `lr0=0.01` and `momentum=0.937`, no
  early stopping; the model is the final weights (`last.pt`). Seeds 0, 1
  and 2.
- **Validation suite** (`scripts/score_suite.py`): `real-val`, ExpDate
  validation as it is (the guard); `real-val-rot`, the same photos turned;
  `real-val-small`, the same photos in 2 × 2 grids, every date at half its
  size; `synth-val`, 2,100 generated photos in 4 held-out fonts on
  validation photos, one target date each, in 7 strata that each change
  one thing from `clean`.
- **Threshold:** each run keeps the confidence with the best F2 for dates
  on the three real sets pooled, unchanged on the store photos.
- **Store photos** (S8, once per run, on the laptop): expiry dates found
  by failure tag and by cause: the month as a word (full name or
  abbreviation) and numeric dates by print height, for upright expiries,
  and sideways ones apart. Counts only.

**Decision rules, written before any run** (`bestbefore.m2`):

- **Pilot gate** (`pilot_passes`): a short `gen` run must box at least
  half of synth-val's targets at any confidence, or the generated arms
  stop.
- **The M2 detector** (`decide_detector`), over `base`, `rot`, `gen`,
  `gen-rot` and `gen-rot-1280` (from
  [`m2-input-size`](../m2-input-size/README.md)): an arm is eligible if
  its mean mAP@0.5 for dates on `real-val` is no more than 1 point under
  `base`'s; among those, the highest mean mAP@0.5 over the three hard
  sets, where any candidate within 1 point of it goes to the cheapest
  (listed cheapest first). The M2 detector is that arm's seed 0. The store
  photos judge every arm afterwards; none is picked by its store score.
- **Rotation:** no rule; read off `rot` against `base` and `gen-rot`
  against `gen`.

The chain applies the rules as their inputs complete and writes them to
`results/runs/m2-decisions.json`; `summarise.py` applies them again
through the same function (`m2.decisions()`) and warns if they differ.

**What we expected, written before any run:**

| Arm | sideways | month as a word | small print | ExpDate validation |
|---|---|---|---|---|
| `base` | still about 0 | low | slightly up (threshold) | as M1 |
| `rot` | up a lot | low | as `base` | as M1 |
| `gen` | up | up a lot | up | as M1, perhaps 1 point lower |
| `gen-rot` | up the most | up a lot | up | as M1 |

If `gen` scores high on synth-val month words but doesn't lift the store's
month-word dates, the generated dates don't look like real ones, and that
gets written up as a finding.

## Result

Rebuild with:

```bash
scripts/pull_cluster_results.sh
python experiments/m2-ablation/summarise.py --write
```

<!-- results:start -->
The pilot gate: **passed**, synth-val targets boxed at any confidence 0.972 (it needs 0.5). The chain's own record (`results/runs/m2-decisions.json`) agrees.

**The M2 detector rule**: mean mAP@0.5 for dates over the 3 seeds, on every candidate:

| arm | seeds | threshold | real-val | real-val-rot | real-val-small | synth-val | hard sets, mean | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | 3/3 | 0.202 (0.150–0.303) | 0.994 (0.994–0.995) | 0.000 | 0.893 (0.885–0.902) | 0.786 (0.782–0.792) | 0.560 (0.559–0.561) | yes |
| rot | 3/3 | 0.208 (0.161–0.251) | 0.990 (0.990–0.991) | 0.984 (0.982–0.987) | 0.825 (0.813–0.834) | 0.834 (0.833–0.836) | 0.881 (0.878–0.883) | yes |
| gen | 3/3 | 0.103 (0.069–0.145) | 0.994 | 0.544 (0.515–0.582) | 0.921 (0.915–0.930) | 0.985 (0.985–0.986) | 0.817 (0.810–0.828) | yes |
| gen-rot | 3/3 | 0.328 (0.316–0.348) | 0.994 (0.994–0.995) | 0.971 (0.967–0.973) | 0.911 (0.907–0.915) | 0.986 (0.984–0.987) | 0.956 (0.953–0.957) | yes |
| gen-rot-1280 | 3/3 | 0.389 (0.342–0.424) | 0.995 (0.994–0.995) | 0.988 (0.984–0.991) | 0.986 (0.985–0.990) | 0.994 (0.994–0.995) | 0.989 (0.989–0.990) | yes |

**The M2 detector, by the rule: `gen-rot-1280`**, the highest mean mAP@0.5 on the hard sets. So the M2 detector is `m2-gen-rot-1280-s0`, with its own threshold, 0.3424. Every candidate is eligible. The chain's own record (`results/runs/m2-decisions.json`) agrees.

**real-val**, at each run's own threshold:

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 0.994 (0.994–0.995) | 186.7/187 (186–187) | 186.7/187 (186–187) | 187/187 | 0.935 (0.917–0.954) | 0.08 (0.05–0.10) |
| rot | 3/3 | 0.990 (0.990–0.991) | 186.3/187 (186–187) | 186.3/187 (186–187) | 187/187 | 0.910 (0.907–0.917) | 0.11 (0.10–0.12) |
| gen | 3/3 | 0.994 | 187/187 | 187/187 | 187/187 | 0.922 (0.903–0.954) | 0.10 (0.05–0.12) |
| gen-rot | 3/3 | 0.994 (0.994–0.995) | 186.3/187 (185–187) | 186.3/187 (185–187) | 187/187 | 0.967 (0.964–0.974) | 0.04 (0.03–0.04) |

**real-val-rot**, at each run's own threshold:

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 0.000 | 0/187 | 0/187 | 0/187 | 0.000 | 0.05 (0.04–0.06) |
| rot | 3/3 | 0.984 (0.982–0.987) | 183.7/187 (183–184) | 184/187 (183–185) | 186/187 (185–187) | 0.862 (0.852–0.868) | 0.18 (0.17–0.19) |
| gen | 3/3 | 0.544 (0.515–0.582) | 103.7/187 (101–105) | 106/187 (102–109) | 153/187 (147–158) | 0.520 (0.467–0.561) | 0.59 (0.50–0.73) |
| gen-rot | 3/3 | 0.971 (0.967–0.973) | 178.7/187 (178–180) | 179/187 (178–180) | 186/187 (185–187) | 0.874 (0.868–0.878) | 0.16 (0.15–0.16) |

**real-val-small**, at each run's own threshold:

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 0.893 (0.885–0.902) | 164.3/186 (162–167) | 168/186 (165–173) | 176.3/186 (174–180) | 0.777 (0.756–0.806) | 1.15 (0.95–1.29) |
| rot | 3/3 | 0.825 (0.813–0.834) | 151/186 (149–155) | 155/186 (152–161) | 175.3/186 (173–177) | 0.719 (0.671–0.752) | 1.46 (1.20–1.85) |
| gen | 3/3 | 0.921 (0.915–0.930) | 169/186 (168–170) | 171.7/186 (171–173) | 179.7/186 (177–181) | 0.737 (0.686–0.786) | 1.49 (1.12–1.88) |
| gen-rot | 3/3 | 0.911 (0.907–0.915) | 160.3/186 (157–162) | 162.3/186 (158–165) | 179.7/186 (179–181) | 0.884 (0.835–0.913) | 0.52 (0.37–0.78) |

**synth-val**, at each run's own threshold:

| arm | seeds | mAP@0.5 date | dates found | targets found | targets boxed at any confidence | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 0.786 (0.782–0.792) | 3335.7/4494 (3287–3369) | 961/2100 (920–990) | 1206.3/2100 (1189–1232) | 3366.7/4494 (3312–3397) | 3594.7/4494 (3578–3619) | 0.934 (0.920–0.956) | 0.11 (0.07–0.14) |
| rot | 3/3 | 0.834 (0.833–0.836) | 3477/4494 (3454–3490) | 1122.3/2100 (1093–1143) | 1479/2100 (1458–1501) | 3512.7/4494 (3486–3528) | 3861.7/4494 (3839–3883) | 0.910 (0.901–0.916) | 0.16 (0.15–0.18) |
| gen | 3/3 | 0.985 (0.985–0.986) | 4409/4494 (4403–4418) | 2016/2100 (2010–2025) | 2062.3/2100 (2062–2063) | 4417.7/4494 (4411–4426) | 4456.3/4494 (4456–4457) | 0.901 (0.871–0.939) | 0.23 (0.14–0.31) |
| gen-rot | 3/3 | 0.986 (0.984–0.987) | 4338.3/4494 (4326–4349) | 1960.7/2100 (1955–1968) | 2063.7/2100 (2062–2067) | 4353/4494 (4337–4367) | 4457.7/4494 (4456–4461) | 0.966 (0.962–0.970) | 0.07 (0.06–0.08) |

**synth-val by stratum**, targets found at the threshold:

| arm | seeds | clean | sideways | month-word | small | dot-matrix | curved | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 217.3/300 (214–222) | 0/300 | 201.7/300 (190–216) | 124/300 (115–131) | 154.7/300 (150–160) | 181.3/300 (173–189) | 82/300 (78–85) |
| rot | 3/3 | 208.3/300 (207–210) | 230.7/300 (229–232) | 185.7/300 (173–202) | 87.7/300 (82–91) | 137.3/300 (131–146) | 169.7/300 (168–171) | 103/300 (100–105) |
| gen | 3/3 | 295.7/300 (295–296) | 296/300 (295–297) | 296.3/300 (296–297) | 273/300 (268–279) | 284.7/300 (284–285) | 293/300 (292–294) | 277.3/300 (275–281) |
| gen-rot | 3/3 | 292.3/300 (291–294) | 294.3/300 (294–295) | 292/300 (291–293) | 261.3/300 (261–262) | 274.7/300 (271–277) | 286.3/300 (283–290) | 259.7/300 (256–262) |

and boxed at any confidence:

| arm | seeds | clean | sideways | month-word | small | dot-matrix | curved | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 247.7/300 (246–249) | 0/300 | 245/300 (241–250) | 173.7/300 (167–182) | 209.7/300 (208–213) | 218.3/300 (212–222) | 112/300 (109–117) |
| rot | 3/3 | 242.7/300 (242–244) | 264/300 | 242.3/300 (241–244) | 157.7/300 (154–163) | 202/300 (197–209) | 209.3/300 (203–213) | 161/300 (157–167) |
| gen | 3/3 | 299/300 | 297/300 | 298.3/300 (298–299) | 287.7/300 (287–288) | 292/300 | 298/300 | 290.3/300 (289–291) |
| gen-rot | 3/3 | 299.7/300 (299–300) | 297.3/300 (297–298) | 299/300 | 289/300 (288–290) | 291.7/300 (291–292) | 298.3/300 (298–299) | 288.7/300 (286–291) |

**Rotation** (no rule; read off): `rot` against `base`, `gen-rot` against `gen`:

| arm | seeds | real-val-rot: mAP@0.5 date | real-val-rot: dates found | synth-val sideways: targets found | store sideways: expiry found |
|---|---:|---:|---:|---:|---:|
| base | 3/3 | 0.000 | 0/187 | 0/300 | 0/15 |
| rot | 3/3 | 0.984 (0.982–0.987) | 183.7/187 (183–184) | 230.7/300 (229–232) | 1/15 |
| gen | 3/3 | 0.544 (0.515–0.582) | 103.7/187 (101–105) | 296/300 (295–297) | 0.7/15 (0–1) |
| gen-rot | 3/3 | 0.971 (0.967–0.973) | 178.7/187 (178–180) | 294.3/300 (294–295) | 0.3/15 (0–1) |

**The store photos** (S8), next to M1:

Each run at its own threshold, IoU 0.5:

| detector | seeds | expiry found | expiry at IoU 0.3 | expiry boxed at any confidence | all dates found | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 21.7/81 (18–25) | 22.7/81 (19–26) | 45/81 (42–50) | 45.7/199 (38–50) | 0.939 (0.926–0.950) | 0.03 (0.02–0.04) |
| base | 3/3 | 27/81 (25–29) | 28/81 (26–30) | 43/81 (39–46) | 59.3/199 (54–65) | 0.763 (0.720–0.806) | 0.19 (0.13–0.23) |
| rot | 3/3 | 26.3/81 (26–27) | 27/81 | 45.7/81 (45–46) | 58/199 (55–61) | 0.728 (0.667–0.833) | 0.23 (0.11–0.29) |
| gen | 3/3 | 38.3/81 (37–40) | 39.3/81 (38–41) | 52.7/81 (50–55) | 92/199 (87–99) | 0.682 (0.634–0.770) | 0.45 (0.26–0.56) |
| gen-rot | 3/3 | 33.7/81 (33–35) | 34.3/81 (33–36) | 55/81 (54–56) | 74.7/199 (70–78) | 0.868 (0.843–0.907) | 0.11 (0.08–0.13) |

**By failure tag**, expiry dates found at the threshold:

| detector | seeds | sideways | dot-matrix | small-print | clean | all |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/15 | 14.7/20 (12–17) | 0/15 | 7/31 (6–8) | 21.7/81 (18–25) |
| base | 3/3 | 0/15 | 17.7/20 (17–19) | 0/15 | 9.3/31 (8–10) | 27/81 (25–29) |
| rot | 3/3 | 1/15 | 17.3/20 (17–18) | 0/15 | 8/31 | 26.3/81 (26–27) |
| gen | 3/3 | 0.7/15 (0–1) | 18.7/20 (18–19) | 0.7/15 (0–1) | 18.3/31 (18–19) | 38.3/81 (37–40) |
| gen-rot | 3/3 | 0.3/15 (0–1) | 17.3/20 (16–18) | 0/15 | 16/31 (15–17) | 33.7/81 (33–35) |

**By failure tag**, expiry dates boxed at any confidence:

| detector | seeds | sideways | dot-matrix | small-print | clean | all |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/15 | 18.7/20 (18–19) | 5.3/15 (4–8) | 21/31 (19–23) | 45/81 (42–50) |
| base | 3/3 | 0/15 | 19/20 | 3.7/15 (3–4) | 20.3/31 (17–23) | 43/81 (39–46) |
| rot | 3/3 | 5.3/15 (4–7) | 18.7/20 (18–19) | 3/15 | 18.7/31 (18–20) | 45.7/81 (45–46) |
| gen | 3/3 | 4/15 (3–5) | 19/20 | 4/15 (3–5) | 25.7/31 (25–27) | 52.7/81 (50–55) |
| gen-rot | 3/3 | 5.7/15 (5–6) | 19.3/20 (19–20) | 4.3/15 (4–5) | 25.7/31 (25–27) | 55/81 (54–56) |

**By cause**, expiry dates found at the threshold (upright expiries by month word or print height; sideways ones apart):

| detector | seeds | month word: full name | month word: abbreviation | numbers: print under 1.5% | numbers: print 1.5 to 2.5% | numbers: print 2.5% and up | sideways |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/19 | 0.7/7 (0–1) | 1/8 | 8.7/19 (5–11) | 11.3/13 (11–12) | 0/15 |
| base | 3/3 | 1/19 (0–2) | 2/7 | 1/8 | 11/19 (10–13) | 12/13 | 0/15 |
| rot | 3/3 | 0/19 | 2/7 | 1/8 | 10.7/19 (10–12) | 11.7/13 (11–12) | 1/15 |
| gen | 3/3 | 6.7/19 (6–7) | 1.3/7 (1–2) | 1.7/8 (1–2) | 16/19 | 12/13 | 0.7/15 (0–1) |
| gen-rot | 3/3 | 5.7/19 (5–6) | 1/7 | 1/8 | 13.7/19 (13–15) | 12/13 | 0.3/15 (0–1) |

**By cause**, expiry dates boxed at any confidence (upright expiries by month word or print height; sideways ones apart):

| detector | seeds | month word: full name | month word: abbreviation | numbers: print under 1.5% | numbers: print 1.5 to 2.5% | numbers: print 2.5% and up | sideways |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 7.3/19 (5–10) | 4/7 (3–5) | 4.3/8 (4–5) | 17.3/19 (17–18) | 12/13 | 0/15 |
| base | 3/3 | 8.7/19 (8–9) | 3.3/7 (2–4) | 2.7/8 (2–3) | 16.3/19 (14–19) | 12/13 | 0/15 |
| rot | 3/3 | 4.7/19 (2–7) | 3.3/7 (2–5) | 3.7/8 (3–4) | 16.7/19 (16–18) | 12/13 | 5.3/15 (4–7) |
| gen | 3/3 | 10/19 | 4.3/7 (4–5) | 4.3/8 (3–5) | 18/19 | 12/13 | 4/15 (3–5) |
| gen-rot | 3/3 | 10.7/19 (10–11) | 3.7/7 (3–4) | 4.7/8 (4–5) | 18/19 | 12.3/13 (12–13) | 5.7/15 (5–6) |

For all 15 runs scored, the causes add up to store.json's counts.
<!-- results:end -->

## What it means

Trained and scored on the validation suite by GPU job A (`72914`) on
`cn5` at `a53c1a9`, 2026-10-24 to 2026-10-26; scored once on the store
photos on the laptop's CPU on 2026-10-28 at `fb5aeaa` (clean tree), each
run at its own threshold from the GPU's `suite.json`, unrounded. The
store set is small (15 sideways and 15 small-print expiries) and no
longer independent of M2's design, so 1 to 3 dates between arms is
noise.

- **The M2 detector is `gen-rot-1280`, seed 0** (threshold 0.3424), by
  the rule written before any run, and the chain's record agrees. Every
  candidate passed the guard: each is within 0.005 of `base` on
  `real-val`. On the hard sets `gen-rot-1280` leads by 3.4 points
  (0.989 against `gen-rot`'s 0.956), well beyond the 1-point tie.
- **Every arm finds more store expiries than M1** (21.7 of 81): `base`
  27, `rot` 26.3, `gen` 38.3, `gen-rot` 33.7 (mean of 3 seeds;
  `gen-rot-1280` 46, [`m2-input-size`](../m2-input-size/README.md)).
  `base` is M1's data with M2's recipe, so its 5.3 more dates come from
  the longer training and the F2 threshold, at a cost in precision (0.94
  to 0.76 over all store dates).
- **The generated dates are what helps on the store photos.** `gen`
  against `base`: clean expiries 18.3 against 9.3 of 31, full month names
  6.7 against 1 of 19, numbers printed 1.5 to 2.5% of the long side 16
  against 11 of 19, dot-matrix 18.7 against 17.7 of 20. The turned copies
  add nothing there: `rot` finds 26.3 against `base`'s 27.
- **Sideways dates are not solved on the store photos.** No 640 px arm
  finds more than 1 of the 15 at its threshold, though `rot` finds 98% of
  `real-val-rot`'s turned dates and `gen-rot` 96%; at any confidence they
  box 5.3 and 5.7 of 15. So turning ExpDate's photos teaches the model
  ExpDate's dates turned, and generated sideways dates teach it the
  generator's (`gen`: 296 of 300 synth-val sideways targets, but 103.7 of
  187 real turned dates); neither is what the store's sideways dates look
  like. Why is not measured here (the plan's Later).
- **Month words carry over only in part.** `gen` finds 296 of 300
  synth-val month-word targets, but 6.7 of the store's 19 full month
  names at its threshold (10 at any confidence) and 1.3 of 7
  abbreviations. That is up from M1's 0 of 19, so not the plan's "no
  lift" case, but far from what synth-val promises: part of the
  generated look doesn't carry over to real packs.
- **Small print isn't helped at 640 px:** 0 to 0.7 of 15 small-print
  expiries for every arm, as M1. Only the 1,280 px input moved it.
- **The threshold still cuts many dates.** Every arm boxes far more store
  expiries at some confidence than it keeps: `gen-rot` 55 against 33.7
  kept, `gen` 52.7 against 38.3. Choosing it for recall on real hard
  validation photos (decision 11) lowered it, but the store's dates are
  still less sure than validation's. Precision fell with it, most for
  `gen` (0.68, 0.45 false boxes a photo, at thresholds of 0.07 to 0.15).
- **Rotation (no rule):** `rot` against `base` lifts `real-val-rot` from
  0 to 98% and `synth-val` sideways from 0 to 231 of 300, but not the
  store's sideways dates (1 against 0 of 15) and costs 6.8 points on
  `real-val-small`. `gen-rot` against `gen` lifts `real-val-rot` from 0.54
  to 0.97 mAP, but finds fewer store expiries at its threshold (33.7
  against 38.3) while boxing more at any confidence (55 against 52.7): its
  thresholds (0.32 to 0.35) are 3 times `gen`'s, and its precision is
  higher (0.87 against 0.68).
- **What we expected, against what happened on the store photos:**
  `base` as expected, except that small print didn't go up with the lower
  threshold (0 of 15, as M1); `rot` up a lot on real turned dates, as
  expected, but not on the store's sideways ones; `gen` up on month words
  (full names), less than "a lot", and not on small print or sideways;
  `gen-rot` not "up the most" on sideways (0.3 of 15), up on full month
  names (5.7 of 19), not on small print. ExpDate validation stayed as M1
  for every arm (mAP@0.5 0.990 to 0.995).
