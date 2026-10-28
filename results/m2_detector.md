# M2: synthetic dates for the detector

M1's detector (`yolo11n`, 640 px) found 97% of the expiry dates on ExpDate
test but about a quarter on the retailer's store photos: it missed
sideways dates, months written as words, and small print. M2 trains the
same detector on other data, every arm with the same budget of 2.56
million images and 3 seeds: M1's data with the M2 recipe (`base`), plus
copies turned 90° or 270° (`rot`), plus 50,000 photos with generated dates drawn
onto ExpDate backgrounds (`gen`), both (`gen-rot`), both at 1,280 px input
(`gen-rot-1280`), and 10,000 generated photos listed five times
(`gen10k`). Every choice is made on validation; ExpDate test and the store
photos are each scored once, afterwards. M2 is judged only on **finding**
dates; reading them is M3's job. Plan and reasoning:
[`docs/m2-synthetic-generator.md`](../docs/m2-synthetic-generator.md).

A date counts as found when a box at or above its run's threshold overlaps
it with IoU 0.5 or more. Each run keeps the threshold with the best F2 on
the three real validation sets pooled. A cell is the mean over the seeds
in, with the lowest and highest in brackets: `178.3/187 (176–181)`; "not
yet" means that run isn't trained, scored or pulled yet.

## Headline

The M1 and M2 detectors are each one run, seed 0, declared before any
run; the 3-seed rows show how much a seed moves the counts.

Expiry dates found at IoU 0.5, each run at its own threshold:

| detector | ExpDate test | store photos | store: boxed at any confidence | store: precision | store: false boxes / photo |
|---|---:|---:|---:|---:|---:|
| the M1 detector (`n640`) | 637/665 | 18/81 | 43/81 | 0.950 | 0.02 |
| M1, 3 seeds | 646/665 (637–651) | 21.7/81 (18–25) | 45/81 (42–50) | 0.939 (0.926–0.950) | 0.03 (0.02–0.04) |
| the M2 detector (`m2-gen-rot-1280-s0`) | 662/665 | 39/81 | 69/81 | 0.902 | 0.11 |
| M2 (`gen-rot-1280`), 3 seeds | 662.7/665 (662–663) | 46/81 (39–53) | 70.3/81 (69–72) | 0.910 (0.890–0.937) | 0.11 (0.08–0.13) |

On the store photos, by failure tag:

| detector | sideways | dot-matrix | small-print | clean | all |
|---|---:|---:|---:|---:|---:|
| the M1 detector (`n640`) | 0/15 | 12/20 | 0/15 | 6/31 | 18/81 |
| M1, 3 seeds | 0/15 | 14.7/20 (12–17) | 0/15 | 7/31 (6–8) | 21.7/81 (18–25) |
| the M2 detector (`m2-gen-rot-1280-s0`) | 3/15 | 19/20 | 2/15 | 15/31 | 39/81 |
| M2 (`gen-rot-1280`), 3 seeds | 2.7/15 (2–3) | 19/20 | 5.3/15 (2–9) | 19/31 (15–23) | 46/81 (39–53) |

## What it shows

Every number here is read from the same records as the tables below. The store set is small (1 to 3 dates between two arms is noise) and less independent than a new one would be: M1's store result chose M2's targets.

**Where M2 beats M1:**

- **More store expiries found.** `gen-rot-1280` finds 46/81 (39–53) against M1's 21.7/81 (18–25). The M2 detector itself, seed 0 by the rule declared before any run, finds 39/81, the fewest of its arm's three seeds (the M1 detector: 18/81). Every arm finds more than M1: from 26.3 to 46 of 81.
- **The generated dates are what helps.** `gen` against `base`: clean expiries 18.3/31 (18–19) against 9.3/31 (8–10), full month names 6.7/19 (6–7) against 1/19 (0–2) (M1 0/19). Turned copies add nothing there: `rot` finds 26.3/81 (26–27), `base` 27/81 (25–29).
- **Small print, only at 1,280 px:** `gen-rot-1280` finds 5.3/15 (2–9) small-print expiries (boxed at any confidence 13/15 (12–14)); every 640 px arm finds 0 to 0.7 of 15, as M1 (0/15).
- **ExpDate kept.** On ExpDate test the M2 detector finds 662/665 expiry dates (the M1 detector 637/665); on ExpDate validation every arm's mean mAP@0.5 for dates is 0.990 to 0.995 (`base` 0.994 (0.994–0.995)).

**Where it doesn't:**

- **Sideways dates are almost unsolved on the store photos:** `gen-rot-1280` finds 2.7/15 (2–3), every 640 px arm at most 1 of 15, though `rot` finds 183.7/187 (183–184) turned validation dates and `gen` 296/300 (295–297) synth-val sideways targets. Neither ExpDate's photos turned nor the generated sideways dates look like the store's to the model.
- **Precision is lower than M1's:** 0.910 (0.890–0.937) against 0.939 (0.926–0.950) over every store date, with 0.11 (0.08–0.13) false boxes a photo against 0.03 (0.02–0.04). Every arm keeps more false boxes than M1: precision 0.682 to 0.910, 0.11 to 0.45 false boxes a photo. That is the price of the F2 threshold (decision 11).
- **Many dates are boxed but cut by the threshold:** `gen-rot-1280` boxes 70.3/81 (69–72) store expiries at some confidence and keeps 46/81 (39–53); small print 13/15 (12–14) against 5.3/15 (2–9). A threshold chosen for recall on real hard validation photos is still above many of the store's dates (cause 4, only partly closed).
- **Large numeric dates** (printed 2.5% and up of the long side) stay where M1 was: `gen-rot-1280` 12/13, M1 11.3/13 (11–12). M1 already found them.

**What we expected, written before any run, against what was measured.** The plan warned that if `gen` scored high on synth-val's month words without lifting the store's, the generated dates wouldn't look like real ones. It is partly that case: the store's month words did go up, but far less than synth-val promised.

| arm | expected | measured | verdict |
|---|---|---|---|
| `base` | sideways: still about 0 | store 0/15; real-val-rot 0/187 | held |
| `base` | small print: slightly up (threshold) | store 0/15 (M1 0/15) | failed |
| `rot` | sideways: up a lot | real-val-rot 183.7/187 (183–184); store 1/15 | held on validation, failed on the store photos |
| `gen` | sideways: up | synth-val 296/300 (295–297); real-val-rot 103.7/187 (101–105); store 0.7/15 (0–1) | held on synth-val only |
| `gen` | month as a word: up a lot | synth-val 296.3/300 (296–297); store full names 6.7/19 (6–7) (M1 0/19), abbreviations 1.3/7 (1–2) (M1 0.7/7 (0–1)) | partly: up on the store, far below synth-val |
| `gen` | small print: up | real-val-small 169/186 (168–170) (`base` 164.3/186 (162–167)); store 0.7/15 (0–1) | slightly on validation, not on the store photos |
| `gen` | ExpDate validation: as M1, perhaps 1 point lower | mAP@0.5 0.994 (`base` 0.994 (0.994–0.995)) | held, no loss |
| `gen-rot` | sideways: up the most | real-val-rot 178.7/187 (178–180); store 0.3/15 (0–1) | failed |
| `gen-rot` | month as a word: up a lot | synth-val 292/300 (291–293); store full names 5.7/19 (5–6) | partly |
| `gen-rot` | small print: up | store 0/15 | failed |
| `gen-rot-1280` | sideways: as `gen-rot` | store 2.7/15 (2–3) (`gen-rot` 0.3/15 (0–1)) | held: both find almost none |
| `gen-rot-1280` | month as a word: as `gen-rot` | store full names 10/19 (6–15) (`gen-rot` 5.7/19 (5–6)) | better than expected |
| `gen-rot-1280` | small print: up the most | real-val-small 181.3/186 (181–182); store 5.3/15 (2–9) | held |
| every arm | ExpDate validation: as M1 | mAP@0.5 0.990 to 0.995 | held |

## The validation suite

`real-val` is ExpDate validation (the guard: must not get worse);
`real-val-rot` the same photos turned 90° or 270°; `real-val-small` the
same photos in 2 × 2 grids, so every date is half its size; `synth-val`
2,100 generated photos in held-out fonts on validation backgrounds, one
target date each, in seven strata that each change one thing from
`clean`.

The pilot gate: **passed**, synth-val targets boxed at any confidence 0.972 (it needs 0.5). The chain's own record (`results/runs/m2-decisions.json`) agrees.

**The M2 detector** (decision 12): eligible if its mean mAP@0.5 for dates
on real-val is no more than 1 point under `base`'s; among those, the best
mean over the hard sets, where a candidate within 1 point goes to the
cheapest.

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
| gen10k | 3/3 | 0.992 (0.989–0.994) | 185.7/187 (185–186) | 185.7/187 (185–186) | 187/187 | 0.943 (0.912–0.964) | 0.07 (0.04–0.11) |
| gen-rot-1280 | 3/3 | 0.995 (0.994–0.995) | 186.7/187 (186–187) | 186.7/187 (186–187) | 187/187 | 0.982 (0.979–0.984) | 0.02 |

**real-val-rot**, at each run's own threshold:

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 0.000 | 0/187 | 0/187 | 0/187 | 0.000 | 0.05 (0.04–0.06) |
| rot | 3/3 | 0.984 (0.982–0.987) | 183.7/187 (183–184) | 184/187 (183–185) | 186/187 (185–187) | 0.862 (0.852–0.868) | 0.18 (0.17–0.19) |
| gen | 3/3 | 0.544 (0.515–0.582) | 103.7/187 (101–105) | 106/187 (102–109) | 153/187 (147–158) | 0.520 (0.467–0.561) | 0.59 (0.50–0.73) |
| gen-rot | 3/3 | 0.971 (0.967–0.973) | 178.7/187 (178–180) | 179/187 (178–180) | 186/187 (185–187) | 0.874 (0.868–0.878) | 0.16 (0.15–0.16) |
| gen10k | 3/3 | 0.509 (0.481–0.527) | 91.3/187 (85–99) | 94/187 (87–103) | 145/187 (140–150) | 0.587 (0.573–0.612) | 0.39 (0.33–0.44) |
| gen-rot-1280 | 3/3 | 0.988 (0.984–0.991) | 183.7/187 (182–185) | 184/187 (183–185) | 187/187 | 0.947 (0.938–0.953) | 0.06 (0.05–0.07) |

**real-val-small**, at each run's own threshold:

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 0.893 (0.885–0.902) | 164.3/186 (162–167) | 168/186 (165–173) | 176.3/186 (174–180) | 0.777 (0.756–0.806) | 1.15 (0.95–1.29) |
| rot | 3/3 | 0.825 (0.813–0.834) | 151/186 (149–155) | 155/186 (152–161) | 175.3/186 (173–177) | 0.719 (0.671–0.752) | 1.46 (1.20–1.85) |
| gen | 3/3 | 0.921 (0.915–0.930) | 169/186 (168–170) | 171.7/186 (171–173) | 179.7/186 (177–181) | 0.737 (0.686–0.786) | 1.49 (1.12–1.88) |
| gen-rot | 3/3 | 0.911 (0.907–0.915) | 160.3/186 (157–162) | 162.3/186 (158–165) | 179.7/186 (179–181) | 0.884 (0.835–0.913) | 0.52 (0.37–0.78) |
| gen10k | 3/3 | 0.913 (0.907–0.923) | 164.3/186 (162–166) | 168.7/186 (168–169) | 178.7/186 (176–182) | 0.806 (0.783–0.818) | 0.97 (0.90–1.10) |
| gen-rot-1280 | 3/3 | 0.986 (0.985–0.990) | 181.3/186 (181–182) | 181.3/186 (181–182) | 185.7/186 (185–186) | 0.954 (0.948–0.958) | 0.21 (0.20–0.24) |

**synth-val**, at each run's own threshold:

| arm | seeds | mAP@0.5 date | dates found | targets found | targets boxed at any confidence | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 0.786 (0.782–0.792) | 3335.7/4494 (3287–3369) | 961/2100 (920–990) | 1206.3/2100 (1189–1232) | 3366.7/4494 (3312–3397) | 3594.7/4494 (3578–3619) | 0.934 (0.920–0.956) | 0.11 (0.07–0.14) |
| rot | 3/3 | 0.834 (0.833–0.836) | 3477/4494 (3454–3490) | 1122.3/2100 (1093–1143) | 1479/2100 (1458–1501) | 3512.7/4494 (3486–3528) | 3861.7/4494 (3839–3883) | 0.910 (0.901–0.916) | 0.16 (0.15–0.18) |
| gen | 3/3 | 0.985 (0.985–0.986) | 4409/4494 (4403–4418) | 2016/2100 (2010–2025) | 2062.3/2100 (2062–2063) | 4417.7/4494 (4411–4426) | 4456.3/4494 (4456–4457) | 0.901 (0.871–0.939) | 0.23 (0.14–0.31) |
| gen-rot | 3/3 | 0.986 (0.984–0.987) | 4338.3/4494 (4326–4349) | 1960.7/2100 (1955–1968) | 2063.7/2100 (2062–2067) | 4353/4494 (4337–4367) | 4457.7/4494 (4456–4461) | 0.966 (0.962–0.970) | 0.07 (0.06–0.08) |
| gen10k | 3/3 | 0.979 (0.978–0.980) | 4349.7/4494 (4326–4366) | 1972/2100 (1957–1983) | 2045.3/2100 (2043–2048) | 4366/4494 (4344–4382) | 4439.3/4494 (4437–4442) | 0.941 (0.928–0.948) | 0.13 (0.11–0.16) |
| gen-rot-1280 | 3/3 | 0.994 (0.994–0.995) | 4427.7/4494 (4417–4436) | 2036.7/2100 (2031–2043) | 2080/2100 | 4432/4494 (4422–4438) | 4474/4494 | 0.988 (0.986–0.989) | 0.03 (0.02–0.03) |

**synth-val by stratum**, targets found at the threshold:

| arm | seeds | clean | sideways | month-word | small | dot-matrix | curved | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 217.3/300 (214–222) | 0/300 | 201.7/300 (190–216) | 124/300 (115–131) | 154.7/300 (150–160) | 181.3/300 (173–189) | 82/300 (78–85) |
| rot | 3/3 | 208.3/300 (207–210) | 230.7/300 (229–232) | 185.7/300 (173–202) | 87.7/300 (82–91) | 137.3/300 (131–146) | 169.7/300 (168–171) | 103/300 (100–105) |
| gen | 3/3 | 295.7/300 (295–296) | 296/300 (295–297) | 296.3/300 (296–297) | 273/300 (268–279) | 284.7/300 (284–285) | 293/300 (292–294) | 277.3/300 (275–281) |
| gen-rot | 3/3 | 292.3/300 (291–294) | 294.3/300 (294–295) | 292/300 (291–293) | 261.3/300 (261–262) | 274.7/300 (271–277) | 286.3/300 (283–290) | 259.7/300 (256–262) |
| gen10k | 3/3 | 293/300 (292–294) | 291/300 (289–292) | 293.7/300 (293–294) | 262.7/300 (257–267) | 278.7/300 (275–281) | 285.7/300 (285–286) | 267.3/300 (264–270) |
| gen-rot-1280 | 3/3 | 295.7/300 (294–297) | 296/300 (295–297) | 297.7/300 (297–298) | 281/300 (279–284) | 289/300 (287–291) | 296.7/300 (295–298) | 280.7/300 (279–284) |

and boxed at any confidence:

| arm | seeds | clean | sideways | month-word | small | dot-matrix | curved | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 3/3 | 247.7/300 (246–249) | 0/300 | 245/300 (241–250) | 173.7/300 (167–182) | 209.7/300 (208–213) | 218.3/300 (212–222) | 112/300 (109–117) |
| rot | 3/3 | 242.7/300 (242–244) | 264/300 | 242.3/300 (241–244) | 157.7/300 (154–163) | 202/300 (197–209) | 209.3/300 (203–213) | 161/300 (157–167) |
| gen | 3/3 | 299/300 | 297/300 | 298.3/300 (298–299) | 287.7/300 (287–288) | 292/300 | 298/300 | 290.3/300 (289–291) |
| gen-rot | 3/3 | 299.7/300 (299–300) | 297.3/300 (297–298) | 299/300 | 289/300 (288–290) | 291.7/300 (291–292) | 298.3/300 (298–299) | 288.7/300 (286–291) |
| gen10k | 3/3 | 296.7/300 (296–297) | 296.7/300 (296–297) | 296/300 | 283/300 (281–286) | 290.7/300 (288–293) | 297/300 (296–298) | 285.3/300 (282–287) |
| gen-rot-1280 | 3/3 | 300/300 | 297.3/300 (297–298) | 299/300 | 293.3/300 (293–294) | 297/300 | 299/300 | 294.3/300 (294–295) |

## The store photos

The frozen store test set v1: 99 photos, 81 printed expiry dates. Each run
at its own threshold, next to M1's three seeds. The set is small: a
difference of 1 to 3 dates between two arms is noise. It was test-only
from M0 to M2, so every number here was measured before any training on
store photos; since 2026-10-28 the retailer's owner allows training on
them, so later milestones are judged on a new set, frozen first.

Each run at its own threshold, IoU 0.5:

| detector | seeds | expiry found | expiry at IoU 0.3 | expiry boxed at any confidence | all dates found | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 21.7/81 (18–25) | 22.7/81 (19–26) | 45/81 (42–50) | 45.7/199 (38–50) | 0.939 (0.926–0.950) | 0.03 (0.02–0.04) |
| base | 3/3 | 27/81 (25–29) | 28/81 (26–30) | 43/81 (39–46) | 59.3/199 (54–65) | 0.763 (0.720–0.806) | 0.19 (0.13–0.23) |
| rot | 3/3 | 26.3/81 (26–27) | 27/81 | 45.7/81 (45–46) | 58/199 (55–61) | 0.728 (0.667–0.833) | 0.23 (0.11–0.29) |
| gen | 3/3 | 38.3/81 (37–40) | 39.3/81 (38–41) | 52.7/81 (50–55) | 92/199 (87–99) | 0.682 (0.634–0.770) | 0.45 (0.26–0.56) |
| gen-rot | 3/3 | 33.7/81 (33–35) | 34.3/81 (33–36) | 55/81 (54–56) | 74.7/199 (70–78) | 0.868 (0.843–0.907) | 0.11 (0.08–0.13) |
| gen10k | 3/3 | 37/81 (34–39) | 38/81 (35–40) | 49/81 (47–51) | 84.7/199 (83–86) | 0.777 (0.741–0.843) | 0.25 (0.16–0.29) |
| gen-rot-1280 | 3/3 | 46/81 (39–53) | 46.7/81 (39–54) | 70.3/81 (69–72) | 108.3/199 (101–119) | 0.910 (0.890–0.937) | 0.11 (0.08–0.13) |

**By failure tag**, expiry dates found at the threshold:

| detector | seeds | sideways | dot-matrix | small-print | clean | all |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/15 | 14.7/20 (12–17) | 0/15 | 7/31 (6–8) | 21.7/81 (18–25) |
| base | 3/3 | 0/15 | 17.7/20 (17–19) | 0/15 | 9.3/31 (8–10) | 27/81 (25–29) |
| rot | 3/3 | 1/15 | 17.3/20 (17–18) | 0/15 | 8/31 | 26.3/81 (26–27) |
| gen | 3/3 | 0.7/15 (0–1) | 18.7/20 (18–19) | 0.7/15 (0–1) | 18.3/31 (18–19) | 38.3/81 (37–40) |
| gen-rot | 3/3 | 0.3/15 (0–1) | 17.3/20 (16–18) | 0/15 | 16/31 (15–17) | 33.7/81 (33–35) |
| gen10k | 3/3 | 0.3/15 (0–1) | 18/20 (17–19) | 0.3/15 (0–1) | 18.3/31 (17–19) | 37/81 (34–39) |
| gen-rot-1280 | 3/3 | 2.7/15 (2–3) | 19/20 | 5.3/15 (2–9) | 19/31 (15–23) | 46/81 (39–53) |

**By failure tag**, expiry dates boxed at any confidence:

| detector | seeds | sideways | dot-matrix | small-print | clean | all |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/15 | 18.7/20 (18–19) | 5.3/15 (4–8) | 21/31 (19–23) | 45/81 (42–50) |
| base | 3/3 | 0/15 | 19/20 | 3.7/15 (3–4) | 20.3/31 (17–23) | 43/81 (39–46) |
| rot | 3/3 | 5.3/15 (4–7) | 18.7/20 (18–19) | 3/15 | 18.7/31 (18–20) | 45.7/81 (45–46) |
| gen | 3/3 | 4/15 (3–5) | 19/20 | 4/15 (3–5) | 25.7/31 (25–27) | 52.7/81 (50–55) |
| gen-rot | 3/3 | 5.7/15 (5–6) | 19.3/20 (19–20) | 4.3/15 (4–5) | 25.7/31 (25–27) | 55/81 (54–56) |
| gen10k | 3/3 | 3.3/15 (3–4) | 19/20 | 3.3/15 (3–4) | 23.3/31 (21–25) | 49/81 (47–51) |
| gen-rot-1280 | 3/3 | 9.3/15 (7–11) | 20/20 | 13/15 (12–14) | 28/31 | 70.3/81 (69–72) |

**By cause**, expiry dates found at the threshold (upright expiries by month word or print height; sideways ones apart):

| detector | seeds | month word: full name | month word: abbreviation | numbers: print under 1.5% | numbers: print 1.5 to 2.5% | numbers: print 2.5% and up | sideways |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/19 | 0.7/7 (0–1) | 1/8 | 8.7/19 (5–11) | 11.3/13 (11–12) | 0/15 |
| base | 3/3 | 1/19 (0–2) | 2/7 | 1/8 | 11/19 (10–13) | 12/13 | 0/15 |
| rot | 3/3 | 0/19 | 2/7 | 1/8 | 10.7/19 (10–12) | 11.7/13 (11–12) | 1/15 |
| gen | 3/3 | 6.7/19 (6–7) | 1.3/7 (1–2) | 1.7/8 (1–2) | 16/19 | 12/13 | 0.7/15 (0–1) |
| gen-rot | 3/3 | 5.7/19 (5–6) | 1/7 | 1/8 | 13.7/19 (13–15) | 12/13 | 0.3/15 (0–1) |
| gen10k | 3/3 | 7.3/19 (7–8) | 1/7 | 1.3/8 (1–2) | 15.3/19 (14–17) | 11.7/13 (11–12) | 0.3/15 (0–1) |
| gen-rot-1280 | 3/3 | 10/19 (6–15) | 3/7 (2–4) | 3/8 (2–4) | 15.3/19 (14–16) | 12/13 | 2.7/15 (2–3) |

**By cause**, expiry dates boxed at any confidence (upright expiries by month word or print height; sideways ones apart):

| detector | seeds | month word: full name | month word: abbreviation | numbers: print under 1.5% | numbers: print 1.5 to 2.5% | numbers: print 2.5% and up | sideways |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 7.3/19 (5–10) | 4/7 (3–5) | 4.3/8 (4–5) | 17.3/19 (17–18) | 12/13 | 0/15 |
| base | 3/3 | 8.7/19 (8–9) | 3.3/7 (2–4) | 2.7/8 (2–3) | 16.3/19 (14–19) | 12/13 | 0/15 |
| rot | 3/3 | 4.7/19 (2–7) | 3.3/7 (2–5) | 3.7/8 (3–4) | 16.7/19 (16–18) | 12/13 | 5.3/15 (4–7) |
| gen | 3/3 | 10/19 | 4.3/7 (4–5) | 4.3/8 (3–5) | 18/19 | 12/13 | 4/15 (3–5) |
| gen-rot | 3/3 | 10.7/19 (10–11) | 3.7/7 (3–4) | 4.7/8 (4–5) | 18/19 | 12.3/13 (12–13) | 5.7/15 (5–6) |
| gen10k | 3/3 | 9.7/19 (9–11) | 3/7 (2–4) | 3.3/8 (3–4) | 17.7/19 (17–18) | 12/13 | 3.3/15 (3–4) |
| gen-rot-1280 | 3/3 | 18.7/19 (18–19) | 5.3/7 (5–6) | 6.3/8 (6–7) | 18.7/19 (18–19) | 12/13 | 9.3/15 (7–11) |

For all 21 runs scored, the causes add up to store.json's counts.

## Side questions

**Rotation** (no rule; read off): `rot` against `base`, `gen-rot` against
`gen`.

| arm | seeds | real-val-rot: mAP@0.5 date | real-val-rot: dates found | synth-val sideways: targets found | store sideways: expiry found |
|---|---:|---:|---:|---:|---:|
| base | 3/3 | 0.000 | 0/187 | 0/300 | 0/15 |
| rot | 3/3 | 0.984 (0.982–0.987) | 183.7/187 (183–184) | 230.7/300 (229–232) | 1/15 |
| gen | 3/3 | 0.544 (0.515–0.582) | 103.7/187 (101–105) | 296/300 (295–297) | 0.7/15 (0–1) |
| gen-rot | 3/3 | 0.971 (0.967–0.973) | 178.7/187 (178–180) | 294.3/300 (294–295) | 0.3/15 (0–1) |

**Data size** (decision 14): `gen` against `gen10k`, the same mix and
budget with 50,000 or 10,000 distinct generated photos.

| arm | seeds | real-val | real-val-rot | real-val-small | synth-val | hard sets, mean |
|---|---:|---:|---:|---:|---:|---:|
| gen | 3/3 | 0.994 | 0.544 (0.515–0.582) | 0.921 (0.915–0.930) | 0.985 (0.985–0.986) | 0.817 (0.810–0.828) |
| gen10k | 3/3 | 0.992 (0.989–0.994) | 0.509 (0.481–0.527) | 0.913 (0.907–0.923) | 0.979 (0.978–0.980) | 0.800 (0.789–0.807) |

**By the rule: more distinct photos help: 200,000 becomes a Later candidate.** Mean mAP@0.5 for dates on the hard sets: `gen` 0.8167, `gen10k` 0.8004 (+0.0163; the rule needs more than +0.01).

**synth-val by stratum**, targets found at the threshold:

| arm | seeds | clean | sideways | month-word | small | dot-matrix | curved | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gen | 3/3 | 295.7/300 (295–296) | 296/300 (295–297) | 296.3/300 (296–297) | 273/300 (268–279) | 284.7/300 (284–285) | 293/300 (292–294) | 277.3/300 (275–281) |
| gen10k | 3/3 | 293/300 (292–294) | 291/300 (289–292) | 293.7/300 (293–294) | 262.7/300 (257–267) | 278.7/300 (275–281) | 285.7/300 (285–286) | 267.3/300 (264–270) |

**The store photos** (S8), expiry dates found at the threshold, by failure tag:

| detector | seeds | sideways | dot-matrix | small-print | clean | all |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/15 | 14.7/20 (12–17) | 0/15 | 7/31 (6–8) | 21.7/81 (18–25) |
| gen | 3/3 | 0.7/15 (0–1) | 18.7/20 (18–19) | 0.7/15 (0–1) | 18.3/31 (18–19) | 38.3/81 (37–40) |
| gen10k | 3/3 | 0.3/15 (0–1) | 18/20 (17–19) | 0.3/15 (0–1) | 18.3/31 (17–19) | 37/81 (34–39) |

**Input size** (decision 13): `gen-rot` at 1,280 px against 640 px.

**real-val-small first** (real print at half its size):

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| gen-rot | 3/3 | 0.911 (0.907–0.915) | 160.3/186 (157–162) | 162.3/186 (158–165) | 179.7/186 (179–181) | 0.884 (0.835–0.913) | 0.52 (0.37–0.78) |
| gen-rot-1280 | 3/3 | 0.986 (0.985–0.990) | 181.3/186 (181–182) | 181.3/186 (181–182) | 185.7/186 (185–186) | 0.954 (0.948–0.958) | 0.21 (0.20–0.24) |

mAP@0.5 for dates on every suite set:

| arm | seeds | real-val | real-val-rot | real-val-small | synth-val | hard sets, mean |
|---|---:|---:|---:|---:|---:|---:|
| gen-rot | 3/3 | 0.994 (0.994–0.995) | 0.971 (0.967–0.973) | 0.911 (0.907–0.915) | 0.986 (0.984–0.987) | 0.956 (0.953–0.957) |
| gen-rot-1280 | 3/3 | 0.995 (0.994–0.995) | 0.988 (0.984–0.991) | 0.986 (0.985–0.990) | 0.994 (0.994–0.995) | 0.989 (0.989–0.990) |

**The store photos** (S8), expiry dates found at the threshold, by failure tag:

| detector | seeds | sideways | dot-matrix | small-print | clean | all |
|---|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0/15 | 14.7/20 (12–17) | 0/15 | 7/31 (6–8) | 21.7/81 (18–25) |
| gen-rot | 3/3 | 0.3/15 (0–1) | 17.3/20 (16–18) | 0/15 | 16/31 (15–17) | 33.7/81 (33–35) |
| gen-rot-1280 | 3/3 | 2.7/15 (2–3) | 19/20 | 5.3/15 (2–9) | 19/31 (15–23) | 46/81 (39–53) |

## ExpDate test

Scored once, for the M2 detector's three seeds, next to M1's.

| detector | seeds | expiry found | expiry at IoU 0.3 | expiry boxed at any confidence | all dates found | precision | false boxes / photo | mAP@0.5 date |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 646/665 (637–651) | 647.7/665 (638–653) | – | 707.3/733 (697–713) | 0.982 (0.974–0.987) | 0.02 (0.01–0.03) | 0.993 (0.992–0.993) |
| gen-rot-1280 | 3/3 | 662.7/665 (662–663) | 663/665 | 664.7/665 (664–665) | 730.7/733 (730–731) | 0.986 (0.981–0.989) | 0.02 (0.01–0.02) | 0.995 |

## The generated data

50,000 photos (spec `3fc14178…`, check run `20261023-234116-m2-generate-check`): 68,480 blocks placed, 1,219 skipped (no free place), 344 skipped (its ink stayed too faint); 109,628 generated dates.

| share | spec | measured | n |
|---|---:|---:|---:|
| photos with 2 blocks | 0.400 | 0.401 | 50000 |
| background from train_real | 0.500 | 0.501 | 50000 |
| background from train_synth_clean | 0.500 | 0.498 | 50000 |
| kind: single | 0.350 | 0.347 | 70043 |
| kind: pair | 0.350 | 0.352 | 70043 |
| kind: panel | 0.200 | 0.201 | 70043 |
| kind: range | 0.050 | 0.050 | 70043 |
| kind: relative | 0.050 | 0.050 | 70043 |
| style: solid | 0.400 | 0.400 | 70043 |
| style: dot-matrix | 0.350 | 0.352 | 70043 |
| style: label | 0.150 | 0.148 | 70043 |
| style: embossed | 0.100 | 0.100 | 70043 |
| band: normal | 0.700 | 0.697 | 70043 |
| band: small | 0.300 | 0.303 | 70043 |
| orientation: upright | 0.550 | 0.557 | 68480 |
| orientation: tilted | 0.150 | 0.149 | 68480 |
| orientation: sideways | 0.300 | 0.294 | 68480 |
| surface: flat | 0.750 | 0.750 | 68480 |
| surface: cylinder | 0.150 | 0.149 | 68480 |
| surface: bend | 0.100 | 0.101 | 68480 |
| perspective | 0.400 | 0.400 | 68480 |
| inks drawn: colourful (solid, dot-matrix) | 0.150 | 0.148 | 56619 |
| inks drawn: low contrast (solid, dot-matrix) | 0.200 | 0.199 | 56619 |
| ink defects: a scratch | 0.100 | 0.098 | 61592 |
| dates: month as a word (ranges aside) | 0.450 | 0.448 | 102782 |
| month words: full name | 0.500 | 0.498 | 46065 |
| numeric dates: a day printed | 0.700 | 0.697 | 56717 |
| month-word dates: a day printed | 0.400 | 0.394 | 46065 |
| month words with a day: an ordinal | 0.500 | 0.496 | 18128 |
| camera: uneven | 0.300 | 0.300 | 50000 |
| camera: dim | 0.100 | 0.100 | 50000 |
| camera: glare | 0.150 | 0.151 | 50000 |
| camera: defocus | 0.400 | 0.400 | 50000 |
| camera: motion | 0.100 | 0.100 | 50000 |
| camera: cast | 0.300 | 0.297 | 50000 |
| camera: noise (and every dim photo) | 0.550 | 0.551 | 50000 |

## Runs

Every M2 run on one H100. Images per second count epochs 2 to the last but
one, without start-up and the final validation.

| run | attempts | state | photos per epoch | epochs | train minutes | images/s | threshold |
|---|---:|---|---:|---:|---:|---:|---:|
| m2-smoke | 1 | scored | 256 | 1 | 2.2 | – | 0.094 |
| m2-speed-a | 1 | finished | 12781 | 3 | 4.3 | 364 | – |
| m2-speed-b | 1 | finished | 12781 | 3 | 2.8 | 364 | – |
| m2-speed-c | 1 | finished | 12781 | 3 | 3.0 | 370 | – |
| m2-pilot | 1 | scored | 66529 | 4 | 17.5 | 359 | 0.222 |
| m2-base-s0 | 1 | scored | 12781 | 200 | 120.0 | 359 | 0.150 |
| m2-base-s1 | 1 | scored | 12781 | 200 | 119.2 | 362 | 0.153 |
| m2-base-s2 | 1 | scored | 12781 | 200 | 120.1 | 359 | 0.303 |
| m2-rot-s0 | 1 | scored | 25562 | 100 | 118.8 | 369 | 0.211 |
| m2-rot-s1 | 1 | scored | 25562 | 100 | 118.9 | 364 | 0.161 |
| m2-rot-s2 | 1 | scored | 25562 | 100 | 119.5 | 362 | 0.251 |
| m2-gen-s0 | 1 | scored | 66529 | 38 | 119.9 | 360 | 0.096 |
| m2-gen-s1 | 1 | scored | 66529 | 38 | 119.6 | 363 | 0.145 |
| m2-gen-s2 | 1 | scored | 66529 | 38 | 119.8 | 360 | 0.069 |
| m2-gen-rot-s0 | 1 | scored | 79310 | 32 | 119.7 | 364 | 0.322 |
| m2-gen-rot-s1 | 1 | scored | 79310 | 32 | 120.0 | 361 | 0.316 |
| m2-gen-rot-s2 | 1 | scored | 79310 | 32 | 118.9 | 364 | 0.348 |
| m2-gen10k-s0 | 1 | scored | 66529 | 38 | 119.4 | 359 | 0.146 |
| m2-gen10k-s1 | 1 | scored | 66529 | 38 | 117.8 | 364 | 0.249 |
| m2-gen10k-s2 | 1 | scored | 66529 | 38 | 117.5 | 365 | 0.153 |
| m2-gen-rot-1280-s0 | 1 | scored | 79310 | 32 | 201.8 | 211 | 0.342 |
| m2-gen-rot-1280-s1 | 1 | scored | 79310 | 32 | 203.4 | 211 | 0.424 |
| m2-gen-rot-1280-s2 | 1 | scored | 79310 | 32 | 204.3 | 211 | 0.399 |

GPU time: 40.5 h training (setup and final validation included), 0.44 h scoring.

The loader (decision 15):

| run | workers | RAM cache | photos | epoch 1 s | epoch 2 s | epoch 3 s | images/s, epoch 2 | train minutes |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| m2-speed-a | 8 | no | 12781 | 55.6 | 35.1 | 50.1 | 364.0 | 4.3 |
| m2-speed-b | 24 | no | 12781 | 54.6 | 35.1 | 50.7 | 364.0 | 2.8 |
| m2-speed-c | 24 | filled (needs 11.2 of 57.6 GB) | 12781 | 55.1 | 34.5 | 51.3 | 370.3 | 3.0 |

**Decision, by the rule: `a` (8 workers, no cache)**, its 364.0 images/s are 98.3% of the fastest, `c`'s 370.3: within 10%, and it is the simplest such setting. The chain's own record (`results/runs/m2-decisions.json`) agrees.

Made by `python scripts/m2_report.py` from the run records in
`results/runs/` and, for the causes, `data/store/runs/`.
