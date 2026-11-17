# M2b / model size: does a bigger detector find more store dates?

## Question

Trained the same way, does a bigger YOLO11 (`s`, and an `m` probe) find
more of v1-val's dates than `n`, at step 1 (public data only) and at step 2
(fine-tuned on store photos), without forgetting ExpDate? And which arm
does R3, written before any run, make the M2b detector?

## Why it matters

M1 couldn't tell sizes apart: ExpDate's validation photos were too easy
(`n640` to `s1280` scored 0.991 to 0.995), so its rule took the cheapest.
Store print is small and soft, and the M2 detector's seeds disagree on it
([`docs/m2b-store-and-size.md`](../../docs/m2b-store-and-size.md), Cause 1).
On COCO `s` beats `n` by about 7.5 mAP, so a bigger model might be less
fragile there. It also costs time on every photo it reads and in training.
R3 weighs the two: the cheapest candidate whose v1-val mAP is within the
noise band of the best.

## Method

| Arm | Starts from | Step | Seeds |
|---|---|---|---|
| `n1280` (the control) | M2's `gen-rot-1280` `last.pt`, not trained again | 1 | 0, 1, 2 |
| `s1280` | `yolo11s` (COCO): `gen-rot-1280`'s lists, mix, budget (2,537,920 images, 32 epochs of 79,310) and recipe | 1 | 0, 1, 2 |
| `m1280` | `yolo11m` (COCO), the same | 1 | 0; 1 and 2 only by R1 |
| `n1280-ft`, `s1280-ft`, `m1280-ft` | each size's step-1 `last.pt`, seed for seed, fine-tuned with R2's recipe (`ft`: the whole network, `lr0` 0.001, 10 epochs of 24,992 photos, 20% of them store lines) | 2 | 0, 1, 2 (`m` only by R1) |

Only the weights differ within a step: the same lists, seeds, batch (16),
input size (1,280 px), MuSGD recipe and final weights (`last.pt`). Step 2
and the M2b suite are as [`m2b-fine-tune`](../m2b-fine-tune/README.md)
describes them. **v1-val** is 34 store photos of 16 pack designs, none of
them in training, with 63 dates, 29 of them expiries. Counts are at each
run's `threshold_store` (the best F2 for dates on v1-val) unless a column
says M2's; mAP doesn't depend on a threshold.

**Decision rules, written before any run** (`bestbefore.m2b`, applied by
the chain on the cluster and worked out again here by `summarise.py` with
the same function, `m2b.decisions()`):

- **The noise band on v1-val**: the spread of mAP@0.5 for dates over the
  control's three seeds, at least 0.03; fixed before any other run was
  scored.
- **The guard** (no forgetting): mean mAP@0.5 for dates on real-val and on
  synth-val each no more than 0.01 under the control's.
- **R1** (the `m` probe): `m1280-s0` earns `m` 2 more seeds and step 2
  only if its v1-val mAP beats every `s1280` seed's and it passes the
  guard.
- **R3** (the M2b detector's arm): the candidates, cheapest first by
  inference cost, then training cost, are `n1280`, `n1280-ft`, `s1280`,
  `s1280-ft`, `m1280`, `m1280-ft` (`-ft` is R2's recipe); only those with
  all 3 seeds scored take part. Of those passing the guard, the highest
  mean v1-val mAP@0.5; any candidate within the band of it goes to the
  cheapest. The M2b detector is that arm's seed 0 at its
  `threshold_store`, unless R4 adopts the ensemble (S13).
- **The side question** (no rule, fixed before any run): `n` against `s`
  against `m`, at step 1 and step 2, on every set, with the seed spread.

**What we expected, written before any run** (the plan's table; the store
columns stand for v1-val's dates, sideways expiries, expiries with month
words and small dates). It has no `m1280-ft` row: `m`'s step 2 was only to
run if R1 passed.

| Run | v1-val mAP | sideways | real-val-rot180 | month words | small dates | real-val |
|---|---|---|---|---|---|---|
| `n1280` (M2's weights) | the baseline | low | about 0 | doubted | low | 0.995 |
| `s1280` (step 1) | a little up | a little up | about 0 | a little up | a little up | as `n` |
| `m1280` (step 1, probe) | about as `s` | | | | | as `n` |
| `n1280-ft` (step 2) | clearly up | up | up a lot | up | up | as `n` |
| `s1280-ft` | the highest | | | | up the most | as `n` |

## Result

Rebuild with:

```bash
scripts/pull_cluster_results.sh
python experiments/m2b-model-size/summarise.py --write
```

<!-- results:start -->
**The rules**, applied by the chain as their inputs came in, and worked out again here:

**The noise band: 0.1414.** The control's three seeds (M2's `gen-rot-1280`, no store training) score 0.6546, 0.7960, 0.7096 on v1-val (mAP@0.5 for dates): a spread of 0.1414, so the band is 0.1414 (the spread, at least 0.03), fixed before any other run was scored. The chain's own record (`results/runs/m2b-decisions.json`) agrees.

**R1, the `m` probe: passes**, so `m` earns 2 more seeds and step 2. `m1280-s0` scores 0.8625 on v1-val against `s1280`'s seeds' 0.7555, 0.7845, 0.8016, above every one, and it passes the guard. The chain's own record (`results/runs/m2b-decisions.json`) agrees.

**R3**, every candidate, cheapest first (inference cost, then training cost; `-ft` stands for R2's recipe): the guard (real-val and synth-val, no more than 0.01 under the control), then the mean on v1-val; mAP@0.5 for dates over the 3 seeds, and each mean's distance from the band's edge (the best eligible mean less the band):

| arm | seeds | real-val | synth-val | v1-val | guard | from the band's edge | R3 |
|---|---:|---:|---:|---:|---|---:|---|
| n1280 (control) | 3/3 | 0.9945 (0.9940–0.9950) | 0.9942 (0.9940–0.9945) | 0.7201 (0.6546–0.7960) | passes | -0.1253 | below the band |
| n1280-ft | 3/3 | 0.9946 (0.9939–0.9950) | 0.9927 (0.9923–0.9933) | 0.9415 (0.9362–0.9514) | passes | +0.0961 | **chosen** |
| s1280 | 3/3 | 0.9948 (0.9947–0.9949) | 0.9949 (0.9948–0.9949) | 0.7805 (0.7555–0.8016) | passes | -0.0648 | below the band |
| s1280-ft | 3/3 | 0.9950 (0.9949–0.9950) | 0.9948 (0.9946–0.9949) | 0.9648 (0.9606–0.9721) | passes | +0.1194 | in the band |
| m1280 | 3/3 | 0.9950 | 0.9950 | 0.8444 (0.8242–0.8625) | passes | -0.0010 | below the band |
| m1280-ft | 3/3 | 0.9950 | 0.9950 (0.9949–0.9950) | 0.9868 (0.9820–0.9919) | passes | +0.1414 | in the band |

**R3, by the rule: `n1280-ft`**, the cheapest candidate within the band (0.1414) of the best, `m1280-ft`'s 0.9868: the band's edge is 0.8454, and `n1280-ft`'s 0.9415 sits 0.0961 above it. Every candidate passes the guard. Within the band too, and dearer: `s1280-ft` (+0.1194), `m1280-ft` (+0.1414). Below the edge: `n1280` (-0.1253), `s1280` (-0.0648), `m1280` (-0.0010). Nearest the edge: `m1280`, 0.0010 below it. So the M2b detector is `n1280-ft-s0` (v1-val 0.9514) at its threshold_store, 0.1641, unless R4 adopts the ensemble (S13). The chain's own record (`results/runs/m2b-decisions.json`) agrees.

Worked out again here with `m2b.decisions()`, the chain's whole record (`results/runs/m2b-decisions.json`: noise_store, R1, R2, R3) is the same.

**mAP@0.5 for dates on every set of the M2b suite**, step 1 (public data; `n1280` is M2's weights) then step 2 (fine-tuned on store photos with R2's recipe):

| arm | seeds | real-val | real-val-rot | real-val-small | synth-val | real-val-rot180 | synth-val-v2 | v1-val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.995 (0.994–0.995) | 0.988 (0.984–0.991) | 0.986 (0.985–0.990) | 0.994 (0.994–0.995) | 0.435 (0.386–0.496) | 0.979 (0.978–0.980) | 0.720 (0.655–0.796) |
| s1280 | 3/3 | 0.995 | 0.993 (0.993–0.994) | 0.994 (0.993–0.994) | 0.995 | 0.295 (0.238–0.337) | 0.980 (0.979–0.980) | 0.781 (0.755–0.802) |
| m1280 | 3/3 | 0.995 | 0.995 | 0.995 | 0.995 | 0.234 (0.215–0.263) | 0.974 | 0.844 (0.824–0.863) |
| n1280-ft | 3/3 | 0.995 (0.994–0.995) | 0.983 (0.976–0.989) | 0.987 (0.985–0.991) | 0.993 (0.992–0.993) | 0.953 (0.947–0.963) | 0.992 (0.992–0.993) | 0.942 (0.936–0.951) |
| s1280-ft | 3/3 | 0.995 | 0.991 (0.988–0.993) | 0.991 (0.985–0.994) | 0.995 | 0.970 (0.964–0.974) | 0.995 | 0.965 (0.961–0.972) |
| m1280-ft | 3/3 | 0.995 | 0.994 (0.994–0.995) | 0.995 | 0.995 | 0.983 (0.982–0.986) | 0.995 | 0.987 (0.982–0.992) |

**v1-val by class**, mAP@0.5: the dates, and the keyword boxes beside them (`due` the expiry's word, `prod` the manufacture's, `code` a batch code or a time):

| arm | seeds | date | due | prod | code |
|---|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.720 (0.655–0.796) | 0.321 (0.275–0.349) | 0.047 (0.030–0.060) | 0.048 (0.031–0.062) |
| s1280 | 3/3 | 0.781 (0.755–0.802) | 0.302 (0.257–0.357) | 0.017 (0.006–0.025) | 0.001 (0.000–0.004) |
| m1280 | 3/3 | 0.844 (0.824–0.863) | 0.302 (0.274–0.334) | 0.055 (0.011–0.117) | 0.003 (0.000–0.009) |
| n1280-ft | 3/3 | 0.942 (0.936–0.951) | 0.950 (0.941–0.960) | 0.717 (0.674–0.751) | 0.408 (0.383–0.425) |
| s1280-ft | 3/3 | 0.965 (0.961–0.972) | 0.955 (0.949–0.965) | 0.736 (0.713–0.753) | 0.538 (0.441–0.621) |
| m1280-ft | 3/3 | 0.987 (0.982–0.992) | 0.957 (0.953–0.963) | 0.884 (0.851–0.936) | 0.529 (0.457–0.620) |

Step 1's early read, M2's suite (`suite.json`, scored as each step-1 run finished; `n`'s are M2's own records), for 9 of 9 step-1 runs: on the four sets it shares with the M2b suite its mAP@0.5 for dates differs by at most 0.0000, and its threshold from the M2b suite's `threshold_m2` by at most 0.0000.

**real-val-rot180** (ExpDate's validation photos turned 180°), at each run's threshold:

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.435 (0.386–0.496) | 94.7/187 (90–97) | 103/187 (100–105) | 130.7/187 (119–144) | 0.457 (0.348–0.608) | 0.76 (0.35–1.10) |
| s1280 | 3/3 | 0.295 (0.238–0.337) | 67/187 (49–89) | 76/187 (56–97) | 86.3/187 (69–101) | 0.440 (0.405–0.470) | 0.53 (0.37–0.79) |
| m1280 | 3/3 | 0.234 (0.215–0.263) | 46/187 (41–49) | 51/187 (46–55) | 58/187 (54–66) | 0.454 (0.329–0.583) | 0.37 (0.21–0.59) |
| n1280-ft | 3/3 | 0.953 (0.947–0.963) | 178.7/187 (177–180) | 181/187 | 185/187 (184–187) | 0.788 (0.723–0.844) | 0.30 (0.20–0.42) |
| s1280-ft | 3/3 | 0.970 (0.964–0.974) | 181.3/187 (179–184) | 182/187 (180–185) | 185.7/187 (185–187) | 0.857 (0.790–0.927) | 0.19 (0.08–0.30) |
| m1280-ft | 3/3 | 0.983 (0.982–0.986) | 176.7/187 (172–182) | 177.3/187 (174–182) | 186/187 (185–187) | 0.955 (0.951–0.961) | 0.05 (0.04–0.05) |

**synth-val-v2 by stratum**, targets found at each run's threshold:

| arm | seeds | upside-down | glued | line-pair | dense-panel | window | embossed | worn | nozzle-rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 188.3/300 (178–196) | 297/300 | 294.3/300 (292–296) | 299/300 (297–300) | 299.7/300 (299–300) | 297.3/300 (296–298) | 297.7/300 (297–298) | 290.3/300 (288–293) |
| s1280 | 3/3 | 163.3/300 (138–183) | 299.7/300 (299–300) | 294.7/300 (293–296) | 299.7/300 (299–300) | 300/300 | 299.3/300 (299–300) | 299.3/300 (299–300) | 293.7/300 (292–296) |
| m1280 | 3/3 | 145.7/300 (134–162) | 300/300 | 294.3/300 (292–297) | 299.7/300 (299–300) | 300/300 | 300/300 | 299.7/300 (299–300) | 294.7/300 (293–296) |
| n1280-ft | 3/3 | 291/300 (288–293) | 296.7/300 (296–297) | 292.7/300 (291–294) | 293.3/300 (290–295) | 299.3/300 (299–300) | 291.3/300 (291–292) | 290.7/300 (288–292) | 278.7/300 (277–281) |
| s1280-ft | 3/3 | 296.7/300 (293–299) | 297.3/300 (296–299) | 295.7/300 (294–298) | 297/300 (295–298) | 300/300 | 296.7/300 (295–299) | 295/300 (292–297) | 285/300 (276–292) |
| m1280-ft | 3/3 | 295/300 (294–296) | 300/300 | 294.7/300 (294–295) | 298.7/300 (298–299) | 299.7/300 (299–300) | 298.7/300 (298–299) | 293/300 (291–294) | 283.7/300 (280–286) |

and boxed at any confidence:

| arm | seeds | upside-down | glued | line-pair | dense-panel | window | embossed | worn | nozzle-rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 226.7/300 (214–238) | 298.7/300 (298–299) | 297.3/300 (296–299) | 300/300 | 299.7/300 (299–300) | 299.3/300 (299–300) | 298.7/300 (298–299) | 293/300 (291–294) |
| s1280 | 3/3 | 199.7/300 (189–206) | 300/300 | 297/300 (295–299) | 300/300 | 300/300 | 299.3/300 (299–300) | 300/300 | 295.7/300 (295–296) |
| m1280 | 3/3 | 166.3/300 (159–174) | 300/300 | 296/300 (295–297) | 300/300 | 300/300 | 300/300 | 299.7/300 (299–300) | 295.7/300 (295–296) |
| n1280-ft | 3/3 | 298.7/300 (298–299) | 298.3/300 (298–299) | 299/300 (298–300) | 299/300 | 300/300 | 296/300 (295–297) | 298/300 (297–299) | 295.3/300 (294–297) |
| s1280-ft | 3/3 | 299.7/300 (299–300) | 299.7/300 (299–300) | 299.3/300 (299–300) | 299.7/300 (299–300) | 300/300 | 300/300 | 299.3/300 (299–300) | 297.7/300 (297–298) |
| m1280-ft | 3/3 | 300/300 | 300/300 | 300/300 | 300/300 | 300/300 | 300/300 | 299.7/300 (299–300) | 297.3/300 (296–298) |

**v1-val at both thresholds**: each run's own (threshold_store, the best F2 on v1-val) and M2's rule's (the best F2 on the three real sets, pooled):

| arm | threshold | seeds | value | dates found | expiry found | precision | false boxes / photo |
|---|---|---:|---:|---:|---:|---:|---:|
| n1280 (control) | store (v1-val) | 3/3 | 0.0191 (0.0055–0.0387) | 45.7/63 (44–48) | 21.7/29 (20–23) | 0.601 (0.530–0.739) | 0.93 (0.50–1.15) |
| n1280 (control) | M2's (real sets) | 3/3 | 0.3885 (0.3424–0.4243) | 31.3/63 (27–38) | 14/29 (11–18) | 0.950 (0.935–0.964) | 0.05 (0.03–0.06) |
| s1280 | store (v1-val) | 3/3 | 0.0089 (0.0028–0.0121) | 48/63 (45–51) | 22.7/29 (21–24) | 0.740 (0.630–0.828) | 0.53 (0.29–0.88) |
| s1280 | M2's (real sets) | 3/3 | 0.3857 (0.3730–0.4108) | 33/63 (32–35) | 15.7/29 (15–17) | 0.970 (0.941–1.000) | 0.03 (0.00–0.06) |
| m1280 | store (v1-val) | 3/3 | 0.0054 (0.0025–0.0073) | 54/63 | 24.3/29 (24–25) | 0.829 (0.794–0.900) | 0.33 (0.18–0.41) |
| m1280 | M2's (real sets) | 3/3 | 0.7329 (0.6801–0.7914) | 37/63 (34–43) | 18/29 (16–22) | 0.990 (0.971–1.000) | 0.01 (0.00–0.03) |
| n1280-ft | store (v1-val) | 3/3 | 0.1711 (0.1280–0.2211) | 60/63 | 27.7/29 (27–29) | 0.776 (0.750–0.789) | 0.51 (0.47–0.59) |
| n1280-ft | M2's (real sets) | 3/3 | 0.3910 (0.2654–0.4836) | 56/63 (55–57) | 25.7/29 (25–26) | 0.878 (0.824–0.932) | 0.24 (0.12–0.35) |
| s1280-ft | store (v1-val) | 3/3 | 0.2960 (0.1276–0.5527) | 59.7/63 (58–62) | 27.7/29 (27–29) | 0.854 (0.775–0.921) | 0.31 (0.15–0.53) |
| s1280-ft | M2's (real sets) | 3/3 | 0.4350 (0.4054–0.4889) | 57.3/63 (57–58) | 26.3/29 (26–27) | 0.882 (0.864–0.892) | 0.23 (0.21–0.26) |
| m1280-ft | store (v1-val) | 3/3 | 0.6514 (0.5841–0.6866) | 61.3/63 (60–62) | 29/29 | 0.995 (0.984–1.000) | 0.01 (0.00–0.03) |
| m1280-ft | M2's (real sets) | 3/3 | 0.7681 (0.7392–0.7997) | 57.7/63 (57–58) | 28/29 (27–29) | 0.994 (0.983–1.000) | 0.01 (0.00–0.03) |

**v1-val by failure tag**, expiry dates found at each run's threshold:

| arm | seeds | sideways | dot-matrix | small-print | clean | all expiry dates | all dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 3.3/6 (3–4) | 3/3 | 3.7/7 (3–4) | 11.7/13 (10–13) | 21.7/29 (20–23) | 45.7/63 (44–48) |
| s1280 | 3/3 | 2.7/6 (2–3) | 3/3 | 4.7/7 (4–5) | 12.3/13 (12–13) | 22.7/29 (21–24) | 48/63 (45–51) |
| m1280 | 3/3 | 3.7/6 (3–4) | 2.7/3 (2–3) | 5/7 (4–6) | 13/13 | 24.3/29 (24–25) | 54/63 |
| n1280-ft | 3/3 | 5.3/6 (5–6) | 3/3 | 6.3/7 (6–7) | 13/13 | 27.7/29 (27–29) | 60/63 |
| s1280-ft | 3/3 | 5.7/6 (5–6) | 2.3/3 (2–3) | 6.7/7 (6–7) | 13/13 | 27.7/29 (27–29) | 59.7/63 (58–62) |
| m1280-ft | 3/3 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 61.3/63 (60–62) |

found at M2's threshold:

| arm | seeds | sideways | dot-matrix | small-print | clean | all expiry dates | all dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 1.7/6 (1–2) | 3/3 | 1.3/7 (0–4) | 8/13 (6–10) | 14/29 (11–18) | 31.3/63 (27–38) |
| s1280 | 3/3 | 1.3/6 (1–2) | 3/3 | 2.3/7 (2–3) | 9/13 (7–11) | 15.7/29 (15–17) | 33/63 (32–35) |
| m1280 | 3/3 | 2.3/6 (2–3) | 2.7/3 (2–3) | 2.7/7 (2–4) | 10.3/13 (9–12) | 18/29 (16–22) | 37/63 (34–43) |
| n1280-ft | 3/3 | 4.3/6 (4–5) | 3/3 | 6/7 | 12.3/13 (11–13) | 25.7/29 (25–26) | 56/63 (55–57) |
| s1280-ft | 3/3 | 5.3/6 (5–6) | 2.3/3 (2–3) | 5.7/7 (5–6) | 13/13 | 26.3/29 (26–27) | 57.3/63 (57–58) |
| m1280-ft | 3/3 | 5.3/6 (5–6) | 2.7/3 (2–3) | 7/7 | 13/13 | 28/29 (27–29) | 57.7/63 (57–58) |

and boxed at any confidence:

| arm | seeds | sideways | dot-matrix | small-print | clean | all expiry dates | all dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 4/6 | 3/3 | 5.7/7 (5–6) | 12.3/13 (12–13) | 25/29 | 53.7/63 (52–56) |
| s1280 | 3/3 | 3.3/6 (3–4) | 3/3 | 4.7/7 (4–5) | 13/13 | 24/29 (23–25) | 51.3/63 (50–52) |
| m1280 | 3/3 | 4/6 | 2.7/3 (2–3) | 5.3/7 (4–6) | 13/13 | 25/29 (24–26) | 55.7/63 (54–58) |
| n1280-ft | 3/3 | 5.7/6 (5–6) | 3/3 | 7/7 | 13/13 | 28.7/29 (28–29) | 62.7/63 (62–63) |
| s1280-ft | 3/3 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 62.7/63 (62–63) |
| m1280-ft | 3/3 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |

**v1-val's dates by size** (Cause 4's bands: a box's short side over its photo's long side), found at each run's threshold:

| arm | seeds | under 1.5% | 1.5-2.5% | 2.5% and up |
|---|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 12.3/23 (12–13) | 22.7/28 (21–24) | 10.7/12 (10–12) |
| s1280 | 3/3 | 14.3/23 (12–16) | 22.3/28 (21–24) | 11.3/12 (11–12) |
| m1280 | 3/3 | 18.3/23 (18–19) | 23.7/28 (23–24) | 12/12 |
| n1280-ft | 3/3 | 21.7/23 (21–22) | 26.3/28 (26–27) | 12/12 |
| s1280-ft | 3/3 | 22/23 (21–23) | 25.7/28 (25–27) | 12/12 |
| m1280-ft | 3/3 | 22.7/23 (22–23) | 26.7/28 (26–27) | 12/12 |

found at M2's threshold:

| arm | seeds | under 1.5% | 1.5-2.5% | 2.5% and up |
|---|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 4.7/23 (3–8) | 16.7/28 (16–18) | 10/12 (8–12) |
| s1280 | 3/3 | 6.3/23 (5–8) | 16.7/28 (14–19) | 10/12 |
| m1280 | 3/3 | 7.3/23 (6–10) | 18.7/28 (17–21) | 11/12 (10–12) |
| n1280-ft | 3/3 | 20/23 (19–21) | 24.3/28 (23–26) | 11.7/12 (11–12) |
| s1280-ft | 3/3 | 20.7/23 (20–21) | 24.7/28 (24–25) | 12/12 |
| m1280-ft | 3/3 | 20.7/23 (20–22) | 25/28 (24–26) | 12/12 |

and boxed at any confidence:

| arm | seeds | under 1.5% | 1.5-2.5% | 2.5% and up |
|---|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 18.3/23 (18–19) | 24.3/28 (23–26) | 11/12 (10–12) |
| s1280 | 3/3 | 16.3/23 (15–17) | 23.7/28 (23–24) | 11.3/12 (11–12) |
| m1280 | 3/3 | 19.3/23 (18–20) | 24.3/28 (23–26) | 12/12 |
| n1280-ft | 3/3 | 23/23 | 27.7/28 (27–28) | 12/12 |
| s1280-ft | 3/3 | 23/23 | 27.7/28 (27–28) | 12/12 |
| m1280-ft | 3/3 | 23/23 | 28/28 | 12/12 |

**Compute by size** (GPU jobs A and B on an H100; minutes and images a second from each run's record and results.csv; the suites' minutes are the only inference time recorded):

`n`'s step 1 is M2's `gen-rot-1280` (3 of 3 seeds, from M2's own records): 203.2 (201.8–204.3) minutes a run at 211 images a second, on the lists, budget and recipe `s1280` and `m1280` trained with.

| run | trained | epochs | photos per epoch | train minutes | setup minutes | epoch 1 s | images/s | M2b suite minutes | M2 suite minutes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m2b-s1280-s0 | yes | 32 | 79,310 | 239.4 | 10.2 | 467 | 185 | 1.9 | 1.6 |
| m2b-s1280-s1 | yes | 32 | 79,310 | 230.3 | 1.9 | 469 | 186 | 2 | 1.6 |
| m2b-s1280-s2 | yes | 32 | 79,310 | 230.6 | 2.0 | 470 | 186 | 2 | 1.6 |
| m2b-m1280-s0 | yes | 32 | 79,310 | 395.6 | 1.9 | 768 | 108 | 2.3 | 2.1 |
| m2b-m1280-s1 | yes | 32 | 79,310 | 388.8 | 4.7 | 748 | 110 | 4 | 4.3 |
| m2b-m1280-s2 | yes | 32 | 79,310 | 392.4 | 8.6 | 754 | 110 | 4 | 4.3 |
| m2b-n1280-ft-s0 | yes | 10 | 24,992 | 25.4 | 4.7 | 141 | 204 | 1.8 | – |
| m2b-n1280-ft-s1 | yes | 10 | 24,992 | 23.1 | 2.5 | 132 | 203 | 1.8 | – |
| m2b-n1280-ft-s2 | yes | 10 | 24,992 | 22.7 | 2.2 | 132 | 204 | 1.7 | – |
| m2b-s1280-ft-s0 | yes | 10 | 24,992 | 23.3 | 1.0 | 141 | 190 | 2 | – |
| m2b-s1280-ft-s1 | yes | 10 | 24,992 | 23.2 | 0.8 | 141 | 189 | 2 | – |
| m2b-s1280-ft-s2 | yes | 10 | 24,992 | 23.1 | 0.7 | 141 | 189 | 1.9 | – |
| m2b-m1280-ft-s0 | yes | 10 | 24,992 | 43 | 4.7 | 238 | 110 | 2.4 | – |
| m2b-m1280-ft-s1 | yes | 10 | 24,992 | 41.3 | 3.1 | 232 | 110 | 2.4 | – |
| m2b-m1280-ft-s2 | yes | 10 | 24,992 | 40.5 | 2.6 | 232 | 111 | 2.4 | – |

Over these runs: 35.71 h of training (setup included) and 0.83 h of scoring, from their records.
<!-- results:end -->

## What it means

Step 1's `s1280` seeds and `m1280-s0` were trained by GPU job A (`73831`,
`cn6`, 2026-11-08 to 11-09, at `2b68e2e`); `m1280-s1` and `-s2` and all of
step 2 by job B (`73832`, `cn5`, 2026-11-13 to 11-14, at `04451ff`,
`4d3166e` and `5739871`, between which no file M2b trains, scores or
chains with changed). So `m1280`'s seeds trained at two commits; their
records show the same Ultralytics arguments, and the same torch and
Ultralytics versions. Every run was scored on the M2b suite by job B.
Pulled and summarised on the laptop on 2026-11-17. v1-val is small: one of
its 63 dates is 1.6 points of recall, so counts stand beside every rate.

- **R3, by the rule: `n1280-ft`.** The best mean on v1-val is
  `m1280-ft`'s 0.9868; the band is 0.1414, the control's seed spread, so
  its edge is 0.8454, and `n1280-ft`, at 0.9415, is the cheapest candidate
  above it (+0.0961). Every candidate passes the guard. `s1280-ft`
  (+0.1194) and `m1280-ft` (+0.1414) are within the band too, and dearer.
  Below the edge: `n1280` (-0.1253), `s1280` (-0.0648) and **`m1280`, only
  0.0010 below**, a sixteenth of one date. Had `m1280` been inside, R3
  wouldn't change, as `n1280-ft` comes first. The M2b detector is
  `n1280-ft-s0` (v1-val 0.9514) at its `threshold_store`, 0.1641, unless
  R4 adopts the ensemble (S13). The chain's record agrees, whole.
- **`m1280-ft` was consistently better.** On v1-val's dates it scores
  0.987 (0.982–0.992), against `s1280-ft`'s 0.965 (0.961–0.972) and
  `n1280-ft`'s 0.942 (0.936–0.951): the ranges are apart, so every `m`
  seed is above every `s` seed, and every `s` seed above every `n` seed.
  The same holds on the `prod` keyword boxes, the manufacture word M3
  reads beside a date: `m` 0.884 (0.851–0.936) against `n` 0.717
  (0.674–0.751) and `s` 0.736 (0.713–0.753). On `code` boxes `m` and `s`
  are level, 0.529 and 0.538, every seed above `n`'s 0.408 (0.383–0.425).
  On `due`, the expiry's word, the three are level (0.950 to 0.957, ranges
  overlapping). At its own threshold `m1280-ft` finds 61.3 of 63 dates and
  all 29 expiries at precision 0.995 (0.01 false boxes a photo), where
  `n1280-ft` finds 60 and 27.7 at 0.776 (0.51 a photo): `m` is surer of
  store print, its `threshold_store` 0.6514 against `n`'s 0.1711. At any
  confidence all three box 62.7 to 63 dates.
- **The band was too coarse to separate fine-tuned arms.** It is the
  control's seed spread, 14.1 points, as the rule says: right for telling
  fine-tuning from no training, which beats it (+22.1 points for `n`,
  [`m2b-fine-tune`](../m2b-fine-tune/README.md)). Fine-tuned seeds spread
  1.5 (`n`), 1.2 (`s`) and 1.0 (`m`) points. `m1280-ft` is 4.5 points
  above `n1280-ft`, about three times those spreads, and still well inside
  the band. So every fine-tuned arm ties under R3, which then falls to the
  cheapest whatever size is best; and `m1280`, with no store training,
  came within 0.1 point of tying with all of them.
- **The same band reaches R4 (S13).** An add-on is adopted only if it
  raises v1-val AP by more than the band. `n1280-ft-s0` scores 0.9514,
  4.9 points short of 1.0, so with a band of 14.1 points no add-on can be
  adopted, whatever it measures. That is the rule as written; nothing is
  changed here.
- **Step 1: bigger finds more store dates, but reads upside-down print
  worse.** On v1-val, `n` 0.720, `s` 0.781 and `m` 0.844: every `m` seed
  above every `s` seed (R1), but `s`'s range (0.755–0.802) overlaps the
  control's (0.655–0.796). At each run's threshold they find 45.7, 48 and
  54 of 63 dates, and of the smallest (under 1.5% of the photo) 12.3, 14.3
  and 18.3 of 23. On ExpDate's turned and shrunk copies bigger helps:
  real-val-rot 0.988, 0.993, 0.995; real-val-small 0.986, 0.994, 0.995.
  But on real-val-rot180 the order turns over: 0.435, 0.295 and 0.234, and
  synth-val-v2's upside-down targets are boxed at any confidence 226.7,
  199.7 and 166.3 times of 300. Step 1 trains on 90° and 270° turns only;
  the bigger the model, the less a half turn it never saw carries over
  (a reading, not tested). It is also why `m1280`'s synth-val-v2 (0.974)
  is under `n`'s (0.979).
- **Step 2 closes that gap and keeps the order.** With the real 180°
  copies in the replay, real-val-rot180 is 0.953, 0.970 and 0.983, and
  upside-down targets are boxed 298.7, 299.7 and 300 times of 300.
  Real-val is 0.995 for all three. On real-val-rot, `n1280-ft` (0.983)
  falls 0.5 points under the control, where `s1280-ft` (0.991) and
  `m1280-ft` (0.994) don't.
- **Where a bigger model doesn't beat a smaller one, or the control:**
  - `s1280` at step 1 against the control: sideways expiries 2.7 of 6
    against 3.3 (boxed at any confidence 3.3 against 4); the smallest dates
    boxed at any confidence 16.3 of 23 against 18.3, and dates 51.3 of 63
    against 53.7;
  - dot-matrix: `m1280` (2.7 of 3) and `s1280-ft` (2.3 of 3) miss a
    dot-matrix expiry in a seed at their thresholds, where every `n` run
    finds 3 of 3; all box 3 at any confidence but `m1280` (2.7). v1-val has
    3 dot-matrix expiries, so this is one date;
  - real-val-rot180 at each run's own threshold: `m1280-ft` finds 176.7 of
    187, fewer than `n1280-ft` (178.7) and `s1280-ft` (181.3), at precision
    0.955 against 0.788 and 0.857; its mAP is still the highest;
  - synth-val-v2 at each run's threshold: `s1280-ft` is ahead of
    `m1280-ft` on upside-down (296.7 against 295), worn (295 against 293)
    and nozzle-row print (285 against 283.7), all within 2 targets;
  - the smallest store dates after fine-tuning: 21.7, 22 and 22.7 of 23,
    one date from `n` to `m`.
- **Against what we expected:** `s1280` at step 1 is "a little up" on
  v1-val (+6.0 points, inside the band), but not on sideways expiries (2.7
  against 3.3 of 6), and its real-val-rot180 isn't "about 0" but 0.295,
  under `n`'s 0.435; on small dates it finds 2 more at its threshold and
  boxes 2 fewer at any confidence; real-val is as `n`'s. `m1280` isn't
  "about as `s`": every seed is above every `s` seed (0.844 against 0.781),
  which is why R1 passed. `s1280-ft` is the highest of the arms planned
  (`n` and `s`), but `m1280-ft`, run by R1, is higher; on small dates it is
  "up the most" by a third of a date (22 of 23 against 21.7). Real-val is
  as `n`'s everywhere. Month words aren't split out by the M2b suite; they
  are read on v2, by cause (S14).
- **Step 1's early read on M2's suite adds nothing**: on the four sets it
  shares with the M2b suite, all 9 step-1 runs give the same mAP and M2's
  threshold to the fourth decimal.
- **Compute by size** (the table). Step 1: `n` 203.2 minutes a run at 211
  images a second (M2's own), `s` 230.3 to 239.4 at 185 to 186, `m` 388.8
  to 395.6 at 108 to 110. Step 2: `n` 22.7 to 25.4 minutes at 203 to 204,
  `s` 23.1 to 23.3 at 189 to 190, `m` 40.5 to 43 at 110 to 111. So `m`
  trains at about half `n`'s speed. S2 estimated about 3.9 hours for an
  `s1280` run (3.8 to 4.0 measured) and about 80 images a second for `m`,
  7.5 to 12.4 hours a run (108 to 110 and 6.5 to 6.6 measured, under the
  plan's 7); S10 estimated step 2's training at 23 minutes for `s` and 39
  for `m` (23.1 to 23.3 and 40.5 to 43, setup included). The only
  inference time recorded is the M2b suite's, 5,070 photos each predicted
  and validated on an H100: 1.7 to 1.8 minutes for `n1280-ft`, 1.9 to 2
  for `s1280-ft` and 2.4 for `m1280-ft`. `m1280-s1` and `-s2` took 4 where
  `-s0` took 2.3, not looked into (S11). Time per photo on a CPU is S13's.

**Decision:** by R3, the M2b detector's arm is `n1280-ft`: `n1280-ft-s0`
at its `threshold_store`, 0.1641, unless R4 adopts the ensemble in S13.
The rule was set in advance, and `n` is the fastest, which live scanning
needs. `m1280-ft` stays the fallback: if M3 loses photos to missed mfg
keyword boxes or missed dates, try it (the plan's Later).
