# M2b / step 2 on `n`: fine-tuning on store photos, and which recipe

## Question

Fine-tuned on store photos with a replay of its old training data, does
M2's detector find more of v1-val's dates than M2's own weights do at a
threshold chosen on store photos, without forgetting ExpDate? Which
recipe: the whole network at a tenth of M2's rate (`ft`), or the head
first with the backbone frozen, then everything (`ft2`)? And how much of
M2's gap on the store was its threshold alone?

## Why it matters

The M2 detector found 39 of store set v1's 81 expiry dates. It boxed 30 of
the rest below its threshold (0.342), which was chosen on ExpDate's
validation photos, where nearly every date is easy, and never boxed 12
([`docs/m2b-store-and-size.md`](../../docs/m2b-store-and-size.md), "What M2
and the store gallery tell us"). Since 2026-10-28 store photos may be
trained on. Fine-tuning brings their look (small, soft print; a few pack
designs) into training. A store-chosen threshold alone may recover the
doubted dates for free, so it is the control every fine-tuned run must
beat.

## Method

| Arm | Starts from | Replay's generated photos | Recipe | Seeds |
|---|---|---|---|---|
| `n1280` (the control) | M2's `gen-rot-1280` `last.pt`, not trained again | – | – | 0, 1, 2 |
| `n1280-ft` | the control's `last.pt`, seed for seed | gen-v2 | the whole network, `lr0` 0.001, 10 epochs | 0, 1, 2 |
| `n1280-ft2` | the same | gen-v2 | layers 0 to 10 frozen for 5 epochs at `lr0` 0.001, then everything for 5 at 0.0005 | 0, 1, 2 |
| `n1280-ft-gen50k` | the same | M2's gen50k | as `ft` | 0, 1, 2 (the side arm: [`m2b-gen-v2`](../m2b-gen-v2/README.md)) |

- **Every epoch lists 24,992 photos**, a fixed sample per run drawn with
  its seed: 4,991 store lines (20%, 161 for each of v1-train's 31 designs,
  spread over its 80 photos in 4 turns: store set v1's training part with
  v3's), then ExpDate's real training photos 709 upright and 709 turned,
  its synthetic ones 2,987 and 2,987, and 12,609 generated photos. The
  turned pools hold ExpDate's 90°, 270° and 180° copies. 10 epochs a run
  (249,920 images), warmup 3% of them, mosaic off for the last epoch (the
  last of each stage in `ft2`). Otherwise M2's recipe: `yolo11n` at 1,280
  px, batch 16, MuSGD with momentum 0.937, the final weights (`last.pt`).
- **The M2b suite** (`scripts/score_suite.py --suite m2b`): M2's four
  sets, `real-val-rot180` (ExpDate's validation photos turned 180°),
  `synth-val-v2` (8 strata of 300 generated photos, each changing one
  thing from M2's `clean`) and **`v1-val`**: 34 store photos of 16 pack
  designs, none of them in training, with 63 dates, 29 of them expiries.
  Each run gets two thresholds, both the best F2 for dates:
  `threshold_store` on v1-val, its own from M2b on, and M2's rule's, on the
  three real sets pooled. Counts are at `threshold_store` unless a column
  says otherwise; mAP doesn't depend on a threshold.

**Decision rules, written before any run** (`bestbefore.m2b`, applied by
the chain on the cluster and worked out again here by `summarise.py` with
the same function, `m2b.decisions()`):

- **The noise band on v1-val** (`noise_store`): the spread of mAP@0.5 for
  dates over the control's three seeds, at least 0.03; fixed before any
  other run was scored.
- **The guard** (no forgetting): mean mAP@0.5 for dates on real-val and on
  synth-val each no more than 0.01 under the control's.
- **R1** (the `m` probe, for step 1): `m1280-s0` earns `m` 2 more seeds
  and step 2 only if its v1-val mAP beats every `s1280` seed's and it
  passes the guard. What it says about model size is read in S12's
  `m2b-model-size`, beside `s` and `m`.
- **R2** (the recipe): of the recipes that pass the guard, the highest
  mean v1-val mAP@0.5; any recipe within the band of it goes to the
  simplest (`ft` before `ft2`). R2's recipe is the one `s` and `m` are
  fine-tuned with.
- **The threshold** (no rule, read off): M2's detector at M2's threshold
  against at `threshold_store`, on v1-val here and once on v2 (S14).

**What we expected, written before any run** (the plan's table; v1-val and
the store columns stand for v1-val's dates, sideways and small ones, and
expiries with month words):

| Run | v1-val mAP | sideways | real-val-rot180 | month words | small dates | real-val |
|---|---|---|---|---|---|---|
| `n1280` (M2's weights) | the baseline | low | about 0 | doubted | low | 0.995 |
| `n1280` at a v1-val threshold | the same | a little up | about 0 | more kept, more false boxes | more kept | the same |
| `n1280-ft` | clearly up | up | up a lot | up | up | as `n` |
| `n1280-ft2` | as `ft`, perhaps lower | | | | | as `n`, perhaps higher |

## Result

Rebuild with:

```bash
scripts/pull_cluster_results.sh
python experiments/m2b-fine-tune/summarise.py --write
```

<!-- results:start -->
**The rules**, applied by the chain as their inputs came in, and worked out again here:

**The noise band: 0.1414.** The control's three seeds (M2's `gen-rot-1280`, no store training) score 0.6546, 0.7960, 0.7096 on v1-val (mAP@0.5 for dates): a spread of 0.1414, so the band is 0.1414 (the spread, at least 0.03), fixed before any other run was scored. The chain's own record (`results/runs/m2b-decisions.json`) agrees.

**R1, the `m` probe: passes**, so `m` earns 2 more seeds and step 2. `m1280-s0` scores 0.8625 on v1-val against `s1280`'s seeds' 0.7555, 0.7845, 0.8016, above every one, and it passes the guard. The chain's own record (`results/runs/m2b-decisions.json`) agrees.

**R2**, `ft` against `ft2` on `n`: the guard (real-val and synth-val, no more than 0.01 under the control), then the mean on v1-val; mAP@0.5 for dates over the 3 seeds:

| arm | seeds | real-val | synth-val | v1-val | guard |
|---|---:|---:|---:|---:|---|
| n1280 (control) | 3/3 | 0.9945 (0.9940–0.9950) | 0.9942 (0.9940–0.9945) | 0.7201 (0.6546–0.7960) | – |
| n1280-ft | 3/3 | 0.9946 (0.9939–0.9950) | 0.9927 (0.9923–0.9933) | 0.9415 (0.9362–0.9514) | passes |
| n1280-ft2 | 3/3 | 0.9944 (0.9941–0.9948) | 0.9928 (0.9921–0.9936) | 0.9369 (0.9332–0.9412) | passes |

**R2, by the rule: `ft`**, the highest mean on v1-val, 0.9415; `ft2` 0.9369 (-0.0046). Both recipes pass the guard (its floors, the control's mean less 0.01: real-val 0.9845, synth-val 0.9842). The chain's own record (`results/runs/m2b-decisions.json`) agrees.

Worked out again here with `m2b.decisions()`, the chain's whole record (`results/runs/m2b-decisions.json`: noise_store, R1, R2, R3) is the same.

**mAP@0.5 for dates on every set of the M2b suite:**

| arm | seeds | real-val | real-val-rot | real-val-small | synth-val | real-val-rot180 | synth-val-v2 | v1-val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.995 (0.994–0.995) | 0.988 (0.984–0.991) | 0.986 (0.985–0.990) | 0.994 (0.994–0.995) | 0.435 (0.386–0.496) | 0.979 (0.978–0.980) | 0.720 (0.655–0.796) |
| n1280-ft | 3/3 | 0.995 (0.994–0.995) | 0.983 (0.976–0.989) | 0.987 (0.985–0.991) | 0.993 (0.992–0.993) | 0.953 (0.947–0.963) | 0.992 (0.992–0.993) | 0.942 (0.936–0.951) |
| n1280-ft2 | 3/3 | 0.994 (0.994–0.995) | 0.978 (0.968–0.985) | 0.988 (0.985–0.991) | 0.993 (0.992–0.994) | 0.892 (0.876–0.911) | 0.992 | 0.937 (0.933–0.941) |
| n1280-ft-gen50k | 3/3 | 0.995 (0.994–0.995) | 0.985 (0.981–0.989) | 0.988 (0.986–0.990) | 0.994 (0.993–0.994) | 0.949 (0.946–0.954) | 0.991 (0.990–0.992) | 0.926 (0.903–0.941) |

**real-val-rot180** (ExpDate's validation photos turned 180°), at each run's threshold:

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.435 (0.386–0.496) | 94.7/187 (90–97) | 103/187 (100–105) | 130.7/187 (119–144) | 0.457 (0.348–0.608) | 0.76 (0.35–1.10) |
| n1280-ft | 3/3 | 0.953 (0.947–0.963) | 178.7/187 (177–180) | 181/187 | 185/187 (184–187) | 0.788 (0.723–0.844) | 0.30 (0.20–0.42) |
| n1280-ft2 | 3/3 | 0.892 (0.876–0.911) | 165/187 (161–169) | 168.7/187 (167–172) | 184/187 (183–185) | 0.676 (0.652–0.714) | 0.48 (0.40–0.52) |
| n1280-ft-gen50k | 3/3 | 0.949 (0.946–0.954) | 174.3/187 (164–181) | 177.3/187 (165–185) | 185.7/187 (184–187) | 0.829 (0.781–0.901) | 0.23 (0.11–0.30) |

**v1-val at both thresholds**: each run's own (threshold_store, the best F2 on v1-val) and M2's rule's (the best F2 on the three real sets, pooled):

| arm | threshold | seeds | value | dates found | expiry found | precision | false boxes / photo |
|---|---|---:|---:|---:|---:|---:|---:|
| n1280 (control) | store (v1-val) | 3/3 | 0.0191 (0.0055–0.0387) | 45.7/63 (44–48) | 21.7/29 (20–23) | 0.601 (0.530–0.739) | 0.93 (0.50–1.15) |
| n1280 (control) | M2's (real sets) | 3/3 | 0.3885 (0.3424–0.4243) | 31.3/63 (27–38) | 14/29 (11–18) | 0.950 (0.935–0.964) | 0.05 (0.03–0.06) |
| n1280-ft | store (v1-val) | 3/3 | 0.1711 (0.1280–0.2211) | 60/63 | 27.7/29 (27–29) | 0.776 (0.750–0.789) | 0.51 (0.47–0.59) |
| n1280-ft | M2's (real sets) | 3/3 | 0.3910 (0.2654–0.4836) | 56/63 (55–57) | 25.7/29 (25–26) | 0.878 (0.824–0.932) | 0.24 (0.12–0.35) |
| n1280-ft2 | store (v1-val) | 3/3 | 0.2187 (0.2062–0.2403) | 59/63 (58–60) | 27.3/29 (27–28) | 0.791 (0.759–0.817) | 0.46 (0.38–0.56) |
| n1280-ft2 | M2's (real sets) | 3/3 | 0.4193 (0.2968–0.5023) | 55.3/63 (54–58) | 25.7/29 (25–26) | 0.889 (0.806–0.931) | 0.22 (0.12–0.41) |
| n1280-ft-gen50k | store (v1-val) | 3/3 | 0.3406 (0.1861–0.6168) | 57/63 (53–60) | 26/29 (24–27) | 0.842 (0.811–0.898) | 0.32 (0.18–0.41) |
| n1280-ft-gen50k | M2's (real sets) | 3/3 | 0.5032 (0.3942–0.5973) | 53.7/63 (51–57) | 23.7/29 (22–25) | 0.897 (0.841–0.944) | 0.19 (0.09–0.29) |

**The threshold on M2's weights**, seed by seed (seed 0 is the M2 detector):

| arm | threshold | value | dates found | expiry found | precision | false boxes / photo |
|---|---|---:|---:|---:|---:|---:|
| m2b-n1280-s0 | store (v1-val) | 0.0055 | 44/63 | 20/29 | 0.530 | 1.15 |
| m2b-n1280-s0 | M2's (real sets) | 0.3424 | 27/63 | 11/29 | 0.964 | 0.03 |
| m2b-n1280-s1 | store (v1-val) | 0.0387 | 48/63 | 23/29 | 0.739 | 0.50 |
| m2b-n1280-s1 | M2's (real sets) | 0.4243 | 38/63 | 18/29 | 0.950 | 0.06 |
| m2b-n1280-s2 | store (v1-val) | 0.0131 | 45/63 | 22/29 | 0.536 | 1.15 |
| m2b-n1280-s2 | M2's (real sets) | 0.3988 | 29/63 | 13/29 | 0.935 | 0.06 |

M2's threshold, worked out again in the M2b suite, is each control seed's own in M2's records (0.3424, 0.4243, 0.3988; 3 of 3 seeds compared).

**v1-val by failure tag**, expiry dates found at each run's threshold:

| arm | seeds | sideways | dot-matrix | small-print | clean | all expiry dates | all dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 3.3/6 (3–4) | 3/3 | 3.7/7 (3–4) | 11.7/13 (10–13) | 21.7/29 (20–23) | 45.7/63 (44–48) |
| n1280-ft | 3/3 | 5.3/6 (5–6) | 3/3 | 6.3/7 (6–7) | 13/13 | 27.7/29 (27–29) | 60/63 |
| n1280-ft2 | 3/3 | 5.7/6 (5–6) | 3/3 | 6/7 | 12.7/13 (12–13) | 27.3/29 (27–28) | 59/63 (58–60) |
| n1280-ft-gen50k | 3/3 | 4.7/6 (4–5) | 3/3 | 6.3/7 (6–7) | 12/13 (11–13) | 26/29 (24–27) | 57/63 (53–60) |

found at M2's threshold:

| arm | seeds | sideways | dot-matrix | small-print | clean | all expiry dates | all dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 1.7/6 (1–2) | 3/3 | 1.3/7 (0–4) | 8/13 (6–10) | 14/29 (11–18) | 31.3/63 (27–38) |
| n1280-ft | 3/3 | 4.3/6 (4–5) | 3/3 | 6/7 | 12.3/13 (11–13) | 25.7/29 (25–26) | 56/63 (55–57) |
| n1280-ft2 | 3/3 | 4/6 | 3/3 | 6/7 | 12.7/13 (12–13) | 25.7/29 (25–26) | 55.3/63 (54–58) |
| n1280-ft-gen50k | 3/3 | 3.7/6 (3–4) | 3/3 | 5.7/7 (5–6) | 11.3/13 (10–13) | 23.7/29 (22–25) | 53.7/63 (51–57) |

and boxed at any confidence:

| arm | seeds | sideways | dot-matrix | small-print | clean | all expiry dates | all dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 4/6 | 3/3 | 5.7/7 (5–6) | 12.3/13 (12–13) | 25/29 | 53.7/63 (52–56) |
| n1280-ft | 3/3 | 5.7/6 (5–6) | 3/3 | 7/7 | 13/13 | 28.7/29 (28–29) | 62.7/63 (62–63) |
| n1280-ft2 | 3/3 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 62.3/63 (61–63) |
| n1280-ft-gen50k | 3/3 | 5.7/6 (5–6) | 3/3 | 7/7 | 13/13 | 28.7/29 (28–29) | 62.3/63 (62–63) |

**v1-val's dates by size** (Cause 4's bands: a box's short side over its photo's long side), found at each run's threshold:

| arm | seeds | under 1.5% | 1.5-2.5% | 2.5% and up |
|---|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 12.3/23 (12–13) | 22.7/28 (21–24) | 10.7/12 (10–12) |
| n1280-ft | 3/3 | 21.7/23 (21–22) | 26.3/28 (26–27) | 12/12 |
| n1280-ft2 | 3/3 | 21.7/23 (21–22) | 25.3/28 (25–26) | 12/12 |
| n1280-ft-gen50k | 3/3 | 21.3/23 (19–23) | 23.7/28 (22–26) | 12/12 |

found at M2's threshold:

| arm | seeds | under 1.5% | 1.5-2.5% | 2.5% and up |
|---|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 4.7/23 (3–8) | 16.7/28 (16–18) | 10/12 (8–12) |
| n1280-ft | 3/3 | 20/23 (19–21) | 24.3/28 (23–26) | 11.7/12 (11–12) |
| n1280-ft2 | 3/3 | 19/23 (17–21) | 24.3/28 (23–25) | 12/12 |
| n1280-ft-gen50k | 3/3 | 18.7/23 (18–19) | 23.3/28 (22–26) | 11.7/12 (11–12) |

**synth-val, M2's strata**, targets found at each run's threshold:

| arm | seeds | clean | sideways | month-word | small | dot-matrix | curved | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 299/300 (298–300) | 297/300 | 299/300 | 291.7/300 (290–293) | 296/300 | 298.7/300 (298–299) | 292.7/300 (291–294) |
| n1280-ft | 3/3 | 296.3/300 (296–297) | 295.7/300 (295–297) | 297/300 (296–298) | 285.3/300 (283–289) | 292/300 (290–294) | 295.7/300 (293–298) | 287/300 (286–289) |
| n1280-ft2 | 3/3 | 296/300 (294–298) | 296.3/300 (296–297) | 297.7/300 (297–298) | 284/300 (283–285) | 291/300 (290–292) | 295.7/300 (295–296) | 285.7/300 (285–287) |
| n1280-ft-gen50k | 3/3 | 295.3/300 (293–297) | 296.3/300 (296–297) | 297/300 (296–298) | 279/300 (270–285) | 288.7/300 (280–293) | 294.3/300 (293–296) | 282.3/300 (271–290) |

and boxed at any confidence:

| arm | seeds | clean | sideways | month-word | small | dot-matrix | curved | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 300/300 | 297.3/300 (297–298) | 299/300 | 293.3/300 (293–294) | 297/300 | 299/300 | 294.3/300 (294–295) |
| n1280-ft | 3/3 | 299.7/300 (299–300) | 297.3/300 (297–298) | 299/300 | 294/300 (291–296) | 296.7/300 (296–297) | 299.3/300 (299–300) | 293.7/300 (293–294) |
| n1280-ft2 | 3/3 | 299.3/300 (299–300) | 297/300 | 299/300 | 294.3/300 (292–296) | 297.3/300 (297–298) | 299.7/300 (299–300) | 293.7/300 (293–294) |
| n1280-ft-gen50k | 3/3 | 299.7/300 (299–300) | 297/300 | 299/300 | 293/300 (292–294) | 297/300 | 299.3/300 (298–300) | 294/300 (293–295) |

**ft2's frozen layers** (stage 1, from the control's weights to its last.pt; layers 0 to 10, the backbone):

| run | frozen layers | convolution weights | batch-norm weights and biases | running means | running variances | values trained |
|---|---:|---:|---:|---:|---:|---:|
| m2b-n1280-ft2-s0 | 0 to 10 | 0 of 1,357,104 | 0 of 8,368 | 0 of 4,184 | 0 of 4,184 | 1,242,729 of 1,247,436 (99.6%) |
| m2b-n1280-ft2-s1 | 0 to 10 | 0 of 1,357,104 | 0 of 8,368 | 0 of 4,184 | 0 of 4,184 | 1,242,755 of 1,247,436 (99.6%) |
| m2b-n1280-ft2-s2 | 0 to 10 | 0 of 1,357,104 | 0 of 8,368 | 0 of 4,184 | 0 of 4,184 | 1,242,641 of 1,247,436 (99.6%) |

**Compute** (GPU job B on an H100; minutes and images a second from each run's record and results.csv):

| run | trained | epochs | photos per epoch | train minutes | setup minutes | epoch 1 s | images/s | M2b suite minutes | M2 suite minutes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m2b-n1280-s0 | no: M2's weights | – | – | – | – | – | – | 5.4 | – |
| m2b-n1280-s1 | no: M2's weights | – | – | – | – | – | – | 1.9 | – |
| m2b-n1280-s2 | no: M2's weights | – | – | – | – | – | – | 1.8 | – |
| m2b-n1280-ft-s0 | yes | 10 | 24,992 | 25.4 | 4.7 | 141 | 204 | 1.8 | – |
| m2b-n1280-ft-s1 | yes | 10 | 24,992 | 23.1 | 2.5 | 132 | 203 | 1.8 | – |
| m2b-n1280-ft-s2 | yes | 10 | 24,992 | 22.7 | 2.2 | 132 | 204 | 1.7 | – |
| m2b-n1280-ft2-s0 | yes | 5 + 5 | 24,992 | 10.8 + 10.3 | 0.6 + 0.3 | 129 + 125 | 204 + 203 | 1.8 | – |
| m2b-n1280-ft2-s1 | yes | 5 + 5 | 24,992 | 10.8 + 10.3 | 0.5 + 0.3 | 130 + 124 | 203 + 203 | 1.8 | – |
| m2b-n1280-ft2-s2 | yes | 5 + 5 | 24,992 | 10.8 + 10.3 | 0.6 + 0.2 | 130 + 125 | 204 + 202 | 1.7 | – |
| m2b-n1280-ft-gen50k-s0 | yes | 10 | 24,992 | 23 | 2.4 | 132 | 203 | 1.8 | – |
| m2b-n1280-ft-gen50k-s1 | yes | 10 | 24,992 | 22.1 | 1.5 | 132 | 203 | 1.8 | – |
| m2b-n1280-ft-gen50k-s2 | yes | 10 | 24,992 | 22 | 1.4 | 133 | 203 | 1.8 | – |

Over these runs: 3.36 h of training (setup included) and 0.42 h of scoring, from their records.
<!-- results:end -->

## What it means

Trained and scored by GPU job B (`73832`) on `cn5` at `04451ff` with a
clean tree, 2026-11-13 (the control's scores and all 9 `n` runs); pulled
and summarised on the laptop on 2026-11-17. v1-val is small: one of its 63
dates is 1.6 points of recall, so counts stand beside every rate.

- **R2, by the rule: `ft`**, and not by the band: `ft` has the highest
  mean on v1-val, 0.9415 against `ft2`'s 0.9369, and both pass the guard.
  Their seed ranges overlap (0.9362–0.9514 and 0.9332–0.9412), so the two
  recipes are level on store dates. `s` and `m` were then fine-tuned with
  `ft` (job B did it; S12 reads them).
- **Fine-tuning lifts v1-val far beyond the band.** Mean mAP@0.5 for dates
  goes from the control's 0.720 (0.655–0.796) to 0.942 (0.936–0.951) with
  `ft`: +22.1 points against a band of 14.1, and every fine-tuned seed is
  above every control seed. At each run's own threshold `ft` finds 60 of
  63 dates and 27.7 of 29 expiries (the control 45.7 and 21.7); it boxes
  62.7 dates at any confidence (53.7). By tag, expiries found: sideways
  5.3 of 6 (3.3), small print 6.3 of 7 (3.7), clean 13 of 13 (11.7);
  dot-matrix 3 of 3 for every run, so v1-val can't show a gain there. By
  size, the smallest dates (under 1.5%) gain most: 21.7 of 23 against
  12.3. v1-val's 16 designs are none of training's 31 (S6's export), so
  the gain isn't packs remembered; v2 is the blind test (S14), as v1-val shaped gen-v2's
  layouts.
- **The threshold alone closes about half of the gap in dates found, and
  pays for it in false boxes.** M2's detector (the control's seed 0) at
  M2's threshold, 0.3424, finds 27 of 63 dates and 11 of 29 expiries at
  precision 0.964; at `threshold_store`, 0.0055, 44 and 20 at precision
  0.530, 1.15 false boxes a photo. Over the three seeds: 31.3 dates at
  M2's threshold, 45.7 at the store's, 60 after fine-tuning; precision
  0.950, 0.601, then 0.776. The store threshold is very low (0.0055 to
  0.0387): it keeps nearly every box the control makes, as M2's weights
  are unsure of store print (Cause 1). mAP, which no threshold moves, rises
  only with fine-tuning. M2's threshold worked out again in the M2b suite
  is exactly M2's own (0.3424, 0.4243, 0.3988).
- **Fine-tuned seeds agree; the control's don't.** v1-val's spread over
  seeds is 14.1 points for the control, 1.5 for `ft`, 0.8 for `ft2` and
  3.8 for `ft-gen50k`. The band is the control's spread, as the rule says,
  so it is wide: right for comparing with the control, coarse between
  fine-tuned arms (R3, S12).
- **Where fine-tuning doesn't beat the control:** real-val 0.995 and
  real-val-small 0.987 against 0.986 are level, and synth-val is 0.15
  points down (0.993 against 0.994), inside the guard. ExpDate turned a
  quarter (`real-val-rot`) is down 0.5 points with `ft` (0.983 against
  0.988) and 1.0 with `ft2` (0.978); the guard doesn't cover it. On
  synth-val's strata every fine-tuned run finds fewer targets at its own
  threshold (small print 285.3 of 300 against 291.7, mixed 287 against
  292.7, dot-matrix 292 against 296), though at any confidence they box as
  many (293.3 and 294 small, 294.3 and 293.7 mixed): those boxes sit under
  the fine-tuned runs' higher thresholds. On synth-val-v2, fine-tuning
  boxes fewer embossed targets even at any confidence (296 of 300 against
  299.3, [`m2b-gen-v2`](../m2b-gen-v2/README.md)). And on v1-val `ft`'s
  precision is 0.776 at its own threshold, 0.51 false boxes a photo: F2
  weighs recall four times as much as precision.
- **`ft2` reads upside-down print worse.** On real-val-rot180, `ft2` scores
  0.892 (0.876–0.911) against `ft`'s 0.953 (0.947–0.963), ranges apart,
  and finds 165 of 187 dates against 178.7 at a lower precision (0.676
  against 0.788). At any confidence it boxes nearly as many (184 against
  185): it is less sure of them, not blind to them. Both replay the same
  180° copies; `ft2` trains its backbone on them for only its second
  half. That is a reading, not tested.
- **The frozen layers didn't move**, in any of `ft2`'s three first stages:
  0 of 1,357,104 convolution weights, of 8,368 batch-norm weights and
  biases, and of 4,184 running means and 4,184 running variances; 99.6% of
  the trained layers' values changed (S10's one-epoch smoke: 87%).
- **Against what we expected:** the control's real-val-rot180 is **not
  about 0 but 0.435** (0.386–0.496): M2's weights already find 94.7 of 187
  upside-down dates at their threshold. The rest held. The control at the
  store threshold keeps more sideways expiries (3.3 of 6 against 1.7 at
  M2's threshold), more small dates (12.3 of 23 against 4.7) and more
  false boxes (0.93 a photo against 0.05), with the same mAP. `ft` is
  clearly up on v1-val, sideways, real-val-rot180 and small dates, and
  real-val is as `n`'s. `ft2` is as `ft`, a little lower, and real-val is
  as `n`'s, not higher. Month words aren't split out by the M2b suite;
  they are read on v2, by cause (S14).
- **Compute** (the table): the 9 runs trained in 21.1 to 25.4 minutes
  each, at 203 to 204 images a second, frozen backbone or not, and each
  scored in 1.7 to 1.8 minutes; with the control's three scores (9.1
  minutes, the first one cold at 5.4), 3.36 GPU hours of training and
  0.42 of scoring. S10 estimated 4.4 hours for `n`'s step 2 (20 minutes
  of training, 5 of setup and 4 of scoring a run): measured, 3.36 hours of
  training and 0.27 of scoring (16 minutes), about 3.6. A first epoch
  took 129 to 141 s against about 123 s for later ones (24,992 photos at
  203 to 204 a second): the cold first read of a run's sample costs 6 to
  18 s. Time outside the epochs was 1.4 to 4.7 minutes for a one-stage
  run and 0.2 to 0.6 for each of `ft2`'s stages; where it goes wasn't
  found (in the job's log the label scan takes seconds and the final
  validation about 12 s).

**Decision:** by R2, step 2's recipe is `ft`, the whole network at a tenth
of M2's rate. Fine-tuning beats the no-training control by more than the
band; which arm becomes the M2b detector is R3's, read in S12.
