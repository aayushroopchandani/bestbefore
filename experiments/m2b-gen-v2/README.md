# M2b / the side arm: what does generator v2 add?

## Question

Fine-tuned the same way (`ft`), does a replay drawn from gen-v2 find more
than one drawn from M2's generated photos (gen50k)? First on gen-v2's own
new layouts, then on small print, the store-like sizes gen-v2 draws, then
on real print turned 180°.

## Why it matters

gen-v2 (S7 to S9) draws what M2's generator never did, all seen failing on
store set v1: upside-down blocks, a date glued to its keyword, two dates on
one line, dense panels, light laser marks on dark windows, worn ink and
missing dot rows. And it sizes each block like v1-train's dates, where M2's
generated dates came out about 1.7 times bigger
([`docs/m2b-store-and-size.md`](../../docs/m2b-store-and-size.md), Causes 2
and 4). It cost three slices, 24 minutes of 24 CPUs and 37 GB. If the store
photos alone do the work, gen-v2's layouts are worth less than they cost.

## Method

| Arm | The replay's generated photos | Everything else | Seeds |
|---|---|---|---|
| `n1280-ft` | 12,609 drawn from gen-v2's 200,000 | as in [`m2b-fine-tune`](../m2b-fine-tune/README.md): from the control's weights, the store lines, ExpDate's photos and their 90°, 270° and 180° copies, 10 epochs at `lr0` 0.001 | 0, 1, 2 |
| `n1280-ft-gen50k` | 12,609 drawn from M2's 50,000 | the same | 0, 1, 2 |

- Each run draws its own fixed sample with its seed, so both arms train on
  12,609 distinct generated photos; only where they come from differs.
- Both replay ExpDate's photos turned 180°, so real upside-down print is in
  both arms; gen-v2 adds generated upside-down blocks (10% of its blocks).
- So the arm measures gen-v2's new layouts and its sizes together.
  Separating the two is in the plan's Later.
- The control (M2's weights, no store training) is shown beside them.
- `n1280-ft-gen50k` is no candidate for the M2b detector (decision 11).

**No rule decides this; the reading was fixed before any run**
(`docs/m2b-store-and-size.md`, "Reading the side questions"): `ft` against
`ft-gen50k` on synth-val-v2 per stratum first (the layouts), then
real-val-small and v1-val's dates by size band (the sizes), then
real-val-rot180. As yardsticks, the band on v1-val (the control's seed
spread) and 0.01 of mAP elsewhere, as the rules use them. Counts are at
each run's own threshold, which differs from run to run (0.128 to 0.617
here, in [`m2b-fine-tune`](../m2b-fine-tune/README.md)'s table), so "boxed
at any confidence" and mAP, which no threshold moves, are the fairer
comparisons.

**What we expected, written before any run:** `n1280-ft-gen50k` a little
below `ft` on v1-val, as `ft` on real-val-rot180, below `ft` on small
store dates (M2's sizes), and as `n` on real-val.

## Result

Rebuild with:

```bash
scripts/pull_cluster_results.sh
python experiments/m2b-gen-v2/summarise.py --write
```

<!-- results:start -->
No rule decides this side question ("Reading the side questions"): it is read in the order the plan fixed. As yardsticks, v1-val's noise band is 0.1414, the control's seed spread; on the real and synthetic sets the rules use 0.01 of mAP@0.5. Worked out again here with `m2b.decisions()`, the chain's whole record (`results/runs/m2b-decisions.json`: noise_store, R1, R2, R3) is the same.

**1. The layouts: synth-val-v2 by stratum**, targets found at each run's threshold:

| arm | seeds | upside-down | glued | line-pair | dense-panel | window | embossed | worn | nozzle-rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 188.3/300 (178–196) | 297/300 | 294.3/300 (292–296) | 299/300 (297–300) | 299.7/300 (299–300) | 297.3/300 (296–298) | 297.7/300 (297–298) | 290.3/300 (288–293) |
| n1280-ft | 3/3 | 291/300 (288–293) | 296.7/300 (296–297) | 292.7/300 (291–294) | 293.3/300 (290–295) | 299.3/300 (299–300) | 291.3/300 (291–292) | 290.7/300 (288–292) | 278.7/300 (277–281) |
| n1280-ft-gen50k | 3/3 | 253.7/300 (245–258) | 295.7/300 (294–297) | 277.7/300 (275–281) | 290.3/300 (286–293) | 299.3/300 (299–300) | 291.7/300 (287–295) | 290/300 (281–295) | 273/300 (257–283) |

and boxed at any confidence:

| arm | seeds | upside-down | glued | line-pair | dense-panel | window | embossed | worn | nozzle-rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 226.7/300 (214–238) | 298.7/300 (298–299) | 297.3/300 (296–299) | 300/300 | 299.7/300 (299–300) | 299.3/300 (299–300) | 298.7/300 (298–299) | 293/300 (291–294) |
| n1280-ft | 3/3 | 298.7/300 (298–299) | 298.3/300 (298–299) | 299/300 (298–300) | 299/300 | 300/300 | 296/300 (295–297) | 298/300 (297–299) | 295.3/300 (294–297) |
| n1280-ft-gen50k | 3/3 | 289.7/300 (286–295) | 298.7/300 (298–299) | 296.3/300 (294–300) | 299.7/300 (299–300) | 299.7/300 (299–300) | 298.7/300 (298–299) | 298.3/300 (297–299) | 293.3/300 (292–295) |

synth-val-v2 as a whole, at each run's threshold:

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.979 (0.978–0.980) | 5718.7/5862 (5697–5730) | 5737/5862 (5720–5748) | 5773.7/5862 (5762–5788) | 0.773 (0.682–0.866) | 0.73 (0.37–1.11) |
| n1280-ft | 3/3 | 0.992 (0.992–0.993) | 5776.3/5862 (5768–5781) | 5788/5862 (5786–5789) | 5842.3/5862 (5837–5848) | 0.926 (0.910–0.938) | 0.19 (0.16–0.24) |
| n1280-ft-gen50k | 3/3 | 0.991 (0.990–0.992) | 5695/5862 (5633–5727) | 5722.7/5862 (5647–5765) | 5830.7/5862 (5824–5837) | 0.954 (0.933–0.988) | 0.12 (0.03–0.17) |

**2. The sizes: real-val-small** (ExpDate's validation photos in 2 × 2 grids, every date at half its size):

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.986 (0.985–0.990) | 184.7/186 (184–186) | 184.7/186 (184–186) | 185.7/186 (185–186) | 0.568 (0.382–0.754) | 4.06 (1.46–7.27) |
| n1280-ft | 3/3 | 0.987 (0.985–0.991) | 183/186 (180–185) | 183.7/186 (182–185) | 185.7/186 (185–186) | 0.841 (0.830–0.853) | 0.85 (0.76–0.93) |
| n1280-ft-gen50k | 3/3 | 0.988 (0.986–0.990) | 180/186 (176–183) | 180.7/186 (176–184) | 185/186 | 0.920 (0.887–0.983) | 0.40 (0.07–0.56) |

and **v1-val's dates by size**, found at each run's threshold:

| arm | seeds | under 1.5% | 1.5-2.5% | 2.5% and up |
|---|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 12.3/23 (12–13) | 22.7/28 (21–24) | 10.7/12 (10–12) |
| n1280-ft | 3/3 | 21.7/23 (21–22) | 26.3/28 (26–27) | 12/12 |
| n1280-ft-gen50k | 3/3 | 21.3/23 (19–23) | 23.7/28 (22–26) | 12/12 |

and boxed at any confidence:

| arm | seeds | under 1.5% | 1.5-2.5% | 2.5% and up |
|---|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 18.3/23 (18–19) | 24.3/28 (23–26) | 11/12 (10–12) |
| n1280-ft | 3/3 | 23/23 | 27.7/28 (27–28) | 12/12 |
| n1280-ft-gen50k | 3/3 | 23/23 | 27.3/28 (27–28) | 12/12 |

v1-val as a whole, at each run's threshold:

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.720 (0.655–0.796) | 45.7/63 (44–48) | 45.7/63 (44–48) | 53.7/63 (52–56) | 0.601 (0.530–0.739) | 0.93 (0.50–1.15) |
| n1280-ft | 3/3 | 0.942 (0.936–0.951) | 60/63 | 60.3/63 (60–61) | 62.7/63 (62–63) | 0.776 (0.750–0.789) | 0.51 (0.47–0.59) |
| n1280-ft-gen50k | 3/3 | 0.926 (0.903–0.941) | 57/63 (53–60) | 57/63 (53–60) | 62.3/63 (62–63) | 0.842 (0.811–0.898) | 0.32 (0.18–0.41) |

**3. real-val-rot180** (ExpDate's validation photos turned 180°):

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.435 (0.386–0.496) | 94.7/187 (90–97) | 103/187 (100–105) | 130.7/187 (119–144) | 0.457 (0.348–0.608) | 0.76 (0.35–1.10) |
| n1280-ft | 3/3 | 0.953 (0.947–0.963) | 178.7/187 (177–180) | 181/187 | 185/187 (184–187) | 0.788 (0.723–0.844) | 0.30 (0.20–0.42) |
| n1280-ft-gen50k | 3/3 | 0.949 (0.946–0.954) | 174.3/187 (164–181) | 177.3/187 (165–185) | 185.7/187 (184–187) | 0.829 (0.781–0.901) | 0.23 (0.11–0.30) |
<!-- results:end -->

## What it means

Trained and scored by GPU job B (`73832`) on `cn5` at `04451ff` with a
clean tree, 2026-11-13; pulled and summarised on the laptop on 2026-11-17.

- **The layouts: gen-v2 adds what it draws, upside-down print and
  line-pairs above all.** On synth-val-v2, at each run's threshold, `ft`
  finds 291 of 300 upside-down targets against `ft-gen50k`'s 253.7, and
  292.7 line-pair targets against 277.7, the seeds' ranges apart on both
  (288–293 against 245–258; 291–294 against 275–281). At any confidence
  too: 298.7 against 289.7, and 299 against 296.3. Fine-tuning on gen50k
  already lifts upside-down targets from the control's 188.3 (the replay's
  real 180° copies); gen-v2's upside-down blocks take them to 291. On the
  other six strata the two are within 3 targets of each other at any
  confidence (glued 298.3 and 298.7, dense panels 299 and 299.7, windows
  300 and 299.7, embossed 296 and 298.7, worn 298 and 298.3, nozzle rows
  295.3 and 293.3); embossed is the one where gen-v2's arm is lower, its
  range apart (295–297 against 298–299). synth-val-v2's mAP over all its
  5,862 dates hides the two strata: 0.992 against 0.991.
- **The sizes: no difference on real small print.** real-val-small scores
  0.987 with `ft` and 0.988 with `ft-gen50k`, both within 0.01 of the
  control's 0.986, and both box 185 or more of 186 at any confidence. On
  v1-val, both box all 23 of the smallest dates (under 1.5%) at any
  confidence, and 27.7 and 27.3 of the middle band's 28. At each run's
  threshold `ft` finds 21.7 and 26.3, `ft-gen50k` 21.3 and 23.7, its
  middle band's range (22–26) reaching `ft`'s (26–27). So gen-v2's
  store-like sizes show nothing measurable on real print here.
- **v1-val as a whole:** 0.942 (0.936–0.951) against 0.926 (0.903–0.941),
  1.6 points, about a ninth of the band, the ranges overlapping; 60
  against 57 of 63 dates found (53–60), 62.7 against 62.3 boxed at any
  confidence.
- **real-val-rot180 is level:** 0.953 (0.947–0.963) against 0.949
  (0.946–0.954), both far above the control's 0.435, and 185 against 185.7
  of 187 boxed at any confidence. The real 180° copies both arms replay do
  this; gen-v2's upside-down blocks lift their own stratum, not real
  upside-down print.
- **Against what we expected:** `ft-gen50k` is a little below `ft` on
  v1-val, as expected, but within the noise; as `ft` on real-val-rot180,
  as expected; **not below `ft` on small store dates** (21.3 against 21.7
  of 23 at each threshold, 23 and 23 at any confidence), against the
  expectation; and as `n` on real-val (0.995 for both, in
  [`m2b-fine-tune`](../m2b-fine-tune/README.md)'s table).
- **v1-val isn't blind to gen-v2** (the plan's Risks): its layouts came
  from looking at all of v1, v1-val included, so a gain there could be
  optimistic. There is none beyond the noise to discount. v2, which shaped
  nothing, shows in S14 whether the layouts help on print never seen.

**Decision:** none by rule. gen-v2 stays in step 2's replay, as it is
`ft`'s, the recipe R2 chose; `ft-gen50k` was never a candidate. On this
validation gen-v2's layouts help where they are drawn, on generated
upside-down print and line-pairs, and nowhere measurable on real print.
Telling its sizes from its layouts stays in the plan's Later.
