# M2b: store photos, gen-v2 and a bigger detector

M2's detector (`yolo11n` at 1,280 px, trained on public data and
generated dates) found 39 of store set v1's 81 expiry dates: a threshold
chosen on ExpDate doubted many it boxed, the generator never drew several
store layouts, and its dates came out bigger than the store's. Once the
retailer's owner allowed training on the store photos, M2b went after all
of it at once: it fine-tunes on store sets v1 and v3 (split by pack
design, a fifth of every epoch, turned four ways) with a replay of the
public data, draws the missing layouts at the store's sizes (gen-v2,
200,000 photos), tries bigger models (`s`, and an `m` probe) and chooses
the threshold on store validation photos (v1-val). Rules R1 to R4, written
as code before any run, chose on validation; store set v2 and ExpDate test
were each scored once, at the end. M2b is judged on **finding** dates;
reading them is M3's job, and the M2b detector is what M3 reads from.
Plan, decisions and every slice's record:
[`docs/m2b-store-and-size.md`](../docs/m2b-store-and-size.md).

A date counts as found when a box at or above its run's threshold overlaps
it with IoU 0.5 or more. Each run keeps its own threshold: M1's the best F1
on ExpDate validation, M2's the best F2 on the three real validation sets
pooled, and every M2b run's `threshold_store`, the best F2 for dates on
v1-val. A cell is the mean over the seeds in, with the lowest and highest
in brackets: `64/68, 94% (62–65)`; "–" means the record doesn't hold that
number. AP (ours) is `detection_eval.average_precision`, which agrees with
Ultralytics' mAP@0.5 within 0.0005 on ExpDate validation (S13); no
threshold moves either.

## Headline

Each milestone's detector is one run, seed 0, declared before any run;
the 3-seed rows show how much a seed moves the counts. Compared on store
set v2 and ExpDate test only: store set v1 trained M2b, so it can't judge
it.

**Store set v2** (99 photos, 68 expiry dates in frame, 212 dates; no model trained or was tuned on them), scored once on the laptop's CPU:

| detector | seeds | threshold | expiry found | all dates found | precision | false boxes / photo | AP@0.5 date (ours) |
|---|---:|---:|---:|---:|---:|---:|---:|
| the M1 detector (`n640`, 640 px) | 1/1 | 0.7279 | 20/68, 29% | 43/212, 20% | 0.956 | 0.02 | 0.440 |
| M1 (`n640`), 3 seeds | 3/3 | 0.6864 (0.6643–0.7279) | 25/68, 37% (20–30) | 52.3/212, 25% (43–62) | 0.957 (0.946–0.969) | 0.02 (0.02–0.03) | 0.438 (0.417–0.457) |
| the M2 detector (`gen-rot-1280-s0`) | 1/1 | 0.3424 | 46/68, 68% | 115/212, 54% | 0.871 | 0.17 | 0.703 |
| M2 (`gen-rot-1280`), 3 seeds | 3/3 | 0.3885 (0.3424–0.4243) | 48.7/68, 72% (46–51) | 121.7/212, 57% (115–127) | 0.903 (0.871–0.934) | 0.13 (0.09–0.17) | 0.733 (0.703–0.769) |
| M2's weights at the store threshold (`n1280-s0`) | 1/1 | 0.0055 | 58/68, 85% | 163/212, 77% | 0.418 | 2.29 | 0.703 |
| the same, 3 seeds | 3/3 | 0.0191 (0.0055–0.0387) | 58.3/68, 86% (58–59) | 161/212, 76% (159–163) | 0.550 (0.418–0.710) | 1.48 (0.66–2.29) | 0.733 (0.703–0.769) |
| the M2b detector (`n1280-ft-s0`) | 1/1 | 0.1641 | 65/68, 96% | 188/212, 89% | 0.761 | 0.60 | 0.895 |
| M2b (`n1280-ft`), 3 seeds | 3/3 | 0.1711 (0.1280–0.2211) | 64/68, 94% (62–65) | 186.3/212, 88% (180–191) | 0.774 (0.758–0.804) | 0.55 (0.44–0.62) | 0.897 (0.889–0.906) |

**ExpDate test** (665 photos, 665 expiry dates in frame, 733 dates), scored once per milestone:

| detector | seeds | threshold | expiry found | all dates found | precision | false boxes / photo | mAP@0.5 date (Ultralytics) |
|---|---:|---:|---:|---:|---:|---:|---:|
| the M1 detector (`n640`, 640 px) | 1/1 | 0.7279 | 637/665, 96% | 697/733, 95% | 0.987 | 0.01 | 0.993 |
| M1 (`n640`), 3 seeds | 3/3 | 0.6864 (0.6643–0.7279) | 646/665, 97% (637–651) | 707.3/733, 96% (697–713) | 0.982 (0.974–0.987) | 0.02 (0.01–0.03) | 0.993 (0.992–0.993) |
| the M2 detector (`gen-rot-1280-s0`) | 1/1 | 0.3424 | 662/665, 100% | 730/733, 100% | 0.988 | 0.01 | 0.995 |
| M2 (`gen-rot-1280`), 3 seeds | 3/3 | 0.3885 (0.3424–0.4243) | 662.7/665, 100% (662–663) | 730.7/733, 100% (730–731) | 0.986 (0.981–0.989) | 0.02 (0.01–0.02) | 0.995 |
| the M2b detector (`n1280-ft-s0`) | 1/1 | 0.1641 | 664/665, 100% | 732/733, 100% | 0.898 | 0.12 | 0.995 |
| M2b (`n1280-ft`), 3 seeds | 3/3 | 0.1711 (0.1280–0.2211) | 663.7/665, 100% (663–664) | 731.7/733, 100% (731–732) | 0.897 (0.876–0.915) | 0.13 (0.10–0.15) | 0.995 (0.994–0.995) |
| the M2b detector at M2's rule's threshold | 1/1 | 0.4836 | 660/665, 99% | 728/733, 99% | 0.968 | 0.04 | 0.995 |
| the same, 3 seeds | 3/3 | 0.3910 (0.2654–0.4836) | 661.7/665, 99% (660–663) | 729/733, 99% (728–730) | 0.950 (0.926–0.968) | 0.06 (0.04–0.09) | 0.995 (0.994–0.995) |

M2's weights at the store threshold weren't scored on ExpDate test (the plan scored only the M2b detector there); the M2b detector cut at M2's rule's threshold, from the same pass, shows what the threshold does there instead.

## What it shows

Every number here is read from the same records as the tables below. Store set v2 holds 68 expiry dates in frame: one is 1.5 points of a rate, and two of its design groups hold 4 and 7 expiries, so counts stand beside the rates.

**Where M2b beats M2:**

- **More of v2's expiry dates found, on photos no model trained or was tuned on.** The M2b detector finds 65/68 against the M2 detector's 46/68 and the M1 detector's 20/68; its arm 64/68 (62–65), against 48.7/68 (46–51) and 25/68 (20–30). Of every date its arm finds 186.3/212, 88% (180–191), against 121.7/212, 57% (115–127) and 52.3/212, 25% (43–62), and it boxes 67/68 expiries at some confidence (M2 61.3/68 (60–64)). By our AP@0.5 for dates, which no threshold moves, the three detectors score 0.895, 0.703 and 0.440.
- **The gains are where M2 was weak**, the M2b arm against M2's at its own threshold: month words 24/25 (23–25) against 14/25 (11–18) (M2's weights at the store threshold 19/25 (18–20), M1 7.3/25 (5–11)); clean print 20.7/24 (19–22) against 12.7/24 (10–17); sideways 9.3/10 (9–10) against 6.3/10 (5–8); dot-matrix 28/28 against 26.7/28 (26–27). v2's one upside-down expiry is found by 3 of the M2b arm's 3 seeds, and by 0, 0 and 0 of M1's, M2's and the control's.
- **Fine-tuning, not the threshold alone.** M2's weights cut at the store threshold find 58.3/68 (58–59) of v2's expiries at precision 0.550 (0.418–0.710), against 48.7/68 (46–51) at 0.903 (0.871–0.934) at M2's own threshold: the threshold alone closes 63% of the gap to the M2b arm's 64/68 (62–65), which keeps precision 0.774 (0.758–0.804). On v1-val the same: 31.3/63 (27–38) dates found at M2's threshold, 45.7/63 (44–48) at the store's, 60/63 fine-tuned.
- **It learned print, not packs.** On v2's designs that no run trained on, 5 of the 5 fine-tuned arms score an AP above every seed of the control's (ranges apart, the reading fixed before any score): the M2b arm 0.909 (0.905–0.915) against 0.726 (0.695–0.767). Its arm finds 53/57 (51–54) of those expiries, the control 47.3/57 (47–48) and M2 at its own threshold 38.7/57 (37–40).
- **ExpDate kept.** On ExpDate test the M2b detector finds 664/665 expiry dates (its arm 663.7/665 (663–664)), the M2 detector 662/665 (662.7/665 (662–663)) and the M1 detector 637/665 (646/665 (637–651)). Every seed of the M2b arm boxes 665/665 at some confidence, and its mAP@0.5 for dates is 0.995 (0.994–0.995), against M2's 0.995.

**Where it doesn't, or didn't do what was expected:**

- **Precision: the store threshold's cost.** On v2 the M2b detector keeps 0.60 false boxes a photo (precision 0.761), the M2 detector 0.17 (0.871) and the M1 detector 0.02 (0.956). On ExpDate test the M2b arm's precision is 0.897 (0.876–0.915) (0.13 (0.10–0.15) false boxes a photo), M2's 0.986 (0.981–0.989) (0.02 (0.01–0.02)); cut at M2's rule's threshold, from the same pass, it finds 661.7/665 (660–663) at 0.950 (0.926–0.968). A wrong date is worse than none, so M3's reader has to reject those boxes.
- **Numeric dates 1.5 to 2.5% of the photo didn't move**: 13/15 found by M2's weights at the store threshold and 13/15 by the M2b arm, which boxes 14/15 at some confidence. The M2b detector's 3 misses, by cause: numbers: print 1.5 to 2.5% (2), sideways (1); by tag: sideways (1), clean (2); by design: unseen (3).
- **The noise band was too coarse to tell the fine-tuned arms apart.** It is the control's seed spread on v1-val, 0.1414; the fine-tuned arms' seeds spread 1.52, 1.15, 0.99 points (`n`, `s`, `m`), and `m1280-ft` sat 4.5 points above `n1280-ft`, inside the band, so R3 took the cheapest, `n1280-ft`, as written. On v2, by our AP: `n1280-ft` 0.897 (0.889–0.906), `s1280-ft` 0.930 (0.921–0.938), `m1280-ft` 0.948 (0.945–0.950), every `m1280-ft` seed above every `s1280-ft` seed, and every `s1280-ft` seed above every `n1280-ft` seed. `m1280-ft` stays M3's fallback (the plan's Later).
- **R4 adopted nothing.** On the detector alone the add-ons gained ensemble +0.0141, rotation +0.0143, imgsz-1536 +0.0256 on v1-val (our AP), against a band of 0.1414. The what-if, with the plan's floor of 0.03, adopts nothing either: imgsz-1536 came closest, 0.0044 short. So the M2b detector is one pass a photo.
- **Designs seen in training aren't above unseen ones**, against "What we expect": the M2b arm's AP is 0.864 (0.780–0.925) on designs seen in training (4 expiries), 0.865 (0.850–0.873) on those seen only in validation (7) and 0.909 (0.905–0.915) on unseen ones (57); it finds 17/21, 81% (14–19), 20.3/23, 88% (20–21) and 149/168, 89% (146–153) of their dates. With no store training, the M2 detector already finds 4/4 of the trained designs' expiries: those few packs are easy ones.
- **Step 1 read upside-down print worse the bigger it was.** On real-val-rot180 (ExpDate's validation photos turned 180°), mAP@0.5 for dates: `n1280` 0.435 (0.386–0.496), `s1280` 0.295 (0.238–0.337), `m1280` 0.234 (0.215–0.263); synth-val-v2's upside-down targets boxed at any confidence 226.7/300 (214–238), 199.7/300 (189–206) and 166.3/300 (159–174). Step 2, with ExpDate's 180° copies in the replay, turned it round: 0.953 (0.947–0.963), 0.970 (0.964–0.974) and 0.983 (0.982–0.986).
- **`n1280-ft` lost a little on ExpDate turned a quarter**: real-val-rot 0.983 (0.976–0.989) against the control's 0.988 (0.984–0.991). The guard watches real-val and synth-val only.
- **The laptop's memory.** Store set v2 and ExpDate test were predicted one photo a call (every record's `batch` is 1): at 16 a call, 1,280 px photos overflowed the laptop's 8 GB, and `m1280-ft-s0`'s device check took 8.4 minutes at 16 a call against 0.7 at one (its memory is in the plan's S14 record).
- **ExpDate test moved from the cluster to the laptop.** It was planned as a CPU job on the cluster; the account's job slots were full, so it was scored on the laptop's CPU, where M2 S8 scored M2's: its records say cpu, 7 torch threads, one photo a call, code `14f718c`.

**What we expected, written before any run, against what was measured.** Store columns are read on v1-val, as the rules read them, but month words, which only v2's labels split out (S11); each run at its own threshold unless a row says M2's.

| run | what | expected | measured | verdict |
|---|---|---|---|---|
| `n1280` (M2's weights) at M2's threshold | v1-val mAP (dates) | the baseline | 0.720 (0.655–0.796) | the baseline; its seeds' spread, 0.1414, is the noise band |
| `n1280` (M2's weights) at M2's threshold | store sideways | low | 1.7/6 (1–2) expiries | held |
| `n1280` (M2's weights) at M2's threshold | real-val-rot180 | about 0 | mAP 0.435 (0.386–0.496); 47/187 (45–49) dates found | failed: M2's weights already find many |
| `n1280` (M2's weights) at M2's threshold | store month words | doubted | v2: 14/25 (11–18) found, 21.3/25 (20–24) boxed at any confidence | held |
| `n1280` (M2's weights) at M2's threshold | small store dates | low | 4.7/23 (3–8) under 1.5% | held |
| `n1280` at a v1-val threshold | v1-val mAP (dates) | the same (mAP ignores the threshold) | 0.720 (0.655–0.796) | held |
| `n1280` at a v1-val threshold | store sideways | a little up | 3.3/6 (3–4) (at M2's threshold 1.7/6 (1–2)) | held |
| `n1280` at a v1-val threshold | real-val-rot180 | about 0 | 94.7/187 (90–97) dates found (at M2's threshold 47/187 (45–49)) | failed |
| `n1280` at a v1-val threshold | store month words | more kept, more false boxes | v2: 19/25 (18–20) found (at M2's threshold 14/25 (11–18)); false boxes a photo 1.48 (0.66–2.29) (at M2's 0.13 (0.09–0.17)) | held |
| `n1280` at a v1-val threshold | small store dates | more kept | 12.3/23 (12–13) under 1.5% (at M2's threshold 4.7/23 (3–8)) | held |
| `s1280` (step 1) | v1-val mAP (dates) | a little up | 0.781 (0.755–0.802) against `n1280`'s 0.720 (0.655–0.796) | held: up, inside the band |
| `s1280` (step 1) | store sideways | a little up | 2.7/6 (2–3) against `n1280`'s 3.3/6 (3–4) | failed: not up |
| `s1280` (step 1) | real-val-rot180 | about 0 | 0.295 (0.238–0.337) against `n1280`'s 0.435 (0.386–0.496) | failed: not 0, and below `n` |
| `s1280` (step 1) | store month words | a little up | v2: 17.7/25 (15–19) against `n1280`'s 19/25 (18–20) | failed: not up |
| `s1280` (step 1) | small store dates | a little up | found 14.3/23 (12–16) against 12.3/23 (12–13); boxed at any confidence 16.3/23 (15–17) against 18.3/23 (18–19) | partly |
| `m1280` (step 1, probe) | v1-val mAP (dates) | about as `s` | 0.844 (0.824–0.863) against `s1280`'s 0.781 (0.755–0.802) | better than expected: R1 passed |
| `n1280-ft` (step 2) | v1-val mAP (dates) | clearly up | 0.942 (0.936–0.951) against `n1280`'s 0.720 (0.655–0.796) | held: up by more than the band |
| `n1280-ft` (step 2) | store sideways | up | 5.3/6 (5–6) against 3.3/6 (3–4) | held |
| `n1280-ft` (step 2) | real-val-rot180 | up a lot | 0.953 (0.947–0.963) against 0.435 (0.386–0.496) | held |
| `n1280-ft` (step 2) | store month words | up | v2: 24/25 (23–25) against 19/25 (18–20) | held |
| `n1280-ft` (step 2) | small store dates | up | 21.7/23 (21–22) against 12.3/23 (12–13) | held |
| `n1280-ft2` (two-stage) | v1-val mAP (dates) | as `ft`, perhaps lower | 0.937 (0.933–0.941) against `ft`'s 0.942 (0.936–0.951) | held |
| `n1280-ft2` (two-stage) | real-val | as `n`, perhaps higher | 0.994 (0.994–0.995) against `n1280`'s 0.995 (0.994–0.995) | as `n`, not higher |
| `n1280-ft-gen50k` (side arm) | v1-val mAP (dates) | a little below `ft` | 0.926 (0.903–0.941) against `ft`'s 0.942 (0.936–0.951) | held, within the noise |
| `n1280-ft-gen50k` (side arm) | real-val-rot180 | as `ft` | 0.949 (0.946–0.954) against 0.953 (0.947–0.963) | held |
| `n1280-ft-gen50k` (side arm) | small store dates | below `ft` (M2's sizes) | 21.3/23 (19–23) against 21.7/23 (21–22) | failed: level |
| `s1280-ft` | v1-val mAP (dates) | the highest | 0.965 (0.961–0.972); `m1280-ft`, run by R1, 0.987 (0.982–0.992) | partly: the highest of the arms planned |
| `s1280-ft` | small store dates | up the most | 22/23 (21–23) against `n1280-ft`'s 21.7/23 (21–22) (`m1280-ft` 22.7/23 (22–23)) | held, barely |
| every arm | real-val | 0.995, as `n` | mAP@0.5 for dates 0.994 to 0.995 (every seed) | held |
| store set v2 | designs seen in training against unseen ones | well above | `n1280-ft`'s AP 0.864 (0.780–0.925) seen in training, 0.865 (0.850–0.873) seen only in validation, 0.909 (0.905–0.915) unseen | failed |
| store set v2 | unseen designs against M2's detector | still above | `n1280-ft` 53/57 (51–54) expiries against M2's 38.7/57 (37–40) | held |
| store set v2 | fine-tuning on unseen designs | lifts them, or it learned packs, not print | 5 of 5 fine-tuned arms above every control seed by AP (`n1280-ft` 0.909 (0.905–0.915), the control 0.726 (0.695–0.767)) | it learned print |

## The validation suite

M2's four sets: `real-val` (ExpDate validation, the guard: must not get
worse), `real-val-rot` (turned 90° or 270°), `real-val-small` (2 × 2
grids, every date half its size) and `synth-val` (M2's generated photos,
7 strata); and three new: `real-val-rot180` (ExpDate's validation photos
turned 180°), `synth-val-v2` (gen-v2's new layouts, 8 strata of 300) and
**`v1-val`**, the store's validation photos (v1's and v3's, 34 photos of
16 pack designs, none of them in training). Every run, the control
included, was scored on all seven by GPU job B.

**The noise band: 0.1414.** The control's three seeds (M2's `gen-rot-1280`, no store training) score 0.6546, 0.7960, 0.7096 on v1-val (mAP@0.5 for dates): a spread of 0.1414, so the band is 0.1414 (the spread, at least 0.03), fixed before any other run was scored. The chain's own record (`results/runs/m2b-decisions.json`) agrees.

**R1, the `m` probe: passes**, so `m` earns 2 more seeds and step 2. `m1280-s0` scores 0.8625 on v1-val against `s1280`'s seeds' 0.7555, 0.7845, 0.8016, above every one, and it passes the guard. The chain's own record (`results/runs/m2b-decisions.json`) agrees.

**mAP@0.5 for dates on every set**, every arm (the control is M2's weights, not trained again):

| arm | seeds | real-val | real-val-rot | real-val-small | synth-val | real-val-rot180 | synth-val-v2 | v1-val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.995 (0.994–0.995) | 0.988 (0.984–0.991) | 0.986 (0.985–0.990) | 0.994 (0.994–0.995) | 0.435 (0.386–0.496) | 0.979 (0.978–0.980) | 0.720 (0.655–0.796) |
| s1280 | 3/3 | 0.995 | 0.993 (0.993–0.994) | 0.994 (0.993–0.994) | 0.995 | 0.295 (0.238–0.337) | 0.980 (0.979–0.980) | 0.781 (0.755–0.802) |
| m1280 | 3/3 | 0.995 | 0.995 | 0.995 | 0.995 | 0.234 (0.215–0.263) | 0.974 | 0.844 (0.824–0.863) |
| n1280-ft | 3/3 | 0.995 (0.994–0.995) | 0.983 (0.976–0.989) | 0.987 (0.985–0.991) | 0.993 (0.992–0.993) | 0.953 (0.947–0.963) | 0.992 (0.992–0.993) | 0.942 (0.936–0.951) |
| n1280-ft2 | 3/3 | 0.994 (0.994–0.995) | 0.978 (0.968–0.985) | 0.988 (0.985–0.991) | 0.993 (0.992–0.994) | 0.892 (0.876–0.911) | 0.992 | 0.937 (0.933–0.941) |
| n1280-ft-gen50k | 3/3 | 0.995 (0.994–0.995) | 0.985 (0.981–0.989) | 0.988 (0.986–0.990) | 0.994 (0.993–0.994) | 0.949 (0.946–0.954) | 0.991 (0.990–0.992) | 0.926 (0.903–0.941) |
| s1280-ft | 3/3 | 0.995 | 0.991 (0.988–0.993) | 0.991 (0.985–0.994) | 0.995 | 0.970 (0.964–0.974) | 0.995 | 0.965 (0.961–0.972) |
| m1280-ft | 3/3 | 0.995 | 0.994 (0.994–0.995) | 0.995 | 0.995 | 0.983 (0.982–0.986) | 0.995 | 0.987 (0.982–0.992) |

**v1-val by class**, mAP@0.5: the dates, and the keyword boxes M3 reads beside them (`due` the expiry's word, `prod` the manufacture's, `code` a batch code or a time):

| arm | seeds | date | due | prod | code |
|---|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.720 (0.655–0.796) | 0.321 (0.275–0.349) | 0.047 (0.030–0.060) | 0.048 (0.031–0.062) |
| s1280 | 3/3 | 0.781 (0.755–0.802) | 0.302 (0.257–0.357) | 0.017 (0.006–0.025) | 0.001 (0.000–0.004) |
| m1280 | 3/3 | 0.844 (0.824–0.863) | 0.302 (0.274–0.334) | 0.055 (0.011–0.117) | 0.003 (0.000–0.009) |
| n1280-ft | 3/3 | 0.942 (0.936–0.951) | 0.950 (0.941–0.960) | 0.717 (0.674–0.751) | 0.408 (0.383–0.425) |
| n1280-ft2 | 3/3 | 0.937 (0.933–0.941) | 0.903 (0.876–0.928) | 0.598 (0.582–0.615) | 0.257 (0.230–0.275) |
| n1280-ft-gen50k | 3/3 | 0.926 (0.903–0.941) | 0.933 (0.918–0.956) | 0.645 (0.616–0.666) | 0.469 (0.382–0.586) |
| s1280-ft | 3/3 | 0.965 (0.961–0.972) | 0.955 (0.949–0.965) | 0.736 (0.713–0.753) | 0.538 (0.441–0.621) |
| m1280-ft | 3/3 | 0.987 (0.982–0.992) | 0.957 (0.953–0.963) | 0.884 (0.851–0.936) | 0.529 (0.457–0.620) |

Step 1's early read, M2's suite (`suite.json`, scored as each step-1 run finished; `n`'s are M2's own records), for 9 of 9 step-1 runs: on the four sets it shares with the M2b suite its mAP@0.5 for dates differs by at most 0.0000, and its threshold from the M2b suite's `threshold_m2` by at most 0.0000.

Worked out again here with `m2b.decisions()`, the chain's whole record (`results/runs/m2b-decisions.json`: noise_store, R1, R2, R3) is the same.

## Model size: step 1 and step 2

Step 1 trains each size on public data with M2's recipe and mix (`n`'s
step 1 is M2's own `gen-rot-1280`); step 2 fine-tunes each seed's step-1
weights on the store photos with R2's recipe. R1 gave `m` its two more
seeds and step 2.

**R3, the M2b detector's arm**, every candidate, cheapest first (inference cost, then training cost): the guard (real-val and synth-val, no more than 0.01 under the control), then the mean on v1-val, mAP@0.5 for dates over the 3 seeds, and each mean's distance from the band's edge (the best eligible mean less the band):

| arm | seeds | real-val | synth-val | v1-val | guard | from the band's edge | R3 |
|---|---:|---:|---:|---:|---|---:|---|
| n1280 (control) | 3/3 | 0.9945 (0.9940–0.9950) | 0.9942 (0.9940–0.9945) | 0.7201 (0.6546–0.7960) | passes | -0.1253 | below the band |
| n1280-ft | 3/3 | 0.9946 (0.9939–0.9950) | 0.9927 (0.9923–0.9933) | 0.9415 (0.9362–0.9514) | passes | +0.0961 | **chosen** |
| s1280 | 3/3 | 0.9948 (0.9947–0.9949) | 0.9949 (0.9948–0.9949) | 0.7805 (0.7555–0.8016) | passes | -0.0648 | below the band |
| s1280-ft | 3/3 | 0.9950 (0.9949–0.9950) | 0.9948 (0.9946–0.9949) | 0.9648 (0.9606–0.9721) | passes | +0.1194 | in the band |
| m1280 | 3/3 | 0.9950 | 0.9950 | 0.8444 (0.8242–0.8625) | passes | -0.0010 | below the band |
| m1280-ft | 3/3 | 0.9950 | 0.9950 (0.9949–0.9950) | 0.9868 (0.9820–0.9919) | passes | +0.1414 | in the band |

**R3, by the rule: `n1280-ft`**, the cheapest candidate within the band (0.1414) of the best, `m1280-ft`'s 0.9868: the band's edge is 0.8454, and `n1280-ft`'s 0.9415 sits 0.0961 above it. Every candidate passes the guard. Within the band too, and dearer: `s1280-ft` (+0.1194), `m1280-ft` (+0.1414). Below the edge: `n1280` (-0.1253), `s1280` (-0.0648), `m1280` (-0.0010). Nearest the edge: `m1280`, 0.0010 below it. So the M2b detector is `n1280-ft-s0` (v1-val 0.9514) at its threshold_store, 0.1641, unless R4 adopts the ensemble (S13). The chain's own record (`results/runs/m2b-decisions.json`) agrees.

**real-val-rot180** (ExpDate's validation photos turned 180°), at each run's threshold:

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.435 (0.386–0.496) | 94.7/187 (90–97) | 103/187 (100–105) | 130.7/187 (119–144) | 0.457 (0.348–0.608) | 0.76 (0.35–1.10) |
| s1280 | 3/3 | 0.295 (0.238–0.337) | 67/187 (49–89) | 76/187 (56–97) | 86.3/187 (69–101) | 0.440 (0.405–0.470) | 0.53 (0.37–0.79) |
| m1280 | 3/3 | 0.234 (0.215–0.263) | 46/187 (41–49) | 51/187 (46–55) | 58/187 (54–66) | 0.454 (0.329–0.583) | 0.37 (0.21–0.59) |
| n1280-ft | 3/3 | 0.953 (0.947–0.963) | 178.7/187 (177–180) | 181/187 | 185/187 (184–187) | 0.788 (0.723–0.844) | 0.30 (0.20–0.42) |
| s1280-ft | 3/3 | 0.970 (0.964–0.974) | 181.3/187 (179–184) | 182/187 (180–185) | 185.7/187 (185–187) | 0.857 (0.790–0.927) | 0.19 (0.08–0.30) |
| m1280-ft | 3/3 | 0.983 (0.982–0.986) | 176.7/187 (172–182) | 177.3/187 (174–182) | 186/187 (185–187) | 0.955 (0.951–0.961) | 0.05 (0.04–0.05) |

**synth-val-v2 by stratum**, targets boxed at any confidence:

| arm | seeds | upside-down | glued | line-pair | dense-panel | window | embossed | worn | nozzle-rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 226.7/300 (214–238) | 298.7/300 (298–299) | 297.3/300 (296–299) | 300/300 | 299.7/300 (299–300) | 299.3/300 (299–300) | 298.7/300 (298–299) | 293/300 (291–294) |
| s1280 | 3/3 | 199.7/300 (189–206) | 300/300 | 297/300 (295–299) | 300/300 | 300/300 | 299.3/300 (299–300) | 300/300 | 295.7/300 (295–296) |
| m1280 | 3/3 | 166.3/300 (159–174) | 300/300 | 296/300 (295–297) | 300/300 | 300/300 | 300/300 | 299.7/300 (299–300) | 295.7/300 (295–296) |
| n1280-ft | 3/3 | 298.7/300 (298–299) | 298.3/300 (298–299) | 299/300 (298–300) | 299/300 | 300/300 | 296/300 (295–297) | 298/300 (297–299) | 295.3/300 (294–297) |
| s1280-ft | 3/3 | 299.7/300 (299–300) | 299.7/300 (299–300) | 299.3/300 (299–300) | 299.7/300 (299–300) | 300/300 | 300/300 | 299.3/300 (299–300) | 297.7/300 (297–298) |
| m1280-ft | 3/3 | 300/300 | 300/300 | 300/300 | 300/300 | 300/300 | 300/300 | 299.7/300 (299–300) | 297.3/300 (296–298) |

**v1-val by failure tag**, expiry dates found at each run's threshold:

| arm | seeds | sideways | dot-matrix | small-print | clean | all expiry dates | all dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 3.3/6 (3–4) | 3/3 | 3.7/7 (3–4) | 11.7/13 (10–13) | 21.7/29 (20–23) | 45.7/63 (44–48) |
| s1280 | 3/3 | 2.7/6 (2–3) | 3/3 | 4.7/7 (4–5) | 12.3/13 (12–13) | 22.7/29 (21–24) | 48/63 (45–51) |
| m1280 | 3/3 | 3.7/6 (3–4) | 2.7/3 (2–3) | 5/7 (4–6) | 13/13 | 24.3/29 (24–25) | 54/63 |
| n1280-ft | 3/3 | 5.3/6 (5–6) | 3/3 | 6.3/7 (6–7) | 13/13 | 27.7/29 (27–29) | 60/63 |
| s1280-ft | 3/3 | 5.7/6 (5–6) | 2.3/3 (2–3) | 6.7/7 (6–7) | 13/13 | 27.7/29 (27–29) | 59.7/63 (58–62) |
| m1280-ft | 3/3 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 61.3/63 (60–62) |

**v1-val's dates by size** (a box's short side over its photo's long side), found at each run's threshold:

| arm | seeds | under 1.5% | 1.5-2.5% | 2.5% and up |
|---|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 12.3/23 (12–13) | 22.7/28 (21–24) | 10.7/12 (10–12) |
| s1280 | 3/3 | 14.3/23 (12–16) | 22.3/28 (21–24) | 11.3/12 (11–12) |
| m1280 | 3/3 | 18.3/23 (18–19) | 23.7/28 (23–24) | 12/12 |
| n1280-ft | 3/3 | 21.7/23 (21–22) | 26.3/28 (26–27) | 12/12 |
| s1280-ft | 3/3 | 22/23 (21–23) | 25.7/28 (25–27) | 12/12 |
| m1280-ft | 3/3 | 22.7/23 (22–23) | 26.7/28 (26–27) | 12/12 |

and boxed at any confidence:

| arm | seeds | under 1.5% | 1.5-2.5% | 2.5% and up |
|---|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 18.3/23 (18–19) | 24.3/28 (23–26) | 11/12 (10–12) |
| s1280 | 3/3 | 16.3/23 (15–17) | 23.7/28 (23–24) | 11.3/12 (11–12) |
| m1280 | 3/3 | 19.3/23 (18–20) | 24.3/28 (23–26) | 12/12 |
| n1280-ft | 3/3 | 23/23 | 27.7/28 (27–28) | 12/12 |
| s1280-ft | 3/3 | 23/23 | 27.7/28 (27–28) | 12/12 |
| m1280-ft | 3/3 | 23/23 | 28/28 | 12/12 |

## Side questions

### The recipe (R2)

`ft` (the whole network at a tenth of M2's rate) against `ft2` (the backbone frozen for half the budget, then everything at half that rate), on `n`: the guard, then the mean on v1-val, mAP@0.5 for dates over the 3 seeds:

| arm | seeds | real-val | synth-val | v1-val | guard |
|---|---:|---:|---:|---:|---|
| n1280 (control) | 3/3 | 0.9945 (0.9940–0.9950) | 0.9942 (0.9940–0.9945) | 0.7201 (0.6546–0.7960) | – |
| n1280-ft | 3/3 | 0.9946 (0.9939–0.9950) | 0.9927 (0.9923–0.9933) | 0.9415 (0.9362–0.9514) | passes |
| n1280-ft2 | 3/3 | 0.9944 (0.9941–0.9948) | 0.9928 (0.9921–0.9936) | 0.9369 (0.9332–0.9412) | passes |

**R2, by the rule: `ft`**, the highest mean on v1-val, 0.9415; `ft2` 0.9369 (-0.0046). Both recipes pass the guard (its floors, the control's mean less 0.01: real-val 0.9845, synth-val 0.9842). The chain's own record (`results/runs/m2b-decisions.json`) agrees.

**ft2's frozen layers** (stage 1, from the control's weights to its last.pt; layers 0 to 10):

| run | frozen layers | convolution weights | batch-norm weights and biases | running means | running variances | values trained |
|---|---:|---:|---:|---:|---:|---:|
| m2b-n1280-ft2-s0 | 0 to 10 | 0 of 1,357,104 | 0 of 8,368 | 0 of 4,184 | 0 of 4,184 | 1,242,729 of 1,247,436 (99.6%) |
| m2b-n1280-ft2-s1 | 0 to 10 | 0 of 1,357,104 | 0 of 8,368 | 0 of 4,184 | 0 of 4,184 | 1,242,755 of 1,247,436 (99.6%) |
| m2b-n1280-ft2-s2 | 0 to 10 | 0 of 1,357,104 | 0 of 8,368 | 0 of 4,184 | 0 of 4,184 | 1,242,641 of 1,247,436 (99.6%) |

### What gen-v2 adds

`n1280-ft` (its replay draws gen-v2) against `n1280-ft-gen50k` (M2's 50,000 generated photos), no rule, read in the order the plan fixed: the layouts, then the sizes, then real print turned 180°. gen-v2 sizes each untilted block like v1-train's dates and a tilted one by its digits, so the arm measures its layouts and sizes together. On store set v2, by our AP, see the next section's tables.

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

**3. real-val-rot180:**

| arm | seeds | mAP@0.5 date | dates found | found at IoU 0.3 | boxed at any confidence | precision | false boxes / photo |
|---|---:|---:|---:|---:|---:|---:|---:|
| n1280 (control) | 3/3 | 0.435 (0.386–0.496) | 94.7/187 (90–97) | 103/187 (100–105) | 130.7/187 (119–144) | 0.457 (0.348–0.608) | 0.76 (0.35–1.10) |
| n1280-ft | 3/3 | 0.953 (0.947–0.963) | 178.7/187 (177–180) | 181/187 | 185/187 (184–187) | 0.788 (0.723–0.844) | 0.30 (0.20–0.42) |
| n1280-ft-gen50k | 3/3 | 0.949 (0.946–0.954) | 174.3/187 (164–181) | 177.3/187 (165–185) | 185.7/187 (184–187) | 0.829 (0.781–0.901) | 0.23 (0.11–0.30) |

### The threshold

M2's detector at M2's threshold against at threshold_store (the best F2 for dates on v1-val), on v1-val, seed by seed (seed 0 is the M2 detector); store set v2's are in the headline:

| arm | threshold | value | dates found | expiry found | precision | false boxes / photo |
|---|---|---:|---:|---:|---:|---:|
| m2b-n1280-s0 | store (v1-val) | 0.0055 | 44/63 | 20/29 | 0.530 | 1.15 |
| m2b-n1280-s0 | M2's (real sets) | 0.3424 | 27/63 | 11/29 | 0.964 | 0.03 |
| m2b-n1280-s1 | store (v1-val) | 0.0387 | 48/63 | 23/29 | 0.739 | 0.50 |
| m2b-n1280-s1 | M2's (real sets) | 0.4243 | 38/63 | 18/29 | 0.950 | 0.06 |
| m2b-n1280-s2 | store (v1-val) | 0.0131 | 45/63 | 22/29 | 0.536 | 1.15 |
| m2b-n1280-s2 | M2's (real sets) | 0.3988 | 29/63 | 13/29 | 0.935 | 0.06 |

M2's threshold, worked out again in the M2b suite, is each control seed's own in M2's records (0.3424, 0.4243, 0.3988; 3 of 3 seeds compared).

**v1-val at both thresholds**, the control and the arms on `n`:

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

**M2's synth-val strata**, targets found at each run's threshold, where the fine-tuned runs' higher thresholds cut boxes they still make:

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

## The inference add-ons

The M2b detector, `n1280-ft-s0`, given more than one look at each photo: its arm's three seeds merged by weighted boxes fusion (the ensemble), the photo also turned 90°, 180° and 270° (rotation), and a 1,536 px input. R4 keeps one only if it raises v1-val AP (ours) by more than the noise band; the what-if, fixed before any add-on was measured, walks the same steps with the plan's floor band and decides nothing.

**The device check passes**: `n1280-ft-s0` on v1-val, on the laptop's CPU against its GPU suite, each at its own F2 threshold (0.1641 on the GPU, 0.1665 here; the GPU's is the run's): dates found 60/63 on the GPU, 60/63 here; expiries found 29/29 on the GPU, 29/29 here; boxed at any confidence 63/63 on the GPU, 63/63 here; mAP@0.5 for dates (Ultralytics') 0.9514 and 0.9514.

**The AP check passes**: on real-val ours is within 0.0005 of Ultralytics' for every seed (the plan asks 0.01).

| run | real-val ours | real-val Ultralytics | real-val difference | v1-val ours | v1-val Ultralytics | v1-val difference |
|---|---:|---:|---:|---:|---:|---:|
| n1280-ft-s0 | 0.9949 | 0.9950 | -0.0001 | 0.9596 | 0.9514 | +0.0082 |
| n1280-ft-s1 | 0.9949 | 0.9949 | -0.0000 | 0.9510 | 0.9369 | +0.0141 |
| n1280-ft-s2 | 0.9944 | 0.9939 | +0.0005 | 0.9513 | 0.9362 | +0.0151 |

**R4** (the noise band, 0.1414):

**R4, as written (band 0.1414, the noise band): adopts nothing**. Gains on v1-val over what it had kept, in R4's order: ensemble +0.0141, rotation +0.0143, imgsz-1536 +0.0256; the largest, imgsz-1536's +0.0256, is within the band. So the M2b detector is `n1280-ft-s0` alone at its threshold_store, 0.1641.

| add-on | against | tried | v1-val before | after | gain | band | real-val change | real-val guard | kept |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| ensemble | none | ensemble | 0.9596 | 0.9737 | +0.0141 | 0.1414 | +0.0001 | passes | no |
| rotation | none | rotation | 0.9596 | 0.9739 | +0.0143 | 0.1414 | -0.0021 | passes | no |
| imgsz-1536 | none | imgsz-1536 | 0.9596 | 0.9852 | +0.0256 | 0.1414 | -0.0021 | passes | no |

**The what-if (band 0.0300, the plan's floor; it decides nothing): adopts nothing**. Gains on v1-val over what it had kept, in R4's order: ensemble +0.0141, rotation +0.0143, imgsz-1536 +0.0256; the largest, imgsz-1536's +0.0256, is within the band. Nothing for the plan's Later.

| add-on | against | tried | v1-val before | after | gain | band | real-val change | real-val guard | kept |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| ensemble | none | ensemble | 0.9596 | 0.9737 | +0.0141 | 0.0300 | +0.0001 | passes | no |
| rotation | none | rotation | 0.9596 | 0.9739 | +0.0143 | 0.0300 | -0.0021 | passes | no |
| imgsz-1536 | none | imgsz-1536 | 0.9596 | 0.9852 | +0.0256 | 0.0300 | -0.0021 | passes | no |

Worked out again here with `m2b.decide_addons()` from the record's APs, R4 and the what-if are the record's, and its arm, run and threshold are R3's in the chain's record.

**Every configuration**, side readings, none adopted (counts on v1-val at each configuration's own F2 threshold; CPU time a photo is the laptop's, its passes' mean plus the fusion):

| configuration | passes | v1-val AP | real-val AP | threshold | dates found | expiry found | precision | false boxes | false / photo | boxed at any confidence | real-val found | CPU ms a photo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 1 | 0.9596 | 0.9949 | 0.1665 | 60/63 | 29/29 | 0.779 | 17 | 0.50 | 63/63 | 187/187 | 167 |
| ensemble | 3 | 0.9737 | 0.9950 | 0.2264 | 61/63 | 27/29 | 0.871 | 9 | 0.26 | 63/63 | 187/187 | 511 |
| rotation | 4 | 0.9739 | 0.9928 | 0.1772 | 63/63 | 29/29 | 0.818 | 14 | 0.41 | 63/63 | 187/187 | 681 |
| imgsz-1536 | 1 | 0.9852 | 0.9929 | 0.4860 | 62/63 | 29/29 | 0.925 | 5 | 0.15 | 63/63 | 187/187 | 270 |
| ensemble+rotation | 12 | 0.9841 | 0.9934 | 0.2284 | 62/63 | 28/29 | 0.912 | 6 | 0.18 | 63/63 | 187/187 | 2053 |
| ensemble+imgsz-1536 | 3 | 0.9790 | 0.9949 | 0.2131 | 63/63 | 29/29 | 0.840 | 12 | 0.35 | 63/63 | 187/187 | 817 |
| rotation+imgsz-1536 | 4 | 0.9856 | 0.9920 | 0.3075 | 62/63 | 29/29 | 0.899 | 7 | 0.21 | 63/63 | 185/187 | 1089 |
| ensemble+rotation+imgsz-1536 | 12 | 0.9841 | 0.9937 | 0.2199 | 63/63 | 29/29 | 0.887 | 8 | 0.24 | 63/63 | 187/187 | 3275 |

Passes: v1-val on Apple M2 (cpu, 7 torch threads, 34 photos, code `e38ae84`); real-val on Intel(R) Xeon(R) Gold 6442Y (cpu, torch's threads not recorded, 165 photos, code `e38ae84`).

**v1-val by failure tag**, expiry dates found at each run's threshold:

| arm | seeds | sideways | dot-matrix | small-print | clean | all expiry dates | all dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 60/63 |
| ensemble | 1/1 | 5/6 | 3/3 | 6/7 | 13/13 | 27/29 | 61/63 |
| rotation | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 62/63 |
| ensemble+rotation | 1/1 | 6/6 | 3/3 | 6/7 | 13/13 | 28/29 | 62/63 |
| ensemble+imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |
| rotation+imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 62/63 |
| ensemble+rotation+imgsz-1536 | 1/1 | 6/6 | 3/3 | 7/7 | 13/13 | 29/29 | 63/63 |

**The detector against `m` on the laptop's CPU**: Apple M2, 7 torch threads, 34 v1-val photos (long side 1600 px, median) loaded first, one photo a call, rect=False; 2 warm-up calls each left out, then 3 repeats.

| model | median ms | mean ms | each repeat's mean | against the first |
|---|---:|---:|---|---:|
| n1280-ft-s0@1280 | 175 | 178 | 170, 179, 185 | 1.00× |
| n1280-ft-s0@1536 | 279 | 312 | 283, 282, 372 | 1.59× |
| m1280-ft-s0@1280 | 819 | 905 | 890, 835, 989 | 4.68× |

## Store set v2, once

Store set v2 holds 99 photos, 68 expiry dates in frame, 212 dates: new photos from the same stores, labelled and frozen (its labels and pack designs on 2026-11-03) before any training on store photos, and never copied to the cluster. Every planned run was scored on it once, on the laptop's CPU, at the threshold its own GPU record holds: M1's `n640` at M1's, M2's `gen-rot-1280` at M2's, and every M2b run at threshold_store, the control being M2's weights cut there. Causes are M2's, with v2's month words as one group (its labels write every month word `MON`) and upside-down print apart from sideways. A design is **seen in training** if v1-train or v3-train holds it, **seen only in validation** if only v1-val or v3-val does, else **unseen**. A group of one or two expiries (curved, upside-down) is a count, not a rate.

**Every record:**

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

**By cause**, expiry dates found at each run's threshold:

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

**By pack design**, expiry dates found at each run's threshold:

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

**Learned packs, not print?** Each fine-tuned arm against the control on v2's unseen designs, by our AP (fixed before any score: lifted only if the seeds' ranges are apart), beside v1-val:

| arm | v1-val mAP | v1-val, against the control | v2 unseen: AP (ours) | v2 unseen, against the control | v2 unseen: dates found | v2 unseen: expiry found |
|---|---:|---|---:|---|---:|---:|
| n1280 (control) | 0.720 (0.655–0.796) | – | 0.726 (0.695–0.767) | – | 123/168, 73% (122–124) | 47.3/57, 83% (47–48) |
| n1280-ft | 0.942 (0.936–0.951) | lifted: ranges apart | 0.909 (0.905–0.915) | lifted: ranges apart | 149/168, 89% (146–153) | 53/57, 93% (51–54) |
| n1280-ft2 | 0.937 (0.933–0.941) | lifted: ranges apart | 0.913 (0.909–0.920) | lifted: ranges apart | 148/168, 88% (144–153) | 52.7/57, 92% (52–53) |
| n1280-ft-gen50k | 0.926 (0.903–0.941) | lifted: ranges apart | 0.890 (0.883–0.899) | lifted: ranges apart | 135.3/168, 81% (122–145) | 49/57, 86% (46–51) |
| s1280-ft | 0.965 (0.961–0.972) | lifted: ranges apart | 0.940 (0.932–0.952) | lifted: ranges apart | 151.3/168, 90% (148–157) | 52/57, 91% (51–53) |
| m1280-ft | 0.987 (0.982–0.992) | lifted: ranges apart | 0.960 (0.957–0.965) | lifted: ranges apart | 148.7/168, 88% (145–151) | 49.7/57, 87% (47–52) |

Every record in so far (30 of 30 on store set v2, 3 of 3 on ExpDate test) was scored on a CPU, at its own GPU record's threshold exactly, by the weights hashed for it, one photo a call; on v2 its counts by cause and by design add up to its own.

For all 30 v2 records in, the per-photo boxes kept in `data/store/v2/runs/` give the record's counts again.

## ExpDate test

Scored once for the M2b detector's three seeds, on the laptop's CPU, one
photo a call; M1's and M2's as their milestones scored them.

| detector | seeds | threshold | expiry found | expiry boxed at any confidence | all dates found | precision | false boxes / photo | mAP@0.5 date (Ultralytics) | AP@0.5 date (ours) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 (`n640`) | 3/3 | 0.6864 (0.6643–0.7279) | 646/665, 97% (637–651) | – | 707.3/733, 96% (697–713) | 0.982 (0.974–0.987) | 0.02 (0.01–0.03) | 0.993 (0.992–0.993) | – |
| M2 (`gen-rot-1280`) at M2's threshold | 3/3 | 0.3885 (0.3424–0.4243) | 662.7/665, 100% (662–663) | 664.7/665, 100% (664–665) | 730.7/733, 100% (730–731) | 0.986 (0.981–0.989) | 0.02 (0.01–0.02) | 0.995 | – |
| M2b (`n1280-ft`) at threshold_store | 3/3 | 0.1711 (0.1280–0.2211) | 663.7/665, 100% (663–664) | 665/665, 100% | 731.7/733, 100% (731–732) | 0.897 (0.876–0.915) | 0.13 (0.10–0.15) | 0.995 (0.994–0.995) | 0.995 |
| M2b (`n1280-ft`) at M2's rule's threshold | 3/3 | 0.3910 (0.2654–0.4836) | 661.7/665, 99% (660–663) | 665/665, 100% | 729/733, 99% (728–730) | 0.950 (0.926–0.968) | 0.06 (0.04–0.09) | 0.995 (0.994–0.995) | 0.995 |

## The device checks

v2 was predicted on the laptop's CPU; each size was checked there first on
v1-val, against its GPU suite, at 16 photos a call and at one (the GPU's
count first). Within two dates, or v2 isn't scored.

| run | photos a call | dates found | expiry found | boxed at any confidence | mAP@0.5 date | moved | check |
|---|---:|---:|---:|---:|---:|---:|---|
| m2b-n1280-ft-s0 | 16 | 60/63 → 60/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9514 → 0.9514 | 0 | passes |
| m2b-n1280-ft-s0 | 1 | 60/63 → 60/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9514 → 0.9511 | 0 | passes |
| m2b-s1280-ft-s0 | 16 | 62/63 → 62/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9721 → 0.9709 | 0 | passes |
| m2b-s1280-ft-s0 | 1 | 62/63 → 62/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9721 → 0.9705 | 0 | passes |
| m2b-m1280-ft-s0 | 16 | 60/63 → 60/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9864 → 0.9866 | 0 | passes |
| m2b-m1280-ft-s0 | 1 | 60/63 → 60/63 | 29/29 → 29/29 | 63/63 → 63/63 | 0.9864 → 0.9867 | 0 | passes |

## Compute

### On the cluster's H100s

Every M2b run's training and scoring, from its record and its
`results.csv`. Images a second count epochs 2 to the last but one, without
start-up and the final validation; a two-stage run's two stages are
joined by +.

`n`'s step 1 is M2's `gen-rot-1280` (3 of 3 seeds, from M2's own records): 203.2 (201.8–204.3) minutes a run at 211 images a second, on the lists, budget and recipe `s1280` and `m1280` trained with.

| run | trained | epochs | photos per epoch | train minutes | setup minutes | epoch 1 s | images/s | M2b suite minutes | M2 suite minutes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m2b-n1280-s0 | no: M2's weights | – | – | – | – | – | – | 5.4 | – |
| m2b-n1280-s1 | no: M2's weights | – | – | – | – | – | – | 1.9 | – |
| m2b-n1280-s2 | no: M2's weights | – | – | – | – | – | – | 1.8 | – |
| m2b-s1280-s0 | yes | 32 | 79,310 | 239.4 | 10.2 | 467 | 185 | 1.9 | 1.6 |
| m2b-s1280-s1 | yes | 32 | 79,310 | 230.3 | 1.9 | 469 | 186 | 2 | 1.6 |
| m2b-s1280-s2 | yes | 32 | 79,310 | 230.6 | 2.0 | 470 | 186 | 2 | 1.6 |
| m2b-m1280-s0 | yes | 32 | 79,310 | 395.6 | 1.9 | 768 | 108 | 2.3 | 2.1 |
| m2b-m1280-s1 | yes | 32 | 79,310 | 388.8 | 4.7 | 748 | 110 | 4 | 4.3 |
| m2b-m1280-s2 | yes | 32 | 79,310 | 392.4 | 8.6 | 754 | 110 | 4 | 4.3 |
| m2b-n1280-ft-s0 | yes | 10 | 24,992 | 25.4 | 4.7 | 141 | 204 | 1.8 | – |
| m2b-n1280-ft-s1 | yes | 10 | 24,992 | 23.1 | 2.5 | 132 | 203 | 1.8 | – |
| m2b-n1280-ft-s2 | yes | 10 | 24,992 | 22.7 | 2.2 | 132 | 204 | 1.7 | – |
| m2b-n1280-ft2-s0 | yes | 5 + 5 | 24,992 | 10.8 + 10.3 | 0.6 + 0.3 | 129 + 125 | 204 + 203 | 1.8 | – |
| m2b-n1280-ft2-s1 | yes | 5 + 5 | 24,992 | 10.8 + 10.3 | 0.5 + 0.3 | 130 + 124 | 203 + 203 | 1.8 | – |
| m2b-n1280-ft2-s2 | yes | 5 + 5 | 24,992 | 10.8 + 10.3 | 0.6 + 0.2 | 130 + 125 | 204 + 202 | 1.7 | – |
| m2b-n1280-ft-gen50k-s0 | yes | 10 | 24,992 | 23 | 2.4 | 132 | 203 | 1.8 | – |
| m2b-n1280-ft-gen50k-s1 | yes | 10 | 24,992 | 22.1 | 1.5 | 132 | 203 | 1.8 | – |
| m2b-n1280-ft-gen50k-s2 | yes | 10 | 24,992 | 22 | 1.4 | 133 | 203 | 1.8 | – |
| m2b-s1280-ft-s0 | yes | 10 | 24,992 | 23.3 | 1.0 | 141 | 190 | 2 | – |
| m2b-s1280-ft-s1 | yes | 10 | 24,992 | 23.2 | 0.8 | 141 | 189 | 2 | – |
| m2b-s1280-ft-s2 | yes | 10 | 24,992 | 23.1 | 0.7 | 141 | 189 | 1.9 | – |
| m2b-m1280-ft-s0 | yes | 10 | 24,992 | 43 | 4.7 | 238 | 110 | 2.4 | – |
| m2b-m1280-ft-s1 | yes | 10 | 24,992 | 41.3 | 3.1 | 232 | 110 | 2.4 | – |
| m2b-m1280-ft-s2 | yes | 10 | 24,992 | 40.5 | 2.6 | 232 | 111 | 2.4 | – |

Over these runs: 37.88 h of training (setup included) and 1.17 h of scoring, from their records.

### On the laptop's CPU

Each final-test record's own prediction time (Apple M2, 7 torch threads,
one photo a call):

| set | weights | passes | photos a pass | photos a call | seconds | ms a photo |
|---|---|---:|---:|---:|---:|---:|
| store set v2 | M1 (`n640`), 640 px | 3 | 99 | 1 | 17.6 | 59 (52–72) |
| store set v2 | M2 (`gen-rot-1280`); the control is cut from these passes | 3 | 99 | 1 | 54.2 | 182 (174–197) |
| store set v2 | s1280 | 3 | 99 | 1 | 275.4 | 927 (368–1640) |
| store set v2 | m1280 | 3 | 99 | 1 | 237.8 | 801 (791–816) |
| store set v2 | n1280-ft | 3 | 99 | 1 | 52.3 | 176 (169–190) |
| store set v2 | n1280-ft2 | 3 | 99 | 1 | 52.3 | 176 (169–189) |
| store set v2 | n1280-ft-gen50k | 3 | 99 | 1 | 56.2 | 189 (177–212) |
| store set v2 | s1280-ft | 3 | 99 | 1 | 106.2 | 358 (343–376) |
| store set v2 | m1280-ft | 3 | 99 | 1 | 238.5 | 803 (787–825) |
| ExpDate test | n1280-ft | 3 | 665 | 1 | 355.4 | 178 (172–187) |

In all, 18.2 minutes of prediction for store set v2 and 5.9 for ExpDate test on the laptop's CPU; the control's 3 records on v2 took no pass of their own.

The device checks on v1-val (34 photos), each run's own record:

| run | photos a call | minutes |
|---|---:|---:|
| m2b-n1280-ft-s0 | 16 | 0.4 |
| m2b-n1280-ft-s0 | 1 | 0.2 |
| m2b-s1280-ft-s0 | 16 | 1.4 |
| m2b-s1280-ft-s0 | 1 | 0.4 |
| m2b-m1280-ft-s0 | 16 | 8.4 |
| m2b-m1280-ft-s0 | 1 | 0.7 |

Whole jobs (Slurm's elapsed and CPU times, the generation job, the CPU
smokes and the cluster's add-on passes) are in the plan's compute budget.

Made by `python scripts/m2b_report.py` from the run records in
`results/runs/` and, for the per-photo check, `data/store/v2/runs/`.
