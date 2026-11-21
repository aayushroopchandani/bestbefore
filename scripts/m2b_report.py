"""Write results/m2b_detector.md: M2b's results, rebuilt from the run records
(docs/m2b-store-and-size.md, "Report" and S15).

    python scripts/m2b_report.py                        # S15: the deliverable
    python scripts/m2b_report.py --out /tmp/m2b.md      # a look, written elsewhere

Reads only what the runs saved: each M2b run's run.json, results.csv and
suite-m2b.json (and step 1's suite.json), the chain's decisions, the
inference add-ons' records (scripts/m2b_inference.py), the device checks,
and the final test's store-v2.json and real-test.json
(scripts/eval_detector.py), with M1's and M2's records beside them. Every
table is a bestbefore.m2b_results block, the experiments' own; every
decision is m2b.py's rule, checked against the chain's record; nothing is
scored again. The per-photo boxes kept in data/store/v2/runs/ are read only
to check that they give each v2 record's counts (the laptop only). Counts
only: no store photo, pack design or brand is named.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bestbefore import final_test, m2b
from bestbefore import m2b_results as r
from bestbefore.config import REPO_ROOT
from bestbefore.m2_results import STRATA, paragraphs, suite_table

OUT = REPO_ROOT / "results" / "m2b_detector.md"
ARMS = (m2b.CONTROL, *m2b.STEP1, *r.STEP2_N, "s1280-ft", "m1280-ft")     # every arm the chain ran, the smoke aside


def suite_section() -> str:
    suites = r.arm_suites(ARMS)
    return paragraphs(
        r.noise_line(), r.r1_line(),
        "**mAP@0.5 for dates on every set**, every arm (the control is M2's weights, not trained again):",
        r.map_table(suites),
        "**v1-val by class**, mAP@0.5: the dates, and the keyword boxes M3 reads beside them (`due` the expiry's "
        "word, `prod` the manufacture's, `code` a batch code or a time):", r.class_table(suites),
        r.early_read_check(), r.record_check())


def size_section() -> str:
    suites = r.arm_suites(r.size_arms())
    return paragraphs(
        "**R3, the M2b detector's arm**, every candidate, cheapest first (inference cost, then training cost): "
        f"the guard (real-val and synth-val, no more than {m2b.NOISE} under the control), then the mean on "
        "v1-val, mAP@0.5 for dates over the 3 seeds, and each mean's distance from the band's edge (the best "
        "eligible mean less the band):", r.r3_table(), r.r3_line(),
        "**real-val-rot180** (ExpDate's validation photos turned 180°), at each run's threshold:",
        suite_table(suites, "real-val-rot180"),
        f"**synth-val-v2 by stratum**, targets {r.LEVELS['any_conf']}:", r.strata_table(suites, "any_conf"),
        f"**v1-val by failure tag**, expiry dates {r.LEVELS['iou0.5']}:", r.tag_table(suites),
        f"**v1-val's dates by size** (a box's short side over its photo's long side), {r.SIZE_COUNTS['found']}:",
        r.size_table(suites),
        f"and {r.SIZE_COUNTS['boxed_at_any_confidence']}:", r.size_table(suites, "boxed_at_any_confidence"))


def side_questions() -> str:
    fine = r.arm_suites(r.FINE_TUNE)
    gen = r.arm_suites(r.GEN_V2)
    return paragraphs(
        "### The recipe (R2)",
        "`ft` (the whole network at a tenth of M2's rate) against `ft2` (the backbone frozen for half the "
        "budget, then everything at half that rate), on `n`: the guard, then the mean on v1-val, mAP@0.5 for "
        "dates over the 3 seeds:", r.r2_table(), r.r2_line(),
        "**ft2's frozen layers** (stage 1, from the control's weights to its last.pt; layers 0 to 10):",
        r.frozen_table(),
        "### What gen-v2 adds",
        "`n1280-ft` (its replay draws gen-v2) against `n1280-ft-gen50k` (M2's 50,000 generated photos), no rule, "
        "read in the order the plan fixed: the layouts, then the sizes, then real print turned 180°. gen-v2 "
        "sizes each untilted block like v1-train's dates and a tilted one by its digits, so the arm measures "
        "its layouts and sizes together. On store set v2, by our AP, see the next section's tables.",
        f"**1. The layouts: synth-val-v2 by stratum**, targets {r.LEVELS['iou0.5']}:", r.strata_table(gen),
        f"and {r.LEVELS['any_conf']}:", r.strata_table(gen, "any_conf"),
        "**2. The sizes: real-val-small** (ExpDate's validation photos in 2 × 2 grids, every date at half its "
        "size):", suite_table(gen, "real-val-small"),
        f"and **v1-val's dates by size**, {r.SIZE_COUNTS['found']}:", r.size_table(gen),
        "**3. real-val-rot180:**", suite_table(gen, "real-val-rot180"),
        "### The threshold",
        "M2's detector at M2's threshold against at threshold_store (the best F2 for dates on v1-val), on "
        "v1-val, seed by seed (seed 0 is the M2 detector); store set v2's are in the headline:",
        r.control_threshold_table(), r.m2_threshold_check(),
        "**v1-val at both thresholds**, the control and the arms on `n`:", r.threshold_table(fine),
        f"**M2's synth-val strata**, targets {r.LEVELS['iou0.5']}, where the fine-tuned runs' higher thresholds "
        "cut boxes they still make:", r.strata_table(fine, "iou0.5", "synth-val", STRATA),
        f"and {r.LEVELS['any_conf']}:", r.strata_table(fine, "any_conf", "synth-val", STRATA))


def addons_section() -> str:
    record, timing = r.inference_record(), r.timing_record()
    if record is None:
        return "_Not yet: scripts/m2b_inference.py predict on both sets, then score (S13)._"
    suites = r.configuration_suites(record)
    noise = m2b.decisions().get("noise_store")
    return paragraphs(
        "The M2b detector, `n1280-ft-s0`, given more than one look at each photo: its arm's three seeds merged "
        "by weighted boxes fusion (the ensemble), the photo also turned 90°, 180° and 270° (rotation), and a "
        "1,536 px input. R4 keeps one only if it raises v1-val AP (ours) by more than the noise band; the "
        "what-if, fixed before any add-on was measured, walks the same steps with the plan's floor band and "
        "decides nothing.",
        r.device_check_line(), r.ap_check_line(record), r.ap_check_table(record),
        f"**R4** (the noise band, {noise['band']:.4f}):" if noise else "**R4:**",
        r.walk_line(record, "R4"), r.walk_table(record, "R4"),
        r.walk_line(record, "what_if"), r.walk_table(record, "what_if"), r.r4_check(record),
        "**Every configuration**, side readings, none adopted (counts on v1-val at each configuration's own F2 "
        "threshold; CPU time a photo is the laptop's, its passes' mean plus the fusion):",
        r.configuration_table(record), r.sources_line(record),
        f"**v1-val by failure tag**, expiry dates {r.LEVELS['iou0.5']}:", r.tag_table(suites),
        *([] if timing is None else [
            f"**The detector against `m` on the laptop's CPU**: {r.timing_line(timing)}", r.timing_table(timing)]))


def v2_section() -> str:
    rows = r.final_rows(final_test.SET)
    return paragraphs(
        f"Store set v2 holds {r.set_counts()}: new photos from the same stores, labelled and frozen (its labels "
        "and pack designs on 2026-11-03) before any training on store photos, and never copied to the cluster. "
        "Every planned run was scored on it once, on the laptop's CPU, at the threshold its own GPU record "
        "holds: M1's `n640` at M1's, M2's `gen-rot-1280` at M2's, and every M2b run at threshold_store, the "
        "control being M2's weights cut there. Causes are M2's, with v2's month words as one group (its labels "
        "write every month word `MON`) and upside-down print apart from sideways. A design is **seen in "
        "training** if v1-train or v3-train holds it, **seen only in validation** if only v1-val or v3-val "
        "does, else **unseen**. A group of one or two expiries (curved, upside-down) is a count, not a rate.",
        "**Every record:**", r.final_table(rows),
        "**By failure tag**, expiry dates found at each run's threshold:", r.found_by(rows, "by_tag"),
        "boxed at any confidence:", r.found_by(rows, "by_tag", "any_conf"),
        "precision (every box on the tag's photos against all their dates):",
        r.value_by(rows, "by_tag", r.precision_of("by_tag")),
        "our AP@0.5 for dates:", r.value_by(rows, "by_tag", r.ap_of("by_tag")),
        "**By cause**, expiry dates found at each run's threshold:", r.found_by(rows, "by_cause"),
        "boxed at any confidence:", r.found_by(rows, "by_cause", "any_conf"),
        "**By pack design**, expiry dates found at each run's threshold:", r.found_by(rows, "by_design"),
        "every date found:", r.found_by(rows, "by_design", what="dates"),
        "precision:", r.value_by(rows, "by_design", r.precision_of("by_design")),
        "our AP@0.5 for dates:", r.value_by(rows, "by_design", r.ap_of("by_design")),
        "**Learned packs, not print?** Each fine-tuned arm against the control on v2's unseen designs, by our "
        "AP (fixed before any score: lifted only if the seeds' ranges are apart), beside v1-val:",
        r.packs_table(),
        r.final_check(), r.private_check())


def report() -> str:
    return f"""# M2b: store photos, gen-v2 and a bigger detector

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

{r.headline_block()}

## What it shows

{r.what_it_shows()}

## The validation suite

M2's four sets: `real-val` (ExpDate validation, the guard: must not get
worse), `real-val-rot` (turned 90° or 270°), `real-val-small` (2 × 2
grids, every date half its size) and `synth-val` (M2's generated photos,
7 strata); and three new: `real-val-rot180` (ExpDate's validation photos
turned 180°), `synth-val-v2` (gen-v2's new layouts, 8 strata of 300) and
**`v1-val`**, the store's validation photos (v1's and v3's, 34 photos of
16 pack designs, none of them in training). Every run, the control
included, was scored on all seven by GPU job B.

{suite_section()}

## Model size: step 1 and step 2

Step 1 trains each size on public data with M2's recipe and mix (`n`'s
step 1 is M2's own `gen-rot-1280`); step 2 fine-tunes each seed's step-1
weights on the store photos with R2's recipe. R1 gave `m` its two more
seeds and step 2.

{size_section()}

## Side questions

{side_questions()}

## The inference add-ons

{addons_section()}

## Store set v2, once

{v2_section()}

## ExpDate test

Scored once for the M2b detector's three seeds, on the laptop's CPU, one
photo a call; M1's and M2's as their milestones scored them.

{r.expdate_table()}

## The device checks

v2 was predicted on the laptop's CPU; each size was checked there first on
v1-val, against its GPU suite, at 16 photos a call and at one (the GPU's
count first). Within two dates, or v2 isn't scored.

{r.device_checks_table()}

## Compute

### On the cluster's H100s

Every M2b run's training and scoring, from its record and its
`results.csv`. Images a second count epochs 2 to the last but one, without
start-up and the final validation; a two-stage run's two stages are
joined by +.

{r.control_compute_line()}

{r.runs_table(r.every_run())}

### On the laptop's CPU

Each final-test record's own prediction time (Apple M2, 7 torch threads,
one photo a call):

{r.laptop_table()}

Whole jobs (Slurm's elapsed and CPU times, the generation job, the CPU
smokes and the cluster's add-on passes) are in the plan's compute budget.

Made by `python scripts/m2b_report.py` from the run records in
`results/runs/` and, for the per-photo check, `data/store/v2/runs/`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT, help=f"default {OUT}")
    args = parser.parse_args()
    md = report()
    args.out.write_text(md)
    print(md)
    print(f"-> {args.out}")
    if "**Warning:**" in md:
        raise SystemExit("warning: a check in the report failed")


if __name__ == "__main__":
    main()
