"""Write results/m2_detector.md: M2's results, rebuilt from the run records
(docs/m2-synthetic-generator.md, S9).

    python scripts/m2_report.py                         # S9: the deliverable
    python scripts/m2_report.py --out /tmp/m2.md        # a look before every run is in

Reads only what the runs saved: each run's run.json, results.csv and
suite.json, the chain's decisions, the real-test.json and store.json that
scripts/eval_detector.py wrote (S8), its per-photo store predictions in
data/store/runs/ (for the causes), and the generation check's records.
Nothing is re-scored, so the report can't drift from the runs, and every
decision is m2.py's rule, checked against the chain's record. A run not in
yet shows as "not yet". Counts only: no store photo is named.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bestbefore import m2, m2_results as r
from bestbefore.config import REPO_ROOT
from bestbefore.tables import markdown_table

OUT = REPO_ROOT / "results" / "m2_detector.md"


def headline() -> str:
    """The M2 detector next to M1, on ExpDate test and the store photos."""
    chosen = m2.decisions().get("m2_detector")
    if chosen is None:
        return ("_Not yet: the M2 detector is chosen on validation once every candidate's 3 "
                "seeds are scored, then scored once on ExpDate test and the store photos (S8)._")
    arm = chosen["arm"]
    test, stores = r.test_rows([arm], "real-test"), r.test_rows([arm], "store")
    groups = r.tag_groups(stores)
    rows = []
    for name, seeds in stores.items():
        tags = r.each(seeds, r.by_tag)
        rows.append({"detector": name if name == r.M1_ROW else f"M2 (`{arm}`)",
                     "seeds": r.seeds_cell(seeds),
                     "ExpDate test": r.count_cell(
                         r.each(test[name], lambda s: r.counted(s["iou0.5"], "expiry"))),
                     **{f"store: {g}": r.count_cell(
                         r.each(tags, lambda s: r.counted(s[g]["iou0.5"], "expiry"))) for g in groups}})
    return markdown_table(rows, left=("detector",))


def report() -> str:
    return f"""# M2: synthetic dates for the detector

M1's detector (`yolo11n`, 640 px) found 97% of the expiry dates on ExpDate
test but about a quarter on the retailer's store photos: it missed
sideways dates, months written as words, and small print. M2 trains the
same detector on other data, every arm with the same budget of 2.56
million images and 3 seeds: M1's data with the M2 recipe (`base`), plus
copies turned 90° (`rot`), plus 50,000 photos with generated dates drawn
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

Expiry dates found, at IoU 0.5:

{headline()}

## The validation suite

`real-val` is ExpDate validation (the guard: must not get worse);
`real-val-rot` the same photos turned 90° or 270°; `real-val-small` the
same photos in 2 × 2 grids, so every date is half its size; `synth-val`
2,100 generated photos in held-out fonts on validation backgrounds, one
target date each, in seven strata that each change one thing from
`clean`.

{r.pilot_line()}

**The M2 detector** (decision 12): eligible if its mean mAP@0.5 for dates
on real-val is no more than 1 point under `base`'s; among those, the best
mean over the hard sets, where a candidate within 1 point goes to the
cheapest.

{r.map_table(m2.arm_suites(m2.CANDIDATES), rule=True)}

{r.detector_decision()}

{r.suite_block(m2.ARMS)}

## The store photos

The frozen store test set v1: 99 photos, 81 printed expiry dates. Each run
at its own threshold, next to M1's three seeds. The set is small: a
difference of 1 to 3 dates between two arms is noise.

{r.store_block(m2.ARMS)}

## Side questions

**Rotation** (no rule; read off): `rot` against `base`, `gen-rot` against
`gen`.

{r.rotation_table()}

**Data size** (decision 14): `gen` against `gen10k`, the same mix and
budget with 50,000 or 10,000 distinct generated photos.

{r.data_size_block()}

**Input size** (decision 13): `gen-rot` at 1,280 px against 640 px.

{r.input_size_block()}

## ExpDate test

Scored once, for the M2 detector's three seeds, next to M1's.

{test_block()}

## The generated data

{r.generation_block()}

## Runs

Every M2 run on one H100. Images per second count epochs 2 to the last but
one, without start-up and the final validation.

{r.runs_table()}

The loader (decision 15):

{r.loader_block()}

Made by `python scripts/m2_report.py` from the run records in
`results/runs/` and, for the causes, `data/store/runs/`.
"""


def test_block() -> str:
    chosen = m2.decisions().get("m2_detector")
    if chosen is None:
        return "_Not yet: scored once the M2 detector is chosen (S8)._"
    return r.scores_table(r.test_rows([chosen["arm"]], "real-test"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT, help=f"default {OUT}")
    args = parser.parse_args()
    md = report()
    args.out.write_text(md)
    print(md)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
