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


NOT_CHOSEN = ("_Not yet: the M2 detector is chosen on validation once every candidate's 3 "
              "seeds are scored, then scored once on ExpDate test and the store photos (S8)._")


def detector_rows(chosen: dict, set_name: str) -> dict[str, list]:
    """M1's and the M2 detector's arm on a test set: each detector alone
    (seed 0, declared in advance in both milestones), then its 3 seeds."""
    arm, rows = chosen["arm"], r.test_rows([chosen["arm"]], set_name)
    m2_seed = r.seed_runs(arm).index(chosen["run"])
    return {f"the M1 detector (`{r.M1_RUNS[0]}`)": rows[r.M1_ROW][:1],
            "M1, 3 seeds": rows[r.M1_ROW],
            f"the M2 detector (`m2-{chosen['run']}`)": rows[arm][m2_seed:m2_seed + 1],
            f"M2 (`{arm}`), 3 seeds": rows[arm]}


def headline() -> str:
    """The M2 detector next to M1, on ExpDate test and the store photos:
    each detector alone and its 3 seeds, overall and by failure tag."""
    chosen = m2.decisions().get("m2_detector")
    if chosen is None:
        return NOT_CHOSEN
    test, stores = detector_rows(chosen, "real-test"), detector_rows(chosen, "store")
    overall, tags = [], []
    for name, seeds in stores.items():
        def expiry(level: str, rows: list = seeds) -> str:
            return r.count_cell(r.each(rows, lambda s: r.counted(s[level], "expiry")))
        overall.append({
            "detector": name,
            "ExpDate test": r.count_cell(
                r.each(test[name], lambda s: r.counted(s["iou0.5"], "expiry"))),
            "store photos": expiry("iou0.5"),
            "store: boxed at any confidence": expiry("any_conf"),
            "store: precision": r.value_cell(r.each(seeds, lambda s: s["iou0.5"]["precision"])),
            "store: false boxes / photo": r.value_cell(
                r.each(seeds, lambda s: s["iou0.5"]["false_boxes_per_photo"]), 2)})
        by_tag = r.each(seeds, r.by_tag)
        tags.append({"detector": name, **{g: r.count_cell(
            r.each(by_tag, lambda s: r.counted(s[g]["iou0.5"], "expiry"))) for g in r.tag_groups(stores)}})
    return r.paragraphs(
        "Expiry dates found at IoU 0.5, each run at its own threshold:",
        markdown_table(overall, left=("detector",)),
        "On the store photos, by failure tag:",
        markdown_table(tags, left=("detector",)))


# --- What it shows: every number read from the records, as the tables are ----------

def _mean(seeds: list, get) -> float:
    """The mean over the seeds of the count `get` reads (found, of n)."""
    return float(sum(k for k, _ in r.each(seeds, get)) / len(seeds))


def _num(x: float) -> str:
    return f"{round(x, 1):g}"


def what_it_shows() -> str:
    """Where M2 beats M1, where it doesn't, and the table written before any
    run against what was measured. The verdicts are judgments; each sits
    beside the numbers it rests on."""
    chosen = m2.decisions().get("m2_detector")
    arms = tuple(m2.ARMS)
    stores = r.test_rows(arms, "store")
    suites = m2.arm_suites(arms)
    if chosen is None or not all(r.complete(s) for s in (*stores.values(), *suites.values())):
        return NOT_CHOSEN
    best, m1 = chosen["arm"], r.M1_ROW
    causes = r.cause_rows(arms)
    test, alone = detector_rows(chosen, "real-test"), detector_rows(chosen, "store")
    m1_one, _, m2_one, _ = alone        # each detector alone, then its 3 seeds
    small_arms = [a for a in arms if m2.ARMS[a].imgsz == 640]

    def test_expiry(name: str) -> str:
        return r.count_cell(r.each(test[name], lambda s: r.counted(s["iou0.5"], "expiry")))

    def tag_get(tag: str, level: str = "iou0.5"):
        return lambda s: r.counted(r.by_tag(s)[tag][level], "expiry")

    def tag(name: str, t: str, level: str = "iou0.5", rows: dict = stores) -> str:
        return r.count_cell(r.each(rows[name], tag_get(t, level)))

    def cause(name: str, c: str, level: str = "iou0.5") -> str:
        return r.count_cell(r.each(causes[name], lambda s: r.counted(s[c][level], "expiry")))

    def store_value(name: str, key: str, digits: int = 3) -> str:
        return r.value_cell(r.each(stores[name], lambda s: s["iou0.5"][key]), digits)

    def suite_dates(arm: str, set_name: str) -> str:
        sets = [r.suite_set(s, set_name) for s in suites[arm]]
        return r.count_cell(r.each(sets, lambda s: r.counted(s["iou0.5"])))

    def stratum(arm: str, name: str) -> str:
        sets = [r.suite_set(s, "synth-val") for s in suites[arm]]
        return r.count_cell(r.each(sets, lambda s: r.counted(s["by_tag"][name]["iou0.5"], "expiry")))

    def real_val_map(arm: str) -> str:
        return r.value_cell(r.each(suites[arm], lambda s: r.date_map(r.suite_set(s, "real-val"))))

    def spread(values: list[float], digits: int | None = None) -> str:
        fmt = (lambda v: f"{v:.{digits}f}") if digits is not None else _num
        lo, hi = fmt(min(values)), fmt(max(values))
        return lo if lo == hi else f"{lo} to {hi}"

    precision = [_mean_value(stores[a], "precision") for a in arms]
    false_boxes = [_mean_value(stores[a], "false_boxes_per_photo") for a in arms]
    small_640 = [_mean(stores[a], tag_get("small-print")) for a in small_arms]
    sideways_640 = [_mean(stores[a], tag_get("sideways")) for a in small_arms]
    val_maps = [_mean_value(suites[a], None) for a in arms]
    large = f"numbers: print {r.BANDS[-1][1]}"
    found = [k for k, _ in r.each(stores[best], tag_get(r.ALL))]
    mine = r.each(alone[m2_one], tag_get(r.ALL))[0][0]
    seed_rank = ("the fewest" if mine == min(found) else "the most" if mine == max(found)
                 else "between the other two")

    beats = "\n".join([
        f"- **More store expiries found.** `{best}` finds {tag(best, r.ALL)} against M1's "
        f"{tag(m1, r.ALL)}. The M2 detector itself, seed 0 by the rule declared before any run, "
        f"finds {tag(m2_one, r.ALL, rows=alone)}, {seed_rank} of its arm's three seeds (the M1 "
        f"detector: {tag(m1_one, r.ALL, rows=alone)}). Every arm "
        f"finds more than M1: from {spread([_mean(stores[a], tag_get(r.ALL)) for a in arms])} of 81.",
        f"- **The generated dates are what helps.** `gen` against `base`: clean expiries "
        f"{tag('gen', 'clean')} against {tag('base', 'clean')}, full month names "
        f"{cause('gen', r.FULL_NAME)} against {cause('base', r.FULL_NAME)} (M1 "
        f"{cause(m1, r.FULL_NAME)}). Turned copies add nothing there: `rot` finds "
        f"{tag('rot', r.ALL)}, `base` {tag('base', r.ALL)}.",
        f"- **Small print, only at 1,280 px:** `{best}` finds {tag(best, 'small-print')} small-print "
        f"expiries (boxed at any confidence {tag(best, 'small-print', 'any_conf')}); every 640 px "
        f"arm finds {spread(small_640)} of 15, as M1 ({tag(m1, 'small-print')}).",
        f"- **ExpDate kept.** On ExpDate test the M2 detector finds {test_expiry(m2_one)} expiry "
        f"dates (the M1 detector {test_expiry(m1_one)}); on ExpDate validation every arm's mean "
        f"mAP@0.5 for dates is {spread(val_maps, 3)} "
        f"(`base` {real_val_map('base')}).",
    ])
    falls_short = "\n".join([
        f"- **Sideways dates are almost unsolved on the store photos:** `{best}` finds "
        f"{tag(best, 'sideways')}, every 640 px arm at most {_num(max(sideways_640))} of 15, though "
        f"`rot` finds {suite_dates('rot', 'real-val-rot')} turned validation dates and `gen` "
        f"{stratum('gen', 'sideways')} synth-val sideways targets. Neither ExpDate's photos "
        "turned nor the generated sideways dates look like the store's to the model.",
        f"- **Precision is lower than M1's:** {store_value(best, 'precision')} against "
        f"{store_value(m1, 'precision')} over every store date, with "
        f"{store_value(best, 'false_boxes_per_photo', 2)} false boxes a photo against "
        f"{store_value(m1, 'false_boxes_per_photo', 2)}. Every arm keeps more false boxes than M1: "
        f"precision {spread(precision, 3)}, {spread(false_boxes, 2)} false boxes a photo. That is "
        "the price of the F2 threshold (decision 11).",
        f"- **Many dates are boxed but cut by the threshold:** `{best}` boxes "
        f"{tag(best, r.ALL, 'any_conf')} store expiries at some confidence and keeps "
        f"{tag(best, r.ALL)}; small print {tag(best, 'small-print', 'any_conf')} against "
        f"{tag(best, 'small-print')}. A threshold chosen for recall on real hard validation "
        "photos is still above many of the store's dates (cause 4, only partly closed).",
        f"- **Large numeric dates** (printed {r.BANDS[-1][1]} of the long side) stay where M1 "
        f"was: `{best}` {cause(best, large)}, M1 {cause(m1, large)}. M1 already found them.",
    ])

    rows = [
        ("`base`", "sideways: still about 0",
         f"store {tag('base', 'sideways')}; real-val-rot {suite_dates('base', 'real-val-rot')}", "held"),
        ("`base`", "small print: slightly up (threshold)",
         f"store {tag('base', 'small-print')} (M1 {tag(m1, 'small-print')})", "failed"),
        ("`rot`", "sideways: up a lot",
         f"real-val-rot {suite_dates('rot', 'real-val-rot')}; store {tag('rot', 'sideways')}",
         "held on validation, failed on the store photos"),
        ("`gen`", "sideways: up",
         f"synth-val {stratum('gen', 'sideways')}; real-val-rot {suite_dates('gen', 'real-val-rot')}; "
         f"store {tag('gen', 'sideways')}", "held on synth-val only"),
        ("`gen`", "month as a word: up a lot",
         f"synth-val {stratum('gen', 'month-word')}; store full names {cause('gen', r.FULL_NAME)} "
         f"(M1 {cause(m1, r.FULL_NAME)}), abbreviations {cause('gen', r.ABBREVIATION)} "
         f"(M1 {cause(m1, r.ABBREVIATION)})", "partly: up on the store, far below synth-val"),
        ("`gen`", "small print: up",
         f"real-val-small {suite_dates('gen', 'real-val-small')} (`base` "
         f"{suite_dates('base', 'real-val-small')}); store {tag('gen', 'small-print')}",
         "slightly on validation, not on the store photos"),
        ("`gen`", "ExpDate validation: as M1, perhaps 1 point lower",
         f"mAP@0.5 {real_val_map('gen')} (`base` {real_val_map('base')})", "held, no loss"),
        ("`gen-rot`", "sideways: up the most",
         f"real-val-rot {suite_dates('gen-rot', 'real-val-rot')}; store {tag('gen-rot', 'sideways')}",
         "failed"),
        ("`gen-rot`", "month as a word: up a lot",
         f"synth-val {stratum('gen-rot', 'month-word')}; store full names "
         f"{cause('gen-rot', r.FULL_NAME)}", "partly"),
        ("`gen-rot`", "small print: up", f"store {tag('gen-rot', 'small-print')}", "failed"),
        (f"`{best}`", "sideways: as `gen-rot`",
         f"store {tag(best, 'sideways')} (`gen-rot` {tag('gen-rot', 'sideways')})",
         "held: both find almost none"),
        (f"`{best}`", "month as a word: as `gen-rot`",
         f"store full names {cause(best, r.FULL_NAME)} (`gen-rot` {cause('gen-rot', r.FULL_NAME)})",
         "better than expected"),
        (f"`{best}`", "small print: up the most",
         f"real-val-small {suite_dates(best, 'real-val-small')}; store {tag(best, 'small-print')}",
         "held"),
        ("every arm", "ExpDate validation: as M1",
         f"mAP@0.5 {spread(val_maps, 3)}", "held"),
    ]
    expected = markdown_table(
        [{"arm": a, "expected": e, "measured": m, "verdict": v} for a, e, m, v in rows],
        left=("arm", "expected", "measured", "verdict"))

    return r.paragraphs(
        "Every number here is read from the same records as the tables below. The store set is "
        "small (1 to 3 dates between two arms is noise) and less independent than a new one "
        "would be: M1's store result chose M2's targets.",
        "**Where M2 beats M1:**", beats,
        "**Where it doesn't:**", falls_short,
        "**What we expected, written before any run, against what was measured.** The plan "
        "warned that if `gen` scored high on synth-val's month words without lifting the store's, "
        "the generated dates wouldn't look like real ones. It is partly that case: the store's "
        "month words did go up, but far less than synth-val promised.",
        expected)


def _mean_value(seeds: list, key: str | None) -> float:
    """The mean over the seeds of a store score (precision, false boxes) or,
    with key None, of a suite's mAP@0.5 for dates on real-val."""
    get = ((lambda s: s["iou0.5"][key]) if key else
           (lambda s: r.date_map(r.suite_set(s, "real-val"))))
    values = r.each(seeds, get)
    return float(sum(values) / len(values))


def report() -> str:
    return f"""# M2: synthetic dates for the detector

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

{headline()}

## What it shows

{what_it_shows()}

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
difference of 1 to 3 dates between two arms is noise. It was test-only
from M0 to M2, so every number here was measured before any training on
store photos; since 2026-10-28 the retailer's owner allows training on
them, so later milestones are judged on a new set, frozen first.

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
