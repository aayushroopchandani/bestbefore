"""M2b's results read back from the run records, as tables
(docs/m2b-store-and-size.md, S11 on), for experiments/m2b-*/summarise.py.

Every rule comes from m2b.py, the code the chain applies on the cluster:
each decision is worked out again with m2b.decisions() and compared with
the chain's own record, results/runs/m2b-decisions.json. The cells are
M2's (bestbefore.m2_results): `60/63 (58–60)` is the mean count over the
seeds in, of 63, with the lowest and highest; `0.942 (0.936–0.951)` a mean
with its range; "not yet" a seed not scored or not pulled. Counts are at
each run's own threshold, threshold_store (the best F2 for dates on
v1-val), unless a column says M2's. The M2b suite holds counts only, so
nothing here names a store photo.

    python -m bestbefore.m2b_results       # every M2b run's compute
"""
from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from . import final_test, m1, m2, m2b, runs, store, yolo
from . import m2_results as m2r
from .detection_eval import Prediction, evaluate, summary
from .m2_results import (NA, NOT_YET, Seeds, complete, count_cell, counted, date_map, each,
                         paragraphs, seeds_cell, suite_set, suite_table, value_cell)
from .synth.spec import SYNTH_VAL_V2
from .tables import markdown_table

STEP2_N = tuple(arm for arm, step in m2b.STEP2.items() if step.size == "n")   # ft, ft2, ft-gen50k
FINE_TUNE = (m2b.CONTROL, *STEP2_N)                     # experiments/m2b-fine-tune
GEN_V2 = (m2b.CONTROL, "n1280-ft", "n1280-ft-gen50k")   # experiments/m2b-gen-v2: ft against gen50k
STRATA_V2 = tuple(name for name, _ in SYNTH_VAL_V2)
LEVELS = {"iou0.5": "found at each run's threshold", "any_conf": "boxed at any confidence",
          "at_threshold_m2": "found at M2's threshold"}


def label(arm: str) -> str:
    return f"{arm} (control)" if arm == m2b.CONTROL else arm


def arm_suites(arms: Sequence[str], root: Path | None = None) -> dict[str, list[dict | None]]:
    """Each arm's M2b suite for every seed, keyed by its label; None where a
    seed isn't scored or pulled yet."""
    return {label(arm): m2b.arm_suites(arm, root) for arm in arms}


def run_names(arms: Sequence[str]) -> list[str]:
    return [f"{arm}-s{s}" for arm in arms for s in m2b.SEEDS]


def r2_recipe(root: Path | None = None) -> str:
    """R2's recipe once decided, else ft (the simplest): what s's and m's
    step-2 runs are named with."""
    return (m2b.decisions(root).get("R2") or {}).get("recipe") or "ft"


def size_arms(root: Path | None = None) -> tuple[str, ...]:
    """The model-size side question's arms (experiments/m2b-model-size): n,
    s and m at step 1 (n1280 is the control, M2's weights), then each
    fine-tuned with R2's recipe."""
    recipe = r2_recipe(root)
    return (m2b.CONTROL, *m2b.STEP1, *(f"{size}1280-{recipe}" for size in m2b.SIZES))


# --- The rules, worked out again and compared with the chain's record -----------------

def chain_check(key: str, root: Path | None = None) -> str:
    return m2r.chain_check(key, root, rules=m2b)


def record_check(root: Path | None = None) -> str:
    """The chain's whole record against m2b.decisions() worked out here:
    every decision it holds, R3 included, must be the same."""
    chain, ours = m2.read_json(m2b.decisions_path(root)), m2b.decisions(root)
    if chain is None:
        return ("_The chain's record isn't here yet: `scripts/pull_cluster_results.sh`._"
                if ours else "")
    if ours == chain:
        return (f"Worked out again here with `m2b.decisions()`, the chain's whole record "
                f"(`results/runs/{m2b.decisions_path(root).name}`: {', '.join(chain)}) is the same.")
    keys = sorted(k for k in ours.keys() | chain.keys() if ours.get(k) != chain.get(k))
    return (f"**Warning:** the chain's record differs from `m2b.decisions()` here in "
            f"{', '.join(keys)}. Pull again and check the runs.")


def noise_line(root: Path | None = None) -> str:
    """The noise band on v1-val (noise_store), fixed by the control."""
    noise = m2b.decisions(root).get("noise_store")
    if noise is None:
        scored = sum(s is not None for s in m2b.arm_suites(m2b.CONTROL, root))
        return f"_The noise band: not yet, {scored} of {len(m2b.SEEDS)} control seeds on the M2b suite._"
    values = ", ".join(f"{v:.4f}" for v in noise["control_v1_val_map50"])
    return (f"**The noise band: {noise['band']:.4f}.** The control's three seeds (M2's `gen-rot-1280`, "
            f"no store training) score {values} on v1-val (mAP@0.5 for dates): a spread of "
            f"{noise['spread']:.4f}, so the band is {noise['band']:.4f} (the spread, at least "
            f"{m2b.NOISE_STORE_MIN}), fixed before any other run was scored. "
            f"{chain_check('noise_store', root)}")


def r1_line(root: Path | None = None) -> str:
    """R1: whether m1280 earns 2 more seeds and step 2."""
    r1 = m2b.decisions(root).get("R1")
    if r1 is None:
        return "_R1: not yet (m1280-s0 and every s1280 seed on the M2b suite)._"
    m, s = r1["m1280_s0_v1_val_map50"], r1["s1280_v1_val_map50"]
    beats = "above every one" if m > max(s) else "not above every one"
    verdict = "`m` earns 2 more seeds and step 2" if r1["passes"] else "`m` stops at its probe"
    return (f"**R1, the `m` probe: {'passes' if r1['passes'] else 'fails'}**, so {verdict}. "
            f"`m1280-s0` scores {m:.4f} on v1-val against `s1280`'s seeds' "
            f"{', '.join(f'{v:.4f}' for v in s)}, {beats}, and it "
            f"{'passes' if r1['guard'] else 'fails'} the guard. {chain_check('R1', root)}")


def r2_table(root: Path | None = None) -> str:
    """R2's inputs: the guard's two sets and v1-val, for the control and
    each recipe on n, mean mAP@0.5 for dates over the seeds (range)."""
    control = m2b.arm_suites(m2b.CONTROL, root)
    rows = []
    for arm in (m2b.CONTROL, *(f"{m2b.CONTROL}-{r}" for r in m2b.RECIPES)):
        seeds = m2b.arm_suites(arm, root)
        row = {"arm": label(arm), "seeds": seeds_cell(seeds)}
        for name in (*m2b.GUARDED, m2b.STORE_SET):
            row[name] = value_cell(each(seeds, lambda s: date_map(suite_set(s, name))), 4)
        if arm == m2b.CONTROL:
            row["guard"] = NA
        elif complete(seeds) and complete(control):
            row["guard"] = "passes" if m2b.guard(seeds, control) else "fails"
        else:
            row["guard"] = NOT_YET
        rows.append(row)
    return markdown_table(rows, left=("arm", "guard"))


def r2_line(root: Path | None = None) -> str:
    """R2's choice, and why."""
    decided = m2b.decisions(root)
    r2 = decided.get("R2")
    if r2 is None:
        return "_R2: not yet (`n1280-ft` and `n1280-ft2`, all three seeds, on the M2b suite)._"
    if r2["recipe"] is None:
        return ("**R2: no recipe passed the guard**, so step 2 on `s` and `m` and R3 wait for a "
                f"person. {chain_check('R2', root)}")
    control = m2b.arm_suites(m2b.CONTROL, root)
    floors = ", ".join(f"{name} {m2.map50_date(control, (name,)) - m2b.NOISE:.4f}" for name in m2b.GUARDED)
    means, chosen = r2["v1_val_map50"], r2["recipe"]
    ok = [r for r in m2b.RECIPES if r2["guard"][r]]
    best = max(ok, key=lambda r: means[r])
    if chosen == best:
        why = f"the highest mean on v1-val, {means[chosen]:.4f}"
    else:
        why = (f"{means[chosen]:.4f} on v1-val, within the band ({r2['band']:.4f}) of the best, "
               f"`{best}`'s {means[best]:.4f}, and the simpler")
    others = "; ".join(f"`{r}` {means[r]:.4f} ({means[r] - means[chosen]:+.4f})"
                       for r in m2b.RECIPES if r != chosen)
    out = [r for r in m2b.RECIPES if r not in ok]
    return (f"**R2, by the rule: `{chosen}`**, {why}; {others}. "
            + (f"Out by the guard: {', '.join(out)}. " if out else
               f"Both recipes pass the guard (its floors, the control's mean less {m2b.NOISE}: {floors}). ")
            + chain_check("R2", root))


def r3_inputs(root: Path | None = None) -> dict | None:
    """R3 as m2b.decide_detector works it out, with what the tables show:
    each candidate's arm (CANDIDATES order, "-ft" as R2's recipe), seeds and
    mean v1-val mAP@0.5 for dates, which pass the guard, the best of those,
    the band, its edge (the best less the band) and the choice. None until
    the chain's rules have decided R3 here."""
    decided = m2b.decisions(root)
    if "R3" not in decided:
        return None
    control = m2b.arm_suites(m2b.CONTROL, root)
    arms = m2b.candidate_runs(decided["R2"]["recipe"], decided["R1"]["passes"]).values()
    suites = {arm: m2b.arm_suites(arm, root) for arm in arms}
    means = {arm: m2b.v1_val_map50(seeds) for arm, seeds in suites.items()}
    eligible = [arm for arm, seeds in suites.items() if m2b.guard(seeds, control)]
    best = max(eligible, key=means.__getitem__)
    band = decided["noise_store"]["band"]
    run = decided["R3"]["run"]
    return {"means": means, "eligible": eligible, "best": best, "band": band, "edge": means[best] - band,
            "run_map50": m2b.suite_of(run, root)[m2b.STORE_SET]["map50"]["date"],
            **{k: decided["R3"][k] for k in ("arm", "run", "threshold")}}


def r3_table(root: Path | None = None) -> str:
    """R3's inputs, one row per candidate, cheapest first: the guard's two
    sets and v1-val, mean mAP@0.5 for dates over the seeds (range), the
    guard, each mean's distance from the band's edge, and what R3 made of
    it. m's rows go once R1 has failed, as R3 leaves m out then."""
    decided, got = m2b.decisions(root), r3_inputs(root)
    r1 = decided.get("R1")
    control = m2b.arm_suites(m2b.CONTROL, root)
    rows = []
    for arm in m2b.candidate_runs(r2_recipe(root), r1 is None or r1["passes"]).values():
        seeds = m2b.arm_suites(arm, root)
        row = {"arm": label(arm), "seeds": seeds_cell(seeds)}
        for name in (*m2b.GUARDED, m2b.STORE_SET):
            row[name] = value_cell(each(seeds, lambda s: date_map(suite_set(s, name))), 4)
        row["guard"] = (("passes" if m2b.guard(seeds, control) else "fails")
                        if complete(seeds) and complete(control) else NOT_YET)
        if got is None:
            row |= {"from the band's edge": NOT_YET, "R3": NOT_YET}
        else:
            mean = got["means"][arm]
            row["from the band's edge"] = f"{mean - got['edge']:+.4f}"
            row["R3"] = ("**chosen**" if arm == got["arm"] else
                         "fails the guard" if arm not in got["eligible"] else
                         "in the band" if m2.at_least(mean, got["edge"]) else "below the band")
        rows.append(row)
    return markdown_table(rows, left=("arm", "guard", "R3"))


def r3_line(root: Path | None = None) -> str:
    """R3's choice, why, and how near each other candidate sits to the band's edge."""
    got = r3_inputs(root)
    if got is None:
        return "_R3: not yet (R1, R2 and every candidate's three seeds on the M2b suite)._"
    means, chosen, edge = got["means"], got["arm"], got["edge"]
    if chosen == got["best"]:
        why = f"the highest mean on v1-val, {means[chosen]:.4f}"
    else:
        why = (f"the cheapest candidate within the band ({got['band']:.4f}) of the best, "
               f"`{got['best']}`'s {means[got['best']]:.4f}: the band's edge is {edge:.4f}, and "
               f"`{chosen}`'s {means[chosen]:.4f} sits {means[chosen] - edge:.4f} above it")
    others = [a for a in means if a != chosen and a in got["eligible"]]
    inside = [a for a in others if m2.at_least(means[a], edge)]
    below = [a for a in others if a not in inside]
    out = [a for a in means if a not in got["eligible"]]
    near = min(others, key=lambda a: abs(means[a] - edge), default=None)

    def listed(arms: Sequence[str]) -> str:
        return ", ".join(f"`{a}` ({means[a] - edge:+.4f})" for a in arms)

    return " ".join(part for part in (
        f"**R3, by the rule: `{chosen}`**, {why}.",
        f"Out by the guard: {', '.join(out)}." if out else "Every candidate passes the guard.",
        f"Within the band too, and dearer: {listed(inside)}." if inside else "",
        f"Below the edge: {listed(below)}." if below else "",
        f"Nearest the edge: `{near}`, {abs(means[near] - edge):.4f} "
        f"{'above' if means[near] >= edge else 'below'} it." if near else "",
        f"So the M2b detector is `{got['run']}` (v1-val {got['run_map50']:.4f}) at its threshold_store, "
        f"{got['threshold']:.4f}, unless R4 adopts the ensemble (S13).",
        chain_check("R3", root)) if part)


# --- The suite, set by set ------------------------------------------------------------

def map_table(suites: Mapping[str, Seeds]) -> str:
    """mAP@0.5 for dates on every set of the M2b suite, one row per arm."""
    rows = []
    for arm, seeds in suites.items():
        row = {"arm": arm, "seeds": seeds_cell(seeds)}
        for name in m2b.SUITE:
            row[name] = value_cell(each(seeds, lambda s: date_map(suite_set(s, name))))
        rows.append(row)
    return markdown_table(rows, left=("arm",))


def class_table(suites: Mapping[str, Seeds], name: str = m2b.STORE_SET) -> str:
    """mAP@0.5 for each class on one set (v1-val), one row per arm: the
    dates and the keyword boxes beside them (yolo.CLASSES); "–" where the
    set holds no box of a class."""
    rows = []
    for arm, seeds in suites.items():
        row = {"arm": arm, "seeds": seeds_cell(seeds)}
        for cls in yolo.CLASSES:
            row[cls] = value_cell(each(seeds, lambda s: (suite_set(s, name) or {})["map50"][cls]))
        rows.append(row)
    return markdown_table(rows, left=("arm",))


def early_read_check(root: Path | None = None) -> str:
    """Step 1's early read, M2's suite (suite.json, scored as each step-1
    run finished; n's are M2's own records), against the M2b suite on the
    four sets both hold: the same weights and sets, so it adds only a
    check. Its largest difference in mAP@0.5 for dates and in M2's
    threshold (threshold_m2 here); more than NOISE is a warning."""
    pairs = []
    for arm in (m2b.CONTROL, *m2b.STEP1):
        for s in m2b.SEEDS:
            ours = m2b.suite_of(f"{arm}-s{s}", root)
            if arm == m2b.CONTROL:
                early = m2.suite_of(f"{m2b.CONTROL_FROM}-s{s}", root)
            else:
                run_dir = m2b.finished_run(f"{arm}-s{s}", root)
                early = m2.read_json(run_dir / "suite.json") if run_dir else None
            if ours and early:
                pairs.append((ours, early))
    if not pairs:
        return ""
    maps = max(abs(date_map(suite_set(e, n)) - date_map(suite_set(o, n))) for o, e in pairs for n in m2.SUITE)
    thresholds = max(abs(e["threshold"] - o["threshold_m2"]) for o, e in pairs)
    head = "**Warning:** step" if max(maps, thresholds) > m2b.NOISE else "Step"
    return (f"{head} 1's early read, M2's suite (`suite.json`, scored as each step-1 run finished; `n`'s "
            f"are M2's own records), for {len(pairs)} of {3 * len(m2b.SEEDS)} step-1 runs: on the four "
            f"sets it shares with the M2b suite its mAP@0.5 for dates differs by at most {maps:.4f}, and "
            f"its threshold from the M2b suite's `threshold_m2` by at most {thresholds:.4f}.")


# The two thresholds every M2b suite records: its own, and M2's rule's.
THRESHOLDS = (("store (v1-val)", "threshold", lambda entry: entry),
              ("M2's (real sets)", "threshold_m2", lambda entry: entry["at_threshold_m2"]))


def threshold_table(suites: Mapping[str, Seeds], name: str = m2b.STORE_SET, seeds_column: bool = True) -> str:
    """One set at both thresholds, two rows per arm: the threshold, dates
    and expiry dates found (IoU 0.5), precision and false boxes per photo."""
    rows = []
    for arm, seeds in suites.items():
        sets = [suite_set(s, name) for s in seeds]
        for words, key, level in THRESHOLDS:
            at = each(sets, lambda entry: level(entry)["iou0.5"])
            row = {"arm": arm, "threshold": words}
            if seeds_column:
                row["seeds"] = seeds_cell(seeds)
            row |= {"value": value_cell(each(seeds, lambda s: s[key]), 4),
                    "dates found": count_cell(each(at, counted)),
                    "expiry found": count_cell(each(at, lambda a: counted(a, "expiry"))),
                    "precision": value_cell(each(at, lambda a: a["precision"])),
                    "false boxes / photo": value_cell(each(at, lambda a: a["false_boxes_per_photo"]), 2)}
            rows.append(row)
    return markdown_table(rows, left=("arm", "threshold"))


def control_threshold_table(root: Path | None = None) -> str:
    """The threshold side question: M2's weights at M2's threshold against
    at threshold_store, on v1-val, seed by seed (seed 0 is M2's detector)."""
    control = m2b.arm_suites(m2b.CONTROL, root)
    return threshold_table({f"m2b-{m2b.CONTROL}-s{s}": [suite] for s, suite in zip(m2b.SEEDS, control)},
                           seeds_column=False)


def m2_threshold_check(root: Path | None = None) -> str:
    """M2's threshold as the M2b suite works it out again for the control
    must be each seed's own M2 record: the same weights, sets and rule."""
    pairs = [(s, m2b.suite_of(f"{m2b.CONTROL}-s{s}", root), m2.suite_of(f"{m2b.CONTROL_FROM}-s{s}", root))
             for s in m2b.SEEDS]
    pairs = [(s, ours, theirs) for s, ours, theirs in pairs if ours and theirs]
    if not pairs:
        return ""
    bad = [f"seed {s}: {ours['threshold_m2']:.6f} here, {theirs['threshold']:.6f} in M2's record"
           for s, ours, theirs in pairs if abs(ours["threshold_m2"] - theirs["threshold"]) > 1e-6]
    if bad:
        return "**Warning:** M2's threshold, worked out again, isn't M2's own: " + "; ".join(bad) + "."
    values = ", ".join(f"{theirs['threshold']:.4f}" for _, _, theirs in pairs)
    return (f"M2's threshold, worked out again in the M2b suite, is each control seed's own in M2's "
            f"records ({values}; {len(pairs)} of {len(m2b.SEEDS)} seeds compared).")


def expiry_tags(suites: Mapping[str, Seeds], name: str = m2b.STORE_SET) -> list[str]:
    """The failure tags with an expiry date to find in a set, in store.TAGS order."""
    found = {tag for seeds in suites.values() for s in seeds if s
             for tag, entry in (suite_set(s, name) or {}).get("by_tag", {}).items()
             if entry["iou0.5"].get("expiry_dates")}
    return [t for t in store.TAGS if t in found]


def at(entry: dict, level: str) -> dict:
    """A set's (or a tag's) summary at one of LEVELS: IoU 0.5 at the run's
    threshold, at any confidence, or at M2's threshold."""
    return entry["at_threshold_m2"]["iou0.5"] if level == "at_threshold_m2" else entry[level]


def _by(seeds: Seeds, name: str, key: str) -> list:
    """Each seed's breakdown of one set (by_tag or by_size); None where a seed isn't in."""
    return [(suite_set(s, name) or {}).get(key) if s else None for s in seeds]


def tag_table(suites: Mapping[str, Seeds], level: str = "iou0.5", name: str = m2b.STORE_SET) -> str:
    """Expiry dates by failure tag on one set (v1-val), then every expiry
    and every date, at one of LEVELS: one row per arm."""
    tags, rows = expiry_tags(suites, name), []
    for arm, seeds in suites.items():
        by_tag, sets = _by(seeds, name, "by_tag"), [suite_set(s, name) for s in seeds]
        row = {"arm": arm, "seeds": seeds_cell(seeds)}
        for tag in tags:
            row[tag] = count_cell(each(by_tag, lambda t: counted(at(t[tag], level), "expiry")))
        row["all expiry dates"] = count_cell(each(sets, lambda e: counted(at(e, level), "expiry")))
        row["all dates"] = count_cell(each(sets, lambda e: counted(at(e, level))))
        rows.append(row)
    return markdown_table(rows, left=("arm",))


SIZE_COUNTS = {"found": "found at each run's threshold", "found_at_threshold_m2": "found at M2's threshold",
               "boxed_at_any_confidence": "boxed at any confidence"}


def size_table(suites: Mapping[str, Seeds], key: str = "found") -> str:
    """v1-val's dates by Cause 4's size bands (m2b.SIZE_BANDS), one row per arm."""
    rows = []
    for arm, seeds in suites.items():
        by_size = _by(seeds, m2b.STORE_SET, "by_size")
        row = {"arm": arm, "seeds": seeds_cell(seeds)}
        for band, _, _ in m2b.SIZE_BANDS:
            row[band] = count_cell(each(by_size, lambda b: (b[band][key], b[band]["dates"])))
        rows.append(row)
    return markdown_table(rows, left=("arm",))


def strata_table(suites: Mapping[str, Seeds], level: str = "iou0.5", name: str = "synth-val-v2",
                 strata: Sequence[str] = STRATA_V2) -> str:
    """A generated set's targets, one column per stratum, one row per arm:
    synth-val-v2's, or M2's synth-val with m2_results.STRATA."""
    rows = []
    for arm, seeds in suites.items():
        by_tag = _by(seeds, name, "by_tag")
        row = {"arm": arm, "seeds": seeds_cell(seeds)}
        for stratum in strata:
            row[stratum] = count_cell(each(by_tag, lambda t: counted(t[stratum][level], "expiry")))
        rows.append(row)
    return markdown_table(rows, left=("arm",))


# --- Training: ft2's frozen layers, and compute ---------------------------------------

def frozen_table(arm: str = "n1280-ft2", root: Path | None = None) -> str:
    """What each two-stage run's first stage moved (m2b.frozen_drift, in
    its run.json): in the frozen layers, and in the layers it trained."""
    rows = []
    for name in run_names((arm,)):
        run_dir = m2b.finished_run(name, root)
        row = {"run": f"m2b-{name}"}
        drift = None if run_dir is None else next(
            (stage["frozen"] for stage in m2.read_json(run_dir / "run.json").get("stages", ())
             if "frozen" in stage), None)
        if drift is None:
            rows.append(row | dict.fromkeys(("frozen layers", "convolution weights",
                                             "batch-norm weights and biases", "running means",
                                             "running variances", "values trained"), NOT_YET))
            continue

        def moved(kind: str) -> str:
            return f"{drift[kind]['changed']:,} of {drift[kind]['values']:,}"

        trained = drift["trained"]
        rows.append(row | {"frozen layers": f"0 to {drift['layers'] - 1}",
                           "convolution weights": moved("conv"),
                           "batch-norm weights and biases": moved("bn_affine"),
                           "running means": moved("bn_running_mean"),
                           "running variances": moved("bn_running_var"),
                           "values trained": f"{moved('trained')} "
                                             f"({100 * trained['changed'] / trained['values']:.1f}%)"})
    return markdown_table(rows, left=("run",))


def _times(path: Path) -> list[float] | None:
    """The cumulative `time` column of a results.csv; None if it isn't here."""
    try:
        with path.open() as f:
            return [float(row["time"]) for row in csv.DictReader(f)]
    except FileNotFoundError:
        return None


def stages(run_dir: Path, record: dict) -> list[dict]:
    """Each Ultralytics training in a run (one, or a two-stage run's two):
    its epochs, minutes, the minutes outside its epochs (setup: lists,
    label scan, loading and saving; results.csv's clock starts with epoch
    1), its first epoch's seconds and images a second over epochs 2 to the
    last but one (m2_results.throughput's measure). A
    stage's results.csv is in the run's folder once pulled (a first stage
    in <id>-stage1/), else beside its weights."""
    parts = record.get("stages") or [record]
    out = []
    for k, stage in enumerate(parts, 1):
        if k == len(parts):
            path = m2.results_csv(run_dir)
        else:
            pulled = run_dir.parent / f"{run_dir.name}-stage{k}" / "results.csv"
            path = pulled if pulled.exists() else Path(stage["applied"]["save_dir"]) / "results.csv"
        times, images = _times(path), stage["applied"]["train_images"]
        speed = (images * (len(times) - 2) / (times[-2] - times[0])
                 if times and len(times) >= 3 else None)
        out.append({"epochs": stage["epochs_run"], "minutes": stage["train_minutes"],
                    "setup_minutes": stage["train_minutes"] - times[-1] / 60 if times else None,
                    "epoch_1_s": times[0] if times else None, "images_per_s": speed})
    return out


COMPUTE = ("trained", "epochs", "photos per epoch", "train minutes", "setup minutes", "epoch 1 s",
           "images/s", "M2b suite minutes", "M2 suite minutes")


def runs_table(names: Sequence[str], root: Path | None = None) -> str:
    """Compute, one row per M2b run (`n1280-ft-s0`, ...): epochs, photos
    per epoch, training minutes and those outside its epochs, epoch 1's
    seconds and images a second (a two-stage run's two stages joined by
    +), and the minutes each suite took to score it; then the sums."""
    rows, train, scoring = [], 0.0, 0.0
    for name in names:
        run_dir = m2b.finished_run(name, root)
        row = {"run": f"m2b-{name}"}
        if run_dir is None:
            rows.append(row | dict.fromkeys(COMPUTE, NOT_YET))
            continue
        record = m2.read_json(run_dir / "run.json")
        suites = [m2.read_json(run_dir / f) for f in (m2b.SUITE_FILE, "suite.json")]
        if record["config"].get("trained") is False:
            row |= {"trained": "no: M2's weights", **dict.fromkeys(COMPUTE[1:7], NA)}
        else:
            parts = stages(run_dir, record)

            def joined(key: str, fmt: Callable[[float], str]) -> str:
                return " + ".join(NOT_YET if p[key] is None else fmt(p[key]) for p in parts)

            row |= {"trained": "yes", "epochs": joined("epochs", str),
                    "photos per epoch": f"{record['applied']['train_images']:,}",
                    "train minutes": joined("minutes", lambda v: f"{v:g}"),
                    "setup minutes": joined("setup_minutes", lambda v: f"{v:.1f}"),
                    "epoch 1 s": joined("epoch_1_s", lambda v: f"{v:.0f}"),
                    "images/s": joined("images_per_s", lambda v: f"{v:.0f}")}
            train += record["train_minutes"]
        for column, suite in zip(COMPUTE[7:], suites):
            row[column] = f"{suite['scored']['minutes']:g}" if suite else NA
            scoring += suite["scored"]["minutes"] if suite else 0.0
        rows.append(row)
    total = (f"Over these runs: {train / 60:.2f} h of training (setup included) and "
             f"{scoring / 60:.2f} h of scoring, from their records.")
    return paragraphs(markdown_table(rows, left=("run", "trained")), total)


def control_compute_line(root: Path | None = None) -> str:
    """n's step 1 is M2's gen-rot-1280, trained in M2: its minutes and
    images a second from M2's own records, beside step 1's other sizes."""
    got = [(m2.read_json(d / "run.json")["train_minutes"], m2r.throughput(d))
           for s in m2b.SEEDS if (d := m2.finished_run(f"{m2b.CONTROL_FROM}-s{s}", root))]
    if not got:
        return ""
    return (f"`n`'s step 1 is M2's `{m2b.CONTROL_FROM}` ({len(got)} of {len(m2b.SEEDS)} seeds, from M2's "
            f"own records): {value_cell([m for m, _ in got], 1)} minutes a run at "
            f"{value_cell([v for _, v in got], 0)} images a second, on the lists, budget and recipe "
            "`s1280` and `m1280` trained with.")


def every_run(root: Path | None = None) -> list[str]:
    """Every M2b run the chain lists, but the smoke, in its order by arm:
    the control, step 1, step 2 on n, then on s and m with R2's recipe."""
    recipe = r2_recipe(root)
    return run_names((m2b.CONTROL, *m2b.STEP1, *STEP2_N, f"s1280-{recipe}", f"m1280-{recipe}"))


# --- The experiments' blocks ------------------------------------------------------------

def fine_tune_block(root: Path | None = None) -> str:
    """experiments/m2b-fine-tune: the noise band, R1 and R2; each n arm
    against the control on every set; the threshold side question; v1-val
    by tag and by size; ft2's frozen layers; compute."""
    suites = arm_suites(FINE_TUNE, root)
    return paragraphs(
        "**The rules**, applied by the chain as their inputs came in, and worked out again here:",
        noise_line(root), r1_line(root),
        "**R2**, `ft` against `ft2` on `n`: the guard (real-val and synth-val, no more than "
        f"{m2b.NOISE} under the control), then the mean on v1-val; mAP@0.5 for dates over the 3 seeds:",
        r2_table(root), r2_line(root), record_check(root),
        "**mAP@0.5 for dates on every set of the M2b suite:**", map_table(suites),
        "**real-val-rot180** (ExpDate's validation photos turned 180°), at each run's threshold:",
        suite_table(suites, "real-val-rot180"),
        "**v1-val at both thresholds**: each run's own (threshold_store, the best F2 on v1-val) and "
        "M2's rule's (the best F2 on the three real sets, pooled):", threshold_table(suites),
        "**The threshold on M2's weights**, seed by seed (seed 0 is the M2 detector):",
        control_threshold_table(root), m2_threshold_check(root),
        f"**v1-val by failure tag**, expiry dates {LEVELS['iou0.5']}:", tag_table(suites),
        f"{LEVELS['at_threshold_m2']}:", tag_table(suites, "at_threshold_m2"),
        f"and {LEVELS['any_conf']}:", tag_table(suites, "any_conf"),
        "**v1-val's dates by size** (Cause 4's bands: a box's short side over its photo's long side), "
        f"{SIZE_COUNTS['found']}:", size_table(suites),
        f"{SIZE_COUNTS['found_at_threshold_m2']}:", size_table(suites, "found_at_threshold_m2"),
        f"**synth-val, M2's strata**, targets {LEVELS['iou0.5']}:",
        strata_table(suites, "iou0.5", "synth-val", m2r.STRATA),
        f"and {LEVELS['any_conf']}:", strata_table(suites, "any_conf", "synth-val", m2r.STRATA),
        "**ft2's frozen layers** (stage 1, from the control's weights to its last.pt; layers 0 to 10, "
        "the backbone):", frozen_table(root=root),
        "**Compute** (GPU job B on an H100; minutes and images a second from each run's record and "
        "results.csv):", runs_table(run_names(FINE_TUNE), root))


def gen_v2_block(root: Path | None = None) -> str:
    """experiments/m2b-gen-v2: n1280-ft against n1280-ft-gen50k in the
    plan's order, the control beside them: synth-val-v2 per stratum (the
    layouts), then real-val-small and v1-val's dates by size (the sizes),
    then real-val-rot180. No rule decides it."""
    suites = arm_suites(GEN_V2, root)
    noise = m2b.decisions(root).get("noise_store")
    band = (f"v1-val's noise band is {noise['band']:.4f}, the control's seed spread; "
            if noise else "v1-val's noise band isn't fixed yet; ")
    return paragraphs(
        "No rule decides this side question (\"Reading the side questions\"): it is read in the "
        f"order the plan fixed. As yardsticks, {band}on the real and synthetic sets the rules use "
        f"{m2b.NOISE} of mAP@0.5. {record_check(root)}",
        f"**1. The layouts: synth-val-v2 by stratum**, targets {LEVELS['iou0.5']}:",
        strata_table(suites), f"and {LEVELS['any_conf']}:", strata_table(suites, "any_conf"),
        "synth-val-v2 as a whole, at each run's threshold:", suite_table(suites, "synth-val-v2"),
        "**2. The sizes: real-val-small** (ExpDate's validation photos in 2 × 2 grids, every date "
        "at half its size):", suite_table(suites, "real-val-small"),
        f"and **v1-val's dates by size**, {SIZE_COUNTS['found']}:", size_table(suites),
        f"and {SIZE_COUNTS['boxed_at_any_confidence']}:", size_table(suites, "boxed_at_any_confidence"),
        "v1-val as a whole, at each run's threshold:", suite_table(suites, "v1-val"),
        "**3. real-val-rot180** (ExpDate's validation photos turned 180°):",
        suite_table(suites, "real-val-rot180"))


def model_size_block(root: Path | None = None) -> str:
    """experiments/m2b-model-size: the noise band and R1 again, R3 read
    back; n, s and m at step 1 and step 2 on every set, with the seed
    spread; v1-val by class, at both thresholds, by tag and by size; step
    1's early read as a check; compute by size."""
    arms = size_arms(root)
    suites = arm_suites(arms, root)
    return paragraphs(
        "**The rules**, applied by the chain as their inputs came in, and worked out again here:",
        noise_line(root), r1_line(root),
        "**R3**, every candidate, cheapest first (inference cost, then training cost; `-ft` stands for "
        f"R2's recipe): the guard (real-val and synth-val, no more than {m2b.NOISE} under the control), "
        "then the mean on v1-val; mAP@0.5 for dates over the 3 seeds, and each mean's distance from the "
        "band's edge (the best eligible mean less the band):",
        r3_table(root), r3_line(root), record_check(root),
        "**mAP@0.5 for dates on every set of the M2b suite**, step 1 (public data; `n1280` is M2's "
        "weights) then step 2 (fine-tuned on store photos with R2's recipe):", map_table(suites),
        "**v1-val by class**, mAP@0.5: the dates, and the keyword boxes beside them (`due` the expiry's "
        "word, `prod` the manufacture's, `code` a batch code or a time):", class_table(suites),
        early_read_check(root),
        "**real-val-rot180** (ExpDate's validation photos turned 180°), at each run's threshold:",
        suite_table(suites, "real-val-rot180"),
        f"**synth-val-v2 by stratum**, targets {LEVELS['iou0.5']}:", strata_table(suites),
        f"and {LEVELS['any_conf']}:", strata_table(suites, "any_conf"),
        "**v1-val at both thresholds**: each run's own (threshold_store, the best F2 on v1-val) and "
        "M2's rule's (the best F2 on the three real sets, pooled):", threshold_table(suites),
        f"**v1-val by failure tag**, expiry dates {LEVELS['iou0.5']}:", tag_table(suites),
        f"{LEVELS['at_threshold_m2']}:", tag_table(suites, "at_threshold_m2"),
        f"and {LEVELS['any_conf']}:", tag_table(suites, "any_conf"),
        "**v1-val's dates by size** (Cause 4's bands: a box's short side over its photo's long side), "
        f"{SIZE_COUNTS['found']}:", size_table(suites),
        f"{SIZE_COUNTS['found_at_threshold_m2']}:", size_table(suites, "found_at_threshold_m2"),
        f"and {SIZE_COUNTS['boxed_at_any_confidence']}:", size_table(suites, "boxed_at_any_confidence"),
        "**Compute by size** (GPU jobs A and B on an H100; minutes and images a second from each run's "
        "record and results.csv; the suites' minutes are the only inference time recorded):",
        control_compute_line(root), runs_table(run_names(arms[1:]), root))


# --- S13: the inference add-ons -----------------------------------------------------------

INFERENCE_RUN, TIMING_RUN = "inference", "inference-timing"     # scripts/m2b_inference.py's records


def inference_record(root: Path | None = None) -> dict | None:
    """The newest finished `score` of scripts/m2b_inference.py (inference.json): counts only."""
    run_dir = m2b.finished_run(INFERENCE_RUN, root)
    return m2.read_json(run_dir / "inference.json") if run_dir else None


def timing_record(root: Path | None = None) -> dict | None:
    """The newest finished `time` of scripts/m2b_inference.py (timing.json)."""
    run_dir = m2b.finished_run(TIMING_RUN, root)
    return m2.read_json(run_dir / "timing.json") if run_dir else None


def addon_aps(record: dict) -> dict[str, dict[str, float]]:
    """Each configuration's AP@0.5 for dates (ours) on R4's sets."""
    return {name: {s: c[s]["map50"]["date"] for s in m2b.R4_SETS} for name, c in record["configurations"].items()}


def r4_check(record: dict, root: Path | None = None) -> str:
    """R4 (the noise band) and the what-if (m2b.WHAT_IF_BAND) worked out
    again with m2b.decide_addons from the record's APs, and its arm, run and
    threshold against R3 as m2b.decisions() works it out here; a warning if
    anything differs."""
    decided, problems = m2b.decisions(root), []
    if "R3" not in decided:
        problems.append("R3 isn't decided here")
    else:
        for key, ours in (("arm", "arm"), ("run", "run"), ("threshold_record", "threshold")):
            if record[key] != decided["R3"][ours]:
                problems.append(f"{key}: the record's {record[key]!r}, R3's {decided['R3'][ours]!r}")
        for key, band in (("R4", decided["noise_store"]["band"]), ("what_if", m2b.WHAT_IF_BAND)):
            again = m2b.decide_addons(addon_aps(record), band)
            if again != {k: record[key][k] for k in again}:
                problems.append(f"{key}: worked out again, it adopts {again['adopted'] or 'nothing'}, "
                                f"the record {record[key]['adopted'] or 'nothing'}")
    if problems:
        return "**Warning:** " + "; ".join(problems) + "."
    return ("Worked out again here with `m2b.decide_addons()` from the record's APs, R4 and the what-if are the "
            "record's, and its arm, run and threshold are R3's in the chain's record.")


def walk_line(record: dict, key: str) -> str:
    """What one walk (R4, or the what-if) adopted, and why."""
    walk, run = record[key], record["run"]
    gains = ", ".join(f"{s['addon']} {s['v1_val_gain']:+.4f}" for s in walk["steps"])
    best = max(walk["steps"], key=lambda s: s["v1_val_gain"])
    if key == "R4":
        head = (f"**R4, as written (band {walk['band']:.4f}, the noise band): adopts "
                f"{', '.join(walk['adopted']) or 'nothing'}**")
        tail = (f"So the M2b detector is `{run}` alone at its threshold_store, {walk['threshold']:.4f}."
                if not walk["adopted"] else
                f"So the M2b detector is `{run}` with {walk['configuration']}, at its threshold {walk['threshold']:.4f}.")
    else:
        head = (f"**The what-if (band {walk['band']:.4f}, the plan's floor; it decides nothing): adopts "
                f"{', '.join(walk['adopted']) or 'nothing'}**")
        tail = ("Nothing for the plan's Later." if not walk["adopted"] else
                f"It would keep {walk['configuration']} at its own threshold, {walk['threshold']:.4f}: "
                "for the plan's Later, not for S14.")
    return (f"{head}. Gains on v1-val over what it had kept, in R4's order: {gains}; the largest, "
            f"{best['addon']}'s {best['v1_val_gain']:+.4f}, is "
            f"{'above' if best['v1_val_gain'] > walk['band'] else 'within'} the band. {tail}")


def walk_table(record: dict, key: str) -> str:
    """One walk's steps: each add-on, what it was measured against, our AP
    on v1-val before and after, the gain, real-val's change and the verdict."""
    aps, walk, rows = addon_aps(record), record[key], []
    for s in walk["steps"]:
        before, after = aps[s["against"]], aps[s["tried"]]
        real_ok = m2.at_least(after["real-val"], before["real-val"] - m2b.NOISE)
        rows.append({"add-on": s["addon"], "against": s["against"], "tried": s["tried"],
                     "v1-val before": f"{before[m2b.STORE_SET]:.4f}", "after": f"{after[m2b.STORE_SET]:.4f}",
                     "gain": f"{s['v1_val_gain']:+.4f}", "band": f"{walk['band']:.4f}",
                     "real-val change": f"{s['real_val_change']:+.4f}",
                     "real-val guard": "passes" if real_ok else "fails",
                     "kept": "**kept**" if s["adopted"] else "no"})
    return markdown_table(rows, left=("add-on", "against", "tried", "real-val guard", "kept"))


def configuration_table(record: dict) -> str:
    """Every configuration, the detector alone first: passes, our AP on
    v1-val and real-val, then on v1-val at its own F2 threshold the dates
    and expiries found, precision, false boxes a photo, boxed at any
    confidence, and CPU time a photo on the laptop."""
    rows = []
    for name, c in record["configurations"].items():
        v1, real = c[m2b.STORE_SET], c["real-val"]
        a = v1["iou0.5"]
        rows.append({"configuration": name, "passes": str(len(c["passes"])),
                     "v1-val AP": f"{v1['map50']['date']:.4f}", "real-val AP": f"{real['map50']['date']:.4f}",
                     "threshold": f"{c['threshold']:.4f}", "dates found": f"{a['dates_found']}/{a['dates']}",
                     "expiry found": f"{a['expiry_found']}/{a['expiry_dates']}",
                     "precision": f"{a['precision']:.3f}",
                     "false boxes": f"{a['boxes_kept'] - a['dates_found']}",    # matching is one to one
                     "false / photo": f"{a['false_boxes_per_photo']:.2f}",
                     "boxed at any confidence": f"{v1['any_conf']['dates_found']}/{v1['any_conf']['dates']}",
                     "real-val found": f"{real['iou0.5']['dates_found']}/{real['iou0.5']['dates']}",
                     "CPU ms a photo": f"{c['cpu_ms_per_photo']:.0f}"})
    return markdown_table(rows, left=("configuration",))


def configuration_suites(record: dict) -> dict[str, list[dict]]:
    """Each configuration as a one-seed M2b suite holding v1-val, so S11's
    tag and size tables read it."""
    return {name: [{m2b.STORE_SET: c[m2b.STORE_SET]}] for name, c in record["configurations"].items()}


def ap_check_table(record: dict) -> str:
    """Each seed's upright pass at 1,280 px: our AP against Ultralytics' mAP
    in its run's GPU suite, on real-val (the check) and v1-val (shown)."""
    rows = []
    for seed, row in record["ap_check"]["seeds"].items():
        out = {"run": f"{record['arm']}-{seed}"}
        for name in ("real-val", m2b.STORE_SET):
            out |= {f"{name} ours": f"{row[name]['ours']:.4f}", f"{name} Ultralytics": f"{row[name]['ultralytics']:.4f}",
                    f"{name} difference": f"{row[name]['difference']:+.4f}"}
        rows.append(out)
    return markdown_table(rows, left=("run",))


def ap_check_line(record: dict) -> str:
    check = record["ap_check"]
    if not check["agrees"]:
        return (f"**Warning:** our AP and Ultralytics' differ by {check['largest_real_val_difference']:.4f} on "
                f"real-val, more than {check['within']}: R4's numbers can't be trusted until that is understood.")
    return (f"**The AP check passes**: on real-val ours is within {check['largest_real_val_difference']:.4f} of "
            f"Ultralytics' for every seed (the plan asks {check['within']}).")


def sources_line(record: dict) -> str:
    """Where each set's passes were made: the machine, torch's threads and the code."""
    parts = []
    for name, s in record["sources"].items():
        threads = (f"{s['torch_threads_used']} torch threads" if s.get("torch_threads_used")
                   else "torch's threads not recorded")
        parts.append(f"{name} on {s['processor']} ({s['device']}, {threads}, {s['photos']} photos, "
                     f"code `{s['git_commit'][:7]}`)")
    return "Passes: " + "; ".join(parts) + "."


def timing_table(timing: dict) -> str:
    """The detector against m on the laptop's CPU: milliseconds a photo, the
    median and mean over every photo and repeat, each repeat's mean, and
    each against the detector at 1,280 px."""
    base = next(iter(timing["timed"].values()))["median_ms"]
    rows = [{"model": key, "median ms": f"{v['median_ms']:.0f}", "mean ms": f"{v['mean_ms']:.0f}",
             "each repeat's mean": ", ".join(f"{x:.0f}" for x in v["repeat_mean_ms"]),
             "against the first": f"{v['median_ms'] / base:.2f}×"}
            for key, v in timing["timed"].items()]
    return markdown_table(rows, left=("model", "each repeat's mean"))


def timing_line(timing: dict) -> str:
    c = timing["config"]
    return (f"{c['processor']}, {c['torch_threads']} torch threads, {c['photos']} v1-val photos (long side "
            f"{c['photo_long_side_median']:.0f} px, median) loaded first, one photo a call, rect=False; "
            f"{c['warmup']} warm-up calls each left out, then {c['repeats']} repeats.")


def device_check_line(root: Path | None = None) -> str:
    """R3's run scored again on v1-val on the laptop's CPU
    (score_suite.py --sets v1-val --out <run>/device-check-cpu) beside its GPU suite."""
    decided = m2b.decisions(root)
    run_dir = m2b.finished_run(decided["R3"]["run"], root) if "R3" in decided else None
    here = m2.read_json(run_dir / "device-check-cpu" / m2b.SUITE_FILE) if run_dir else None
    if here is None:
        return "_The device check: not yet._"
    gpu = m2.read_json(run_dir / m2b.SUITE_FILE)
    a, b = gpu[m2b.STORE_SET], here[m2b.STORE_SET]
    pairs = [("dates found", a["iou0.5"]["dates_found"], b["iou0.5"]["dates_found"], a["iou0.5"]["dates"]),
             ("expiries found", a["iou0.5"]["expiry_found"], b["iou0.5"]["expiry_found"], a["iou0.5"]["expiry_dates"]),
             ("boxed at any confidence", a["any_conf"]["dates_found"], b["any_conf"]["dates_found"], a["any_conf"]["dates"])]
    moved = sum(abs(x - y) for _, x, y, _ in pairs)
    counts = "; ".join(f"{label} {x}/{n} on the GPU, {y}/{n} here" for label, x, y, n in pairs)
    return (f"**The device check {'passes' if moved <= 2 else 'fails'}**: `{decided['R3']['run']}` on v1-val, on the "
            f"laptop's CPU against its GPU suite, each at its own F2 threshold ({gpu['threshold']:.4f} on the GPU, "
            f"{here['threshold']:.4f} here; the GPU's is the run's): {counts}; mAP@0.5 for dates (Ultralytics') "
            f"{a['map50']['date']:.4f} and {b['map50']['date']:.4f}." + ("" if moved <= 2 else " **Warning:** more "
                                                                         "than two dates moved."))


def inference_block(root: Path | None = None) -> str:
    """experiments/m2b-inference: the device check, the AP check, R4 and the
    what-if worked out again, every configuration on v1-val (by tag and by
    size) and real-val, and n timed against m on the laptop's CPU."""
    record, timing = inference_record(root), timing_record(root)
    noise = m2b.decisions(root).get("noise_store")
    if record is None:
        parts = ["_The add-ons: not yet (scripts/m2b_inference.py predict on both sets, then score)._"]
    else:
        suites = configuration_suites(record)
        parts = [ap_check_line(record), ap_check_table(record),
                 f"**R4, worked out again** (the noise band {noise['band']:.4f}; R4's own AP, ours, before and "
                 "after, like with like):" if noise else "**R4:**",
                 walk_line(record, "R4"), walk_table(record, "R4"),
                 walk_line(record, "what_if"), walk_table(record, "what_if"), r4_check(record, root),
                 "**Every configuration**, side readings, none adopted (counts on v1-val at each configuration's "
                 "own F2 threshold; real-val's dates found at that threshold too; CPU time a photo is the "
                 "laptop's, its passes' mean plus the fusion):", configuration_table(record), sources_line(record),
                 f"**v1-val by failure tag**, expiry dates {LEVELS['iou0.5']}:", tag_table(suites),
                 f"and {LEVELS['any_conf']}:", tag_table(suites, "any_conf"),
                 f"**v1-val's dates by size**, {SIZE_COUNTS['found']}:", size_table(suites)]
    timing_parts = (["_The timing: not yet (scripts/m2b_inference.py time)._"] if timing is None else
                    [f"**The detector against `m` on the laptop's CPU**: {timing_line(timing)}", timing_table(timing)])
    return paragraphs(device_check_line(root), *parts, *timing_parts, record_check(root))


# --- S14: the final test, store set v2 and ExpDate test ------------------------------------

M2_ROW = "M2 (`gen-rot-1280`) at M2's threshold"
FINAL_LABELS = {m2b.CONTROL: "n1280 (control: M2's weights at threshold_store)",
                "n1280-ft": "n1280-ft (the M2b detector's arm)"}
FINE_TUNED = ("n1280-ft", "n1280-ft2", "n1280-ft-gen50k", "s1280-ft", "m1280-ft")
GROUP_ORDER = {"by_tag": store.TAGS, "by_design": final_test.DESIGN_GROUPS, "by_cause": final_test.CAUSES}


def final_run_dir(name: str, root: Path | None = None) -> Path | None:
    """A final-test record's run folder, by scripts/eval_detector.py's name:
    m2b-<run> (M2b), m2-<run> (M2), else M1's."""
    if name.startswith("m2b-"):
        return m2b.finished_run(name.removeprefix("m2b-"), root)
    if name.startswith("m2-"):
        return m2.finished_run(name.removeprefix("m2-"), root)
    return m1.finished_run(name.removeprefix("m1-"), root)


def final_record(name: str, set_name: str = final_test.SET, root: Path | None = None) -> dict | None:
    """A run's scores on a final-test set (scripts/eval_detector.py), or None."""
    run_dir = final_run_dir(name, root)
    return m2.read_json(run_dir / f"{set_name}.json") if run_dir else None


def final_rows(set_name: str = final_test.SET, root: Path | None = None) -> dict[str, list[dict | None]]:
    """The final test's records on a set, three seeds a row: M1, M2 at its
    own threshold, then every M2b arm at threshold_store, the control first."""
    groups = {m2r.M1_ROW: final_test.M1_RUNS, M2_ROW: final_test.M2_RUNS}
    groups |= {FINAL_LABELS.get(arm, arm): tuple(f"m2b-{arm}-s{s}" for s in m2b.SEEDS)
               for arm in final_test.M2B_ARMS}
    return {row: [final_record(n, set_name, root) for n in names] for row, names in groups.items()}


def rate_cell(pairs: Sequence) -> str:
    """count_cell with the rate beside the count: '41.3/57, 72% (39–44)'."""
    text = count_cell(pairs)
    got = [p for p in pairs if isinstance(p, tuple)]
    if not got or not got[0][1]:
        return text
    rate = 100 * float(np.mean([k for k, _ in got])) / got[0][1]
    count, _, spread = text.partition(" ")
    return f"{count}, {rate:.0f}%" + (f" {spread}" if spread else "")


def final_table(rows: Mapping[str, Seeds]) -> str:
    """One row per arm on store set v2, at each run's own threshold (IoU
    0.5 unless named): expiry dates found, at IoU 0.3 and boxed at any
    confidence, every date found, precision, false boxes a photo, and our
    AP@0.5 for dates, which no threshold moves."""
    out = []
    for name, seeds in rows.items():
        out.append({"detector": name, "seeds": seeds_cell(seeds),
                    "threshold": value_cell(each(seeds, lambda s: s["scored"]["threshold"]), 4),
                    "expiry found": rate_cell(each(seeds, lambda s: counted(s["iou0.5"], "expiry"))),
                    "expiry at IoU 0.3": rate_cell(each(seeds, lambda s: counted(s["iou0.3"], "expiry"))),
                    "expiry boxed at any confidence": rate_cell(
                        each(seeds, lambda s: counted(s["any_conf"], "expiry"))),
                    "all dates found": rate_cell(each(seeds, lambda s: counted(s["iou0.5"]))),
                    "precision": value_cell(each(seeds, lambda s: s["iou0.5"]["precision"])),
                    "false boxes / photo": value_cell(
                        each(seeds, lambda s: s["iou0.5"]["false_boxes_per_photo"]), 2),
                    "AP@0.5 date (ours)": value_cell(each(seeds, lambda s: s["ap50"]["date"]))})
    return markdown_table(out, left=("detector",))


def final_groups(rows: Mapping[str, Seeds], key: str, level: str = "iou0.5", expiry: bool = False) -> list[str]:
    """The groups of a breakdown (by_tag, by_design or by_cause) the records
    hold, in their own order; with `expiry`, only those with an expiry date."""
    held = {g: entry for seeds in rows.values() for s in seeds if s for g, entry in s.get(key, {}).items()}
    return [g for g in GROUP_ORDER[key] if g in held
            and (not expiry or any(((s or {}).get(key, {}).get(g, {}).get(level) or {}).get("expiry_dates")
                                   for seeds in rows.values() for s in seeds))]


def found_by(rows: Mapping[str, Seeds], key: str, level: str = "iou0.5", what: str = "expiry") -> str:
    """Expiry dates (or every date, what="dates") found at `level`, one
    column per group of a breakdown, one row per arm."""
    out = []
    for name, seeds in rows.items():
        row = {"detector": name, "seeds": seeds_cell(seeds)}
        for g in final_groups(rows, key, level, expiry=what == "expiry"):
            row[g] = rate_cell(each(seeds, lambda s: counted(s[key][g][level], what)))
        out.append(row)
    return markdown_table(out, left=("detector",))


def value_by(rows: Mapping[str, Seeds], key: str, get: Callable[[dict, str], object], digits: int = 3) -> str:
    """One number per group of a breakdown (`get(record, group)`: precision,
    false boxes a photo or our AP), one row per arm, mean and range."""
    out = []
    for name, seeds in rows.items():
        row = {"detector": name}
        for g in final_groups(rows, key):
            row[g] = value_cell(each(seeds, lambda s: get(s, g)), digits)
        out.append(row)
    return markdown_table(out, left=("detector",))


def precision_of(key: str) -> Callable[[dict, str], object]:
    return lambda s, g: s[key][g]["iou0.5"]["precision"]


def false_boxes_of(key: str) -> Callable[[dict, str], object]:
    return lambda s, g: s[key][g]["iou0.5"]["false_boxes_per_photo"]


def ap_of(key: str) -> Callable[[dict, str], object]:
    return lambda s, g: s["ap50"][key][g]


def compare(values: Sequence, control: Sequence) -> str:
    """S14's reading, fixed before any score: an arm lifts a measure if its
    seeds' range is apart from the control's, above every one of them."""
    ours = [v for v in values if isinstance(v, (int, float))]
    theirs = [v for v in control if isinstance(v, (int, float))]
    if len(ours) < len(values) or len(theirs) < len(control) or not ours or not theirs:
        return NOT_YET
    if min(ours) > max(theirs):
        return "lifted: ranges apart"
    if max(ours) < min(theirs):
        return "lowered: ranges apart"
    return "not lifted: ranges overlap"


def packs_table(root: Path | None = None) -> str:
    """"Learned packs, not print" (S14's note): each fine-tuned arm against
    the control on v2's unseen designs, by our AP@0.5 for dates, with every
    date and expiry found there beside it; and on v1-val, by mAP@0.5 for
    dates, as R1 to R3 read it."""
    def records(arm: str) -> list[dict | None]:
        return [final_record(f"m2b-{arm}-s{s}", root=root) for s in m2b.SEEDS]

    def v1(arm: str) -> list:
        return each(m2b.arm_suites(arm, root), lambda s: date_map(suite_set(s, m2b.STORE_SET)))

    def unseen_ap(seeds) -> list:
        return each(seeds, lambda s: s["ap50"]["by_design"][final_test.UNSEEN])

    control = records(m2b.CONTROL)
    rows = []
    for arm in (m2b.CONTROL, *FINE_TUNED):
        seeds = records(arm)
        is_control = arm == m2b.CONTROL
        rows.append({"arm": label(arm), "v1-val mAP": value_cell(v1(arm)),
                     "v1-val, against the control": NA if is_control else compare(v1(arm), v1(m2b.CONTROL)),
                     "v2 unseen: AP (ours)": value_cell(unseen_ap(seeds)),
                     "v2 unseen, against the control": NA if is_control else compare(unseen_ap(seeds),
                                                                                     unseen_ap(control)),
                     "v2 unseen: dates found": rate_cell(
                         each(seeds, lambda s: counted(s["by_design"][final_test.UNSEEN]["iou0.5"]))),
                     "v2 unseen: expiry found": rate_cell(
                         each(seeds, lambda s: counted(s["by_design"][final_test.UNSEEN]["iou0.5"], "expiry")))})
    return markdown_table(rows, left=("arm", "v1-val, against the control", "v2 unseen, against the control"))


def expdate_table(root: Path | None = None) -> str:
    """ExpDate test: M1's and M2's detectors' arms as M1 and M2 S8 scored
    them, and the M2b detector's arm at its threshold_store and, from the
    same pass, at M2's rule's threshold."""
    def row(name: str, names: Sequence[str], at_m2: bool = False) -> dict[str, str]:
        seeds = [final_record(n, "real-test", root) for n in names]

        def level(s: dict) -> dict:
            return s["at_threshold_m2"]["iou0.5"] if at_m2 else s["iou0.5"]

        return {"detector": name, "seeds": seeds_cell(seeds),
                "threshold": value_cell(each(seeds, lambda s: (s["at_threshold_m2"] if at_m2 else s)["threshold"]), 4),
                "expiry found": rate_cell(each(seeds, lambda s: counted(level(s), "expiry"))),
                "expiry boxed at any confidence": rate_cell(each(seeds, lambda s: counted(s["any_conf"], "expiry"))),
                "all dates found": rate_cell(each(seeds, lambda s: counted(level(s)))),
                "precision": value_cell(each(seeds, lambda s: level(s)["precision"])),
                "false boxes / photo": value_cell(each(seeds, lambda s: level(s)["false_boxes_per_photo"]), 2),
                "mAP@0.5 date (Ultralytics)": value_cell(each(seeds, lambda s: s["map50"]["date"])),
                "AP@0.5 date (ours)": value_cell(each(seeds, lambda s: s["ap50_ours"]["date"]))}

    return markdown_table([row(m2r.M1_ROW, final_test.M1_RUNS), row(M2_ROW, final_test.M2_RUNS),
                           row("M2b (`n1280-ft`) at threshold_store", final_test.EXPDATE_TEST),
                           row("M2b (`n1280-ft`) at M2's rule's threshold", final_test.EXPDATE_TEST, at_m2=True)],
                          left=("detector",))


def headline_table(root: Path | None = None) -> str:
    """Each milestone's detector, seed 0 at its own threshold, on store set
    v2 and on ExpDate test, with its arm's three seeds beside it."""
    rows = []
    for name, names in ((f"{m2r.M1_ROW}, seed 0", final_test.M1_RUNS),
                        ("M2 (`gen-rot-1280-s0`)", final_test.M2_RUNS),
                        ("M2b (`n1280-ft-s0`)", final_test.EXPDATE_TEST)):
        v2 = [final_record(n, root=root) for n in names]
        test = [final_record(n, "real-test", root) for n in names]

        def expiry(seeds) -> str:
            return rate_cell(each(seeds, lambda s: counted(s["iou0.5"], "expiry")))

        rows.append({"detector": name, "v2: expiry found": expiry(v2[:1]), "v2: its arm's 3 seeds": expiry(v2),
                     "v2: precision": value_cell(each(v2[:1], lambda s: s["iou0.5"]["precision"])),
                     "v2: false boxes / photo": value_cell(
                         each(v2[:1], lambda s: s["iou0.5"]["false_boxes_per_photo"]), 2),
                     "v2: AP (ours)": value_cell(each(v2[:1], lambda s: s["ap50"]["date"])),
                     "ExpDate test: expiry found": expiry(test[:1]),
                     "ExpDate test: its arm's 3 seeds": expiry(test)})
    return markdown_table(rows, left=("detector",))


def final_expected(name: str, root: Path | None = None) -> tuple[float | None, str | None]:
    """The threshold a final-test record must be cut at (its GPU record's,
    unrounded) and the SHA-256 its weights must have."""
    run_dir = final_run_dir(name, root)
    if run_dir is None:
        return None, None
    if name.startswith("m2b-"):
        run = name.removeprefix("m2b-")
        sha = (m2.LAST_PT_SHA256.get(f"{m2b.CONTROL_FROM}-s{run.rsplit('-s', 1)[1]}")
               if run.startswith(f"{m2b.CONTROL}-s") else m2b.LAST_PT_SHA256.get(run))
        return (m2.read_json(run_dir / m2b.SUITE_FILE) or {}).get("threshold"), sha
    if name.startswith("m2-"):
        return (m2.read_json(run_dir / "suite.json") or {}).get("threshold"), m2.LAST_PT_SHA256.get(name[3:])
    return ((m2.read_json(run_dir / "val.json") or {}).get("threshold"),
            m1.BEST_PT_SHA256.get(name.removeprefix("m1-")))


def breakdown_problems(name: str, record: dict) -> list[str]:
    """A v2 record's breakdowns must add up to its counts: by cause to its
    expiry dates, by design to its expiry dates and to every date."""
    out = []
    for level in final_test.LEVELS:
        total, dates = counted(record[level], "expiry"), counted(record[level])
        causes = tuple(sum(record["by_cause"][c][level][k] for c in final_test.CAUSES)
                       for k in ("expiry_found", "expiry_dates"))
        designs = tuple(sum(record["by_design"][g][level].get(k, 0) for g in final_test.DESIGN_GROUPS)
                        for k in ("expiry_found", "expiry_dates"))
        design_dates = tuple(sum(record["by_design"][g][level][k] for g in final_test.DESIGN_GROUPS)
                             for k in ("dates_found", "dates"))
        for what, got, want in (("causes", causes, total), ("designs", designs, total),
                                ("designs' dates", design_dates, dates)):
            if got != want:
                out.append(f"{name}, {level}: its {what} give {got[0]}/{got[1]}, its counts {want[0]}/{want[1]}")
    return out


def final_check(root: Path | None = None) -> str:
    """Every final-test record in so far: scored on a CPU, at its GPU
    record's own threshold exactly, by the weights hashed for it; on v2,
    its breakdowns adding up. A warning names any that aren't."""
    problems, scored = [], {}
    for set_name, names in ((final_test.SET, final_test.RUNS), ("real-test", final_test.EXPDATE_TEST)):
        for name in names:
            record = final_record(name, set_name, root)
            if record is None:
                continue
            scored[set_name] = scored.get(set_name, 0) + 1
            threshold, sha = final_expected(name, root)
            got = record.get("scored", {})
            checks = [("threshold", got.get("threshold"), threshold),
                      ("weights' SHA-256", got.get("weights_sha256"), sha), ("device", got.get("device"), "cpu")]
            checks.append(("photos a call", got.get("batch"), final_test.BATCH))     # v2's and ExpDate test's
            if set_name == final_test.SET:
                problems += breakdown_problems(name, record)
            problems += [f"{name} on {set_name}: {what} {ours!r}, not {want!r}"
                         for what, ours, want in checks if ours != want]
    if problems:
        return "**Warning:** " + "; ".join(problems) + "."
    if not scored:
        return ""
    return (f"Every record in so far ({scored.get(final_test.SET, 0)} of {len(final_test.RUNS)} on store set v2, "
            f"{scored.get('real-test', 0)} of {len(final_test.EXPDATE_TEST)} on ExpDate test) was scored on a CPU, "
            "at its own GPU record's threshold exactly, by the weights hashed for it, one photo a call; on v2 "
            "its counts by cause and by design add up to its own.")


# Each size's device check on v1-val (score_suite.py --sets v1-val --out <run>/<folder>), the
# laptop's CPU against the run's GPU suite: S13's for n and S14's for s and m at 16 photos a
# call, then S14's at one a call, as store set v2 is predicted.
DEVICE_CHECK_RUNS = ("n1280-ft-s0", "s1280-ft-s0", "m1280-ft-s0")
DEVICE_CHECKS = (("device-check-cpu", 16), ("device-check-cpu-batch1", 1))


def device_checks_table(root: Path | None = None) -> str:
    """Each device check beside its run's GPU suite, each at its own F2
    threshold on v1-val: dates and expiry dates found, dates boxed at any
    confidence and Ultralytics' mAP@0.5 for dates, and how many dates moved
    in all; within two it passes (S13). A record that doesn't say how many
    photos a call it took was made at 16, the only setting then."""
    rows = []
    for run in DEVICE_CHECK_RUNS:
        run_dir = m2b.finished_run(run, root)
        gpu = m2.read_json(run_dir / m2b.SUITE_FILE) if run_dir else None
        for folder, batch in DEVICE_CHECKS:
            here = m2.read_json(run_dir / folder / m2b.SUITE_FILE) if run_dir else None
            row = {"run": f"m2b-{run}", "photos a call": str(batch)}
            if gpu is None or here is None:
                rows.append(row | dict.fromkeys(("dates found", "expiry found", "boxed at any confidence",
                                                 "mAP@0.5 date", "moved", "check"), NOT_YET))
                continue
            a, b = gpu[m2b.STORE_SET], here[m2b.STORE_SET]
            counts = {"dates found": ("iou0.5", "dates_found", "dates"),
                      "expiry found": ("iou0.5", "expiry_found", "expiry_dates"),
                      "boxed at any confidence": ("any_conf", "dates_found", "dates")}
            moved = sum(abs(a[lv][k] - b[lv][k]) for lv, k, _ in counts.values())
            made = here.get("scored", {}).get("batch", 16)
            row |= {what: f"{a[lv][k]}/{a[lv][n]} → {b[lv][k]}/{b[lv][n]}" for what, (lv, k, n) in counts.items()}
            row |= {"mAP@0.5 date": f"{a['map50']['date']:.4f} → {b['map50']['date']:.4f}", "moved": str(moved),
                    "check": ("passes" if moved <= 2 else "**fails**") if made == batch
                             else f"**made at {made} a call**"}
            rows.append(row)
    bad = [f"{r['run']} at {r['photos a call']} a call: {r['check'].strip('*')}" for r in rows
           if r["check"] not in ("passes", NOT_YET)]
    warning = f"\n\n**Warning:** a device check didn't pass: {'; '.join(bad)}." if bad else ""
    return markdown_table(rows, left=("run", "check")) + warning


def private_check(root: Path | None = None, load: Callable[[], final_test.V2] | None = None) -> str:
    """The per-photo boxes eval_detector.py kept for each v2 record, in
    data/store/v2/runs/, give the record's counts again, so the kept boxes
    are the scored ones. Reads v2's frozen labels: the laptop only."""
    names = [n for n in final_test.RUNS if final_record(n, root=root)]
    if not names:
        return ""
    try:
        v2 = (load or final_test.load)()
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        return f"_The per-photo boxes weren't checked here: store set v2 can't be read ({type(e).__name__})._"
    bad = []
    for name in names:
        record = final_record(name, root=root)
        kept = runs.latest_run(f"detector-{final_test.SET}-{name}", final_test.RUNS_DIR)
        if kept is None:
            bad.append(f"{name}: no per-photo boxes")
            continue
        boxes = m2.read_json(kept / "predictions.json")
        preds = {pid: [Prediction("date", tuple(b[:4]), b[4]) for b in rows] for pid, rows in boxes.items()}
        again = summary(evaluate(v2.samples, preds, 0.5), record["scored"]["threshold"])
        bad += [f"{name}: {k} {again.get(k)} from its boxes, {record['iou0.5'].get(k)} in its record"
                for k in ("dates_found", "expiry_found", "boxes_kept") if again.get(k) != record["iou0.5"].get(k)]
    if bad:
        return "**Warning:** the kept boxes don't give the records' counts: " + "; ".join(bad) + "."
    return (f"For all {len(names)} v2 records in, the per-photo boxes kept in `data/store/v2/runs/` give the "
            "record's counts again.")


def final_test_block(root: Path | None = None) -> str:
    """experiments/m2b-final-test: store set v2 scored once (S14) by tag,
    by cause and by pack design, every M2b run beside M1's and M2's
    detectors; "learned packs, not print"; ExpDate test for the M2b
    detector; and the checks."""
    rows = final_rows(final_test.SET, root)
    return paragraphs(
        "**The headline**: each milestone's detector, seed 0 at its own threshold, on store set v2 and on ExpDate "
        "test; its arm's three seeds beside it:", headline_table(root),
        "**The device checks** (v1-val on the laptop's CPU against each run's GPU suite, the GPU's count first; "
        "v2 is predicted on that CPU one photo a call):", device_checks_table(root),
        "**Store set v2, every record** (M1's and M2's detectors at their own thresholds; every M2b run at "
        "threshold_store, the control being M2's weights cut there):", final_table(rows),
        "**By failure tag**, expiry dates found at each run's threshold:", found_by(rows, "by_tag"),
        "boxed at any confidence:", found_by(rows, "by_tag", "any_conf"),
        "precision (every box on the tag's photos against all their dates):",
        value_by(rows, "by_tag", precision_of("by_tag")),
        "false boxes a photo:", value_by(rows, "by_tag", false_boxes_of("by_tag"), 2),
        "our AP@0.5 for dates:", value_by(rows, "by_tag", ap_of("by_tag")),
        "**By cause** (M2's causes, its month-word groups as one and upside-down print apart; S14's note), expiry "
        "dates found at each run's threshold:", found_by(rows, "by_cause"),
        "boxed at any confidence:", found_by(rows, "by_cause", "any_conf"),
        "**By pack design** (seen in training: a design in v1-train or v3-train; seen only in validation; "
        "unseen), expiry dates found at each run's threshold:", found_by(rows, "by_design"),
        "every date found:", found_by(rows, "by_design", what="dates"),
        "precision:", value_by(rows, "by_design", precision_of("by_design")),
        "false boxes a photo:", value_by(rows, "by_design", false_boxes_of("by_design"), 2),
        "our AP@0.5 for dates:", value_by(rows, "by_design", ap_of("by_design")),
        "**Learned packs, not print?** Each fine-tuned arm against the control on v2's unseen designs, by our AP "
        "(read as fixed before any score: lifted only if the ranges are apart), beside v1-val:",
        packs_table(root),
        "**ExpDate test** (665 expiry dates, 733 dates; the M2b detector's arm once, M1's and M2's as their "
        "milestones scored them):", expdate_table(root),
        final_check(root), private_check(root), record_check(root))


def main() -> None:
    print(runs_table(every_run()))


if __name__ == "__main__":
    main()
