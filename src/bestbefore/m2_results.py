"""M2's results read back from the run records, as tables
(docs/m2-synthetic-generator.md, S7), for experiments/m2-*/summarise.py
and scripts/m2_report.py.

Every rule comes from m2.py, the code the chain applies on the cluster:
each decision is worked out again with it and compared with the chain's
own record. Only the laptop runs this module: the store tables read the
frozen store labels and the per-photo predictions eval_detector.py keeps
in data/store/runs/, and print counts only, never a photo's name.

A table takes what exists so far: a run not yet trained, scored or pulled
shows as "not yet", never as an error. A cell summarises the seeds in so
far: `178.3/187 (176–181)` is the mean count over them, of 187, with the
lowest and highest; the "seeds" column says how many of the 3 are in. "–"
marks a number the record doesn't hold (M1's test scores have no "boxed
at any confidence").
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from . import m1, m2, runs, store
from .config import store_photos_dir
from .detection_eval import Prediction, match_photo
from .images import upright_size
from .samples import DATE_LABELS
from .store_import import SMALL_PRINT, print_height
from .synth.spec import SYNTH_VAL
from .tables import markdown_table

Seeds = Sequence[dict | None]       # one record per seed; None where it isn't in yet

NOT_YET = "not yet"
NA = "–"
_NA = object()                      # a record that doesn't hold the number asked for
M1_RUNS = ("n640", "n640-seed1", "n640-seed2")    # the M1 detector's three seeds (M1 D2, D3)
M1_ROW = "M1 (`n640`)"
ABLATION = ("base", "rot", "gen", "gen-rot")      # the four main arms
STRATA = tuple(name for name, _ in SYNTH_VAL)
ALL = "all"
LEVELS = {"iou0.5": "found at the threshold", "any_conf": "boxed at any confidence"}


# --- Reading the records ------------------------------------------------------

def seed_runs(arm: str) -> list[str]:
    return [f"{arm}-s{s}" for s in m2.SEEDS]


def m2_scores(run: str, set_name: str, root: Path | None = None) -> dict | None:
    """An M2 run's scores on a test set (scripts/eval_detector.py), or None."""
    run_dir = m2.finished_run(run, root)
    return m2.read_json(run_dir / f"{set_name}.json") if run_dir else None


def m1_scores(name: str, set_name: str, root: Path | None = None) -> dict | None:
    run_dir = m1.finished_run(name, root)
    return m2.read_json(run_dir / f"{set_name}.json") if run_dir else None


def test_rows(arms: Sequence[str], set_name: str, root: Path | None = None) -> dict[str, list]:
    """M1's seeds, then each arm's, scored on a test set (store, real-test)."""
    rows = {M1_ROW: [m1_scores(n, set_name, root) for n in M1_RUNS]}
    return rows | {arm: [m2_scores(r, set_name, root) for r in seed_runs(arm)] for arm in arms}


def suite_set(suite: dict | None, name: str) -> dict | None:
    """One set's scores in a suite.json; None if not scored, or missing then."""
    entry = (suite or {}).get(name)
    return None if entry is None or "missing" in entry else entry


def date_map(entry: dict | None) -> float | None:
    """mAP@0.5 for dates on one set; 0 when Ultralytics found no date at all."""
    return None if entry is None else entry["map50"].get("date", 0.0)


def hard_map(suite: dict) -> float | None:
    """One run's mean mAP@0.5 for dates over the HARD sets, as the rule weighs them."""
    values = [date_map(suite_set(suite, name)) for name in m2.HARD]
    return None if None in values else float(np.mean(values))


def counted(summary: dict | None, what: str = "dates") -> tuple[int, int] | None:
    """(found, out of) in a summary (detection_eval.summary): every date, or
    the expiry dates (synth-val's targets)."""
    if summary is None:
        return None
    found, total = ("expiry_found", "expiry_dates") if what == "expiry" else ("dates_found", "dates")
    return summary.get(found, 0), summary.get(total, 0)


def results_times(run_dir: Path) -> list[float] | None:
    """The cumulative `time` column of a run's results.csv; None if it
    isn't here (not pulled)."""
    try:
        with m2.results_csv(run_dir).open() as f:
            return [float(row["time"]) for row in csv.DictReader(f)]
    except FileNotFoundError:
        return None


def throughput(run_dir: Path) -> float | None:
    """Training images per second over epochs 2 to the last but one:
    without epoch 1's start-up and the last epoch's final validation. For a
    3-epoch speed run that is its epoch 2, the loader rule's measure."""
    times = results_times(run_dir)
    if times is None or len(times) < 3:
        return None
    images = m2.read_json(run_dir / "run.json")["applied"]["train_images"]
    return images * (len(times) - 2) / (times[-2] - times[0])


# --- Cells ------------------------------------------------------------------------

def each(seeds: Seeds, get: Callable[[dict], object]) -> list:
    """get(record) for every seed in; None for a seed not in yet, and _NA
    where its record doesn't hold the number."""
    out = []
    for s in seeds:
        if s is None or s is _NA:
            out.append(s)
            continue
        try:
            value = get(s)
        except KeyError:
            value = None
        out.append(_NA if value is None else value)
    return out


def _absent(values: Sequence) -> str:
    return NA if any(v is _NA for v in values) else NOT_YET


def seeds_cell(seeds: Seeds) -> str:
    return f"{sum(s is not None for s in seeds)}/{len(seeds)}"


def count_cell(pairs: Sequence) -> str:
    """'178.3/187 (176–181)': the mean count over the seeds in, of n, with
    the lowest and highest."""
    got = [p for p in pairs if p is not None and p is not _NA]
    if not got:
        return _absent(pairs)
    ks, ns = [k for k, _ in got], {n for _, n in got}
    if len(ns) > 1:
        raise ValueError(f"seeds scored on different sets: n = {sorted(ns)}")
    text = f"{round(float(np.mean(ks)), 1):g}/{ns.pop()}"
    return text if min(ks) == max(ks) else f"{text} ({min(ks)}–{max(ks)})"


def value_cell(values: Sequence, digits: int = 3) -> str:
    """'0.991 (0.989–0.993)': the mean over the seeds in, with the range."""
    got = [v for v in values if v is not None and v is not _NA]
    if not got:
        return _absent(values)

    def fmt(v: float) -> str:
        return f"{v:.{digits}f}"

    text, lo, hi = fmt(float(np.mean(got))), fmt(min(got)), fmt(max(got))
    return text if lo == hi else f"{text} ({lo}–{hi})"


def paragraphs(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


# --- The validation suite ---------------------------------------------------------

def suite_table(suites: Mapping[str, Seeds], name: str) -> str:
    """One row per arm on one suite set, at each run's own threshold (IoU
    0.5 unless named), and its mAP@0.5 for dates."""
    rows = []
    for arm, seeds in suites.items():
        sets = [suite_set(s, name) for s in seeds]
        row = {"arm": arm, "seeds": seeds_cell(sets),
               "mAP@0.5 date": value_cell(each(sets, date_map)),
               "dates found": count_cell(each(sets, lambda s: counted(s["iou0.5"])))}
        if name == "synth-val":
            row["targets found"] = count_cell(each(sets, lambda s: counted(s["iou0.5"], "expiry")))
            row["targets boxed at any confidence"] = count_cell(
                each(sets, lambda s: counted(s["any_conf"], "expiry")))
        row |= {"found at IoU 0.3": count_cell(each(sets, lambda s: counted(s["iou0.3"]))),
                "boxed at any confidence": count_cell(each(sets, lambda s: counted(s["any_conf"]))),
                "precision": value_cell(each(sets, lambda s: s["iou0.5"]["precision"])),
                "false boxes / photo": value_cell(
                    each(sets, lambda s: s["iou0.5"]["false_boxes_per_photo"]), 2)}
        rows.append(row)
    return markdown_table(rows, left=("arm",))


def complete(seeds: Seeds) -> bool:
    return all(s is not None for s in seeds)


def map_table(suites: Mapping[str, Seeds], rule: bool = False) -> str:
    """mAP@0.5 for dates on every suite set and over the HARD sets, one row
    per arm: what the M2 detector rule and the side questions read. With
    `rule`, also each run's threshold and whether the arm is eligible."""
    done = {arm: list(seeds) for arm, seeds in suites.items() if complete(seeds)}
    ok = m2.eligible(done) if "base" in done else None
    rows = []
    for arm, seeds in suites.items():
        row = {"arm": arm, "seeds": seeds_cell(seeds)}
        if rule:
            row["threshold"] = value_cell(each(seeds, lambda s: s["threshold"]))
        for name in ("real-val", *m2.HARD):
            row[name] = value_cell(each(seeds, lambda s: date_map(suite_set(s, name))))
        row["hard sets, mean"] = value_cell(each(seeds, hard_map))
        if rule:
            row["eligible"] = NOT_YET if ok is None or arm not in done else ("yes" if arm in ok else "no")
        rows.append(row)
    return markdown_table(rows, left=("arm", "eligible"))


def strata_table(suites: Mapping[str, Seeds], level: str = "iou0.5") -> str:
    """synth-val's targets, one column per stratum (each changes one thing
    from clean), one row per arm."""
    rows = []
    for arm, seeds in suites.items():
        strata = [(suite_set(s, "synth-val") or {}).get("by_tag") if s else None for s in seeds]
        row = {"arm": arm, "seeds": seeds_cell(strata)}
        for stratum in STRATA:
            row[stratum] = count_cell(each(strata, lambda t: counted(t[stratum][level], "expiry")))
        rows.append(row)
    return markdown_table(rows, left=("arm",))


# --- Decisions, worked out again and compared with the chain's record ---------------

def chain_check(key: str, root: Path | None = None) -> str:
    """How decision `key`, worked out again here with m2.decisions(),
    compares with the chain's record, results/runs/m2-decisions.json."""
    ours = m2.decisions(root).get(key)
    chain = (m2.read_json(m2.decisions_path(root)) or {}).get(key)
    if ours is None and chain is None:
        return ""
    if ours == chain:
        return "The chain's own record (`results/runs/m2-decisions.json`) agrees."
    if chain is None:
        return "The chain hasn't recorded this decision yet."
    if ours is None:
        return (f"**Warning:** the chain has decided this ({chain}), but the runs here don't "
                "complete it: pull again (`scripts/pull_cluster_results.sh`).")
    fields = sorted(k for k in ours.keys() | chain.keys() if ours.get(k) != chain.get(k))
    return ("**Warning:** the chain's record differs: "
            + "; ".join(f"{k}: chain {chain.get(k)!r}, here {ours.get(k)!r}" for k in fields)
            + ". Check the runs.")


def loader_block(root: Path | None = None) -> str:
    """The speed test (decision 15): the three loader settings' epochs and
    the rule's choice."""
    rows, speed = [], {}
    for k, loader in m2.LOADERS.items():
        run = f"speed-{k}"
        row = {"run": f"m2-{run}", "workers": str(loader["workers"])}
        run_dir = m2.finished_run(run, root)
        times = results_times(run_dir) if run_dir else None
        if times is None or len(times) < 3:
            rows.append(row | dict.fromkeys(("RAM cache", "photos", "epoch 1 s", "epoch 2 s",
                                             "epoch 3 s", "images/s, epoch 2", "train minutes"),
                                            NOT_YET))
            continue
        record = m2.read_json(run_dir / "run.json")
        applied, check = record["applied"], record["config"]["ram_check"]
        speed[k] = m2.speed_of(run, root)
        cache = "no"
        if loader["cache"]:
            cache = ("filled" if applied["ram_cache"] else "asked, not filled") + (
                f" (needs {check['need_gb']} of {check['limit_gb']} GB)" if check else "")
        epochs = [t - (times[i - 1] if i else 0.0) for i, t in enumerate(times)]
        row |= {"RAM cache": cache, "photos": str(applied["train_images"]),
                **{f"epoch {i + 1} s": f"{s:.1f}" for i, s in enumerate(epochs[:3])},
                "images/s, epoch 2": f"{speed[k]:.1f}", "train minutes": str(record["train_minutes"])}
        rows.append(row)
    table = markdown_table(rows, left=("run", "RAM cache"))
    if len(speed) < len(m2.LOADERS):
        verdict = (f"_Not decided yet: {len(speed)} of {len(m2.LOADERS)} speed runs finished "
                   "and pulled._")
    else:
        choice = m2.loader_choice(speed)
        fastest = max(speed, key=speed.get)
        loader = m2.LOADERS[choice]
        what = f"{loader['workers']} workers, {'RAM cache' if loader['cache'] else 'no cache'}"
        if choice == fastest:
            why = "the fastest, with no simpler setting within 10% of it"
        else:
            why = (f"its {speed[choice]:.1f} images/s are {100 * speed[choice] / speed[fastest]:.1f}% "
                   f"of the fastest, `{fastest}`'s {speed[fastest]:.1f}: within 10%, and it is the "
                   "simplest such setting")
        verdict = f"**Decision, by the rule: `{choice}` ({what})**, {why}."
    return paragraphs(table, verdict + " " + chain_check("loader", root))


def pilot_line(root: Path | None = None) -> str:
    pilot = m2.decisions(root).get("pilot")
    if pilot is None:
        text = "_The pilot gate: not yet (m2-pilot, scored on synth-val)._"
    else:
        text = (f"The pilot gate: **{'passed' if pilot['passes'] else 'failed'}**, synth-val "
                f"targets boxed at any confidence {pilot['synth_val_any_conf']:.3f} "
                f"(it needs {m2.PILOT_RECALL}).")
    return f"{text} {chain_check('pilot', root)}".strip()


def detector_decision(root: Path | None = None) -> str:
    """The M2 detector rule (decision 12), applied once every candidate's
    three seeds are scored."""
    suites = m2.arm_suites(m2.CANDIDATES, root)
    scored = sum(s is not None for seeds in suites.values() for s in seeds)
    ours = m2.decisions(root).get("m2_detector")
    if ours is None:
        text = (f"_The M2 detector: not decided yet, {scored} of "
                f"{len(m2.CANDIDATES) * len(m2.SEEDS)} candidate runs scored._")
    else:
        arm, hard, ok = ours["arm"], ours["hard_map50"], m2.eligible(suites)
        best = max(ok, key=hard.get)
        why = ("the highest mean mAP@0.5 on the hard sets" if arm == best else
               f"within {hard[best] - hard[arm]:.4f} of the best ({best}, {hard[best]:.4f}), "
               "which is under 1 point, and cheaper")
        left_out = [a for a in m2.CANDIDATES if a not in ok]
        text = (f"**The M2 detector, by the rule: `{arm}`**, {why}. So the M2 detector is "
                f"`m2-{ours['run']}`, with its own threshold, {ours['threshold']:.4f}. "
                + (f"Not eligible, more than 1 point under `base` on real-val: "
                   f"{', '.join(left_out)}." if left_out else "Every candidate is eligible."))
    return f"{text} {chain_check('m2_detector', root)}".strip()


# --- The store photos: by failure tag, and by cause ------------------------------

def by_tag(scores: dict) -> dict[str, dict]:
    """A store.json's summaries by failure tag, and every photo as ALL."""
    return {**scores.get("by_tag", {}), ALL: scores}


def tag_groups(rows: Mapping[str, Seeds]) -> list[str]:
    """The failure tags with an expiry date to find, in store.TAGS order, then ALL."""
    found = {tag for seeds in rows.values() for s in seeds if s
             for tag, entry in s.get("by_tag", {}).items() if entry["iou0.5"].get("expiry_dates")}
    return [t for t in store.TAGS if t in found] + [ALL]


def group_table(rows: Mapping[str, Seeds], groups: Sequence[str], level: str) -> str:
    """Expiry dates found, one row per detector (M1 first) and one column
    per group, a failure tag or a cause. Each seed's record maps a group
    to its summaries by level."""
    out = []
    for name, seeds in rows.items():
        row = {"detector": name, "seeds": seeds_cell(seeds)}
        for group in groups:
            row[group] = count_cell(each(seeds, lambda s: counted(s[group][level], "expiry")))
        out.append(row)
    return markdown_table(out, left=("detector",))


def scores_table(rows: Mapping[str, Seeds]) -> str:
    """One row per detector (M1 first) on a test set: expiry dates found at
    its threshold, at IoU 0.3 and at any confidence; all dates found,
    precision, false boxes per photo; mAP@0.5 for dates where the set has
    YOLO labels."""
    with_map = any(s and "map50" in s for seeds in rows.values() for s in seeds)
    out = []
    for name, seeds in rows.items():
        row = {"detector": name, "seeds": seeds_cell(seeds),
               "expiry found": count_cell(each(seeds, lambda s: counted(s["iou0.5"], "expiry"))),
               "expiry at IoU 0.3": count_cell(each(seeds, lambda s: counted(s["iou0.3"], "expiry"))),
               "expiry boxed at any confidence": count_cell(
                   each(seeds, lambda s: counted(s["any_conf"], "expiry"))),
               "all dates found": count_cell(each(seeds, lambda s: counted(s["iou0.5"]))),
               "precision": value_cell(each(seeds, lambda s: s["iou0.5"]["precision"])),
               "false boxes / photo": value_cell(
                   each(seeds, lambda s: s["iou0.5"]["false_boxes_per_photo"]), 2)}
        if with_map:
            row["mAP@0.5 date"] = value_cell(each(seeds, lambda s: s["map50"]["date"]))
        out.append(row)
    return markdown_table(out, left=("detector",))


# The causes M2 targets ("How M2 closes the gap"), as experiments/m1-store-misses
# split them, with sideways expiries as a group of their own.
BANDS = ((SMALL_PRINT, "under 1.5%"), (0.025, "1.5 to 2.5%"), (math.inf, "2.5% and up"))
FULL_NAME, ABBREVIATION, SIDEWAYS = "month word: full name", "month word: abbreviation", "sideways"
CAUSES = (FULL_NAME, ABBREVIATION, *(f"numbers: print {name}" for _, name in BANDS), SIDEWAYS)


def cause_of(label: dict, size: tuple[int, int]) -> str | None:
    """The cause group of a store photo's expiry; None for an other-face
    photo, which doesn't show it. Upright expiries split by format, as
    experiments/m1-store-misses does (a month word when the format has MON
    in any case, a full name when it has `Month`), and numeric ones by
    print height: the expiry box's height over the photo's long side."""
    if label["tag"] == "other-face":
        return None
    if label["tag"] == "sideways":
        return SIDEWAYS
    fmt = label["format"] or ""
    if "MON" in fmt.upper():
        return FULL_NAME if "Month" in fmt else ABBREVIATION
    height = print_height(label, size) or 0.0
    return "numbers: print " + next(name for top, name in BANDS if height < top)


def store_causes(labels: Mapping[str, dict], sizes: Mapping[str, tuple[int, int]],
                 boxes: Mapping[str, list], threshold: float) -> dict[str, dict]:
    """One run's printed store expiries by cause: {cause: {level:
    {"expiry_dates", "expiry_found"}}} at its threshold and at any
    confidence. `boxes` are the per-photo date boxes eval_detector.py kept,
    [x1, y1, x2, y2, conf]. One matching at every confidence gives both
    levels: a prediction's match depends only on the more confident ones
    (detection_eval)."""
    out = {c: {level: {"expiry_dates": 0, "expiry_found": 0} for level in LEVELS} for c in CAUSES}
    for photo, label in labels.items():
        truths = [b for b in label["boxes"] if b["label"] in DATE_LABELS]
        expiry = [i for i, b in enumerate(truths) if b["label"] == "expiry"]
        group = cause_of(label, sizes[photo])
        if group is None or not expiry:         # not in frame, or worked out from a mfg date
            continue
        preds = [Prediction("date", tuple(b[:4]), b[4]) for b in boxes.get(photo, [])]
        _, taken = match_photo(preds, [b["xyxy"] for b in truths], 0.5)
        for level, cut in (("iou0.5", threshold), ("any_conf", 0.0)):
            entry = out[group][level]
            entry["expiry_dates"] += len(expiry)
            entry["expiry_found"] += sum(taken[i] is not None and taken[i] >= cut for i in expiry)
    return out


def private_predictions(name: str, store_root: Path | None = None) -> tuple[float, dict] | None:
    """The threshold and per-photo date boxes eval_detector.py kept for a
    run on the store photos (n640, m2-base-s0), or None if not scored."""
    run = runs.latest_run(f"detector-store-{name}", store_root or store.RUNS_DIR)
    if run is None:
        return None
    return (m2.read_json(run / "run.json")["config"]["threshold"],
            m2.read_json(run / "predictions.json"))


def cause_rows(arms: Sequence[str], store_root: Path | None = None) -> dict[str, list]:
    """Each detector's store expiries by cause (M1's seeds, then each
    arm's); None for a seed not scored on the store photos yet."""
    names = {M1_ROW: list(M1_RUNS)} | {arm: [f"m2-{r}" for r in seed_runs(arm)] for arm in arms}
    kept = {row: [private_predictions(n, store_root) for n in ns] for row, ns in names.items()}
    if all(k is None for ks in kept.values() for k in ks):
        return {row: [None] * len(ks) for row, ks in kept.items()}
    labels, photos = store.labels(), store_photos_dir()
    sizes = {p: upright_size(photos / p) for p in labels}
    return {row: [None if k is None else store_causes(labels, sizes, k[1], k[0]) for k in ks]
            for row, ks in kept.items()}


def cause_check(rows: Mapping[str, Seeds], causes: Mapping[str, Seeds]) -> str:
    """Each run's causes must add up to the expiries its store.json counts:
    otherwise its kept predictions aren't the ones scored."""
    checked, bad = 0, []
    for name, seeds in rows.items():
        for scores, split in zip(seeds, causes.get(name, ())):
            if scores is None or split is None:
                continue
            checked += 1
            for level in LEVELS:
                ours = tuple(sum(split[c][level][k] for c in CAUSES)
                             for k in ("expiry_found", "expiry_dates"))
                theirs = counted(scores[level], "expiry")
                if ours != theirs:
                    bad.append(f"{name}, {LEVELS[level]}: {ours[0]}/{ours[1]} by cause, "
                               f"{theirs[0]}/{theirs[1]} in store.json")
    if bad:
        return "**Warning:** the causes don't add up to store.json: " + "; ".join(bad) + "."
    return f"For all {checked} runs scored, the causes add up to store.json's counts." if checked else ""


def store_block(arms: Sequence[str], root: Path | None = None,
                store_root: Path | None = None) -> str:
    """The store photos (S8), next to M1: overall, by failure tag and by cause."""
    rows = test_rows(arms, "store", root)
    tags = {name: each(seeds, by_tag) for name, seeds in rows.items()}
    groups = tag_groups(rows)
    causes = cause_rows(arms, store_root)
    parts = ["Each run at its own threshold, IoU 0.5:", scores_table(rows)]
    for level, words in LEVELS.items():
        parts += [f"**By failure tag**, expiry dates {words}:", group_table(tags, groups, level)]
    for level, words in LEVELS.items():
        parts += [f"**By cause**, expiry dates {words} (upright expiries by month word or print "
                  "height; sideways ones apart):", group_table(causes, CAUSES, level)]
    return paragraphs(*parts, cause_check(rows, causes))


# --- The experiments' blocks -------------------------------------------------------

def suite_block(arms: Sequence[str], root: Path | None = None) -> str:
    suites = m2.arm_suites(arms, root)
    parts = []
    for name in m2.SUITE:
        parts += [f"**{name}**, at each run's own threshold:", suite_table(suites, name)]
    parts += [f"**synth-val by stratum**, targets {LEVELS['iou0.5']}:", strata_table(suites),
              f"and {LEVELS['any_conf']}:", strata_table(suites, "any_conf")]
    return paragraphs(*parts)


def rotation_table(root: Path | None = None) -> str:
    """Rotation, read off (the plan fixes no rule for it): rot against
    base, gen-rot against gen, on every sideways measure."""
    suites = m2.arm_suites(ABLATION, root)
    stores = test_rows(ABLATION, "store", root)
    rows = []
    for arm in ABLATION:
        rot = [suite_set(s, "real-val-rot") for s in suites[arm]]
        synth = [suite_set(s, "synth-val") for s in suites[arm]]
        rows.append({
            "arm": arm, "seeds": seeds_cell(suites[arm]),
            "real-val-rot: mAP@0.5 date": value_cell(each(rot, date_map)),
            "real-val-rot: dates found": count_cell(each(rot, lambda s: counted(s["iou0.5"]))),
            "synth-val sideways: targets found": count_cell(
                each(synth, lambda s: counted(s["by_tag"]["sideways"]["iou0.5"], "expiry"))),
            "store sideways: expiry found": count_cell(
                each(stores[arm], lambda s: counted(s["by_tag"]["sideways"]["iou0.5"], "expiry")))})
    return markdown_table(rows, left=("arm",))


def ablation_block(root: Path | None = None, store_root: Path | None = None) -> str:
    """experiments/m2-ablation: the M2 detector rule over every candidate,
    then the four main arms on validation and on the store photos."""
    return paragraphs(
        pilot_line(root),
        "**The M2 detector rule**: mean mAP@0.5 for dates over the 3 seeds, on every candidate:",
        map_table(m2.arm_suites(m2.CANDIDATES, root), rule=True),
        detector_decision(root),
        suite_block(ABLATION, root),
        "**Rotation** (no rule; read off): `rot` against `base`, `gen-rot` against `gen`:",
        rotation_table(root),
        "**The store photos** (S8), next to M1:",
        store_block(ABLATION, root, store_root))


def data_size_block(root: Path | None = None) -> str:
    """experiments/m2-data-size: 50,000 distinct generated photos against
    10,000 listed five times (decision 14)."""
    arms = ("gen", "gen10k")
    suites = m2.arm_suites(arms, root)
    if all(complete(seeds) for seeds in suites.values()):
        gen, small = (m2.map50_date(suites[a], m2.HARD) for a in arms)
        verdict = ("more distinct photos help: 200,000 becomes a Later candidate"
                   if m2.more_photos_help(suites["gen"], suites["gen10k"])
                   else "50,000 is already enough")
        text = (f"**By the rule: {verdict}.** Mean mAP@0.5 for dates on the hard sets: "
                f"`gen` {gen:.4f}, `gen10k` {small:.4f} ({gen - small:+.4f}; the rule needs more "
                "than +0.01).")
    else:
        done = sum(s is not None for seeds in suites.values() for s in seeds)
        text = f"_Not answered yet: {done} of {len(arms) * len(m2.SEEDS)} runs scored._"
    stores = test_rows(arms, "store", root)
    tags = {name: each(seeds, by_tag) for name, seeds in stores.items()}
    return paragraphs(map_table(suites), text,
                      f"**synth-val by stratum**, targets {LEVELS['iou0.5']}:", strata_table(suites),
                      f"**The store photos** (S8), expiry dates {LEVELS['iou0.5']}, by failure tag:",
                      group_table(tags, tag_groups(stores), "iou0.5"))


def input_size_block(root: Path | None = None) -> str:
    """experiments/m2-input-size: gen-rot at 1,280 px against 640 px,
    real-val-small first (the plan fixes no rule beyond that reading; the
    M2 detector rule decides whether 1,280 px is used)."""
    arms = ("gen-rot", "gen-rot-1280")
    suites = m2.arm_suites(arms, root)
    stores = test_rows(arms, "store", root)
    tags = {name: each(seeds, by_tag) for name, seeds in stores.items()}
    return paragraphs("**real-val-small first** (real print at half its size):",
                      suite_table(suites, "real-val-small"),
                      "mAP@0.5 for dates on every suite set:", map_table(suites),
                      f"**The store photos** (S8), expiry dates {LEVELS['iou0.5']}, by failure tag:",
                      group_table(tags, tag_groups(stores), "iou0.5"))


# --- The report's own tables ---------------------------------------------------------

RUN_ORDER = ("smoke", "speed-a", "speed-b", "speed-c", "pilot",
             *(f"{arm}-s{s}" for arm in m2.ARMS for s in m2.SEEDS))


def runs_table(root: Path | None = None) -> str:
    """Every M2 run: its newest attempt's state, training time and speed,
    and its threshold once scored."""
    rows, train, score = [], 0.0, 0.0
    for run in RUN_ORDER:
        attempts = m2.run_dirs(run, root)
        row = {"run": f"m2-{run}", "attempts": str(len(attempts))}
        run_dir = m2.finished_run(run, root)
        if run_dir is None:
            last = (m2.read_json(attempts[-1] / "run.json") or {}) if attempts else {}
            status = {"running": "training"}.get(last.get("status"), last.get("status", NOT_YET))
            rows.append(row | {"state": status} | dict.fromkeys(
                ("photos per epoch", "epochs", "train minutes", "images/s", "threshold"), "–"))
            continue
        record, suite = m2.read_json(run_dir / "run.json"), m2.read_json(run_dir / "suite.json")
        speed = throughput(run_dir)
        train += record["train_minutes"]
        score += suite["scored"]["minutes"] if suite else 0.0
        rows.append(row | {
            "state": "scored" if suite else "finished",
            "photos per epoch": str(record["applied"]["train_images"]),
            "epochs": str(record["epochs_run"]), "train minutes": str(record["train_minutes"]),
            "images/s": f"{speed:.0f}" if speed else "–",
            "threshold": f"{suite['threshold']:.3f}" if suite else "–"})
    done = all(row["state"] in ("scored", "finished") for row in rows)
    total = (f"GPU time{'' if done else ' so far'}: {train / 60:.1f} h training (setup and "
             f"final validation included), {score / 60:.2f} h scoring.")
    return paragraphs(markdown_table(rows, left=("run", "state")), total)


def generation_block(root: Path | None = None) -> str:
    """The generated training photos as S6's check measured them."""
    for run_dir in reversed(m2.run_dirs("generate-check", root)):
        gen = m2.read_json(run_dir / "gen.json")
        if gen and gen["complete"]:
            break
    else:
        return f"_{NOT_YET}: no complete check of the generated photos here._"
    c = gen["counts"]
    rows = [{"share": s["what"], "spec": f"{s['spec']:.3f}", "measured": f"{s['measured']:.3f}",
             "n": str(s["n"])} for s in gen["shares"]]
    return paragraphs(
        f"{gen['photos']:,} photos (spec `{gen['spec_sha256'][:8]}…`, check run "
        f"`{run_dir.name}`): {c['blocks']:,} blocks placed, "
        + ", ".join(f"{n:,} skipped ({why})" for why, n in c["skipped"].items())
        + f"; {c['generated_dates']:,} generated dates.",
        markdown_table(rows, left=("share",)))


# --- The command line every experiments/m2-*/summarise.py shares --------------------

def summarise(block: Callable[[Path], str], readme: Path, about: str) -> None:
    """Print an experiment's results block, rebuilt from the run records;
    with --write, also put it between its README's results markers. Exits
    with 1 after printing when a decision disagrees with the chain's."""
    parser = argparse.ArgumentParser(description=about)
    parser.add_argument("--write", action="store_true", help="update README.md")
    parser.add_argument("--runs-dir", type=Path, default=runs.RUNS_DIR)
    args = parser.parse_args()
    text = block(args.runs_dir)
    print(text)
    if args.write:
        readme.write_text(m1.replace_results(readme.read_text(), text))
        print(f"\nupdated {readme}")
    if "**Warning:**" in text:
        print("\nwarning: a check above failed", file=sys.stderr)
        raise SystemExit(1)
