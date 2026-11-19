"""The final test (docs/m2b-store-and-size.md, "The final test" and S14):
store set v2, scored once, and the groups its results are read by.

    RUNS            the 30 records scored on v2, by scripts/eval_detector.py's
                    names, in the plan's order: M1's n640 at M1's threshold,
                    M2's gen-rot-1280 at M2's, then every M2b run at its
                    threshold_store (the control first: M2's weights again)
    expiry_causes   each expiry box's cause: M2's cause_of, its two month-word
                    groups as one, and upside-down print apart from sideways
    load            v2 from its frozen files: its samples, each photo's design
                    group (seen in training, seen only in validation, unseen)
                    and each expiry box's cause
    breakdown       one run's counts by design and by cause, and our AP

v2's photos and labels stay on the laptop (decision 3). This module reads
the frozen labels and design files, and what it returns holds counts only;
scripts/eval_detector.py puts the per-photo boxes in RUNS_DIR,
data/store/v2/runs/, never in results/.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from . import m2b, store, store_train
from . import m2_results as m2r
from .config import store_photos_dir
from .detection_eval import EXPIRY_LABELS, Matches, average_precision, evaluate, restricted, summary
from .images import upright_size
from .samples import DATE_LABELS, Sample
from .tables import markdown_table

SET = "store-v2"
RUNS_DIR = store.V2.data_dir / "runs"

# The records, fixed before any score (S14's note). The control's runs are M2's
# weights cut at threshold_store, so they share M2's runs' predictions.
M1_RUNS = m2r.M1_RUNS
M2_RUNS = tuple(f"m2-{m2b.CONTROL_FROM}-s{s}" for s in m2b.SEEDS)
M2B_ARMS = (m2b.CONTROL, *m2b.STEP1, "n1280-ft", "n1280-ft2", "n1280-ft-gen50k", "s1280-ft", "m1280-ft")
RUNS = (*M1_RUNS, *M2_RUNS, *(f"m2b-{arm}-s{s}" for arm in M2B_ARMS for s in m2b.SEEDS))
DETECTOR = "m2b-n1280-ft-s0"                    # R3's run; R4 adopted no add-on (S13)
# Photos a call: store set v2 is predicted one at a time, as 16 at 1,280 px
# overflow the 8 GB laptop's memory (S14's change of 2026-11-19).
BATCH = 1
EXPDATE_TEST = tuple(f"m2b-n1280-ft-s{s}" for s in m2b.SEEDS)    # the M2b detector's 3 seeds

# --- By cause -----------------------------------------------------------------------

MONTH_WORD, SIDEWAYS, UPSIDE_DOWN = "month word", m2r.SIDEWAYS, "upside-down"
NUMBERS = tuple(f"numbers: print {name}" for _, name in m2r.BANDS)
CAUSES = (MONTH_WORD, *NUMBERS, SIDEWAYS, UPSIDE_DOWN)
LEVELS = ("iou0.5", "any_conf")


def tall(xyxy: Sequence[float], size: tuple[int, int]) -> bool:
    """Is a box (0..1 of the upright photo) taller than wide in pixels?
    S3's measure (experiments/m2b-store-look)."""
    x1, y1, x2, y2 = xyxy
    width, height = size
    return (y2 - y1) * height > (x2 - x1) * width


def expiry_causes(label: dict, size: tuple[int, int]) -> list[str]:
    """Each expiry box's cause, in the label's box order: M2's cause_of
    (m2_results), with two changes made before v2 was scored. Its full-name
    and abbreviation groups are one, MONTH_WORD: v2's labels write every
    month word as MON, so they can't tell the two apart. And a sideways-
    tagged expiry whose box isn't taller than wide is UPSIDE_DOWN (S3's
    rule, which found v1's 2). An other-face photo has none; nor has a
    derived expiry, which has no box."""
    expiries = [b for b in label["boxes"] if b["label"] == "expiry"]
    cause = m2r.cause_of(label, size)
    if cause is None:
        if expiries:
            raise ValueError("an other-face photo with an expiry box (import check 3 forbids it)")
        return []
    if cause in (m2r.FULL_NAME, m2r.ABBREVIATION):
        return [MONTH_WORD] * len(expiries)
    if cause == m2r.SIDEWAYS:
        return [SIDEWAYS if tall(b["xyxy"], size) else UPSIDE_DOWN for b in expiries]
    return [cause] * len(expiries)


# --- By design ----------------------------------------------------------------------

SEEN, VALIDATION, UNSEEN = "seen in training", "seen only in validation", "unseen"
DESIGN_GROUPS = (SEEN, VALIDATION, UNSEEN)


def design_sides(designs: Mapping[str, str], split: Mapping[str, str],
                 designs_v3: Mapping[str, str], split_v3: Mapping[str, str]) -> tuple[set[str], set[str]]:
    """The designs trained on, and those only validated on: v1's split and
    v3's, as v3-train is training too (S5b decision 6)."""
    def side(part: str) -> set[str]:
        return ({designs[p] for p, s in split.items() if s == part}
                | {designs_v3[p] for p, s in split_v3.items() if s == part})
    train = side("train")
    return train, side("val") - train


def design_group(design: str, train: set[str], validation: set[str]) -> str:
    return SEEN if design in train else VALIDATION if design in validation else UNSEEN


def frozen_sides() -> tuple[set[str], set[str]]:
    """design_sides from the frozen design files and both splits, each
    checked against its hash (DESIGNS_SHA256, DESIGNS_V3_SHA256,
    SPLIT_SHA256, SPLIT_V3_SHA256)."""
    designs_v3 = store_train.read_designs(store_train.DESIGNS_V3_PATH, store.DESIGNS_V3_SHA256)
    return design_sides(store_train.read_designs(), store_train.v1_split(), designs_v3, store_train.v3_split())


# --- Store set v2, as the final test reads it ----------------------------------------

@dataclass(frozen=True)
class V2:
    samples: list[Sample]
    groups: dict[str, str]          # photo -> its design's group
    causes: dict[str, list[str]]    # photo -> each expiry box's cause


def load() -> V2:
    """v2 from its frozen files: the labels, once their file, every photo
    and the photo list are shown unchanged (store.read_frozen); the design
    ids and both splits, each with its hash. Reads each photo's header for
    its upright size; decodes no photo."""
    labels = store.read_frozen(store.V2)["photos"]
    root = store_photos_dir()
    sizes = {p: upright_size(root / p) for p in labels}
    designs = store_train.read_designs()
    if missing := sorted(set(labels) - set(designs)):
        raise ValueError(f"{len(missing)} v2 photos have no design")
    train, validation = frozen_sides()
    return V2(store.to_samples(labels, root),
              {p: design_group(designs[p], train, validation) for p in labels},
              {p: expiry_causes(label, sizes[p]) for p, label in labels.items()})


def counts(v2: V2) -> str:
    """What v2 holds, by failure tag, design group and cause: photos,
    expiry boxes and date boxes. Counts only."""
    def row(kind: str, name: str, samples: Sequence[Sample]) -> dict[str, str]:
        return {"by": kind, "group": name, "photos": str(len(samples)),
                "expiries in frame": str(sum(b.label in EXPIRY_LABELS for s in samples for b in s.boxes)),
                "dates": str(sum(b.label in DATE_LABELS for s in samples for b in s.boxes))}

    rows = [row("tag", tag, [s for s in v2.samples if s.tag == tag]) for tag in store.TAGS]
    rows += [row("design", g, [s for s in v2.samples if v2.groups[s.id] == g]) for g in DESIGN_GROUPS]
    per_cause = Counter(c for cs in v2.causes.values() for c in cs)
    rows += [{"by": "cause", "group": c, "photos": "–", "expiries in frame": str(per_cause[c]), "dates": "–"}
             for c in CAUSES]
    rows.append(row("all", "every photo", v2.samples))
    return markdown_table(rows, left=("by", "group"))


# --- One run's breakdown --------------------------------------------------------------

def cause_counts(samples: Sequence[Sample], at_05: Matches, conf: float,
                 causes: Mapping[str, list[str]]) -> dict[str, dict]:
    """Expiry dates by cause, found at IoU 0.5 at `conf` and at any
    confidence, from `at_05`, evaluate's matching of these samples (it
    lists their dates in this same order)."""
    out = {c: {level: {"expiry_dates": 0, "expiry_found": 0} for level in LEVELS} for c in CAUSES}
    i = 0
    for s in samples:
        mine = iter(causes.get(s.id, ()))
        for b in s.boxes:
            if b.label not in DATE_LABELS:
                continue
            matched, i = at_05.truth_conf[i], i + 1
            if b.label not in EXPIRY_LABELS:
                continue
            cause = next(mine)
            for level, cut in (("iou0.5", conf), ("any_conf", 0.0)):
                out[cause][level]["expiry_dates"] += 1
                out[cause][level]["expiry_found"] += int(matched >= cut)
        if next(mine, None) is not None:
            raise ValueError(f"{s.id}: more causes than expiry boxes")
    if i != len(at_05.truth_conf):
        raise ValueError(f"{i} dates walked, {len(at_05.truth_conf)} matched")
    return out


def breakdown(v2: V2, predictions: Mapping[str, list], conf: float) -> dict:
    """One run's counts beyond its failure tags (detector.score's): by
    design group, as by tag (found at IoU 0.5 and 0.3 at `conf`, boxed at
    any confidence, precision, false boxes a photo); by cause, expiry dates
    found at `conf` and at any confidence (precision belongs to a photo,
    not a cause); and our AP@0.5 for dates (detection_eval.average_precision),
    overall, by tag and by design group. Counts only."""
    at_05 = evaluate(v2.samples, predictions, 0.5)
    grouped = [replace(s, tag=v2.groups[s.id]) for s in v2.samples]
    g05, g03 = evaluate(grouped, predictions, 0.5), evaluate(grouped, predictions, 0.3)
    tags = [t for t in store.TAGS if any(s.tag == t for s in v2.samples)]
    return {
        "by_design": {g: {"iou0.5": summary(g05, conf, {g}), "iou0.3": summary(g03, conf, {g}),
                          "any_conf": summary(g05, 0.0, {g})} for g in DESIGN_GROUPS},
        "by_cause": cause_counts(v2.samples, at_05, conf, v2.causes),
        "ap50": {"date": average_precision(at_05),
                 "by_tag": {t: average_precision(restricted(at_05, {t})) for t in tags},
                 "by_design": {g: average_precision(restricted(g05, {g})) for g in DESIGN_GROUPS}},
    }
