"""M3's settings and decision rules (docs/m3-date-reader.md, "Decision
rules"). Each rule is written here before the run it judges. So far: S2's
crop experiment, R1. R2 to R6 come with S9's chain.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .reader import parse
from .runs import RUNS_DIR

# M3's decisions, one file, each key written by the step that decides it
# (R1 by S2's scoring; the chain's later). Pulled with the run records.
DECISIONS_PATH = RUNS_DIR / "m3-decisions.json"

# reader/parse.py's SHA-256, frozen before S2's job ran (2026-11-13). A
# change to the parser is a new version: a new hash, recorded and re-scored.
PARSE_SHA256: str | None = "026755390a0392eca42dbf01391b9628f0ac27b434743b655fc3b576244613ea"


def parse_sha256() -> str:
    with open(parse.__file__, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_parser() -> str:
    """The parser's hash, once it is shown to be the frozen one."""
    digest = parse_sha256()
    if PARSE_SHA256 is None:
        raise RuntimeError("reader/parse.py isn't frozen yet: PARSE_SHA256 is unset in m3.py")
    if digest != PARSE_SHA256:
        raise RuntimeError(f"reader/parse.py has changed since it was frozen ({digest[:12]} is not "
                           f"{PARSE_SHA256[:12]}): a new parser is a new version, recorded first")
    return digest


# --- S2, the crop experiment ------------------------------------------------------

PADS = (0.0, 0.15, 0.4)            # padding, as a share of the box's short side
CROP_READERS = ("rapidocr", "ppocrv6")
# What runs on a crop: the reader's full pipeline; its recogniser on the
# crop as cut; its recogniser read both ways (decision 14).
CROP_MODES = ("full", "alone", "turned")


def crop_recipe(rows: Iterable[dict]) -> dict:
    """R1. `rows`: one per crop and configuration, each with `reader`,
    `mode`, `pad`, `shape` (wide or tall) and `outcome` (right, wrong,
    none, or None where the truth gives no date).

    pad: the most right reads, summed over both untrained readers alone
    with turning on, over every crop given (real-val and every store crop
    with a value; none of them test); a tie goes to the smaller pad.
    turn: kept if, at that pad, on tall crops (both readers summed), turning
    adds right reads and adds no wrong read."""
    rows = [r for r in rows if r["reader"] in CROP_READERS]

    def count(mode: str, pad: float, outcome: str, tall_only: bool = False) -> int:
        return sum(r["mode"] == mode and r["pad"] == pad and r["outcome"] == outcome
                   and (not tall_only or r["shape"] == "tall") for r in rows)

    right = {pad: count("turned", pad, "right") for pad in PADS}
    pad = max(PADS, key=lambda p: (right[p], -p))
    tall = {mode: {o: count(mode, pad, o, tall_only=True) for o in ("right", "wrong", "none")}
            for mode in ("alone", "turned")}
    turn = tall["turned"]["right"] > tall["alone"]["right"] and tall["turned"]["wrong"] <= tall["alone"]["wrong"]
    return {"pad": pad, "turn": turn, "right_alone_turned_by_pad": {str(p): right[p] for p in PADS},
            "tall_at_pad": tall}


# The owner's departures from R1, each tied to the run it was made on
# (docs/m3-date-reader.md, S2). m3-decisions.json keeps R1's own result.
# On run 20261113-185541, R1 dropped turning: on the 41 tall crops (82
# reads, both readers, pad 0) it took 0 / 0 / 82 to 71 / 2 / 9. Both wrong
# reads are one misread, PP-OCRv6 reading 'FEB 2025' as 'FE8 2025' in the
# right orientation; turning chose the right way up on every tall crop.
# Without it M3 reads none of them. Kept by the owner on 2026-11-13: a
# misread is for R4's threshold, not the orientation step.
CROP_DEPARTURES = {"20261113-185541-m3-crop-experiment": {
    "turn": True, "date": "2026-11-13",
    "why": "R1's two added wrong reads are one misread (FEB as FE8) in the right orientation; "
           "without turning no sideways crop is read"}}


def crop_recipe_used(decisions: dict) -> dict:
    """The crop recipe every M3 crop uses (S3's piles, baseline C): R1's
    result in m3-decisions.json, with the owner's recorded departures for
    that run applied. A departure made on another run doesn't carry over:
    if R1 is run again (a new parser, say), its result stands until the
    owner decides again."""
    recipe = decisions.get("crop_recipe")
    if recipe is None:
        raise RuntimeError("no crop recipe yet: S2's crop experiment writes it into m3-decisions.json")
    departure = CROP_DEPARTURES.get(recipe["run"], {})
    return {"pad": recipe["pad"], "turn": departure.get("turn", recipe["turn"]),
            "run": recipe["run"], "departures": sorted(k for k in departure if k in ("pad", "turn"))}


# --- S3, the crop shards ------------------------------------------------------------

# The loader's jitter (docs/m3-date-reader.md, "Augmentation"): each box
# edge moved by up to this share of the box's short side (its print height)
# round the recipe's padding. Crops are cut once with crops.MARGIN of the
# box's short side, so make_crops.py refuses to cut if the recipe's padding
# plus this won't fit.
EDGE_JITTER = 0.10


# --- S5, the training stream ----------------------------------------------------------

SEEDS = (0, 1, 2)
# Each sample picks a pile by weight, normalised over its arm's piles, then
# a record; the store pile picks a pack design first (docs/m3-date-reader.md, "The mix").
PILE_WEIGHT = {"expdate-real": 10.0, "expdate-synth": 20.0, "gen": 69.5, "store": 0.5}
ARMS = {"exp": ("expdate-real", "expdate-synth"),
        "exp+gen": ("expdate-real", "expdate-synth", "gen"),
        "all": ("expdate-real", "expdate-synth", "gen", "store")}
BUDGET = {"crnn": 10_000_000, "ppocrv6": 2_000_000}     # crops seen per run
EVALS = 20                     # checkpoints scored per run (every 5%), decision 17
CHECKPOINT_SUITES = ("real-val", "synth-val-v2")        # decision 17: pooled exact text
# S5: training crops (piles.VISIBILITY_FILTERED) whose visibility
# (reader/data.py) is under this are never drawn. Set by eye with the owner
# on 2026-11-16, before any training, on the sheets of job 73966 (commit
# a15ee90): gen-bands, expdate-synth-bands and expdate-real-bands, then
# scripts/visibility.py check --cut 3.0 (gen-dropped-3, gen-kept-above-3), all
# in $GENERATED_DIR/m3/crops/sheets/visibility/. Under 3, nearly every crop
# is empty (noise, flat, black); above it, nearly every one is readable.
# It drops gen 1,168 of 676,218 (0.17%), expdate-synth 28 of 16,652 (0.17%),
# expdate-real none. None would refuse to train.
VISIBLE_MIN: float | None = 3.0
