"""S2's crop experiment, scored (docs/m3-date-reader.md, S2;
experiments/m3-crop-reading/README.md): each crop's read in each
configuration through the parser, against its truth.

A configuration is a reader (rapidocr, ppocrv6), what runs (full: its own
pipeline; alone: its recogniser on the crop as cut; turned: its recogniser
read both ways, decision 14) and a padding. Pure functions over the cut
step's crop records and the read step's reads, so the job, the
experiment's summarise.py and the tests all score alike.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .. import m3
from ..samples import Expiry
from ..scoring import OUTCOMES, score
from ..store import parse_date, TAGS
from . import crops, parse

# The parser's patterns that only a crop's whole read allows (parse._whole).
WHOLE_READ = ("MM.YY", "MON.YY", "DDMMYY", "MM.YY-MM.YY")


def fold(text: str) -> str:
    """Case folded, runs of spaces folded to one (decision 10)."""
    return " ".join((text or "").split()).upper()


def config_name(reader: str, mode: str, pad: float) -> str:
    return f"{reader}/{mode}/{pad:g}"


def truth_of(crop: dict) -> tuple[Expiry | None, bool]:
    """The crop's true date, and whether it took the parser's whole-read
    rule to get it. A store crop's is its frozen value; a real-val crop's
    is the parser's one date in its transcription, None if it gives none."""
    if crop["set"] == "store":
        return parse_date(crop["truth"]), False
    found = parse.read_dates(crop["truth_text"])
    date = parse.crop_date(found)
    longest = min(found, key=lambda f: (f.start - f.end, f.start)) if found else None
    return date, bool(longest and longest.pattern in WHOLE_READ)


def holds_date(read: dict) -> bool:
    return bool(parse.read_dates(read["text"]))


def choose_turn(reads: list[dict]) -> dict:
    """Decision 14: of a crop's two reads, the one that holds a date; if
    both or neither do, the more confident; a tie, the first."""
    def key(r: dict) -> tuple:
        return (holds_date(r), r["score"] if r["score"] is not None else -1.0)
    return max(reads, key=key)       # max keeps the first of equals


def the_read(crop: dict, reads: dict, reader: str, mode: str, pad: float) -> dict:
    """The read a configuration scores, from the crop's reads keyed by
    (reader, full or alone, pad, quarter turn); a turned one carries its
    two reads' milliseconds summed and the turn it chose."""
    if mode == "full":
        return reads[(reader, "full", pad, 0)]
    if mode == "alone":
        return reads[(reader, "alone", pad, 0)]
    both = [reads[(reader, "alone", pad, k)] for k in crops.turns(crop["shape"] == "tall")]
    chosen = choose_turn(both)
    return {**chosen, "ms": sum(r["ms"] for r in both)}


def score_rows(crop_records: list[dict], reads: Iterable[dict]) -> list[dict]:
    """One row per crop and configuration: the read, the parser's dates, the
    outcome against the truth (None when the truth gives no date) and, for
    real-val, whether the text is exact."""
    by_crop: dict[str, dict] = {}
    for r in reads:
        by_crop.setdefault(r["key"], {})[(r["reader"], r["mode"], r["pad"], r["k"])] = r
    rows = []
    for c in crop_records:
        truth, whole = truth_of(c)
        for reader in m3.CROP_READERS:
            for mode in m3.CROP_MODES:
                for pad in m3.PADS:
                    read = the_read(c, by_crop[c["key"]], reader, mode, pad)
                    found = parse.read_dates(read["text"])
                    date = parse.crop_date(found)
                    rows.append({
                        "key": c["key"], "set": c["set"], "part": c["part"], "tag": c.get("tag"),
                        "shape": c["shape"], "label": c["label"], "photo": c["photo"],
                        "reader": reader, "mode": mode, "pad": pad, "config": config_name(reader, mode, pad),
                        "text": read["text"], "score": read["score"], "k": read["k"], "ms": read["ms"],
                        "dates": len(found), "date": str(date) if date else None,
                        "truth": str(truth) if truth else None, "truth_whole_read": whole,
                        "outcome": score(truth, date) if truth else None,
                        "exact": fold(read["text"]) == fold(c["truth_text"]) if c.get("truth_text") else None})
    return rows


def group_of(row: dict) -> list[str]:
    """The groups a row counts in: its set and part, and for store crops its
    tag, shape and label."""
    if row["set"] != "store":
        return [row["set"]]
    return ["store", f"store-{row['part']}", f"tag:{row['tag']}", f"shape:{row['shape']}",
            f"label:{row['label']}"]


def tally(rows: Iterable[dict]) -> dict[str, dict[str, dict]]:
    """config -> group -> {n, right, wrong, none, no_truth, exact, multi}:
    `none` holds reads with no date; `no_truth` crops whose truth gives no
    date (left out of right, wrong and none); `exact` real-val's exact
    texts; `multi` reads holding more than one date."""
    out: dict[str, dict[str, Counter]] = {}
    for r in rows:
        for g in group_of(r):
            c = out.setdefault(r["config"], {}).setdefault(g, Counter())
            c["n"] += 1
            if r["outcome"] is None:
                c["no_truth"] += 1
            else:
                c[r["outcome"]] += 1
            c["exact"] += bool(r["exact"])
            c["multi"] += r["dates"] > 1
    keys = ("n", *OUTCOMES, "no_truth", "exact", "multi")
    return {cfg: {g: {k: c[k] for k in keys} for g, c in groups.items()} for cfg, groups in out.items()}


def mean_ms(rows: Iterable[dict]) -> dict[str, float]:
    """Milliseconds a crop, by configuration (a turned one: both reads)."""
    total: dict[str, list[float]] = {}
    for r in rows:
        total.setdefault(r["config"], []).append(r["ms"])
    return {cfg: round(sum(v) / len(v), 1) for cfg, v in total.items()}


def group_order(groups: Iterable[str]) -> list[str]:
    """Groups in report order: sets, then tags in store.TAGS order, shapes, labels."""
    order = ["real-val", "store", "store-train", "store-val",
             *(f"tag:{t}" for t in TAGS), "shape:wide", "shape:tall", "label:expiry", "label:mfg"]
    groups = set(groups)
    return [g for g in order if g in groups] + sorted(groups - set(order))
