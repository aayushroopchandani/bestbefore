"""One scoring rule for every reader (docs/m0-data-and-test-set.md, C1).

A reading is `right`, `wrong` or `none`, and wrong is kept apart from none:
reading 08/2027 as 08/2021 marks a whole batch as expired, while no date
just sends a person to look.
"""
from __future__ import annotations

from collections import Counter

from .samples import Expiry

RIGHT, WRONG, NONE = "right", "wrong", "none"
OUTCOMES = (RIGHT, WRONG, NONE)


def score(truth: Expiry, read: Expiry | None) -> str:
    """Right needs the year, month and day; only year and month when the
    pack prints no day, so a reader answering with the month's last day
    (as Scanify does) still scores. Any other date, the mfg date included,
    is wrong."""
    if read is None:
        return NONE
    if (read.year, read.month) != (truth.year, truth.month):
        return WRONG
    return RIGHT if truth.day is None or read.day == truth.day else WRONG


def tally(outcomes: list[str]) -> dict[str, int]:
    """{'n': 87, 'right': 31, 'wrong': 4, 'none': 52}."""
    counts = Counter(outcomes)
    return {"n": len(outcomes), **{o: counts[o] for o in OUTCOMES}}


def summarise(rows: list[dict]) -> dict:
    """Counts over per-photo rows holding `tag`, `score` and `read`.

    A row with no score had nothing to score against (no expiry marked, or
    a store photo that doesn't show one). Those are counted apart, by tag,
    with how many still got a date read: a date read from a photo with no
    expiry in it can only be the wrong one.
    """
    scored = [r for r in rows if r["score"]]
    tags = dict.fromkeys(r["tag"] for r in scored if r["tag"])
    not_scored: dict[str, dict[str, int]] = {}
    for r in rows:
        if not r["score"]:
            group = not_scored.setdefault(r["tag"] or "no expiry marked", {"n": 0, "date_read": 0})
            group["n"] += 1
            group["date_read"] += r["read"] is not None
    return {"all": tally([r["score"] for r in scored]),
            "by_tag": {t: tally([r["score"] for r in scored if r["tag"] == t]) for t in tags},
            "not_scored": not_scored}
