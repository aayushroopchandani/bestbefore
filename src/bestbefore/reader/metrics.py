"""How a read's text is scored against its truth (docs/m3-date-reader.md,
"Shared"), written against tests/test_reader_metrics.py. Both texts are
folded first (charset.fold: case, runs of spaces, characters outside the
set), for every reader alike.

Date-level scores (right, wrong, none) are scoring.score on the parser's
dates, in S10's scoring.
"""
from __future__ import annotations

from .charset import fold


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: the fewest single-character insertions,
    deletions and substitutions that turn `a` into `b`. Dynamic
    programming, one row at a time: row[j] is the distance between the
    part of `a` read so far and b[:j]."""
    row = list(range(len(b) + 1))                     # "" -> b[:j]: j insertions
    for i, ca in enumerate(a, start=1):
        diagonal, row[0] = row[0], i                  # a[:i] -> "": i deletions
        for j, cb in enumerate(b, start=1):
            diagonal, row[j] = row[j], min(
                row[j] + 1,                           # delete ca
                row[j - 1] + 1,                       # insert cb
                diagonal + (ca != cb),                # substitute ca by cb (free if equal)
            )
    return row[-1]


def cer(read: str, truth: str) -> float:
    """Character error rate: edits from the read to the truth, over the
    truth's length. An empty read scores 1; a read with extra characters
    can score above 1."""
    truth = fold(truth)
    if not truth:
        raise ValueError("no character error rate without a truth")
    return edit_distance(fold(read), truth) / len(truth)


def exact(read: str, truth: str) -> bool:
    """The whole text right, case and spaces folded."""
    return fold(read) == fold(truth)
