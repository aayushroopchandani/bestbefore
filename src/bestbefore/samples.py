"""The one shape every dataset is loaded into.

ExpDate and the store photos both become lists of `Sample`, so the contact
sheet, the stats and the baselines are written once and run on either.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

# Box labels that mark a printed date, in either dataset.
DATE_LABELS = {"date", "exp",                   # ExpDate
               "expiry", "mfg", "other_date"}   # store photos


@dataclass(frozen=True)
class Box:
    label: str                                  # e.g. date, exp, due, prod, code
    xyxy: tuple[float, float, float, float]     # 0..1 of the upright image
    text: str | None = None                     # the transcription, if any


@dataclass(frozen=True)
class Expiry:
    """An expiry date at the level of detail the pack prints."""
    year: int
    month: int
    day: int | None = None      # None: the pack prints no day

    def __post_init__(self):
        if not 1 <= self.month <= 12:
            raise ValueError(f"month {self.month}")
        if self.day is not None:
            last = calendar.monthrange(self.year, self.month)[1]
            if not 1 <= self.day <= last:
                raise ValueError(f"day {self.day} in {self.year}-{self.month:02d}")

    def __str__(self) -> str:
        ym = f"{self.year:04d}-{self.month:02d}"
        return ym if self.day is None else f"{ym}-{self.day:02d}"


@dataclass(frozen=True)
class Sample:
    id: str                     # unique within its set, e.g. "test_00001.jpg"
    image: Path
    width: int                  # upright, as load_upright returns it
    height: int
    boxes: tuple[Box, ...]
    expiry: Expiry | None = None    # None: nothing to score a reader against
    tag: str | None = None          # failure tag (store photos only)


def parse_month(text: str) -> int | None:
    """'08' -> 8, 'AUG' or 'August' -> 8, anything else -> None."""
    text = text.strip().upper()
    if text.isdigit():
        month = int(text)
        return month if 1 <= month <= 12 else None
    return MONTHS.index(text[:3]) + 1 if text[:3] in MONTHS else None


def parse_year(text: str) -> int | None:
    """'2027' -> 2027, '27' -> 2027."""
    text = text.strip()
    if not text.isdigit() or len(text) not in (2, 4):
        return None
    return int(text) + 2000 if len(text) == 2 else int(text)
