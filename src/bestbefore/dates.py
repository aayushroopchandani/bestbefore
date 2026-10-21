"""Find printed dates in OCR text: the simple regex behind baseline A
(docs/m0-data-and-test-set.md, C1).

Written from two sources only, and frozen before any test set was read:
the date shapes in ExpDate's real-train transcriptions, and the format list
in CLAUDE.md. Never tuned on a test split or the store photos.

Each pattern lists the orders its parts may be read in, most likely first;
the first order that makes a real date in YEARS wins. Where the order is
ambiguous:
  - day before month (04/12/2026 is 4 December), as Indian packs print it;
  - three 2-digit parts joined by dots are year first (21.08.15 is
    2021-08-15): 137 of the 147 such dates in real-train start with 19 to 22;
  - with any other separator they're day first (21/08/15 is 2015-08-21).
Two-part dates with no 4-digit year (08/27) are skipped: they're as likely
a price, a weight or a day and month.

A found date is an `Expiry` only in the sense of its type: year, month and
a day if one is printed. Whether it's the expiry is up to the reader.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .samples import Expiry, parse_month, parse_year

# Plausible years on a pack in this project: real-train's 4-digit years run
# from 2018 to 2025, and an expiry can be ten years out.
YEARS = range(2015, 2040)

_MONTHS = ("jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?"
           "|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
N = r"(\d{1,2})"                            # a day or a month number
N2 = r"(\d{2})"
Y4 = r"(20\d{2})"
Y2 = r"(\d{2})"
MON = rf"(?<![a-z])({_MONTHS})(?![a-z])"
S = r"(?:\s*[./\-]\s*|\s+)"                 # . / - or spaces between parts

# (name, regex, orders to try). Earlier patterns win ties (see find_dates).
_PATTERNS = (
    ("YYYY.MM.DD", rf"{Y4}{S}{N}{S}{N}", ("ymd",)),
    ("YYYY.MON.DD", rf"{Y4}{S}{MON}{S}{N}", ("ymd",)),
    ("YYYYMMDD", r"(20\d{2})(\d{2})(\d{2})", ("ymd",)),
    ("DD.MM.YYYY", rf"{N}{S}{N}{S}{Y4}", ("dmy", "mdy")),
    ("DD.MON.YYYY", rf"{N}{S}{MON}{S}{Y4}", ("dmy",)),
    ("MON.DD.YYYY", rf"{MON}{S}{N}{S}{Y4}", ("mdy",)),
    ("YY.MM.DD", rf"{N2}\s*\.\s*{N2}\s*\.\s*{N2}", ("ymd", "dmy", "mdy")),
    ("DD.MM.YY", rf"{N}{S}{N}{S}{Y2}", ("dmy", "ymd", "mdy")),
    ("DD.MON.YY", rf"{N}{S}{MON}{S}{Y2}", ("dmy",)),
    ("MON.DD.YY", rf"{MON}{S}{N}{S}{Y2}", ("mdy",)),
    ("MM.YYYY", rf"{N}{S}{Y4}", ("my",)),
    ("MON.YYYY", rf"{MON}{S}?{Y4}", ("my",)),
    ("YYYY.MM", rf"{Y4}{S}{N2}", ("ym",)),
)
# No digit may touch a date's ends: 120/08/2027 holds no date.
_COMPILED = tuple((name, re.compile(rf"(?<!\d){rx}(?!\d)", re.IGNORECASE), orders)
                  for name, rx, orders in _PATTERNS)


@dataclass(frozen=True)
class Found:
    date: Expiry | None     # None only inside find_dates: shaped like a date, but isn't one
    text: str               # the matched text, e.g. "15/08/2027"
    pattern: str        # e.g. "DD.MM.YYYY"
    start: int
    end: int


def _make(parts: tuple[str, ...], order: str) -> Expiry | None:
    """The parts read in this order ('dmy', 'my', ...), if they make a real
    date in YEARS."""
    fields = dict(zip(order, parts))
    year = parse_year(fields["y"])
    month = parse_month(fields["m"])
    if year not in YEARS or month is None:
        return None
    try:
        return Expiry(year, month, int(fields["d"]) if "d" in fields else None)
    except ValueError:
        return None


def _matches(text: str):
    """Every match of every pattern, with its date, or None when no order
    makes a real date."""
    for priority, (name, rx, orders) in enumerate(_COMPILED):
        for m in rx.finditer(text):
            date = next(filter(None, (_make(m.groups(), o) for o in orders)), None)
            yield priority, Found(date, m.group(0), name, m.start(), m.end())


def find_dates(text: str) -> list[Found]:
    """Every date in the text, left to right. Where matches overlap, the
    longest wins (15/08/2027 over 08/2027), then the earlier pattern. A
    match that isn't a real date still takes its place, so 31/02/2027
    gives nothing rather than 02/2027."""
    taken: list[Found] = []
    for _, f in sorted(_matches(text), key=lambda c: (c[1].start - c[1].end, c[0], c[1].start)):
        if all(f.end <= t.start or f.start >= t.end for t in taken):
            taken.append(f)
    return sorted((f for f in taken if f.date), key=lambda f: f.start)


def latest(found: list[Found]) -> Expiry | None:
    """The latest date. A month-only date counts as its month's end."""
    def key(f: Found) -> tuple[int, int, int]:
        return f.date.year, f.date.month, f.date.day or 31
    return max(found, key=key).date if found else None
