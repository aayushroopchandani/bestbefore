"""M3's parser: the dates in one crop's read, the role words inside it, and
a shelf life (docs/m3-date-reader.md, "The parser").

Built on dates.py, which stays frozen as baseline A's: its patterns are used
as they are, in their order, with its rules for ties. Written in M3 S2 from
training data only: the shapes of ExpDate's real-train dates (without
real-val) and Products-Synth's, the expiry formats of store sets v1-train
and v3-train, and gen-v2's format list (synth/text.py: FORMATS, RANGES,
the words before a date, the relative block). Frozen with PARSE_SHA256 in
m3.py before S2's job ran; any change is a new version, recorded and
re-scored.

What it reads that dates.find_dates doesn't:

- an ordinal day, '19th December 2026', '1st Jan 2027' (gen-v2's DDth
  formats): dates.py reads 'December 2026' and drops the day;
- a comma after the day, 'March 09, 2027' (gen-v2's Month DD, YYYY);
- a day glued to its month word, '09MAR27', '9MAR2027' (gen-v2's DDMONYY);

and, in the read of one crop (whole_crop=True), what a box of one date
allows and free OCR text doesn't:

- DDMMYYYY with no separators, '04122026' (v1-train, gen-v2), where
  YYYYMMDD makes no date (both never can: a YYYYMMDD year starts 20, and a
  DDMMYYYY month can't be 20 to 29);
- when the read is one date and nothing else (a role word before it
  allowed): a month and a 2-digit year, '08/27', 'MAR 27' (gen-v2's MM/YY,
  MON YY), two of them as a range, '03/25-03/27' (gen-v2's MM/YY-MM/YY),
  and six digits as DDMMYY, '090327' (gen-v2's), day first as the store's
  packs print. In a box the detector calls a date this is the natural
  reading, where in free text dates.py rightly skips it.

Kept from dates.py, known: three 2-digit parts joined by dots are year first
('23.11.26' is 2023-11-26), as ExpDate prints them, though gen-v2's
DD.MM.YY prints that shape day first; where both make a date, year first
wins. And 26 of real-train's 1,057 dates without real-val are two parts
with no year ('02.18', '01.31': month and day); in a crop the whole-read
rule reads them as month and year (2018-02).

`dates._matches` and `dates._make` are dates.py's own private helpers, used
so its patterns race against the ones added here by one rule.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import replace

from .. import dates
from ..dates import MON, N, S, Y2, Y4, Found
from ..samples import Expiry

# --- Normalising, length kept so positions still point into the read -----------

_MONTH_START = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
# '19th', '1st', '23RD', glued or not to the month word that follows.
_ORDINAL = re.compile(rf"(?<!\d)(\d{{1,2}})(st|nd|rd|th)(?=[^a-z]|$|{_MONTH_START})", re.I)
# The comma in 'March 09, 2027'.
_COMMA = re.compile(rf"{MON}\s*{N}\s*,(?=\s*20\d{{2}}(?!\d))", re.I)


def _normalised(text: str) -> str:
    text = _ORDINAL.sub(lambda m: m.group(1) + " " * len(m.group(2)), text)
    return _COMMA.sub(lambda m: m.group(0)[:-1] + " ", text)


# --- Patterns added to dates.py's, raced against them ------------------------------

_SEP = r"[\s./\-]*"           # glued, or any separators
_ALWAYS = (
    ("DDMONYYYY", rf"{N}{_SEP}{MON}{_SEP}{Y4}", ("dmy",)),
    ("DDMONYY", rf"{N}{_SEP}{MON}{_SEP}{Y2}", ("dmy",)),
)
_CROP = (
    ("DDMMYYYY", r"(\d{2})(\d{2})(20\d{2})", ("dmy",)),
)


def _compile(patterns):
    return tuple((name, re.compile(rf"(?<!\d){rx}(?!\d)", re.I), orders)
                 for name, rx, orders in patterns)


_ALWAYS_RX, _CROP_RX = _compile(_ALWAYS), _compile(_CROP)


def _candidates(text: str, extra):
    """dates.py's matches with their priority, then the added ones after them."""
    yield from dates._matches(text)
    base = len(dates._PATTERNS)
    for k, (name, rx, orders) in enumerate(extra):
        for m in rx.finditer(text):
            date = next(filter(None, (dates._make(m.groups(), o) for o in orders)), None)
            yield base + k, Found(date, m.group(0), name, m.start(), m.end())


def _race(candidates) -> list[Found]:
    """dates.find_dates' rule: where matches overlap, the longest wins, then
    the earlier pattern, and a match that isn't a real date still takes its
    place. One thing added: of two matches over the same characters, one
    that makes a date beats one that doesn't ('20112026' is DDMMYYYY, as
    YYYYMMDD gives month 20). Among dates.py's own patterns two such
    matches always read their parts alike, so its results don't change."""
    taken: list[Found] = []
    for _, f in sorted(candidates, key=lambda c: (c[1].start - c[1].end, c[1].date is None, c[0], c[1].start)):
        if all(f.end <= t.start or f.start >= t.end for t in taken):
            taken.append(f)
    return sorted((f for f in taken if f.date), key=lambda f: f.start)


# --- Role words ---------------------------------------------------------------------
#
# From gen-v2's DUE, PROD and IMPORT words and ExpDate's due and prod marks.
# A letter may not touch a word on either side (EXPORT is no EXP), but a
# date may (EXP.FEB 2027, MFG:01/25).

_DUE = (r"EXP(?:IRY|IRES|IRE|D)?", r"BEST\s*BEFORE", r"B\.?\s?B\.?", r"USE\s*(?:BY|BEFORE)")
_PROD = (r"MFG", r"MFD", r"MANUF[A-Z]*", r"PKD", r"PACKED", r"PACKING", r"DOM",
         r"IMP(?:ORT(?:ED)?)?")


def _words(words) -> re.Pattern:
    return re.compile(rf"(?<![a-z])(?:{'|'.join(words)})(?![a-z])", re.I)


_DUE_RX, _PROD_RX = _words(_DUE), _words(_PROD)
# What may stand before the one date of a whole read: role words, DATE, DT,
# ON, OF, each with a full stop, colon or dash.
_LEAD = re.compile(rf"(?:(?:{'|'.join((*_DUE, *_PROD, 'DATE', 'DT', 'ON', 'OF'))})(?![a-z])\s*[.:\-]?\s*)+", re.I)
_ENDS = " \t.:;,-_|"


def roles_in(text: str) -> str | None:
    """'expiry' if the read holds EXP, BB, BEST BEFORE, USE BY or USE
    BEFORE; 'mfg' if it holds MFG, MFD, MANUF..., PKD, PACKED, PACKING, DOM
    or IMPORT (an import date comes before the expiry too); None if it holds
    neither, or both (it can't tell which is the date's)."""
    due, prod = bool(_DUE_RX.search(text or "")), bool(_PROD_RX.search(text or ""))
    return "expiry" if due and not prod else "mfg" if prod and not due else None


# --- Whole reads -------------------------------------------------------------------

_MY = rf"(\d{{1,2}}){S}(\d{{2}})"
_WHOLE = _compile((("MM.YY", _MY, ("my",)),
                   ("MON.YY", rf"{MON}{_SEP}(\d{{2}})", ("my",)),
                   ("DDMMYY", r"(\d{2})(\d{2})(\d{2})", ("dmy",))))
_RANGE = re.compile(rf"{_MY}\s*-\s*{_MY}", re.I)


def _bare(text: str) -> tuple[str, int]:
    """The read without a role word before it and punctuation at its ends,
    and where that starts in the read."""
    start = len(text) - len(text.lstrip(_ENDS))
    lead = _LEAD.match(text, start)
    if lead:
        start = lead.end()
    rest = text[start:].strip(_ENDS)
    return rest, text.index(rest, start) if rest else start


def _whole(text: str) -> list[Found] | None:
    """The whole read's one date (or range) in the forms only a crop allows;
    None if the read isn't one of those forms."""
    bare, at = _bare(text)
    if not bare:
        return None
    if m := _RANGE.fullmatch(bare):
        found = [Found(dates._make(m.groups()[i:i + 2], "my"), m.group(0), "MM.YY-MM.YY",
                       at + m.start(1 + i), at + m.end(2 + i)) for i in (0, 2)]
        return [replace(f, text=text[f.start:f.end]) for f in found if f.date]
    for name, rx, orders in _WHOLE:
        if m := rx.fullmatch(bare):
            date = next(filter(None, (dates._make(m.groups(), o) for o in orders)), None)
            return [Found(date, bare, name, at, at + len(bare))] if date else []
    return None


# --- The parser ----------------------------------------------------------------------

def read_dates(text: str, whole_crop: bool = True) -> list[Found]:
    """Every date in one read, left to right, each with its text as read
    and its place in it. whole_crop: the read is one crop's (a box the
    detector, or a person, called a date), which allows DDMMYYYY anywhere
    in it, and the whole-read forms (see the module's docstring)."""
    if not text or not text.strip():
        return []
    if whole_crop and (found := _whole(text)) is not None:
        return found
    found = _race(_candidates(_normalised(text), _ALWAYS_RX + (_CROP_RX if whole_crop else ())))
    return [replace(f, text=text[f.start:f.end]) for f in found]


def crop_date(found: list[Found]) -> Expiry | None:
    """The one date a crop's read gives: the longest match (padding can
    reach a neighbouring date, cut short), then the leftmost."""
    return min(found, key=lambda f: (f.start - f.end, f.start)).date if found else None


# --- Shelf life ---------------------------------------------------------------------

_LIFE = re.compile(r"(?<!\d)(\d{1,3})\s*(months?|mths?|mnths?|years?|yrs?)\.?\s*(?:from|after|of)\s*"
                   r"(?:the\s*)?(?:date\s*of\s*)?(?:mfg|mfd|manuf[a-z]*|pkd|packing|packaging|production)",
                   re.I)


def shelf_life(text: str) -> int | None:
    """The months of a shelf life: '24 months from Mfg.' 24, '1 Year from
    Mfg. Dt.' 12, '18 MONTHS FROM MANUFACTURE' 18. None for anything else,
    a number with no 'from (the date of) manufacture' after it included,
    and for more than 10 years."""
    m = _LIFE.search(text or "")
    if not m:
        return None
    months = int(m.group(1)) * (12 if m.group(2)[0] in "yY" else 1)
    return months if 1 <= months <= 120 else None


def add_months(d: Expiry, months: int) -> Expiry:
    """`months` after `d`, its day kept when printed, clipped to the
    month's end (31 Jan + 1 month = 28 or 29 Feb); a month-only date stays
    month-only ('2024-12' + 18 = '2026-06')."""
    year, month = divmod(d.month - 1 + months, 12)
    year, month = d.year + year, month + 1
    day = None if d.day is None else min(d.day, calendar.monthrange(year, month)[1])
    return Expiry(year, month, day)
