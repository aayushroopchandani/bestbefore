"""The text the generator prints: dates in every format, the words and codes
around them, and blocks of lines like the panel on a pack
(docs/m2-synthetic-generator.md, "Text").

Every random choice comes from the numpy Generator passed in; the shares
are spec.TextSpec's. The formats come from CLAUDE.md's list, ExpDate's own
shapes and general packaging practice, never from a store photo.

A format is written as the plan writes it and is its own template: YYYY,
YY, MM and DD are zero-padded numbers, DDth a day with its ordinal suffix
(never padded), Month a full month name and MON a three-letter one.
Everything else prints as it stands.
"""
from __future__ import annotations

import calendar
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .spec import TEXT, Shares, TextSpec

MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December")

# The plan's format table, by family. A range prints two month-only dates,
# mfg then expiry, joined by a dash: two date tokens, two boxes.
FORMATS = {
    "number-day": ("DD/MM/YYYY", "DD-MM-YYYY", "DD.MM.YYYY", "DD MM YYYY", "DD/MM/YY",
                   "DD.MM.YY", "DD-MM-YY", "YYYY.MM.DD", "YYYY/MM/DD", "YYYY-MM-DD",
                   "YY.MM.DD", "YYYYMMDD", "DDMMYYYY", "DDMMYY"),
    "number-month": ("MM/YYYY", "MM-YYYY", "MM.YYYY", "MM/YY", "YYYY.MM", "YYYY/MM"),
    "word-full": ("Month YYYY", "DD Month YYYY", "DDth Month YYYY", "Month DD, YYYY"),
    "word-abbr": ("MON YYYY", "MON-YYYY", "MON.YYYY", "MON/YYYY", "MON YY", "DD MON YYYY",
                  "DD-MON-YYYY", "DD/MON/YYYY", "DD.MON.YYYY", "DDMONYY", "DD MON YY",
                  "DDth MON YYYY", "MON DD YYYY", "YYYY.MON.DD", "YYYY MON DD"),
}
RANGES = ("MM/YYYY-MM/YYYY", "MM/YY-MM/YY")
DASH = "-"
FAMILY_OF = {f: family for family, formats in FORMATS.items() for f in formats}

_FIELD = re.compile(r"DDth|YYYY|YY|MM|DD|Month|MON")

# The words before a date (the plan's examples, as printed).
DUE = ("EXP", "EXP.", "EXP:", "EXP DATE", "EXPIRY DATE:", "Best Before", "BEST BEFORE:",
       "BB", "B.B.", "USE BY", "Use before")
PROD = ("MFG", "MFG:", "MFD", "Mfg. Date:", "MFG DT", "PKD", "PKD:", "Packed on:", "DOM",
        "Date of Mfg.")
# Lot and batch codes, each a word and a value: A is a capital letter, 9 a digit.
CODES = (("LOT ", "A99A9"), ("B.No. ", "9999A"), ("BATCH: ", "A9999"), ("L ", "99999"))
# Text with numbers that is not a date, and gets no box.
DISTRACTORS = ("MRP Rs. {price}.00", "(Incl. of all taxes)", "Net Wt. {kg} kg",
               "Net Qty {grams} g", "{hh}:{mm}", "1800 {ddd} {dddd}", "Lic. No. {licence}")
_FILL = {
    "price": lambda r: str(r.integers(20, 2501)),
    "kg": lambda r: f"{r.integers(1, 201) / 10:.1f}",
    "grams": lambda r: str(r.integers(5, 101) * 10),
    "hh": lambda r: f"{r.integers(24):02d}",
    "mm": lambda r: f"{r.integers(60):02d}",
    "ddd": lambda r: f"{r.integers(1000):03d}",
    "dddd": lambda r: f"{r.integers(10000):04d}",
    "licence": lambda r: str(r.integers(10**13, 10**14)),
}
# The relative block's second line: only the words get a box (due).
RELATIVE_DUE = "Best Before"
RELATIVE = "{months} months from Mfg."


@dataclass(frozen=True)
class Token:
    text: str
    cls: str | None                  # "date", "due", "prod", "code" or None (no box)
    role: str | None = None          # dates only: "expiry", "mfg" or "target"
    parts: tuple[int, int, int | None] | None = None   # dates only: year, month, day
    format: str | None = None        # dates only, e.g. "DDth Month YYYY"
    space_before: bool = True        # False: printed against the token before (a range)


@dataclass(frozen=True)
class Block:
    lines: tuple[tuple[Token, ...], ...]
    kind: str                # single, pair, panel, range, relative

    @property
    def tokens(self) -> tuple[Token, ...]:
        """Every token, line by line: the order of the sidecar and the label lines."""
        return tuple(t for line in self.lines for t in line)

    @property
    def text(self) -> str:
        return "\n".join(line_text(line) for line in self.lines)


def line_text(line: Sequence[Token]) -> str:
    return "".join((" " if i and t.space_before else "") + t.text for i, t in enumerate(line))


# --- Random choices -----------------------------------------------------------

def pick(rng: np.random.Generator, shares: Shares) -> str:
    """A name, drawn with its share (shares are scaled to sum to 1)."""
    u = rng.random() * sum(share for _, share in shares)
    for name, share in shares:
        u -= share
        if u < 0:
            return name
    return shares[-1][0]


def choice(rng: np.random.Generator, options: Sequence):
    """One of the options, each equally likely."""
    return options[int(rng.integers(len(options)))]


# --- Dates --------------------------------------------------------------------

def ordinal(day: int) -> str:
    """The suffix: 1st 2nd 3rd 4th, 11th 12th 13th, 21st 22nd 23rd, 31st."""
    if 11 <= day % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _cased(word: str, case: str) -> str:
    """'upper', 'title' (as written: December, Dec, th) or 'lower'."""
    return word.upper() if case == "upper" else word.lower() if case == "lower" else word


def format_date(d: date, fmt: str, case: str = "title") -> str:
    """The date printed in format `fmt`, month words and suffixes in `case`."""
    def field(m: re.Match) -> str:
        f = m.group()
        if f == "YYYY":
            return f"{d.year:04d}"
        if f == "YY":
            return f"{d.year % 100:02d}"
        if f == "MM":
            return f"{d.month:02d}"
        if f == "DD":
            return f"{d.day:02d}"
        if f == "DDth":
            return f"{d.day}{_cased(ordinal(d.day), case)}"
        name = MONTH_NAMES[d.month - 1]
        return _cased(name if f == "Month" else name[:3], case)
    return _FIELD.sub(field, fmt)


def prints_day(fmt: str) -> bool:
    return "DD" in fmt


def random_date(rng: np.random.Generator, spec: TextSpec = TEXT, last: date | None = None) -> date:
    """Any real day of the spec's years (up to `last`), all equally likely,
    so 29 February comes up in leap years and never in others."""
    first = date(spec.years[0], 1, 1)
    last = last or date(spec.years[1], 12, 31)
    return first + timedelta(days=int(rng.integers((last - first).days + 1)))


def add_months(d: date, months: int) -> date:
    """`months` later, the day clamped to the month's end (31 Jan + 1 = 28 or 29 Feb)."""
    year, month = divmod(d.month - 1 + months, 12)
    year, month = d.year + year, month + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def date_pair(rng: np.random.Generator, spec: TextSpec = TEXT) -> tuple[date, date]:
    """(mfg, expiry): the expiry 6 to 36 months after mfg, both in the
    spec's years, so mfg is drawn only up to where its expiry still fits."""
    lo, hi = spec.shelf_months
    months = int(rng.integers(lo, hi + 1))
    year, month = divmod(spec.years[1] * 12 + 11 - months, 12)
    last = date(year, month + 1, calendar.monthrange(year, month + 1)[1])
    mfg = random_date(rng, spec, last)
    return mfg, add_months(mfg, months)


def date_token(d: date, fmt: str, case: str, role: str, space_before: bool = True) -> Token:
    """A date as a token; a month-only format records no day."""
    return Token(format_date(d, fmt, case), "date", role,
                 (d.year, d.month, d.day if prints_day(fmt) else None), fmt, space_before)


# Month-word formats by (family, prints a day, ordinal): the last choice is uniform.
_WORD_CHOICES: dict[tuple[str, bool, bool], tuple[str, ...]] = {}
for _family in ("word-full", "word-abbr"):
    for _f in FORMATS[_family]:
        _WORD_CHOICES.setdefault((_family, prints_day(_f), "DDth" in _f), ())
        _WORD_CHOICES[_family, prints_day(_f), "DDth" in _f] += (_f,)


def random_format(rng: np.random.Generator, spec: TextSpec = TEXT) -> str:
    """Numbers or a month word, full name or abbreviation, a day or not, an
    ordinal or not, each by its share; then any format that fits."""
    if rng.random() < spec.month_word:
        family = "word-full" if rng.random() < spec.full_name else "word-abbr"
        day = bool(rng.random() < spec.day_words)
        return choice(rng, _WORD_CHOICES[family, day, day and bool(rng.random() < spec.ordinal)])
    return choice(rng, FORMATS["number-day" if rng.random() < spec.day_numbers else "number-month"])


# --- Other tokens -------------------------------------------------------------

def code(rng: np.random.Generator) -> str:
    word, pattern = choice(rng, CODES)
    return word + "".join(choice(rng, string.ascii_uppercase) if c == "A"
                          else str(rng.integers(10)) for c in pattern)


def distractor(rng: np.random.Generator) -> str:
    template = choice(rng, DISTRACTORS)
    fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    return template.format_map({name: _FILL[name](rng) for name in sorted(fields)})


# --- Blocks -------------------------------------------------------------------

def _labelled(rng, spec, fmt, case, mfg: date, expiry: date) -> list[tuple[Token, ...]]:
    """An mfg line and an expiry line, each with its word, mfg first by its share."""
    lines = [(Token(choice(rng, PROD), "prod"), date_token(mfg, fmt, case, "mfg")),
             (Token(choice(rng, DUE), "due"), date_token(expiry, fmt, case, "expiry"))]
    return lines if rng.random() < spec.mfg_first else lines[::-1]


def _single(rng, spec, fmt, case):
    line = (date_token(random_date(rng, spec), fmt, case, "expiry"),)
    if rng.random() < spec.single_word:
        line = (Token(choice(rng, DUE), "due"),) + line
    return [line]


def _pair(rng, spec, fmt, case):
    return _labelled(rng, spec, fmt, case, *date_pair(rng, spec))


def _panel(rng, spec, fmt, case):
    """A pair, plus codes and distractors in the same font, each line put
    anywhere among the others (the pair keeps its order)."""
    lines = _pair(rng, spec, fmt, case)
    lo, hi = spec.panel_lines
    for _ in range(int(rng.integers(lo, hi + 1))):
        token = (Token(code(rng), "code") if rng.random() < spec.panel_codes
                 else Token(distractor(rng), None))
        lines.insert(int(rng.integers(len(lines) + 1)), (token,))
    return lines


def _range(rng, spec, fmt, case):
    one = fmt.split(DASH)[0]                   # MM/YYYY-MM/YYYY prints MM/YYYY twice
    mfg, expiry = date_pair(rng, spec)
    line = (date_token(mfg, one, case, "mfg"), Token(DASH, None, space_before=False),
            date_token(expiry, one, case, "expiry", space_before=False))
    if rng.random() < spec.range_word:
        line = (Token(choice(rng, DUE), "due"),) + line
    return [line]


def _relative(rng, spec, fmt, case):
    """An mfg line and 'Best Before N months from Mfg.': one date only."""
    mfg = random_date(rng, spec)
    rule = (Token(RELATIVE_DUE, "due"),
            Token(RELATIVE.format(months=choice(rng, spec.relative_months)), None))
    lines = [(Token(choice(rng, PROD), "prod"), date_token(mfg, fmt, case, "mfg")), rule]
    return lines if rng.random() < spec.mfg_first else lines[::-1]


_KINDS = {"single": _single, "pair": _pair, "panel": _panel, "range": _range,
          "relative": _relative}


def random_block(rng: np.random.Generator, spec: TextSpec = TEXT) -> Block:
    """A block of text: its kind, its date format and the case of its month
    words are drawn once, so every date in it looks alike."""
    kind = pick(rng, spec.kinds)
    case = pick(rng, spec.case)
    fmt = choice(rng, RANGES) if kind == "range" else random_format(rng, spec)
    return Block(tuple(_KINDS[kind](rng, spec, fmt, case)), kind)


def _literal(template: str) -> str:
    """A template's fixed characters: its {fields} and format fields removed."""
    return _FIELD.sub("", re.sub(r"\{\w+\}", "", template))


# Every character the text can print: what each font must have (fonts.py).
CHARSET = frozenset(
    string.digits + string.ascii_uppercase + " "                    # numbers, codes' letters
    + "".join(n + n.upper() + n.lower() for n in MONTH_NAMES)
    + "stndrhSTNDRH"                                                # ordinal suffixes
    + "".join(_literal(f) for f in (*FAMILY_OF, *RANGES, *DISTRACTORS, RELATIVE))
    + "".join(DUE + PROD + (RELATIVE_DUE,) + tuple(w for w, _ in CODES)))
