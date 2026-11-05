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

gen-v2's text (docs/m2b-store-and-size.md, "The generator, v2", S7) comes
with a V2Text: a date glued to its word (`EXP.FEB 2027`), two word-and-date
pairs on one line (`line-pair`), dense panels, import dates and their words,
barcode numbers and unit prices. Without one, every draw is M2's, in M2's
order. Its words and examples are made up, as M2's are.
"""
from __future__ import annotations

import calendar
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np

from .spec import TEXT, V2, Shares, TextSpec, V2Text

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
# Lot and batch codes, and the time a coding printer prints beside a date:
# each gets a `code` box, as ExpDate's photos box them.
CODES = ("LOT {lot}", "B.No. {batch}", "BATCH: {batch_no}", "L {lot_no}", "{hh}:{mm}")
# Text with numbers that is not a date, and gets no box.
DISTRACTORS = ("MRP Rs. {price}.00", "(Incl. of all taxes)", "Net Wt. {kg} kg",
               "Net Qty {grams} g", "1800 {ddd} {dddd}", "Lic. No. {licence}")
# How each {field} is drawn; in a pattern, A is a capital letter and 9 a digit.
_FILL = {
    "lot": lambda r: _pattern(r, "A99A9"),
    "batch": lambda r: _pattern(r, "9999A"),
    "batch_no": lambda r: _pattern(r, "A9999"),
    "lot_no": lambda r: _pattern(r, "99999"),
    "price": lambda r: str(r.integers(20, 2501)),
    "kg": lambda r: f"{r.integers(1, 201) / 10:.1f}",
    "grams": lambda r: str(r.integers(5, 101) * 10),
    "hh": lambda r: f"{r.integers(24):02d}",
    "mm": lambda r: f"{r.integers(60):02d}",
    "ddd": lambda r: f"{r.integers(1000):03d}",
    "dddd": lambda r: f"{r.integers(10000):04d}",
    "licence": lambda r: str(r.integers(10**13, 10**14)),
    # gen-v2's (S7):
    "ean13": lambda r: ean13(r),
    "ean13_spaced": lambda r: _spaced(ean13(r)),
    "usp": lambda r: f"{r.integers(10, 1000) / 100:.2f}",
    "usp_short": lambda r: f"{r.integers(1, 100) / 10:.1f}",
    "unit": lambda r: choice(r, UNITS),
}
# The relative block's second line: only the words get a box (due).
RELATIVE_DUE = "Best Before"
RELATIVE = "{months} months from Mfg."

# gen-v2's (S7). Its two new kinds; the import words, boxed prod like the
# manufacture words, their date's role "import"; and the store's commonest
# false boxes as distractors: 13-digit barcode numbers, run together or
# spaced 1-6-6, and unit prices, money in V2Text.currency.
V2_KINDS = ("dense-panel", "line-pair")
IMPORT = ("IMPORT:", "Imp.", "Date of Import:", "Imported on:", "Month & Year of Import:")
DISTRACTORS_V2 = ("{ean13}", "{ean13_spaced}", "USP {currency} {usp}/{unit}",
                  "Unit Sale Price: {currency} {usp_short} per {unit}")
UNITS = ("g", "ml")
GLUE = (":", ".")           # a date is glued only to a word that ends in one of these
_GLUABLE = {words: tuple(w for w in words if w.endswith(GLUE)) for words in (DUE, PROD, IMPORT)}


@dataclass(frozen=True)
class Token:
    text: str
    cls: str | None                  # "date", "due", "prod", "code" or None (no box)
    role: str | None = None          # dates only: "expiry", "mfg" or "target"; gen-v2's "import"
    parts: tuple[int, int, int | None] | None = None   # dates only: year, month, day
    format: str | None = None        # dates only, e.g. "DDth Month YYYY"
    space_before: bool = True        # False: printed against the token before (a range; gen-v2's glued date)


@dataclass(frozen=True)
class Block:
    lines: tuple[tuple[Token, ...], ...]
    kind: str                # single, pair, panel, range, relative; gen-v2's dense-panel, line-pair

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

def _pattern(rng: np.random.Generator, pattern: str) -> str:
    return "".join(choice(rng, string.ascii_uppercase) if c == "A" else str(rng.integers(10))
                   for c in pattern)


def _filled(rng: np.random.Generator, template: str) -> str:
    """The template with each of its {fields} drawn."""
    fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    return template.format_map({name: _FILL[name](rng) for name in sorted(fields)})


def code(rng: np.random.Generator) -> str:
    return _filled(rng, choice(rng, CODES))


def distractor(rng: np.random.Generator, v2: V2Text | None = None) -> str:
    """Text with numbers that is no date: M2's templates or, in gen-v2's
    text, M2's and the new ones, every template equally likely."""
    if v2 is None:
        return _filled(rng, choice(rng, DISTRACTORS))
    return _filled(rng, choice(rng, DISTRACTORS + DISTRACTORS_V2).replace("{currency}", v2.currency))


def ean13(rng: np.random.Generator) -> str:
    """A barcode number: 12 random digits and their EAN-13 check digit,
    which makes the digits' sum, weighted 1, 3, 1, 3 ... from the left, a
    multiple of 10."""
    digits = [int(d) for d in rng.integers(0, 10, 12)]
    check = -sum(d * (3 if i % 2 else 1) for i, d in enumerate(digits)) % 10
    return "".join(map(str, digits)) + str(check)


def _spaced(number: str) -> str:
    """A barcode number spaced 1-6-6, as printed under the bars."""
    return f"{number[0]} {number[1:7]} {number[7:]}"


# --- Blocks -------------------------------------------------------------------

def _word(rng, words: tuple[str, ...], v2: V2Text | None) -> tuple[str, bool]:
    """A word to print before a date, and whether the date is glued to it.
    Only gen-v2's text glues: v2.glued of its dates, each to a word that
    ends in ':' or '.'. M2's text draws the word alone, as it always did."""
    if v2 is None:
        return choice(rng, words), False
    glued = bool(rng.random() < v2.glued)
    return choice(rng, _GLUABLE[words] if glued else words), glued


def _labelled_line(rng, words, cls, d: date, fmt, case, role, v2) -> tuple[Token, Token]:
    """A word and its date: `EXP: 11/2027` or, glued, `EXP:11/2027`."""
    word, glued = _word(rng, words, v2)
    return Token(word, cls), date_token(d, fmt, case, role, space_before=not glued)


def _labelled(rng, spec, fmt, case, mfg: date, expiry: date, v2=None) -> list[tuple[Token, ...]]:
    """An mfg line and an expiry line, each with its word, mfg first by its share."""
    lines = [_labelled_line(rng, PROD, "prod", mfg, fmt, case, "mfg", v2),
             _labelled_line(rng, DUE, "due", expiry, fmt, case, "expiry", v2)]
    return lines if rng.random() < spec.mfg_first else lines[::-1]


def _single(rng, spec, fmt, case, v2=None):
    d = random_date(rng, spec)
    if rng.random() < spec.single_word:
        return [_labelled_line(rng, DUE, "due", d, fmt, case, spec.single_role, v2)]
    return [(date_token(d, fmt, case, spec.single_role),)]


def _pair(rng, spec, fmt, case, v2=None):
    return _labelled(rng, spec, fmt, case, *date_pair(rng, spec), v2=v2)


def _extra(rng, spec, v2) -> Token:
    """One of a panel's extra lines: a code or a distractor, by spec.panel_codes."""
    if rng.random() < spec.panel_codes:
        return Token(code(rng), "code")
    return Token(distractor(rng, v2), None)


def _panel(rng, spec, fmt, case, v2=None):
    """A pair, plus codes and distractors in the same font, each line put
    anywhere among the others (the pair keeps its order)."""
    lines = _pair(rng, spec, fmt, case, v2)
    lo, hi = spec.panel_lines
    for _ in range(int(rng.integers(lo, hi + 1))):
        token = _extra(rng, spec, v2)                    # drawn before its place, as M2 drew it
        lines.insert(int(rng.integers(len(lines) + 1)), (token,))
    return lines


def _range(rng, spec, fmt, case, v2=None):
    one = fmt.split(DASH)[0]                   # MM/YYYY-MM/YYYY prints MM/YYYY twice
    mfg, expiry = date_pair(rng, spec)
    rest = (Token(DASH, None, space_before=False), date_token(expiry, one, case, "expiry", space_before=False))
    if rng.random() < spec.range_word:
        return [_labelled_line(rng, DUE, "due", mfg, one, case, "mfg", v2) + rest]
    return [(date_token(mfg, one, case, "mfg"),) + rest]


def _relative(rng, spec, fmt, case, v2=None):
    """An mfg line and 'Best Before N months from Mfg.': one date only."""
    mfg = random_date(rng, spec)
    rule = (Token(RELATIVE_DUE, "due"),
            Token(RELATIVE.format(months=choice(rng, spec.relative_months)), None))
    lines = [_labelled_line(rng, PROD, "prod", mfg, fmt, case, "mfg", v2), rule]
    return lines if rng.random() < spec.mfg_first else lines[::-1]


def import_date(rng: np.random.Generator, mfg: date, expiry: date, v2: V2Text) -> date:
    """An import date, v2.import_months whole months after mfg (its day
    kept, as add_months keeps it) and in a month before the expiry's."""
    shelf = (expiry.year - mfg.year) * 12 + expiry.month - mfg.month
    lo, hi = v2.import_months
    return add_months(mfg, int(rng.integers(lo, min(hi, shelf - 1) + 1)))


def _line_pair(rng, spec, fmt, case, v2):
    """Two word-and-date pairs on one line, `MFG:FEB 2025 | IMPORT: MAR
    2025`: the mfg date and, with share v2.pair_import, an import date,
    else the expiry; the mfg pair first by spec.mfg_first, as a pair's
    lines are. Between them a separator, printed as written: its mark a
    token with no box (an empty one for two spaces), its spaces those
    around it."""
    mfg, expiry = date_pair(rng, spec)
    first = _labelled_line(rng, PROD, "prod", mfg, fmt, case, "mfg", v2)
    if rng.random() < v2.pair_import:
        second = _labelled_line(rng, IMPORT, "prod", import_date(rng, mfg, expiry, v2), fmt, case, "import", v2)
    else:
        second = _labelled_line(rng, DUE, "due", expiry, fmt, case, "expiry", v2)
    if rng.random() >= spec.mfg_first:
        first, second = second, first
    sep = choice(rng, v2.separators)
    word, d = second
    mark = Token(sep.strip(), None, space_before=sep.startswith(" "))
    return [first + (mark, replace(word, space_before=sep.endswith(" ")), d)]


def _dense_panel(rng, spec, fmt, case, v2):
    """A pair plus v2.dense_lines lines, each put anywhere among the others
    (the pair keeps its order): with share v2.dense_import one of them is an
    import line, the rest codes and distractors as a panel's. Printed close
    together (V2Style.line_spacing)."""
    mfg, expiry = date_pair(rng, spec)
    lines = _labelled(rng, spec, fmt, case, mfg, expiry, v2)
    lo, hi = v2.dense_lines
    count = int(rng.integers(lo, hi + 1))
    extra = []
    if rng.random() < v2.dense_import:
        extra.append(_labelled_line(rng, IMPORT, "prod", import_date(rng, mfg, expiry, v2), fmt, case, "import", v2))
    while len(extra) < count:
        extra.append((_extra(rng, spec, v2),))
    for line in extra:
        lines.insert(int(rng.integers(len(lines) + 1)), line)
    return lines


_KINDS = {"single": _single, "pair": _pair, "panel": _panel, "range": _range,
          "relative": _relative, "dense-panel": _dense_panel, "line-pair": _line_pair}


def random_block(rng: np.random.Generator, spec: TextSpec = TEXT, v2: V2Text | None = None) -> Block:
    """A block of text: its kind, its date format and the case of its month
    words are drawn once, so every date in it looks alike. With `v2`
    (gen-v2's text) dates glue to their words, distractors include barcode
    numbers and unit prices, and spec.kinds may hold the new kinds; without
    it every draw is M2's."""
    if v2 is None and (new := [name for name, _ in spec.kinds if name in V2_KINDS]):
        raise ValueError(f"{', '.join(new)}: gen-v2's kinds need its text (v2)")
    kind = pick(rng, spec.kinds)
    case = pick(rng, spec.case)
    fmt = choice(rng, RANGES) if kind == "range" else random_format(rng, spec)
    return Block(tuple(_KINDS[kind](rng, spec, fmt, case, v2)), kind)


def _literal(template: str) -> str:
    """A template's fixed characters: its {fields} and format fields removed."""
    return _FIELD.sub("", re.sub(r"\{\w+\}", "", template))


# Every character the text can print, M2's and gen-v2's: what each font
# must have (fonts.py). gen-v2's money is V2's currency.
CHARSET = frozenset(
    string.digits + string.ascii_uppercase + " "                    # numbers, codes' letters
    + "".join(n + n.upper() + n.lower() for n in MONTH_NAMES)
    + "stndrhSTNDRH"                                                # ordinal suffixes
    + "".join(_literal(f) for f in (*FAMILY_OF, *RANGES, *CODES, *DISTRACTORS, RELATIVE))
    + "".join(DUE + PROD + (RELATIVE_DUE,))
    + "".join(_literal(t.replace("{currency}", V2.text.currency)) for t in DISTRACTORS_V2)
    + "".join(IMPORT + UNITS + V2.text.separators))
