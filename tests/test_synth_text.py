"""The generator's text (docs/m2-synthetic-generator.md, "Text"): every format
prints a known date as written and reads back with a pattern built from the
format's name alone; ordinals; month-only formats drop the day; only real
days; a pair's expiry 6 to 36 months after its mfg date; each block kind's
tokens; and the sampled shares within 3 points of the spec."""
import calendar
import random
import re
from collections import Counter
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from bestbefore.synth import text as T
from bestbefore.synth.spec import TEXT

KNOWN = date(2026, 12, 4)
# Every format in the plan's table, printing 4 December 2026, in title case.
EXPECTED = {
    "DD/MM/YYYY": "04/12/2026", "DD-MM-YYYY": "04-12-2026", "DD.MM.YYYY": "04.12.2026",
    "DD MM YYYY": "04 12 2026", "DD/MM/YY": "04/12/26", "DD.MM.YY": "04.12.26",
    "DD-MM-YY": "04-12-26", "YYYY.MM.DD": "2026.12.04", "YYYY/MM/DD": "2026/12/04",
    "YYYY-MM-DD": "2026-12-04", "YY.MM.DD": "26.12.04", "YYYYMMDD": "20261204",
    "DDMMYYYY": "04122026", "DDMMYY": "041226",
    "MM/YYYY": "12/2026", "MM-YYYY": "12-2026", "MM.YYYY": "12.2026", "MM/YY": "12/26",
    "YYYY.MM": "2026.12", "YYYY/MM": "2026/12",
    "Month YYYY": "December 2026", "DD Month YYYY": "04 December 2026",
    "DDth Month YYYY": "4th December 2026", "Month DD, YYYY": "December 04, 2026",
    "MON YYYY": "Dec 2026", "MON-YYYY": "Dec-2026", "MON.YYYY": "Dec.2026",
    "MON/YYYY": "Dec/2026", "MON YY": "Dec 26", "DD MON YYYY": "04 Dec 2026",
    "DD-MON-YYYY": "04-Dec-2026", "DD/MON/YYYY": "04/Dec/2026", "DD.MON.YYYY": "04.Dec.2026",
    "DDMONYY": "04Dec26", "DD MON YY": "04 Dec 26", "DDth MON YYYY": "4th Dec 2026",
    "MON DD YYYY": "Dec 04 2026", "YYYY.MON.DD": "2026.Dec.04", "YYYY MON DD": "2026 Dec 04",
}
MONTHS = {name.lower(): i + 1 for i, name in enumerate(T.MONTH_NAMES)}
MONTHS.update({name[:3].lower(): i + 1 for i, name in enumerate(T.MONTH_NAMES)})

# A format's parts as regular expressions, from its name alone.
_PART = {"DDth": r"(?P<day>\d{1,2})(?P<suffix>st|nd|rd|th)", "YYYY": r"(?P<year>\d{4})",
         "YY": r"(?P<yy>\d{2})", "MM": r"(?P<month>\d{2})", "DD": r"(?P<day>\d{2})",
         "Month": r"(?P<name>[A-Za-z]{3,9})", "MON": r"(?P<name>[A-Za-z]{3})"}


def pattern(fmt: str) -> re.Pattern:
    out, end = "", 0
    for m in re.finditer("|".join(_PART), fmt):
        out += re.escape(fmt[end:m.start()]) + _PART[m.group()]
        end = m.end()
    return re.compile(out + re.escape(fmt[end:]), re.IGNORECASE)


def read_back(fmt: str, printed: str) -> tuple[int, int, int | None]:
    m = pattern(fmt).fullmatch(printed)
    assert m, f"{printed!r} does not match {fmt}"
    g = m.groupdict()
    year = int(g["year"]) if g.get("year") else 2000 + int(g["yy"])
    month = int(g["month"]) if g.get("month") else MONTHS[g["name"].lower()]
    day = int(g["day"]) if g.get("day") else None
    if g.get("suffix"):
        assert g["suffix"].lower() == T.ordinal(day)
    return year, month, day


# --- Formats -------------------------------------------------------------------

def test_the_table_has_every_format_of_the_plan():
    assert set(T.FAMILY_OF) == set(EXPECTED)
    assert [len(T.FORMATS[f]) for f in T.FORMATS] == [14, 6, 4, 15]
    assert T.RANGES == ("MM/YYYY-MM/YYYY", "MM/YY-MM/YY")


@pytest.mark.parametrize("fmt", EXPECTED)
def test_every_format_prints_a_known_date(fmt):
    assert T.format_date(KNOWN, fmt) == EXPECTED[fmt]
    assert read_back(fmt, EXPECTED[fmt]) == (2026, 12, 4 if T.prints_day(fmt) else None)


def test_month_words_and_suffixes_follow_the_case():
    assert T.format_date(KNOWN, "DDth Month YYYY", "upper") == "4TH DECEMBER 2026"
    assert T.format_date(KNOWN, "DDth Month YYYY", "lower") == "4th december 2026"
    assert T.format_date(date(2031, 5, 22), "DDMONYY", "upper") == "22MAY31"
    assert T.format_date(date(2031, 9, 1), "Month DD, YYYY", "title") == "September 01, 2031"


def test_every_format_reads_back_in_every_case():
    rng = np.random.default_rng(0)
    for _ in range(300):
        d = T.random_date(rng)
        for fmt in T.FAMILY_OF:
            for case in ("upper", "title", "lower"):
                assert read_back(fmt, T.format_date(d, fmt, case)) == (
                    d.year, d.month, d.day if T.prints_day(fmt) else None)


@pytest.mark.parametrize("day, suffix", [
    (1, "st"), (2, "nd"), (3, "rd"), (4, "th"), (11, "th"), (12, "th"), (13, "th"),
    (21, "st"), (22, "nd"), (23, "rd"), (24, "th"), (30, "th"), (31, "st")])
def test_ordinal_suffixes(day, suffix):
    assert T.ordinal(day) == suffix
    assert T.format_date(date(2027, 1, day), "DDth MON YYYY") == f"{day}{suffix} Jan 2027"


def test_month_only_formats_drop_the_day():
    d = date(2029, 3, 17)
    for fmt in T.FAMILY_OF:
        token = T.date_token(d, fmt, "upper", "expiry")
        assert token.parts == (2029, 3, 17 if T.prints_day(fmt) else None)
        assert ("17" in token.text) == T.prints_day(fmt)


# --- Dates ---------------------------------------------------------------------

def test_dates_are_real_days_and_29_february_only_in_leap_years():
    rng = np.random.default_rng(1)
    days = [T.random_date(rng) for _ in range(60_000)]
    assert min(days) >= date(2018, 1, 1) and max(days) <= date(2035, 12, 31)
    leap_days = {d.year for d in days if (d.month, d.day) == (2, 29)}
    assert leap_days and all(calendar.isleap(y) for y in leap_days)
    # Every day equally likely: each year's share follows its length.
    years = Counter(d.year for d in days)
    assert max(years.values()) / min(years.values()) < 1.15


@pytest.mark.parametrize("start, months, end", [
    (date(2027, 1, 31), 1, date(2027, 2, 28)), (date(2028, 1, 31), 1, date(2028, 2, 29)),
    (date(2026, 3, 31), 1, date(2026, 4, 30)), (date(2026, 8, 31), 6, date(2027, 2, 28)),
    (date(2025, 11, 15), 36, date(2028, 11, 15)), (date(2025, 12, 1), 13, date(2027, 1, 1))])
def test_add_months_clamps_to_the_months_last_day(start, months, end):
    assert T.add_months(start, months) == end


def test_a_pairs_expiry_is_6_to_36_months_after_mfg():
    rng = np.random.default_rng(2)
    gaps = Counter()
    for _ in range(20_000):
        mfg, expiry = T.date_pair(rng)
        gap = (expiry.year - mfg.year) * 12 + expiry.month - mfg.month
        gaps[gap] += 1
        assert 6 <= gap <= 36
        assert expiry.day == min(mfg.day, calendar.monthrange(expiry.year, expiry.month)[1])
        assert date(2018, 1, 1) <= mfg < expiry <= date(2035, 12, 31)
    assert set(gaps) == set(range(6, 37))


# --- Blocks --------------------------------------------------------------------

def _blocks(n: int, seed: int = 0, spec=TEXT) -> list[T.Block]:
    rng = np.random.default_rng(seed)
    return [T.random_block(rng, spec) for _ in range(n)]


def _dates(block: T.Block) -> list[T.Token]:
    return [t for t in block.tokens if t.cls == "date"]


def _months_apart(a: T.Token, b: T.Token) -> int:
    return (b.parts[0] - a.parts[0]) * 12 + b.parts[1] - a.parts[1]


def test_every_kind_of_block_has_the_right_tokens():
    for block in _blocks(3000, seed=3):
        dates = _dates(block)
        for t in dates:
            assert read_back(t.format, t.text)[:2] == t.parts[:2]
        if block.kind == "single":
            assert [t.role for t in dates] == ["expiry"]
            assert len(block.lines) == 1 and block.tokens[-1] is dates[0]
        elif block.kind in ("pair", "panel"):
            mfg, expiry = sorted(dates, key=lambda t: t.role != "mfg")
            assert (mfg.role, expiry.role) == ("mfg", "expiry") and mfg.format == expiry.format
            assert 6 <= _months_apart(mfg, expiry) <= 36
            for line in block.lines:
                if any(t.cls == "date" for t in line):    # each date after its own word
                    assert [t.cls for t in line] == ["prod" if line[1] is mfg else "due", "date"]
            extra = len(block.lines) - 2
            assert extra == 0 if block.kind == "pair" else 1 <= extra <= 3
        elif block.kind == "range":
            (line,) = block.lines
            mfg, dash, expiry = line[-3:]
            assert (mfg.role, dash.cls, expiry.role) == ("mfg", None, "expiry")
            assert not dash.space_before and not expiry.space_before
            assert re.fullmatch(r"\d\d/\d{2,4}-\d\d/\d{2,4}", T.line_text(line[-3:]))
            assert 6 <= _months_apart(mfg, expiry) <= 36 and mfg.parts[2] is None
        else:                                              # relative: one date only
            assert [t.role for t in dates] == ["mfg"]
            (rule,) = [line for line in block.lines if line[0].text == T.RELATIVE_DUE]
            assert [t.cls for t in rule] == ["due", None]
            months = int(rule[1].text.split()[0])
            assert months in TEXT.relative_months


def test_every_printed_character_is_in_the_charset():
    printed = set("".join(b.text for b in _blocks(10_000, seed=4))) - {"\n"}
    assert printed <= T.CHARSET
    assert set("0123456789/-.:, ") <= printed      # the charset isn't vacuous


def test_a_time_is_a_code_box_and_a_real_time():
    """Times are `code`, as ExpDate's photos box them (decided in S4)."""
    tokens = [t for b in _blocks(10_000, seed=6) for t in b.tokens]
    times = [t for t in tokens if re.fullmatch(r"\d\d:\d\d", t.text)]
    assert times and all(t.cls == "code" for t in times)
    assert all(int(t.text[:2]) < 24 and int(t.text[3:]) < 60 for t in times)
    assert not any(re.search(r"\d:\d", t.text) for t in tokens if t.cls is None)


def _case(token: T.Token) -> str:
    letters = "".join(c for c in token.text if c.isalpha())
    return "upper" if letters.isupper() else "lower" if letters.islower() else "title"


def test_sampled_shares_are_within_3_points_of_the_spec():
    blocks = _blocks(10_000)
    shares = {}

    def share(name: str, hits: list[bool]) -> None:
        shares[name] = sum(hits) / len(hits)

    kinds = Counter(b.kind for b in blocks)
    for kind, want in TEXT.kinds:
        shares[f"kind {kind}"] = kinds[kind] / len(blocks)
    firsts = [_dates(b)[0] for b in blocks if b.kind != "range"]
    word = [T.FAMILY_OF[t.format].startswith("word") for t in firsts]
    share("month word", word)
    words = [t for t, w in zip(firsts, word) if w]
    numbers = [t for t, w in zip(firsts, word) if not w]
    share("full name", [T.FAMILY_OF[t.format] == "word-full" for t in words])
    share("day, numbers", [t.parts[2] is not None for t in numbers])
    share("day, words", [t.parts[2] is not None for t in words])
    share("ordinal", ["DDth" in t.format for t in words if t.parts[2] is not None])
    for case, _ in TEXT.case:
        share(f"case {case}", [_case(t) == case for t in words])
    share("single word", [b.tokens[0].cls == "due" for b in blocks if b.kind == "single"])
    share("mfg first", [next(t.role for t in b.tokens if t.role) == "mfg"
                        for b in blocks if b.kind in ("pair", "panel", "relative")])
    extra = [len(b.lines) - 2 for b in blocks if b.kind == "panel"]
    for n in (1, 2, 3):
        share(f"panel +{n}", [e == n for e in extra])
    share("panel codes", [line[0].cls == "code" for b in blocks if b.kind == "panel"
                          for line in b.lines if not any(t.cls == "date" for t in line)])

    want = {**{f"kind {k}": v for k, v in TEXT.kinds}, "month word": TEXT.month_word,
            "full name": TEXT.full_name, "day, numbers": TEXT.day_numbers,
            "day, words": TEXT.day_words, "ordinal": TEXT.ordinal,
            **{f"case {c}": v for c, v in TEXT.case}, "single word": TEXT.single_word,
            "mfg first": TEXT.mfg_first, "panel +1": 1 / 3, "panel +2": 1 / 3,
            "panel +3": 1 / 3, "panel codes": TEXT.panel_codes}
    assert set(shares) == set(want)
    off = {k: (round(shares[k], 3), want[k]) for k in want if abs(shares[k] - want[k]) > 0.03}
    assert not off, off


def test_range_shares_over_10000_ranges():
    ranges = _blocks(10_000, seed=5, spec=replace(TEXT, kinds=(("range", 1.0),)))
    with_word = sum(b.tokens[0].cls == "due" for b in ranges) / len(ranges)
    two_digit = sum(len(_dates(b)[0].text) == 5 for b in ranges) / len(ranges)
    assert abs(with_word - TEXT.range_word) <= 0.03 and abs(two_digit - 0.5) <= 0.03


def test_the_same_seed_gives_the_same_blocks_whatever_the_global_state():
    random.seed(1)
    np.random.seed(1)
    first = _blocks(200, seed=7)
    random.seed(2)
    np.random.seed(2)
    assert _blocks(200, seed=7) == first
    assert _blocks(200, seed=8) != first
