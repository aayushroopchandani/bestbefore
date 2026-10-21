import pytest

from bestbefore.dates import find_dates, latest
from bestbefore.samples import Expiry


def one(text: str) -> Expiry | None:
    found = find_dates(text)
    assert len(found) <= 1, found
    return found[0].date if found else None


@pytest.mark.parametrize("text, expected", [
    # The formats in the plan and CLAUDE.md
    ("15/08/2027", Expiry(2027, 8, 15)),
    ("15.08.2027", Expiry(2027, 8, 15)),
    ("15-08-2027", Expiry(2027, 8, 15)),
    ("15 08 2027", Expiry(2027, 8, 15)),
    ("04/12/2026", Expiry(2026, 12, 4)),            # day before month
    ("08/25/2027", Expiry(2027, 8, 25)),            # only month-first is a date
    ("15/08/27", Expiry(2027, 8, 15)),
    ("10/2027", Expiry(2027, 10)),
    ("20260518", Expiry(2026, 5, 18)),
    ("2027.08.15", Expiry(2027, 8, 15)),
    ("2027 / 08 / 15", Expiry(2027, 8, 15)),
    ("2028.JAN.21", Expiry(2028, 1, 21)),
    ("21 Aug 2027", Expiry(2027, 8, 21)),
    ("21 AUG 27", Expiry(2027, 8, 21)),
    ("April 2031", Expiry(2031, 4)),
    ("APR.2031", Expiry(2031, 4)),
    ("Sept 15 2027", Expiry(2027, 9, 15)),
    ("2027.08", Expiry(2027, 8)),
    # Three 2-digit parts: dots read year first, anything else day first
    ("21.08.15", Expiry(2021, 8, 15)),
    ("21/08/15", Expiry(2015, 8, 21)),
    ("31.08.21", Expiry(2031, 8, 21)),              # year first, even where day first works
    ("21.08.03", Expiry(2021, 8, 3)),
    # Leap years
    ("29/02/2028", Expiry(2028, 2, 29)),
    ("29/02/2027", None),                           # not 02/2027: the failed date blocks it
    # Not dates
    ("08/27", None),                                # two parts, no 4-digit year
    ("Rs. 12.50", None),
    ("15/08/20271", None),                          # a digit touching the end
    ("15/08/1999", None),                           # outside YEARS
    ("MARKET 2027", None),                          # MAR inside a word
    ("8801007166421", None),                        # a barcode number
])
def test_formats(text, expected):
    assert one(text) == expected


def test_surrounding_text():
    assert one("EXP:15/08/2027 L23") == Expiry(2027, 8, 15)
    assert one("：2022.06.20") == Expiry(2022, 6, 20)        # a full-width colon


def test_longest_match_wins():
    [f] = find_dates("15/08/2027")
    assert f.pattern == "DD.MM.YYYY"                    # not 08/2027 as MM.YYYY


def test_a_range_gives_two_dates():
    found = find_dates("05/2025-05/2027")
    assert [f.date for f in found] == [Expiry(2025, 5), Expiry(2027, 5)]


def test_latest():
    found = find_dates("MFG 21/05/2025 EXP 20/05/2027")
    assert latest(found) == Expiry(2027, 5, 20)
    assert latest([]) is None
    # A month-only date counts as its month's end.
    assert latest(find_dates("15/05/2027 05/2027")) == Expiry(2027, 5)
