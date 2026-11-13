"""M3's parser (reader/parse.py): dates.py's dates unchanged, plus what one
crop's read allows; role words; shelf life; adding months."""
from datetime import date, timedelta

import numpy as np
import pytest

from bestbefore import dates
from bestbefore.reader import parse
from bestbefore.samples import Expiry
from bestbefore.synth.text import FAMILY_OF, RANGES, format_date, prints_day


def one(text: str, whole_crop: bool = True) -> str | None:
    d = parse.crop_date(parse.read_dates(text, whole_crop))
    return str(d) if d else None


def all_dates(text: str, whole_crop: bool = True) -> list[str]:
    return [str(f.date) for f in parse.read_dates(text, whole_crop)]


@pytest.mark.parametrize("text", [
    "2021.08.15", "15/08/2027", "AUG 2027", "21.08.15", "21/08/15", "MFG 2025.03.01 EXP 2027.03.01",
    "120/08/2027", "31/02/2027", "08/2027", "2027.08", "Aug. 15. 2027", "15 AUG 27", "20270815"])
def test_dates_py_reads_unchanged(text):
    """Where nothing added applies, the race is dates.find_dates exactly."""
    assert parse._race(dates._matches(text)) == dates.find_dates(text)
    assert all_dates(text, whole_crop=False) == [str(f.date) for f in dates.find_dates(text)]


@pytest.mark.parametrize("text, want", [
    ("19th December 2026", "2026-12-19"), ("1st Jan 2027", "2027-01-01"),
    ("23RD NOV 2026", "2026-11-23"), ("9thMarch 2027", "2027-03-09"), ("2nd feb 2028", "2028-02-02"),
    ("March 09, 2027", "2027-03-09"), ("NOVEMBER 23, 2026", "2026-11-23"),
    ("09MAR27", "2027-03-09"), ("9MAR2027", "2027-03-09"), ("09-Mar-27", "2027-03-09"),
])
def test_added_everywhere(text, want):
    assert one(text) == want
    assert one(text, whole_crop=False) == want


def test_found_text_is_as_read():
    (f,) = parse.read_dates("EXP 19th December 2026")
    assert f.text == "19th December 2026" and (f.start, f.end) == (4, 22)


@pytest.mark.parametrize("text, want", [
    ("04122026", "2026-12-04"), ("EXP 04122026", "2026-12-04"),
    ("20261120", "2026-11-20"),          # YYYYMMDD makes a date: it stands
    ("20112026", "2026-11-20"),          # YYYYMMDD gives month 20: DDMMYYYY
    ("31022027", None), ("123456789", None),
])
def test_ddmmyyyy_in_a_crop(text, want):
    assert one(text) == want


def test_ddmmyyyy_not_in_free_text():
    assert all_dates("04122026", whole_crop=False) == []


@pytest.mark.parametrize("text, want", [
    ("08/27", ["2027-08"]), ("8/27", ["2027-08"]), ("03.27", ["2027-03"]), ("03-27", ["2027-03"]),
    ("MAR 27", ["2027-03"]), ("Mar-27", ["2027-03"]), ("mar27", ["2027-03"]),
    ("EXP: 03.27", ["2027-03"]), ("Mfg. Date: 01/25", ["2025-01"]), ("BEST BEFORE 08/27.", ["2027-08"]),
    ("Packed on: 11/26", ["2026-11"]),
    ("03/25-03/27", ["2025-03", "2027-03"]), ("03/25 - 11/26", ["2025-03", "2026-11"]),
    ("090327", ["2027-03-09"]),
    ("13/27", []), ("123456", []), ("21.08", []),
])
def test_whole_read(text, want):
    assert all_dates(text) == want


@pytest.mark.parametrize("text", ["08/27", "MAR 27", "03/25-03/27", "090327"])
def test_whole_read_forms_need_a_crop(text):
    assert all_dates(text, whole_crop=False) == []


@pytest.mark.parametrize("text", ["EXP 08/27 LOT 5", "08/27 Rs. 120", "B.No. 12345", "MRP Rs. 120.00", "", "   "])
def test_not_a_whole_date(text):
    assert all_dates(text) == []


def test_two_part_expdate_text_is_read_as_month_and_year():
    """Known: ExpDate's month-and-day texts read as month and year in a crop."""
    assert one("02.18") == "2018-02"


def test_every_generator_format():
    """gen-v2's formats (synth/text.py), in every case, read to the date
    printed, but for DD.MM.YY where year first also makes a date
    (dates.py's rule, kept): there the year-first date."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        d = date(2023, 1, 1) + timedelta(days=int(rng.integers(0, 365 * 9)))
        for fmt in FAMILY_OF:
            for case in ("title", "upper", "lower"):
                text = format_date(d, fmt, case)
                want = Expiry(d.year, d.month, d.day if prints_day(fmt) else None)
                got = parse.crop_date(parse.read_dates(text))
                if fmt == "DD.MM.YY" and got != want:
                    yy, mm, dd = text.split(".")
                    assert got == Expiry(2000 + int(yy), int(mm), int(dd))
                else:
                    assert got == want, (fmt, text)
        for fmt in RANGES:
            first, second = fmt.split("-")
            mfg = date(d.year - 2, d.month, 1)
            text = f"{format_date(mfg, first)}-{format_date(d, second)}"
            assert all_dates(text) == [f"{mfg.year}-{mfg.month:02d}", f"{d.year}-{d.month:02d}"], text


@pytest.mark.parametrize("fmt, text, want", [
    # store set v1-train's and v3-train's expiry formats, written out here.
    ("DD/MM/YYYY", "04/12/2026", "2026-12-04"), ("MON YYYY", "APR 2031", "2031-04"),
    ("Month YYYY", "April 2031", "2031-04"), ("DDMMYYYY", "18052026", "2026-05-18"),
    ("DD/MM/YY", "04/12/26", "2026-12-04"), ("YYYYMMDD", "20260518", "2026-05-18"),
    ("DD-MM-YYYY", "04-12-2026", "2026-12-04"), ("DD Month YYYY", "4 December 2026", "2026-12-04"),
    ("YYYY.MON.DD", "2028.JAN.21", "2028-01-21"), ("MON-YYYY", "JAN-2028", "2028-01"),
    ("MM/YYYY", "10/2027", "2027-10"),
])
def test_store_training_formats(fmt, text, want):
    assert one(text) == want


def test_crop_date_longest_then_leftmost():
    assert one("5/2025 15/08/2027") == "2027-08-15"
    assert all_dates("05/2025 05/2027") == ["2025-05", "2027-05"]
    assert one("05/2025 05/2027") == "2025-05"
    assert parse.crop_date([]) is None


@pytest.mark.parametrize("text, role", [
    ("EXP: 03/27", "expiry"), ("EXP.FEB 2027", "expiry"), ("Expiry Date", "expiry"), ("Best Before", "expiry"),
    ("B.B. 03/27", "expiry"), ("BB 03/27", "expiry"), ("Use by 03/27", "expiry"), ("USE BEFORE", "expiry"),
    ("MFG DT 01/25", "mfg"), ("Mfd. 01/25", "mfg"), ("MANUFACTURED ON", "mfg"), ("PKD: 01/25", "mfg"),
    ("Packed on: 01/25", "mfg"), ("DOM 01/25", "mfg"), ("Imp. 03/2025", "mfg"), ("Date of Import:", "mfg"),
    ("MFG 01/25 EXP 01/27", None), ("EXPORT QUALITY", None), ("03/27", None), ("", None),
])
def test_roles_in(text, role):
    assert parse.roles_in(text) == role


@pytest.mark.parametrize("text, months", [
    ("Best Before 24 months from Mfg.", 24), ("1 Year from Mfg. Dt.", 12),
    ("18 MONTHS FROM MANUFACTURE", 18), ("BEST BEFORE 6 MONTHS FROM THE DATE OF MANUFACTURE", 6),
    ("2 Years from Mfg", 24), ("24 months", None), ("MRP 24.00", None), ("200 months from mfg", None), ("", None),
])
def test_shelf_life(text, months):
    assert parse.shelf_life(text) == months


@pytest.mark.parametrize("start, months, want", [
    (Expiry(2025, 1, 10), 24, Expiry(2027, 1, 10)), (Expiry(2024, 12), 18, Expiry(2026, 6)),
    (Expiry(2027, 1, 31), 1, Expiry(2027, 2, 28)), (Expiry(2028, 1, 31), 1, Expiry(2028, 2, 29)),
    (Expiry(2025, 11, 30), 3, Expiry(2026, 2, 28)),
])
def test_add_months(start, months, want):
    assert parse.add_months(start, months) == want
