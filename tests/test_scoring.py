import pytest

from bestbefore.samples import Expiry
from bestbefore.scoring import NONE, RIGHT, WRONG, score, summarise, tally


@pytest.mark.parametrize("truth, read, expected", [
    (Expiry(2027, 8, 15), Expiry(2027, 8, 15), RIGHT),
    (Expiry(2027, 8, 15), Expiry(2027, 8, 16), WRONG),      # one day off
    (Expiry(2027, 8, 15), Expiry(2027, 8), WRONG),          # day printed but not read
    (Expiry(2027, 8, 15), Expiry(2021, 8, 15), WRONG),      # the batch looks expired
    (Expiry(2027, 8), Expiry(2027, 8, 31), RIGHT),          # Scanify's last-day answer
    (Expiry(2027, 8), Expiry(2027, 8), RIGHT),
    (Expiry(2027, 8), Expiry(2025, 8), WRONG),              # e.g. the mfg date
    (Expiry(2027, 8), None, NONE),
    (Expiry(2028, 2, 29), Expiry(2028, 2, 29), RIGHT),
])
def test_score(truth, read, expected):
    assert score(truth, read) == expected


def test_tally():
    assert tally([RIGHT, RIGHT, NONE]) == {"n": 3, "right": 2, "wrong": 0, "none": 1}


def test_summarise():
    rows = [
        {"tag": "clean", "score": RIGHT, "read": "2027-08"},
        {"tag": "clean", "score": WRONG, "read": "2025-08"},
        {"tag": "sideways", "score": NONE, "read": None},
        {"tag": "other-face", "score": None, "read": "2025-01"},   # nothing to score
        {"tag": None, "score": None, "read": None},                # ExpDate: no expiry marked
    ]
    s = summarise(rows)
    assert s["all"] == {"n": 3, "right": 1, "wrong": 1, "none": 1}
    assert s["by_tag"]["clean"] == {"n": 2, "right": 1, "wrong": 1, "none": 0}
    assert s["not_scored"] == {"other-face": {"n": 1, "date_read": 1},
                               "no expiry marked": {"n": 1, "date_read": 0}}
