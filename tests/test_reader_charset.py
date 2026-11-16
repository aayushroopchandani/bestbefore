"""M3's character set (reader/charset.py, decision 10)."""
import pytest

from bestbefore.reader import charset


def test_48_classes_blank_first_unknown_last():
    assert len(charset.CLASSES) == 48
    assert charset.CLASSES[charset.BLANK] == "" and charset.BLANK == 0
    assert charset.CLASSES[-1] == "?"
    assert len(charset.CHARACTERS) == 46 == len(set(charset.CHARACTERS))
    assert set(charset.CHARACTERS) == set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ./-:,()&|")


@pytest.mark.parametrize("text, folded", [
    ("Jan 2027", "JAN 2027"),
    ("  exp:\t04/12/2026  ", "EXP: 04/12/2026"),       # runs of white space become one space
    ("Best Before 1 Year from Mfg. Dt.", "BEST BEFORE 1 YEAR FROM MFG. DT."),
    ("Rs. 5.69/ml", "RS. 5.69/ML"),
    ("08–2027", "08?2027"),                             # an en dash is outside the set
    ("é", "?"), ("ß", "?"),                              # one '?' each, never upper-cased to 'SS'
    ("#12*", "?12?"),
    ("", ""),
])
def test_fold(text, folded):
    assert charset.fold(text) == folded


def test_encode_and_decode():
    ids = charset.encode("Exp 08/2027")
    assert charset.BLANK not in ids
    assert charset.decode(ids) == "EXP 08/2027"
    assert charset.encode("é") == [len(charset.CLASSES) - 1]


def test_index():
    assert charset.index("0") == 1
    assert charset.index("a") == charset.index("A")
    with pytest.raises(KeyError):
        charset.index("")
