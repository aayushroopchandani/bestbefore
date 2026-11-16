"""M3's one character set (docs/m3-date-reader.md, decision 10).

Digits, the 26 letters, space and `. / - : , ( ) & |`: 46 characters. The
CRNN reads them as 48 classes: class 0 is CTC's blank ("no character in
this frame"), then the 46, then `?` for any character outside them. Case is
folded (`Jan` and `JAN` read alike), so the letters are upper case.

`fold` is how every score compares texts, for every reader (PP-OCRv6 keeps
its own dictionary and case, and is folded the same way when scored).
"""
from __future__ import annotations

BLANK = 0
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ./-:,()&|"
UNKNOWN = "?"
CLASSES = ("",) + tuple(CHARACTERS) + (UNKNOWN,)      # 48; CLASSES[BLANK] is the blank
_INDEX = {c: i for i, c in enumerate(CLASSES) if i != BLANK}


def _fold_char(c: str) -> str:
    """One character in the set: upper case, or '?'. Only ASCII is
    upper-cased, so one character always stays one ('ß' would become 'SS')."""
    u = c.upper() if c.isascii() else UNKNOWN
    return u if u in _INDEX else UNKNOWN


def fold(text: str) -> str:
    """Upper case, runs of white space as one space, the ends stripped, and
    any character outside the set as '?'."""
    return "".join(_fold_char(c) for c in " ".join(text.split()))


def index(char: str) -> int:
    """The class of one character, folded."""
    if len(char) != 1:
        raise KeyError(f"not one character: {char!r}")
    return _INDEX[_fold_char(char)]


def encode(text: str) -> list[int]:
    """A text as the classes a CTC loss trains on: folded, never a blank."""
    return [_INDEX[c] for c in fold(text)]


def decode(classes) -> str:
    """Classes back to text (blanks give nothing)."""
    return "".join(CLASSES[k] for k in classes)
