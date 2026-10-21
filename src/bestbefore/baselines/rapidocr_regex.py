"""Baseline A: RapidOCR off the shelf, then the date regex (dates.py), then
the latest date found taken as the expiry, since an expiry comes after the
manufacture date. No label words, no resizing, no rules.

Same RapidOCR as Scanify (`rapidocr-onnxruntime==1.2.3`, default settings),
given the whole upright photo as RGB, which is what RapidOCR itself makes
of a file path, minus the EXIF turn. So A and baseline B differ only in
what Scanify adds (decision 9).
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path

import numpy as np

from ..dates import find_dates, latest
from ..images import load_upright


@cache
def _engine():
    # Imported here: loading the models takes seconds, and only this reader needs them.
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr_lines(path: Path) -> list[str]:
    """Every text line RapidOCR reads in the photo, in its order."""
    result, _ = _engine()(np.asarray(load_upright(path)))
    return [text for _box, text, _conf in result or []]


def read(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        lines = ocr_lines(path)
        found = [f for line in lines for f in find_dates(line)]
        yield {"expiry": latest(found), "lines": lines,
               "dates": [{"text": f.text, "date": str(f.date), "pattern": f.pattern}
                         for f in found]}
