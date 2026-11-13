"""Cutting a date box out of a photo, with padding, and the quarter turns a
crop is read at (docs/m3-date-reader.md, decisions 4 and 14).

Boxes here are in pixels of the image they are cut from, (x1, y1, x2, y2),
floats as a label gives them. Padding is a share of the box's short side,
which for a sideways date is its print height, so a padding means the same
margin round the print whichever way the date runs.
"""
from __future__ import annotations

import math

XYXY = tuple[float, float, float, float]


def to_pixels(xyxy: XYXY, size: tuple[int, int]) -> XYXY:
    """A box in 0..1 of the image (as every label in this project) in pixels."""
    w, h = size
    return xyxy[0] * w, xyxy[1] * h, xyxy[2] * w, xyxy[3] * h


def padded(box: XYXY, pad: float, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The box grown by `pad` x its short side on every side, rounded out to
    whole pixels and clipped to the image of `size`: what to cut."""
    x1, y1, x2, y2 = box
    m = pad * min(x2 - x1, y2 - y1)
    w, h = size
    return (max(0, math.floor(x1 - m)), max(0, math.floor(y1 - m)),
            min(w, math.ceil(x2 + m)), min(h, math.ceil(y2 + m)))


def is_tall(box: XYXY) -> bool:
    """Taller than wide: a date running up or down the photo."""
    return box[3] - box[1] > box[2] - box[0]


def turns(tall: bool) -> tuple[int, int]:
    """The quarter turns (anticlockwise) a crop is read at, decision 14: a
    wide one upright and upside down, a tall one either way on its side."""
    return (1, 3) if tall else (0, 2)
