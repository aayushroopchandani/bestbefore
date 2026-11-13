"""Cutting a date box out of a photo, with padding, and the quarter turns a
crop is read at (docs/m3-date-reader.md, decisions 4 and 14).

Boxes here are in pixels of the image they are cut from, (x1, y1, x2, y2),
floats as a label gives them. Padding is a share of the box's short side,
which for a sideways date is its print height, so a padding means the same
margin round the print whichever way the date runs.

`cut` makes a stored crop (decision 4): the box with MARGIN of its short
side on every side, shrunk so the box's short side is at most MAX_SHORT
pixels (never enlarged), turned upright by a known quarter. The padding a
reader sees is cut from that margin when the crop is loaded.
"""
from __future__ import annotations

import math

from PIL import Image

XYXY = tuple[float, float, float, float]

MARGIN = 0.5            # decision 4: half the box's short side, on every side
MAX_SHORT = 64          # decision 4: the box's short side at most this, in pixels
# PIL's transposes that turn anticlockwise, as np.rot90 and derived.turn do.
_TRANSPOSE = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
              3: Image.Transpose.ROTATE_270}


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


def turn_box(box: XYXY, size: tuple[int, int], quarter: int) -> XYXY:
    """A box in pixels of an image of `size`, after the image turns
    `quarter` x 90 degrees anticlockwise. One quarter sends (x, y) to
    (y, W - x): (x1, y1, x2, y2) becomes (y1, W - x2, y2, W - x1), and the
    image becomes H x W."""
    x1, y1, x2, y2 = box
    w, h = size
    for _ in range(quarter % 4):
        x1, y1, x2, y2 = y1, w - x2, y2, w - x1
        w, h = h, w
    return x1, y1, x2, y2


def cut(image: Image.Image, box: XYXY, turn: int = 0, margin: float = MARGIN,
        max_short: int = MAX_SHORT) -> tuple[Image.Image, XYXY]:
    """The stored crop of `box` (pixels of `image`) and the box inside it:
    `margin` x the box's short side round it (clipped to the image), shrunk
    with an area filter so the box's short side is at most `max_short`
    pixels (a smaller box is kept as it is), then turned `turn` quarters
    anticlockwise. Raises ValueError for a box with no area inside the image.
    The box is first rounded to 1e-6 px, so one that went through 0..1 and
    back (79.9999999 for 80) rounds out to the same pixels."""
    box = tuple(round(v, 6) for v in box)
    rect = padded(box, margin, image.size)
    x0, y0 = rect[0], rect[1]
    inside = (max(box[0], x0) - x0, max(box[1], y0) - y0,
              min(box[2], rect[2]) - x0, min(box[3], rect[3]) - y0)
    if inside[2] <= inside[0] or inside[3] <= inside[1]:
        raise ValueError(f"box {box} has no area inside the image of {image.size}")
    crop = image.crop(rect)
    short = min(inside[2] - inside[0], inside[3] - inside[1])
    scale = min(1.0, max_short / short)
    if scale < 1.0:
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        sx, sy = size[0] / crop.width, size[1] / crop.height
        crop = crop.resize(size, Image.Resampling.BOX)
        inside = (inside[0] * sx, inside[1] * sy, inside[2] * sx, inside[3] * sy)
    q = turn % 4
    if q:
        inside = turn_box(inside, crop.size, q)
        crop = crop.transpose(_TRANSPOSE[q])
    return crop, inside
