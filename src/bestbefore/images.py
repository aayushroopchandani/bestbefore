"""Opening images. Every image in this project goes through `load_upright`.

Phone cameras often store a photo sideways plus an EXIF flag saying how to
turn it. Boxes are drawn on the upright view, so if one script applies the
flag and another doesn't, boxes land in the wrong place with no error.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

ORIENTATION = 0x0112          # the EXIF tag
SIDEWAYS = {5, 6, 7, 8}       # orientations that swap width and height


def load_upright(path: str | Path) -> Image.Image:
    """The image as RGB, turned the way its EXIF flag says."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def orientation(path: str | Path) -> int:
    """The EXIF orientation flag, 1 (already upright) when there is none."""
    with Image.open(path) as im:
        return int(im.getexif().get(ORIENTATION, 1))


def upright_size(path: str | Path) -> tuple[int, int]:
    """(width, height) after `load_upright`, without decoding the pixels."""
    with Image.open(path) as im:
        w, h = im.size
        flag = int(im.getexif().get(ORIENTATION, 1))
    return (h, w) if flag in SIDEWAYS else (w, h)
