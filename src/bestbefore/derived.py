"""M2's derived photos: ExpDate photos turned a quarter, and 2 x 2 grids
(docs/m2-synthetic-generator.md, S2). Also reading and writing any YOLO
set as `Sample`s, so the evaluator, the contact sheet and the box stats
run on derived and generated sets too.

Boxes are xyxy in 0..1 of the upright image, as everywhere in this
project. Turning moves them by a formula, never by looking at the pixels,
so tests/test_derived.py checks the formula against drawn pixels.
"""
from __future__ import annotations

import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from PIL import Image

from .images import upright_size
from .samples import Box, Sample
from .yolo import CLASSES, LABEL_TO_CLASS, label_lines, read_label_file

JPEG_QUALITY = 95
PAD = (114, 114, 114)      # the grey Ultralytics pads with

# PIL's ROTATE_* turn anticlockwise.
_TRANSPOSE = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
              3: Image.Transpose.ROTATE_270}

XYXY = tuple[float, float, float, float]


# --- Turning ----------------------------------------------------------------

def turn_box(xyxy: XYXY, quarter: int) -> XYXY:
    """The box after its image turns `quarter` x 90 degrees anticlockwise.

    One quarter turn sends a point (x, y) to (y, 1 - x): the right edge
    becomes the top. So (x1, y1, x2, y2) becomes (y1, 1 - x2, y2, 1 - x1),
    and 270 degrees is three of those: (1 - y2, x1, 1 - y1, x2).
    """
    x1, y1, x2, y2 = xyxy
    for _ in range(quarter % 4):
        x1, y1, x2, y2 = y1, 1 - x2, y2, 1 - x1
    return (x1, y1, x2, y2)


def turn(image: Image.Image, boxes: Iterable[Box],
         quarter: int) -> tuple[Image.Image, tuple[Box, ...]]:
    """The image and its boxes turned `quarter` x 90 degrees anticlockwise.
    Exact: a quarter turn moves pixels without resampling them."""
    q = quarter % 4
    turned = image.transpose(_TRANSPOSE[q]) if q else image.copy()
    return turned, tuple(replace(b, xyxy=turn_box(b.xyxy, q)) for b in boxes)


def quarter_for(key: str, seed: int) -> int:
    """1 (90 degrees) or 3 (270 degrees): a fair coin seeded by the photo's
    key, so a photo turns the same way however the work is split."""
    return random.Random(f"{seed}/{key}").choice((1, 3))


# --- Grids ------------------------------------------------------------------

def grid(tiles: Sequence[tuple[Image.Image, Iterable[Box]]]) -> tuple[Image.Image, tuple[Box, ...]]:
    """Four photos in a 2 x 2 grid, filled left to right, top to bottom.

    Each photo is centred, unscaled, in a C x C cell on grey, C being the
    longest side of the four. Its print keeps every pixel but covers about
    half as much of the picture: a small-print test made of real print.
    """
    if len(tiles) != 4:
        raise ValueError(f"a grid takes 4 photos, not {len(tiles)}")
    c = max(max(im.size) for im, _ in tiles)
    side = 2 * c
    canvas = Image.new("RGB", (side, side), PAD)
    placed: list[Box] = []
    for i, (im, boxes) in enumerate(tiles):
        w, h = im.size
        x0, y0 = (i % 2) * c + (c - w) // 2, (i // 2) * c + (c - h) // 2
        canvas.paste(im, (x0, y0))
        placed += [replace(b, xyxy=((x0 + b.xyxy[0] * w) / side, (y0 + b.xyxy[1] * h) / side,
                                    (x0 + b.xyxy[2] * w) / side, (y0 + b.xyxy[3] * h) / side))
                   for b in boxes]
    return canvas, tuple(placed)


def grid_groups(samples: Iterable[Sample]) -> tuple[list[list[Sample]], list[Sample]]:
    """The samples in fours of similar size (sorted by long side, then id),
    so every grid shrinks its dates by close to half; and the ones left over."""
    ordered = sorted(samples, key=lambda s: (max(s.width, s.height), s.id))
    n = len(ordered) // 4 * 4
    return [ordered[i:i + 4] for i in range(0, n, 4)], ordered[n:]


# --- YOLO sets on disk --------------------------------------------------------

def label_path(image: Path) -> Path:
    """Where Ultralytics looks for an image's labels: the last /images/ in
    its path becomes /labels/, and the extension .txt."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return Path(sb.join(str(image).rsplit(sa, 1))).with_suffix(".txt")


def read_boxes(image: Path) -> tuple[Box, ...]:
    """An image's boxes from its YOLO label file, labelled by class name."""
    return tuple(Box(CLASSES[cls], xyxy)
                 for cls, xyxy in read_label_file(label_path(image).read_text()))


def read_list(path: Path) -> list[Path]:
    """The images in a list file, in order, repeats kept. As in Ultralytics,
    a line starting with ./ is relative to the list's folder."""
    lines = (line.strip() for line in path.read_text().splitlines())
    return [path.parent / line[2:] if line.startswith("./") else Path(line)
            for line in lines if line]


def write_list(path: Path, images: Iterable[Path]) -> int:
    """Write a list file, one image path per line; returns how many."""
    lines = [f"{p}\n" for p in images]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    return len(lines)


def load_sample(image: Path) -> Sample:
    w, h = upright_size(image)
    return Sample(id=image.name, image=image, width=w, height=h, boxes=read_boxes(image))


def load_yolo_set(list_path: Path) -> list[Sample]:
    """Every image in a list file, with its labels, as `Sample`s."""
    return [load_sample(p) for p in read_list(list_path)]


def write_yolo(root: Path, set_name: str, name: str, image: Image.Image,
               boxes: Iterable[Box]) -> Sample:
    """Save a photo as `root/images/<set>/<name>` (JPEG, quality 95) and its
    labels where Ultralytics will look; returns it as a `Sample`."""
    path = root / "images" / set_name / name
    sample = Sample(id=name, image=path, width=image.width, height=image.height,
                    boxes=tuple(boxes))
    lines, _ = label_lines(sample)
    labels = label_path(path)
    for folder in (path.parent, labels.parent):
        folder.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=JPEG_QUALITY)
    labels.write_text("".join(f"{line}\n" for line in lines))
    return sample


def check_written(sample: Sample) -> float:
    """Read a written sample back and compare it with what was meant: the
    image size, and every box with an area, in order, with its class.
    Returns the worst box error in pixels; raises on any other difference."""
    back = load_sample(sample.image)
    w, h = sample.width, sample.height
    if (back.width, back.height) != (w, h):
        raise ValueError(f"{sample.image}: {back.width} x {back.height} on disk, meant {w} x {h}")
    meant = [b for b in sample.boxes if b.xyxy[2] > b.xyxy[0] and b.xyxy[3] > b.xyxy[1]]
    if len(back.boxes) != len(meant):
        raise ValueError(f"{sample.image}: {len(back.boxes)} boxes on disk, meant {len(meant)}")
    worst = 0.0
    for m, b in zip(meant, back.boxes):
        if CLASSES[LABEL_TO_CLASS[m.label]] != b.label:
            raise ValueError(f"{sample.image}: class {b.label} on disk, meant {m.label}")
        worst = max(worst, *(abs(p - q) * size for p, q, size
                             in zip(m.xyxy, b.xyxy, (w, h, w, h))))
    return worst
