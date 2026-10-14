"""YOLO label format: one `class cx cy w h` line per box, all 0..1."""
from __future__ import annotations

from .samples import Sample

CLASSES = ("date", "due", "prod", "code")

# To a detector the test split's expiry is a date like any other; telling
# expiry from manufacture is the reader's job (M3).
LABEL_TO_CLASS = {"date": 0, "exp": 0, "due": 1, "prod": 2, "code": 3}


def to_yolo(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def from_yolo(cxcywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = cxcywh
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def label_lines(sample: Sample) -> tuple[list[str], int]:
    """The sample's label file lines, and how many zero-area boxes were dropped."""
    lines, dropped = [], 0
    for b in sample.boxes:
        cx, cy, w, h = to_yolo(b.xyxy)
        if w <= 0 or h <= 0:
            dropped += 1
            continue
        lines.append(f"{LABEL_TO_CLASS[b.label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, dropped


def list_ids(yolo_dir, list_name: str) -> set[str]:
    """Photo ids in one of the converter's image lists, e.g. 'val'."""
    lines = (yolo_dir / f"{list_name}.txt").read_text().splitlines()
    return {line.strip().rsplit("/", 1)[-1] for line in lines if line.strip()}


def read_label_file(text: str) -> list[tuple[int, tuple[float, float, float, float]]]:
    """(class, xyxy) per line."""
    out = []
    for line in text.splitlines():
        cls, *vals = line.split()
        out.append((int(cls), from_yolo(tuple(float(v) for v in vals))))
    return out
