"""The ExpDate Products sets (Seker & Ahn, 2022; CC BY 4.0, credit KIST).

Each split is a folder holding `annotations.json` and `images/`. The JSON
maps a file name to its `width`, `height` and `ann` list; each `ann` has
`cls`, a pixel `bbox` [x1, y1, x2, y2], and for dates a `transcription`.
Notes on what is in the files: docs/datasets.md.

Classes as they appear in the files:
    date   a printed date (train and synth: every date; test: non-expiry dates)
    exp    the expiry date (test split only, exactly one per image), which
           also carries `dmy_ann`: its day / month / year parts
    due    an expiry mark, the "EXP" or "Best before" words
    prod   a production mark, the "MFG" words
    code   a batch or lot code
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import expdate_dir
from .samples import Box, Expiry, Sample, parse_month, parse_year

SPLITS = {
    "real-train": "products-real/Products-Real/train",
    "real-test": "products-real/Products-Real/evaluation",
    "synth": "products-synth/Products-Synth",
}
CLASSES = ("date", "exp", "due", "prod", "code")


def split_dir(split: str, root: Path | None = None) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; one of {', '.join(SPLITS)}")
    return (root or expdate_dir()) / SPLITS[split]


def read_annotations(split: str, root: Path | None = None) -> dict:
    return json.loads((split_dir(split, root) / "annotations.json").read_text())


def load_split(split: str, root: Path | None = None) -> list[Sample]:
    """Every image in the split, sorted by file name."""
    folder = split_dir(split, root)
    ann = read_annotations(split, root)
    on_disk = {p.name for p in (folder / "images").iterdir()
               if p.suffix.lower() == ".jpg"}
    if set(ann) != on_disk:
        raise ValueError(
            f"{split}: {len(set(ann) - on_disk)} annotated images missing, "
            f"{len(on_disk - set(ann))} images without annotations")
    return [_sample(name, ann[name], folder / "images" / name)
            for name in sorted(ann)]


def _sample(name: str, item: dict, path: Path) -> Sample:
    w, h = item["width"], item["height"]
    boxes = tuple(
        Box(a["cls"], normalise(a["bbox"], w, h), a.get("transcription"))
        for a in item["ann"])
    exp = [a for a in item["ann"] if a["cls"] == "exp"]
    expiry = expiry_from_parts(exp[0].get("dmy_ann", [])) if exp else None
    return Sample(id=name, image=path, width=w, height=h,
                  boxes=boxes, expiry=expiry)


def normalise(bbox, width: int, height: int) -> tuple[float, float, float, float]:
    """Pixel [x1, y1, x2, y2] to 0..1, clipped to the image."""
    x1, y1, x2, y2 = bbox

    def clip(v: float) -> float:
        return min(max(v, 0.0), 1.0)

    return (clip(x1 / width), clip(y1 / height),
            clip(x2 / width), clip(y2 / height))


def expiry_from_parts(parts: list[dict]) -> Expiry | None:
    """The expiry from its day / month / year annotations.

    None when the year or month is missing or doesn't parse, since then
    there is no full date to score a reader against. A missing day is fine:
    the pack prints a month only.
    """
    text = {p["cls"]: p["transcription"] for p in parts}
    year = parse_year(text.get("year", ""))
    month = parse_month(text.get("month", ""))
    if year is None or month is None:
        return None
    day = text.get("day")
    if day is not None and not day.strip().isdigit():
        return None
    try:
        return Expiry(year, month, int(day) if day is not None else None)
    except ValueError:
        return None
