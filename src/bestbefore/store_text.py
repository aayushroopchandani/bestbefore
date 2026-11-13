"""The store dates' values, for the cluster (docs/m3-date-reader.md, "Store
text"; M3 S2).

The cluster holds store sets v1 and v3 as M2b S6 exported them
($STORE_TRAIN_DIR): numbered photos, upright, each box as a YOLO class, so
an expiry, an mfg date and any other date are all `date` there. `text.json`,
written beside the export, says which is which and what each one says, by
numbered photo and the box's line in its label file: never a photo's name.

S2 writes the values-only version (`scripts/export_store_text.py
--values`): each date box's label (expiry, mfg, other_date) and, for an
expiry or mfg box, its photo's frozen value; and each photo's month, from
its EXIF capture date, for the plausibility check (decision 13). S4 adds
the printed text and each box's turn, typed by the owner.

Kept out of store.py and store_train.py, which M2b's running job imports:
a new file changes none of its code.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from . import store
from .store_train import ExportPhoto

NAME = "text.json"
VALUE_LABELS = ("expiry", "mfg")
_EXIF_IFD, _TAKEN, _DATETIME = 0x8769, 0x9003, 0x0132


def photo_month(path: Path) -> str | None:
    """'2026-08' from the photo's EXIF capture date (DateTimeOriginal, else
    DateTime); None when it carries neither."""
    with Image.open(path) as im:
        exif = im.getexif()
        taken = exif.get_ifd(_EXIF_IFD).get(_TAKEN) or exif.get(_DATETIME)
    if not taken or len(taken) < 7 or not taken[:4].isdigit():
        return None
    return f"{taken[:4]}-{taken[5:7]}"


def box_values(p: ExportPhoto, label: dict) -> list[dict]:
    """The photo's date boxes as text.json lists them: the line of each in
    its label file (`p.boxes` in order, slips already left out), its label,
    and for an expiry or mfg box its frozen value. A derived expiry is
    worked out, not printed, so an expiry box on a derived photo (none
    exists) would get no value."""
    values = {"expiry": None if label["derived"] else label["expiry"], "mfg": label["mfg"]}
    return [{"index": k, "label": b.label, "value": values.get(b.label)}
            for k, b in enumerate(p.boxes) if b.label in store.BOX_LABELS]


def values_file(photos: dict[str, dict], manifest_sha256: str, sources: dict, made: str,
                provenance: dict) -> dict:
    """text.json's content: `photos` maps a numbered photo to its part,
    month and boxes; the export's manifest hash ties it to one export."""
    return {"version": "values", "made": made, **provenance,
            "export_manifest_sha256": manifest_sha256, "sources": sources,
            "photos": dict(sorted(photos.items()))}


def write_private(path: Path, data: dict) -> str:
    """Write JSON readable by its owner only (mode 600, whatever the umask)
    and return its SHA-256."""
    text = json.dumps(data, indent=1, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(path, 0o600)
    return hashlib.sha256(text.encode()).hexdigest()


def check(text: dict, manifest: dict, manifest_sha256: str) -> list[str]:
    """Problems with a text.json against the export it says it belongs to:
    the same manifest, every exported photo once, each box's line holding a
    date of that label in the manifest, and a value on every expiry and mfg
    box that parses."""
    problems = []
    if text.get("export_manifest_sha256") != manifest_sha256:
        problems.append("text.json was made for another export (manifest hashes differ)")
    photos, listed = text.get("photos", {}), manifest["photos"]
    problems += [f"{n}: in the export, not in text.json" for n in sorted(set(listed) - set(photos))]
    problems += [f"{n}: in text.json, not in the export" for n in sorted(set(photos) - set(listed))]
    for name in sorted(set(photos) & set(listed)):
        labels = listed[name]["labels"]
        if photos[name]["part"] != listed[name]["part"]:
            problems.append(f"{name}: part {photos[name]['part']}, the export's {listed[name]['part']}")
        dates = [k for k, label in enumerate(labels) if label in store.BOX_LABELS]
        if [b["index"] for b in photos[name]["boxes"]] != dates:
            problems.append(f"{name}: date lines {[b['index'] for b in photos[name]['boxes']]}, the export's {dates}")
            continue
        for b in photos[name]["boxes"]:
            if b["label"] != labels[b["index"]]:
                problems.append(f"{name}: line {b['index']} is {labels[b['index']]}, text.json says {b['label']}")
            if b["label"] in VALUE_LABELS and b["value"] is not None:
                try:
                    store.parse_date(b["value"])
                except ValueError:
                    problems.append(f"{name}: line {b['index']}'s value {b['value']!r} isn't a date")
    return problems
