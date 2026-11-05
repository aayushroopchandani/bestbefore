"""v1-train's date sizes, measured as the store measures them, and their
deciles for gen-v2's sizes (docs/m2b-store-and-size.md, "The generator, v2",
S8).

    python scripts/store_sizes.py

Reads the training part of store sets v1 and v3's export ($STORE_TRAIN_DIR,
S6), checked against its manifest first: each training photo once, upright
(train_0), and its date boxes (class 0: expiry, mfg and other dates, those
S5's review added, slips under 2 px left out), as the detector trains on
them. Never v1-val, v3's validation part or v2: the export holds no v2
photo, and its validation photos are not read. Each date's size is its
box's short side over its photo's long side (Cause 4's measure, as
experiments/m2b-store-look/measure.py has it): for a sideways date, the
short side is its print height.

Prints counts, the size bands, the median and the deciles only, never a
photo: the deciles, their ends clipped to M2's floor and ceiling, are what
spec.V2Size holds, and the script says whether they match it.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from bestbefore.config import store_train_dir
from bestbefore.store_train import check_export
from bestbefore.synth.spec import SIZE, V2Size

BANDS = ((0.015, "under 1.5%"), (0.025, "1.5–2.5%"), (np.inf, "2.5% and up"))   # Cause 4's
DATE = "0"                      # yolo.CLASSES: date
PLACES = 5                      # the deciles' decimals in V2Size: 0.001% of the long side


def size(w: float, h: float, width: int, height: int) -> float:
    """A box (w, h as fractions of its photo) as the store measures it:
    its short side over its photo's long side."""
    return min(w * width, h * height) / max(width, height)


def training_sizes(folder: Path) -> tuple[list[float], Counter]:
    """Every date of the export's training part, once (its upright copies):
    their sizes, and their labels (expiry, mfg, other_date), counted from
    the manifest, whose box labels are in label-file order."""
    manifest = json.loads((folder / "manifest.json").read_text())
    upright = next(name for name, quarter in manifest["settings"]["folders"]["train"].items() if quarter == 0)
    sizes, roles = [], Counter()
    for name, photo in sorted(manifest["photos"].items()):
        if photo["part"] != "train":
            continue
        with Image.open(folder / "images" / upright / f"{name}.jpg") as im:
            width, height = im.size
        lines = (folder / "labels" / upright / f"{name}.txt").read_text().splitlines()
        if len(lines) != len(photo["labels"]):
            raise SystemExit(f"{name}: {len(lines)} label lines, the manifest names {len(photo['labels'])} boxes")
        for line, label in zip(lines, photo["labels"]):
            cls, _, _, w, h = line.split()
            if cls == DATE:
                sizes.append(size(float(w), float(h), width, height))
                roles[label] += 1
    return sizes, roles


def deciles(sizes, floor: float = SIZE.small[0], ceiling: float = SIZE.normal[1]) -> tuple[float, ...]:
    """The 11 edges of the sizes' deciles (numpy's linear percentiles), the
    two ends clipped to M2's floor and ceiling, rounded to PLACES."""
    edges = np.percentile(np.asarray(sizes), np.arange(0, 101, 10))
    edges[0], edges[-1] = np.clip(edges[0], floor, ceiling), np.clip(edges[-1], floor, ceiling)
    return tuple(round(float(e), PLACES) for e in edges)


def bands(sizes) -> dict[str, float]:
    v, out, lo = np.asarray(sizes), {}, 0.0
    for top, label in BANDS:
        out[label] = float(np.mean((v >= lo) & (v < top)))
        lo = top
    return out


def main() -> None:
    folder = store_train_dir()
    if problems := check_export(folder):
        sys.exit(f"the store export doesn't match its manifest ({len(problems)} problems), stopping:\n  "
                 + "\n  ".join(problems[:10]))
    sizes, roles = training_sizes(folder)
    v = np.asarray(sizes)
    edges = deciles(sizes)
    print(f"v1-train's dates (v3's training part included): {len(v)} "
          f"({', '.join(f'{k} {n}' for k, n in sorted(roles.items()))})")
    print("bands: " + ", ".join(f"{k} {100 * share:.1f}%" for k, share in bands(sizes).items())
          + f"; median {100 * np.median(v):.3f}%")
    print(f"under M2's floor ({100 * SIZE.small[0]:.1f}%): {int((v < SIZE.small[0]).sum())}; "
          f"over its ceiling ({100 * SIZE.normal[1]:.1f}%): {int((v > SIZE.normal[1]).sum())}")
    raw = np.percentile(v, np.arange(0, 101, 10))
    print("deciles, %: " + " ".join(f"{100 * e:.3f}" for e in raw))
    print(f"for V2Size, ends clipped: deciles={edges}")
    print("spec.V2Size holds them" if edges == V2Size().deciles else "spec.V2Size holds other deciles: update it")


if __name__ == "__main__":
    main()
