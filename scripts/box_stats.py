"""Summary stats on a dataset's photos and date boxes, as a markdown table.

    python scripts/box_stats.py real-train real-test synth --out results/m0_expdate_stats.md

One column per set. The date-box sizes decide M1's input size: a date
whose box is only a few pixels tall after resizing can't be detected.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from bestbefore.images import orientation
from bestbefore.samples import DATE_LABELS, Sample
from bestbefore.sets import NAMES, load_set

INPUT_SIZES = (640, 1280)   # YOLO input sizes worth comparing for M1


def stats(samples: list[Sample], labels_shown: list[str]) -> dict[str, str]:
    long_sides = np.array([max(s.width, s.height) for s in samples])
    labels = Counter(b.label for s in samples for b in s.boxes)
    per_photo = Counter(min(3, sum(b.label in DATE_LABELS for b in s.boxes))
                        for s in samples)

    area, short_px, tall = [], {n: [] for n in INPUT_SIZES}, 0
    for s in samples:
        for b in s.boxes:
            if b.label not in DATE_LABELS:
                continue
            w = (b.xyxy[2] - b.xyxy[0]) * s.width
            h = (b.xyxy[3] - b.xyxy[1]) * s.height
            area.append(w * h / (s.width * s.height))
            tall += h > w
            for n in INPUT_SIZES:
                short_px[n].append(min(w, h) * n / max(s.width, s.height))
    area = np.array(area) * 100

    n = len(samples)
    row = {
        "photos": f"{n}",
        "photo long side, median (min to max)":
            f"{np.median(long_sides):.0f} ({long_sides.min()} to {long_sides.max()})",
        "photos with an EXIF rotation flag":
            f"{sum(orientation(s.image) != 1 for s in samples)}",
    }
    for label in labels_shown:
        row[f"boxes: `{label}`"] = f"{labels[label]}"
    for k, name in ((0, "0 dates"), (1, "1 date"), (2, "2 dates"), (3, "3+ dates")):
        row[f"photos with {name}"] = f"{per_photo[k]} ({100 * per_photo[k] / n:.0f}%)"
    row["date box area, % of photo: p10 / median / p90"] = (
        " / ".join(f"{np.percentile(area, p):.2f}" for p in (10, 50, 90)))
    for size in INPUT_SIZES:
        px = np.array(short_px[size])
        row[f"date box short side at {size} px: p10 / median"] = (
            f"{np.percentile(px, 10):.1f} / {np.median(px):.1f} px")
        row[f"date boxes under 8 px at {size}"] = f"{100 * np.mean(px < 8):.1f}%"
    row["tall date boxes (height > width)"] = f"{100 * tall / len(area):.1f}%"
    return row


def table(sets: dict[str, dict[str, str]]) -> str:
    keys = list(dict.fromkeys(k for row in sets.values() for k in row))
    lines = ["| | " + " | ".join(sets) + " |",
             "|---|" + "---:|" * len(sets)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(r.get(k, "") for r in sets.values()) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sets", nargs="+", choices=NAMES)
    parser.add_argument("--out", help="write the table here as well as printing it")
    args = parser.parse_args()

    loaded = {name: load_set(name) for name in args.sets}
    labels = sorted({b.label for ss in loaded.values() for s in ss for b in s.boxes})
    md = table({name: stats(ss, labels) for name, ss in loaded.items()})
    print(md)
    if args.out:
        header = (f"# Box stats: {', '.join(args.sets)}\n\n"
                  f"Made by `python scripts/box_stats.py {' '.join(args.sets)}"
                  f" --out {args.out}`.\n\n")
        with open(args.out, "w") as f:
            f.write(header + md + "\n")


if __name__ == "__main__":
    main()
