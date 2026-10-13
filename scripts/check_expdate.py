"""Check the unpacked ExpDate splits and print what is in them.

    python scripts/check_expdate.py

For every split: images, box counts per class, and any problem worth
knowing before training: a JSON size that doesn't match the upright image,
boxes outside the image, and (test split) expiries with no full date to
score against. The numbers are written up in docs/datasets.md.
"""
from __future__ import annotations

from collections import Counter

from bestbefore import expdate
from bestbefore.images import orientation, upright_size


def check(split: str) -> None:
    raw = expdate.read_annotations(split)
    samples = expdate.load_split(split)

    classes = Counter(b.label for s in samples for b in s.boxes)
    size_mismatch = [s.id for s in samples
                     if upright_size(s.image) != (s.width, s.height)]
    flagged = sum(orientation(s.image) != 1 for s in samples)
    outside = sum(
        1 for item in raw.values() for a in item["ann"]
        if min(a["bbox"]) < 0 or a["bbox"][2] > item["width"]
        or a["bbox"][3] > item["height"])

    print(f"\n{split}: {len(samples)} images")
    print("  boxes:", ", ".join(f"{c} {classes[c]}" for c in expdate.CLASSES
                                if classes[c]))
    print(f"  images with an EXIF rotation flag: {flagged}")
    print(f"  JSON size != upright image size: {len(size_mismatch)}",
          size_mismatch[:5] if size_mismatch else "")
    print(f"  boxes outside the image (clipped): {outside}")

    if split == "real-test":
        scorable = [s for s in samples if s.expiry]
        month_only = sum(s.expiry.day is None for s in scorable)
        print(f"  expiry with a full date: {len(scorable)} "
              f"({month_only} month-only); without: "
              f"{len(samples) - len(scorable)}")


def main() -> None:
    for split in expdate.SPLITS:
        check(split)


if __name__ == "__main__":
    main()
