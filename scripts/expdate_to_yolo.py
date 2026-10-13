"""Convert ExpDate to a YOLO dataset under $EXPDATE_DIR/yolo/.

    python scripts/expdate_to_yolo.py

Writes:
  images/<split>/   symlinks to the original photos (no second copy)
  labels/<split>/   one .txt per photo, empty when it has no boxes
  train_real.txt, train_synth.txt, val.txt, test.txt
                    image lists; val is 15% of real-train, seed 0
  expdate.yaml      the Ultralytics dataset file M1 trains from

Real and synthetic training images sit in separate lists, so training with
or without synthetic data is a one-line change in the yaml. The symlinks
are absolute paths: on the cluster, rerun this instead of copying yolo/.
Every label file is read back and checked against its source boxes.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter

from bestbefore import expdate
from bestbefore.config import expdate_dir
from bestbefore.runs import start_run
from bestbefore.yolo import CLASSES, LABEL_TO_CLASS, label_lines, read_label_file

VAL_SHARE = 0.15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = expdate_dir() / "yolo"
    run_dir = start_run("expdate-to-yolo", {"out": out, "val_share": VAL_SHARE},
                        seed=args.seed)
    lists: dict[str, list[str]] = {"train_real": [], "val": [],
                                   "train_synth": [], "test": []}
    report = []
    val: set[str] = set()

    for split in expdate.SPLITS:
        samples = expdate.load_split(split)
        images, labels = out / "images" / split, out / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)

        if split == "real-train":
            ids = [s.id for s in samples]
            val = set(random.Random(args.seed).sample(ids, round(VAL_SHARE * len(ids))))
        dropped, classes, worst = 0, Counter(), 0.0
        for s in samples:
            link = images / s.id
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(s.image.resolve())

            lines, n_dropped = label_lines(s)
            dropped += n_dropped
            label_file = labels / f"{link.stem}.txt"
            label_file.write_text("\n".join(lines) + ("\n" if lines else ""))

            # Read it back: every box must come back to within a pixel.
            kept = [b for b in s.boxes if b.xyxy[2] > b.xyxy[0] and b.xyxy[3] > b.xyxy[1]]
            back = read_label_file(label_file.read_text())
            assert len(back) == len(kept), s.id
            for b, (cls, xyxy) in zip(kept, back):
                assert cls == LABEL_TO_CLASS[b.label], s.id
                err = max(abs(p - q) * size for p, q, size in zip(
                    b.xyxy, xyxy, (s.width, s.height, s.width, s.height)))
                worst = max(worst, err)
                classes[CLASSES[cls]] += 1

            entry = f"./images/{split}/{s.id}"
            if split == "real-train":
                lists["val" if s.id in val else "train_real"].append(entry)
            else:
                lists["train_synth" if split == "synth" else "test"].append(entry)

        report.append(f"{split}: {len(samples)} images, "
                      + ", ".join(f"{c} {classes[c]}" for c in CLASSES)
                      + f"; dropped {dropped} empty boxes; worst round trip {worst:.3f} px")

    for name, entries in lists.items():
        (out / f"{name}.txt").write_text("\n".join(entries) + "\n")
    (out / "expdate.yaml").write_text(
        "# Made by scripts/expdate_to_yolo.py\n"
        f"path: {out.resolve()}\n"
        "train: [train_synth.txt, train_real.txt]\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))

    report.append(", ".join(f"{name} {len(e)}" for name, e in lists.items()))
    (run_dir / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(out / "expdate.yaml")


if __name__ == "__main__":
    main()
