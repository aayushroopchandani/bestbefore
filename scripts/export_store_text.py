"""Write `text.json` beside the store export: each numbered photo's date
boxes, by their line in its label file, with their values and the photo's
month (docs/m3-date-reader.md, S2 and "Store text"); with --text, also each
box's printed text and turn and each shelf-life box (S4).

    python scripts/export_store_text.py --values   # S2's values-only version
    python scripts/export_store_text.py --text     # S4's, from the frozen data/store/text.json

Reads only the frozen files the export was made from (store_train.
export_inputs: v1's and v3's labels, keywords, designs and splits, each
checked against its hash), and the laptop's export in $STORE_TRAIN_DIR,
whose manifest it is tied to. --text reads data/store/text.json only once its hash is TEXT_SHA256, and
its values, months and boxes must equal what --values works out from the
frozen files. Store set v2 is never opened. Writes
$STORE_TRAIN_DIR/text.json with mode 600; copy it to the cluster's export
by CLAUDE.md's owner-only rules. Prints counts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date

from bestbefore import store, store_text, store_train
from bestbefore.config import store_photos_dir, store_train_dir
from bestbefore.images import upright_size
from bestbefore.runs import code_sha256, git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--values", action="store_true", help="the values-only version (S2)")
    which.add_argument("--text", action="store_true", help="with the typed text and turns (S4)")
    args = parser.parse_args()

    folder, root = store_train_dir(), store_photos_dir()
    manifest_bytes = (folder / store_train.MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if problems := store_train.check_export(folder):
        sys.exit("the laptop's export doesn't match its manifest:\n  " + "\n  ".join(problems))

    sets, designs, split, sources = store_train.export_inputs()
    labels = {p: lab for given in sets.values() for p, lab in given["labels"].items()}
    photos, counts = {}, Counter()
    for p in store_train.export_plan(sets, designs, split):
        first = next(iter(store_train.FOLDERS[p.part]))
        p = store_train.without_slips(p, upright_size(folder / "images" / first / f"{p.name}.jpg"))
        if [b.label for b in p.boxes] != manifest["photos"][p.name]["labels"]:
            sys.exit(f"{p.name}: its boxes don't follow the export's label file; nothing written")
        month = store_text.photo_month(root / p.photo)
        boxes = store_text.box_values(p, labels[p.photo])
        photos[p.name] = {"part": p.part, "month": month, "boxes": boxes}
        counts[(p.part, "photos")] += 1
        counts[(p.part, "no month")] += month is None
        for b in boxes:
            counts[(p.part, b["label"], "value" if b["value"] else "no value")] += 1

    provenance = {**git_state(), "code_sha256": code_sha256(store_text, store_train, store)}
    text = store_text.values_file(photos, manifest_sha256, sources, date.today().isoformat(), provenance)
    if args.text:
        typed = store_text.read_text()
        numbered = store_text.numbered_file(typed, store_text.TEXT_SHA256, manifest_sha256,
                                            date.today().isoformat(), provenance)
        if typed["sources"] != sources:
            sys.exit("data/store/text.json was made from other frozen files than the export's; nothing written")
        for name, entry in photos.items():
            got = numbered["photos"].get(name, {})
            if (got.get("part"), got.get("month")) != (entry["part"], entry["month"]) or \
                    [{k: b[k] for k in ("index", "label", "value")} for b in got.get("boxes", [])] != entry["boxes"]:
                sys.exit(f"{name}: data/store/text.json disagrees with the frozen files; nothing written")
        for name, entry in numbered["photos"].items():
            for b in entry["boxes"]:
                counts[(entry["part"], "turn", str(b["turn"]))] += 1
            counts[(entry["part"], "life boxes")] += len(entry["life"])
        text = numbered
    if problems := store_text.check(text, manifest, manifest_sha256):
        sys.exit("text.json fails its own check:\n  " + "\n  ".join(problems))
    path = folder / store_text.NAME
    digest = store_text.write_private(path, text)
    print(f"{path}: sha256 {digest}")
    for key in sorted(counts):
        print(f"  {' / '.join(key)}: {counts[key]}")


if __name__ == "__main__":
    main()
