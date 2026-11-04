"""Turn a reviewed keywords project ("Store v1 keywords", "Store v3 keywords")
into its keyword file, checking it first (docs/m2b-store-and-size.md: S5, S5b).

    python scripts/import_store_keywords.py                 # v1: data/store/v1_keywords.json
    python scripts/import_store_keywords.py --version v3    # v3: data/store/v3/keywords.json, from data/store/v3/keywords/

The file is dated by the review's last change in the export, so it
rebuilds byte for byte from the same export. Never v2.

Reads Label Studio's full JSON export of the project
(data/store/keywords/export.json: the JSON export, not JSON-MIN, as only it
keeps each region's id), OCR's proposals (data/store/keywords/proposals.json)
and store set v1's frozen labels, and writes data/store/v1_keywords.json
only when every check passes:

  1. every v1 photo reviewed: one submitted annotation, no skip, no draft,
     and no photo from outside v1;
  2. every keyword box a due, prod or code box, and every date box an
     expiry, mfg or other_date box, upright, drawn on the upright photo and
     inside it;
  3. no keyword box over a date box (frozen or added) at IoU 0.5 or more;
  4. every frozen date box still there, unchanged.

A date box the reviewer added, a date v1's labels missed, goes into the
file's added_dates; v1's frozen labels don't change. Then it counts the
boxes by label, and what the review did to OCR's proposals (kept, moved,
deleted, added). v1's frozen labels and its own project are only read.
Problems name photos, so they're printed on screen only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from bestbefore import store, store_train
from bestbefore.config import store_photos_dir
from bestbefore.images import upright_size
from bestbefore.tables import markdown_table

OUTCOMES = ("kept", "moved", "deleted", "added")


def run_checks(tasks: list[dict], labels: dict[str, dict], root: Path, store_set: store.StoreSet = store.V1
               ) -> tuple[dict[str, list[str]], dict[str, list[dict]], dict[str, list[dict]]]:
    """The checks by name, each photo's reviewed keyword boxes, and the date
    boxes the reviewer added."""
    v = store_set.version
    photos = store.photos_on_disk(store_set, root)
    upright = {p: upright_size(root / p) for p in photos}
    boxes = store_train.keyword_boxes(tasks)
    added = store_train.added_dates(tasks, labels)
    every_date = {p: {"boxes": labels.get(p, {}).get("boxes", []) + added.get(p, [])} for p in photos}
    checks = {
        f"1. all {len(photos)} {v} photos reviewed: one submitted annotation each, no skip, no draft":
            store_train.check_reviewed(tasks, photos, f"store set {v}"),
        "2. boxes labelled due, prod or code (dates: expiry, mfg, other_date), upright, on the upright photo and inside it":
            store_train.check_keyword_boxes(tasks, upright)
            + store_train.check_keyword_boxes(tasks, upright, store_train.REFERENCE_CONTROL, store.BOX_LABELS),
        f"3. no keyword box over a date box (IoU {store_train.OVER_DATE} or more)":
            store_train.check_off_dates(boxes, every_date),
        "4. every frozen date box still there, unchanged":
            store_train.check_reference(tasks, labels),
    }
    return checks, boxes, added


def count_rows(boxes: dict[str, list[dict]]) -> list[dict[str, str]]:
    """Boxes and photos with one, by label."""
    def row(name: str, keep) -> dict[str, str]:
        return {"label": name, "boxes": f"{sum(keep(b) for bs in boxes.values() for b in bs)}",
                "photos": f"{sum(any(keep(b) for b in bs) for bs in boxes.values())}"}
    return ([row(c, lambda b, c=c: b["label"] == c) for c in store_train.KEYWORD_LABELS]
            + [row("**all**", lambda b: True)])


def help_rows(counts: Counter) -> list[dict[str, str]]:
    """What the review did to OCR's proposals, by outcome and label."""
    labels = store_train.KEYWORD_LABELS
    return [{"": outcome, **{c: f"{counts[outcome, c]}" for c in labels},
             "all": f"{sum(counts[outcome, c] for c in labels)}"}
            for outcome in (*OUTCOMES, "relabelled")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", default="v1", help="the store set: v1 or v3 (never v2)")
    parser.add_argument("--export", type=Path, help="default: export.json in the set's keywords folder")
    parser.add_argument("--proposals", type=Path, help="default: proposals.json in the set's keywords folder")
    parser.add_argument("--out", type=Path, help="default: v1_keywords.json, or the set's keywords.json")
    args = parser.parse_args()

    store_set = store_train.keyword_set(args.version)
    folder, v = store_train.keywords_dir(store_set), store_set.version
    export = args.export or folder / "export.json"
    proposals_path = args.proposals or folder / "proposals.json"
    out = args.out or store_train.keywords_path(store_set)
    tasks = json.loads(export.read_text())
    proposals = json.loads(proposals_path.read_text())
    if proposals["test_set_sha256"] != store_set.test_set_sha256:
        raise SystemExit(f"{proposals_path} was made against other {v} labels than the frozen ones")
    labels = store.labels(store_set)
    checks, boxes, added = run_checks(tasks, labels, store_photos_dir(), store_set)
    for name, problems in checks.items():
        print(f"{'✗' if problems else '✓'} {name}" + (f": {len(problems)}" if problems else ""))
        for p in problems:
            print(f"    {p}")
    failed = sum(bool(p) for p in checks.values())
    if failed:
        raise SystemExit(f"\n{failed} checks failed, so {out} was not written. "
                         "Fix the boxes in Label Studio, export again and rerun.")

    hashes = {"test_set_sha256": store_set.test_set_sha256, "export_sha256": store.sha256(export),
              "proposals_sha256": store.sha256(proposals_path)}
    content = store_train.keywords_file(boxes, hashes, store_train.reviewed_on(tasks) or date.today().isoformat(),
                                        added, v)
    out.write_text(json.dumps(content, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {len(content['photos'])} photos to {out}\n")
    print(markdown_table(count_rows(boxes), left=("label",)))
    print(f"\n{sum(map(len, added.values()))} date boxes added on {len(added)} photos (dates {v}'s labels "
          "missed), kept in added_dates: " + (", ".join(f"{c} {n}" for c, n in sorted(
              Counter(b['label'] for bs in added.values() for b in bs).items())) or "none"))
    proposed = {p: proposals["photos"][p]["boxes"] for p in boxes}
    counts = store_train.ocr_help(proposed, boxes)
    print(f"\nOCR's {sum(map(len, proposed.values()))} proposals against the reviewed boxes "
          "(by region id; relabelled is among moved):\n")
    print(markdown_table(help_rows(counts), left=("",)))
    bands = store_train.overlap_bands(proposed, boxes)
    print(f"\nkept or moved, by IoU with the final box: "
          + ", ".join(f"{b} {bands[b]}" for b in ("0.7+", "0.5-0.7", "under 0.5")))


if __name__ == "__main__":
    main()
