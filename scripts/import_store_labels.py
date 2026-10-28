"""Turn the Label Studio export into data/store/labels.json, checking it first.

    python scripts/import_store_labels.py

Reads data/store/label_studio_export.json (Label Studio's JSON-MIN export),
runs the B2 checks in docs/m0-data-and-test-set.md, and writes labels.json
only when every check passes. `small-print` vs `clean` is then set from the
expiry box (store_import.SMALL_PRINT), not taken from the hand tag.
Problems name photos, so they're printed on screen only, never written to
a file in the repo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bestbefore import store
from bestbefore import store_import as imp
from bestbefore.config import scanify_dir, store_photos_dir
from bestbefore.images import upright_size
from bestbefore.tables import markdown_table


def report(name: str, problems: list[str]) -> None:
    print(f"{'✗' if problems else '✓'} {name}" + (f": {len(problems)}" if problems else ""))
    for p in problems:
        print(f"    {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--export", type=Path,
                        default=store.DATA_DIR / "label_studio_export.json")
    parser.add_argument("--out", type=Path, default=store.LABELS_PATH)
    args = parser.parse_args()

    tasks = json.loads(args.export.read_text())
    root = store_photos_dir()
    on_disk = store.photos_on_disk(store.V1, root)
    labels = imp.from_export(tasks)
    upright = {p: upright_size(root / p) for p in labels.keys() & set(on_disk)}
    truth = imp.read_scanify_truth(scanify_dir() / imp.SCANIFY_TRUTH)

    checks = {
        "export well formed: known tags and labels, upright non-empty boxes":
            imp.check_export(tasks),
        f"1. all {len(on_disk)} photos labelled once, no extras":
            imp.check_photos([imp.photo_id(t["image"]) for t in tasks], on_disk),
        "2. dates parse, expiry years 2024 to 2035, mfg before expiry":
            imp.check_dates(labels),
        "3. expiry, tag and boxes agree":
            imp.check_consistency(labels),
        f"4. the {len(truth)} expiries in Scanify's ground truth match":
            imp.check_scanify(labels, truth),
        "5. boxes drawn on the upright photo":
            imp.check_orientation(imp.drawn_sizes(tasks), upright),
    }
    for name, problems in checks.items():
        report(name, problems)
    failed = sum(bool(p) for p in checks.values())
    if failed:
        raise SystemExit(f"\n{failed} checks failed, so {args.out} was not written. "
                         "Fix the labels in Label Studio, export again and rerun.")

    labels, moved = imp.apply_print_rule(labels, upright)
    print(f"✓ 6. small-print rule (expiry under {100 * imp.SMALL_PRINT:.1f}% of the "
          f"long side): {', '.join(f'{k}: {n}' for k, n in moved.items()) or 'no tag changed'}")
    args.out.write_text(json.dumps(labels, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {len(labels)} photos to {args.out}\n")
    print(markdown_table(store.tag_counts(labels), left=("tag",)))


if __name__ == "__main__":
    main()
