"""Turn a store test set's Label Studio export into its labels.json, checking it first.

    python scripts/import_store_labels.py                  # v1 (B2)
    python scripts/import_store_labels.py --version v2     # v2 (Part D, D6)

Reads the set's label_studio_export.json (Label Studio's JSON-MIN export:
data/store/ for v1, data/store/v2/ for v2), runs the B2 checks in
docs/m0-data-and-test-set.md, and writes the set's labels.json only when
every check passes. `small-print` vs `clean` is then set from the expiry
box (store_import.SMALL_PRINT), not taken from the hand tag. Check 4,
agreement with Scanify's ground truth, is for v1 only. For v2, check 1
compares with its frozen photo list, which must still match the photos.
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


def expected_photos(store_set: store.StoreSet, root: Path) -> tuple[list[str], str]:
    """The photos the set's export must label, and where that list comes
    from: the frozen photo list for a set that has one (v2), else its folders."""
    if store_set.photo_list:
        return (sorted(store.read_photo_list(store_set, root)["photos"]),
                f"store test set {store_set.version}'s frozen photo list")
    return store.photos_on_disk(store_set, root), "$STORE_PHOTOS_DIR"


def run_checks(store_set: store.StoreSet, tasks: list[dict], root: Path
               ) -> tuple[dict[str, list[str]], dict[str, dict], dict[str, tuple[int, int]]]:
    """The B2 checks by name, the labels and the upright size of each labelled photo."""
    on_disk, where = expected_photos(store_set, root)
    labels = imp.from_export(tasks)
    upright = {p: upright_size(root / p) for p in labels.keys() & set(on_disk)}

    checks = {
        "export well formed: known tags and labels, upright non-empty boxes":
            imp.check_export(tasks),
        f"1. all {len(on_disk)} photos labelled once, no extras":
            imp.check_photos([imp.photo_id(t["image"]) for t in tasks], on_disk, where),
        "2. dates parse, expiry years 2024 to 2035, mfg before expiry":
            imp.check_dates(labels),
        "3. expiry, tag and boxes agree":
            imp.check_consistency(labels),
    }
    if store_set.scanify_check:
        truth = imp.read_scanify_truth(scanify_dir() / imp.SCANIFY_TRUTH)
        checks[f"4. the {len(truth)} expiries in Scanify's ground truth match"] = \
            imp.check_scanify(labels, truth)
    checks["5. boxes drawn on the upright photo"] = \
        imp.check_orientation(imp.drawn_sizes(tasks), upright)
    return checks, labels, upright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", choices=store.SETS, default="v1",
                        help="which store test set (default: v1)")
    parser.add_argument("--export", type=Path, help="default: the set's label_studio_export.json")
    parser.add_argument("--out", type=Path, help="default: the set's labels.json")
    args = parser.parse_args()

    store_set = store.SETS[args.version]
    export, out = args.export or store_set.export_path, args.out or store_set.labels_path
    tasks = json.loads(export.read_text())
    checks, labels, upright = run_checks(store_set, tasks, store_photos_dir())
    if not store_set.scanify_check:
        print(f"store test set {store_set.version}: check 4, agreement with Scanify's "
              "ground truth, is for v1 only\n")
    for name, problems in checks.items():
        report(name, problems)
    failed = sum(bool(p) for p in checks.values())
    if failed:
        raise SystemExit(f"\n{failed} checks failed, so {out} was not written. "
                         "Fix the labels in Label Studio, export again and rerun.")

    labels, moved = imp.apply_print_rule(labels, upright)
    print(f"✓ 6. small-print rule (expiry under {100 * imp.SMALL_PRINT:.1f}% of the "
          f"long side): {', '.join(f'{k}: {n}' for k, n in moved.items()) or 'no tag changed'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {len(labels)} photos to {out}\n")
    print(markdown_table(store.tag_counts(labels), left=("tag",)))


if __name__ == "__main__":
    main()
