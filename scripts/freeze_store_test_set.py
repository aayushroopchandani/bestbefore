"""Freeze a store test set: the labels plus a SHA-256 of every photo (B3),
and for v2, the photo list first (Part D).

    python scripts/freeze_store_test_set.py --version v2 --photos   # D3: before labelling
    python scripts/freeze_store_test_set.py --version v2            # D6: once imported

`--photos` writes the set's photos.json: every photo in its folders and
its SHA-256. Without it, the script writes the set's test_set.json: the
labels plus each photo's SHA-256 (for v2, only when the labels cover
exactly the frozen photo list, still unchanged). Neither file is
committed. results/store_test_set.md (v1) or results/store_test_set_v2.md
(committed: hashes and counts only) records what was frozen, and the
script prints the line to paste into src/bestbefore/store.py. From then on
every load checks the file and every photo against it.

A frozen file is never overwritten. v1 was frozen with `--version v1` on
2026-10-21; a label fix to a frozen set will be a revision of it (Part D,
Later).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

from bestbefore import store
from bestbefore.config import REPO_ROOT
from bestbefore.tables import markdown_table

SUMMARIES = {"v1": REPO_ROOT / "results" / "store_test_set.md",
             "v2": REPO_ROOT / "results" / "store_test_set_v2.md"}
HISTORY = "## History\n\n"
INTROS = {
    "v1": "# Store test set\n\n"
          "The retailer's photos, labelled by hand in M0\n"
          "([plan](../docs/m0-data-and-test-set.md), Part B). The photos and the\n"
          "labels never enter git. This file records what was frozen, so any change\n"
          "shows: `store.read_test_set()` refuses a file or photo that differs.\n\n",
    "v2": "# Store test set v2\n\n"
          "New photos from the retailer's stores, the test set once store photos are\n"
          "trained on ([plan](../docs/m0-data-and-test-set.md), Part D). The photos\n"
          "and the labels never enter git. This file records what was frozen, so any\n"
          "change shows: `store.read_photo_list()` and `store.read_frozen()` refuse a\n"
          "file or photo that differs.\n\n",
}


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def summary(store_set: store.StoreSet, history: list[str],
            photo_list: dict | None = None, test_set: dict | None = None,
            digest: str | None = None) -> str:
    """The committed record: hashes and counts, never a photo name."""
    text = INTROS[store_set.version]
    if photo_list is not None:
        text += (f"- **Photo list:** {len(photo_list['photos'])} photos, frozen {photo_list['frozen']}\n"
                 f"- **SHA-256 of `{_rel(store_set.photos_path)}`:** `{store_set.photos_sha256}`\n")
    if test_set is None:
        text += "- **Labels:** not frozen yet\n\n"
    else:
        text += (f"- **Version:** {test_set['version']}, frozen {test_set['frozen']}\n"
                 f"- **Photos:** {len(test_set['photos'])}\n"
                 f"- **SHA-256 of `{_rel(store_set.test_set_path)}`:** `{digest}`\n\n"
                 "## Counts\n\n"
                 + markdown_table(store.tag_counts(test_set["photos"]), left=("tag",))
                 + "\n\nEach photo has the first tag that applies, in the order above.\n"
                 "`small-print` vs `clean` comes from the expiry box: under 1.5% of the\n"
                 "photo's long side is small print.\n\n")
    return text + HISTORY + "\n".join(history) + "\n"


def write(path: Path, content: dict) -> str:
    """Write a frozen file and return its SHA-256."""
    data = (json.dumps(content, indent=1, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def history(path: Path) -> list[str]:
    return path.read_text().split(HISTORY, 1)[1].strip().splitlines() if path.exists() else []


def freeze_photos(store_set: store.StoreSet, today: str) -> None:
    v = store_set.version
    if not store_set.photo_list:
        raise SystemExit(f"{v} has no photo list of its own: its photos were frozen with its labels")
    if store_set.photos_path.exists():
        raise SystemExit(f"{v}'s photo list is already frozen: {store_set.photos_path}")
    photo_list = store.freeze_photos(store_set, today)
    if not photo_list["photos"]:
        raise SystemExit(f"no photos in {', '.join(store_set.folders)}")
    digest = write(store_set.photos_path, photo_list)
    line = f"- **{v} photo list** ({today}): {len(photo_list['photos'])} photos, before any labelling."
    summary_path = SUMMARIES[v]
    summary_path.write_text(summary(replace(store_set, photos_sha256=digest),
                                    history(summary_path) + [line], photo_list))
    print(f"froze {v}'s photo list, {len(photo_list['photos'])} photos: {store_set.photos_path}")
    print(f"wrote {summary_path}")
    print(f'\nnow set, in src/bestbefore/store.py:\nPHOTOS_{v.upper()}_SHA256 = "{digest}"')


def freeze_labels(store_set: store.StoreSet, today: str, why: str | None) -> None:
    v = store_set.version
    if store_set.test_set_path.exists():
        raise SystemExit(f"{v} is already frozen. A label fix is a revision (Part D, Later).")
    labels = store.read_labels(store_set.labels_path)
    photo_list = None
    if store_set.photo_list:
        photo_list = store.read_photo_list(store_set)
        if set(labels) != set(photo_list["photos"]):
            raise SystemExit(f"{store_set.labels_path} doesn't label exactly {v}'s frozen photo "
                             f"list: rerun scripts/import_store_labels.py --version {v}")
    test_set = store.freeze(labels, v, today)
    digest = write(store_set.test_set_path, test_set)
    summary_path = SUMMARIES[v]
    line = f"- **{v}** ({today}): {why or ('labels frozen.' if photo_list else 'first version.')}"
    summary_path.write_text(summary(store_set, history(summary_path) + [line],
                                    photo_list, test_set, digest))
    constant = "TEST_SET_SHA256" if store_set is store.V1 else f"TEST_SET_{v.upper()}_SHA256"
    print(f"froze {len(test_set['photos'])} photos as {v}: {store_set.test_set_path}")
    print(f"wrote {summary_path}")
    print(f'\nnow set, in src/bestbefore/store.py:\n{constant} = "{digest}"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", required=True, choices=store.SETS)
    parser.add_argument("--photos", action="store_true",
                        help="freeze the photo list, before labelling (v2 on)")
    parser.add_argument("--why", help="the history line, if not the default")
    args = parser.parse_args()

    store_set, today = store.SETS[args.version], date.today().isoformat()
    if args.photos:
        freeze_photos(store_set, today)
    else:
        freeze_labels(store_set, today, args.why)


if __name__ == "__main__":
    main()
