"""Freeze the store test set: the labels plus a SHA-256 of every photo (B3).

    python scripts/freeze_store_test_set.py --version v1
    python scripts/freeze_store_test_set.py --version v2 --why "sample 12: expiry misread"

Writes data/store/test_set.json (never committed) and
results/store_test_set.md (committed: hashes and counts only), then prints
the line to paste into src/bestbefore/store.py. From then on every load
checks the file and every photo against it.

A frozen version is never overwritten. A label change after freezing is a
new version, with a line saying what changed and why, and every baseline is
rerun on it (docs/m0-data-and-test-set.md, B3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date

from bestbefore import store
from bestbefore.config import REPO_ROOT
from bestbefore.m1 import markdown_table

SUMMARY = REPO_ROOT / "results" / "store_test_set.md"
HISTORY = "## History\n\n"


def summary(test_set: dict, digest: str, history: list[str]) -> str:
    return (
        "# Store test set\n\n"
        "The retailer's photos, labelled by hand in M0\n"
        "([plan](../docs/m0-data-and-test-set.md), Part B). The photos and the\n"
        "labels never enter git. This file records what was frozen, so any change\n"
        "shows: `store.read_test_set()` refuses a file or photo that differs.\n\n"
        f"- **Version:** {test_set['version']}, frozen {test_set['frozen']}\n"
        f"- **Photos:** {len(test_set['photos'])}\n"
        f"- **SHA-256 of `data/store/test_set.json`:** `{digest}`\n\n"
        "## Counts\n\n"
        + markdown_table(store.tag_counts(test_set["photos"]))
        + "\n\nEach photo has the first tag that applies, in the order above.\n"
        "`small-print` vs `clean` comes from the expiry box: under 1.5% of the\n"
        "photo's long side is small print.\n\n"
        + HISTORY + "\n".join(history) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", required=True, help="e.g. v1")
    parser.add_argument("--why", help="what changed since the last version (from v2 on)")
    args = parser.parse_args()

    if store.TEST_SET_PATH.exists():
        frozen = json.loads(store.TEST_SET_PATH.read_text())["version"]
        if frozen == args.version:
            raise SystemExit(f"{args.version} is already frozen. A change is a new version.")
        if not args.why:
            raise SystemExit("a new version needs --why: what changed and why")

    test_set = store.freeze(store.read_labels(), args.version, date.today().isoformat())
    data = (json.dumps(test_set, indent=1, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    store.TEST_SET_PATH.write_bytes(data)

    old = SUMMARY.read_text().split(HISTORY, 1)[1].strip().splitlines() if SUMMARY.exists() else []
    line = f"- **{args.version}** ({test_set['frozen']}): {args.why or 'first version.'}"
    SUMMARY.write_text(summary(test_set, digest, old + [line]))

    print(f"froze {len(test_set['photos'])} photos as {args.version}: {store.TEST_SET_PATH}")
    print(f"wrote {SUMMARY}")
    print(f'\nnow set, in src/bestbefore/store.py:\nTEST_SET_SHA256 = "{digest}"')


if __name__ == "__main__":
    main()
