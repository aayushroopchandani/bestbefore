"""Freeze the store pack designs, then split store set v1 by design (M2b S4),
and store set v3 the same way beside it (S5b).

    python scripts/store_designs.py             # check data/store/designs.csv; writes nothing
    python scripts/store_designs.py --freeze    # write data/store/designs.json, print DESIGNS_SHA256
    python scripts/store_designs.py --report    # split v1, write results/store_designs.md
    python scripts/store_designs.py --version v3 [--freeze | --report]   # v3's: designs_v3.csv,
        designs_v3.json, DESIGNS_V3_SHA256; its split in its own section of the report

v3's ids are frozen apart from v1's and v2's, which don't change: a pack
v1 or v2 already has must carry exactly its id (the freeze refuses one
typed another way). A v3 design that v1 has keeps v1's side of the split;
the rest are split by the same rule (store_train.split_v3). Its section
is written after v1's, which stays byte for byte as it was.

designs.csv holds one `photo,design` row per photo of store sets v1 and
v2, named by looking at the photos (docs/m2b-store-and-size.md, "Design
ids"; drafted by Claude and approved by the owner, S4's record). The
freeze needs both sets' labels frozen, every photo of each exactly once
and no id empty, and never overwrites designs.json. The report splits v1
with store_train.split_v1 (seed 0) and writes counts and hashes only;
problems name photos, so they're printed on screen only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import date

from bestbefore import store, store_train
from bestbefore.config import REPO_ROOT
from bestbefore.tables import markdown_table

SUMMARY = REPO_ROOT / "results" / "store_designs.md"
MIN_VAL_EXPIRIES = 20       # S4's check: fewer is a finding, not a reason to change the rule


def frozen_sets() -> dict[str, dict[str, dict]]:
    """v1's and v2's frozen labels; the designs are typed against both."""
    return {s.version: store.read_frozen(s)["photos"] for s in (store.V1, store.V2)}


def expiries(label: dict) -> int:
    """Expiry dates in frame: the expiry boxes, as M2's 81 on v1."""
    return sum(b["label"] == "expiry" for b in label["boxes"])


def check(rows: list[list[str]], photos: list[str]) -> list[str]:
    problems = store_train.check_designs(rows, photos)
    empty = [p for p in problems if ": no design for " in p]
    typed = {row[1] for row in rows if len(row) == 2 and row[1]}
    print(f"{len(photos) - len(empty)} of {len(photos)} photos have a design; {len(typed)} designs")
    if empty:
        print(f"  {len(empty)} rows with no design yet")
    for p in problems:
        if p not in empty:
            print(f"  {p}")
    return problems


def freeze(today: str) -> None:
    if store_train.DESIGNS_PATH.exists():
        raise SystemExit(f"the designs are already frozen: {store_train.DESIGNS_PATH}")
    sets = frozen_sets()
    rows = store_train.read_designs_csv()
    if check(rows, sorted(p for labels in sets.values() for p in labels)):
        raise SystemExit(f"\n{store_train.DESIGNS_PATH} was not written: fix designs.csv and rerun.")
    content = store_train.freeze(rows, {"v1": store.TEST_SET_SHA256, "v2": store.TEST_SET_V2_SHA256}, today)
    data = (json.dumps(content, indent=1, sort_keys=True) + "\n").encode()
    store_train.DESIGNS_PATH.write_bytes(data)
    print(f"\nfroze {len(content['designs'])} photos' designs: {store_train.DESIGNS_PATH}")
    print(f'\nnow set, in src/bestbefore/store.py:\nDESIGNS_SHA256 = "{hashlib.sha256(data).hexdigest()}"')


def design_rows(designs: dict[str, str], sets: dict[str, dict[str, dict]],
                split: dict[str, str]) -> list[dict[str, str]]:
    """Designs, photos and expiries in v1 and v2, v2 by whether its design
    is trained on (in v1-train), only in v1-val, or not in v1 at all."""
    v1, v2 = sets["v1"], sets["v2"]
    train = {designs[p] for p in v1 if split[p] == "train"}
    in_v1 = {designs[p] for p in v1}

    def row(name: str, labels: dict[str, dict]) -> dict[str, str]:
        return {"": name, "designs": f"{len({designs[p] for p in labels})}",
                "photos": f"{len(labels)}", "expiries in frame": f"{sum(map(expiries, labels.values()))}"}

    def part(test) -> dict[str, dict]:
        return {p: lab for p, lab in v2.items() if test(designs[p])}

    return [row("v1", v1), row("v2", v2),
            row("v2, design in v1-train (seen)", part(lambda d: d in train)),
            row("v2, design in v1-val only", part(lambda d: d in in_v1 - train)),
            row("v2, design not in v1 (unseen)", part(lambda d: d not in in_v1)),
            row("designs in both, their v1 and v2 photos", {p: lab for p, lab in {**v1, **v2}.items()
                                    if designs[p] in in_v1 & {designs[q] for q in v2}})]


def concentration(designs: dict[str, str], labels: dict[str, dict]) -> dict[str, str]:
    """How few designs hold a set's expiries; no design is named."""
    per = Counter()
    for p, lab in labels.items():
        per[designs[p]] += expiries(lab)
    photos = sorted(Counter(designs[p] for p in labels).values())
    top = sorted(per.values(), reverse=True)
    return {"photos per design (median, most)": f"{statistics.median(photos):g}, {photos[-1]}",
            "designs with one photo": f"{sum(n == 1 for n in photos)}",
            "expiries: largest design, largest 4": f"{top[0]}, {sum(top[:4])} of {sum(top)}",
            "designs with an expiry in frame": f"{sum(n > 0 for n in top)}"}


def split_rows(designs: dict[str, str], labels: dict[str, dict], split: dict[str, str]
               ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The split by design tag (how it was stratified) and by photo tag
    (how results are read): designs, photos, expiries and dates per part."""
    photos_of: dict[str, list[str]] = {}
    for p in labels:
        photos_of.setdefault(designs[p], []).append(p)
    tag_of = {d: store_train.design_tag([labels[p]["tag"] for p in ps]) for d, ps in photos_of.items()}
    by_design = []
    for tag in (*store.TAGS, "**all**"):
        ds = [d for d in photos_of if tag in (tag_of[d], "**all**")]
        n_val = sum(split[photos_of[d][0]] == "val" for d in ds)
        by_design.append({"design tag": tag, "designs": f"{len(ds)}",
                          "train": f"{len(ds) - n_val}", "val": f"{n_val}"})
    by_photo = []
    for tag in (*store.TAGS, "**all**"):
        row = {"photo tag": tag}
        for part in ("train", "val"):
            group = [lab for p, lab in labels.items() if split[p] == part and tag in (lab["tag"], "**all**")]
            row[f"photos {part}"] = f"{len(group)}"
            row[f"expiries {part}"] = f"{sum(map(expiries, group))}"
            row[f"dates {part}"] = f"{sum(len(lab['boxes']) for lab in group)}"
        by_photo.append(row)
    return by_design, by_photo


def summary(designs: dict[str, str], sets: dict[str, dict[str, dict]], split: dict[str, str],
            frozen: str, digest: str) -> str:
    by_design, by_photo = split_rows(designs, sets["v1"], split)
    val = sum(expiries(lab) for p, lab in sets["v1"].items() if split[p] == "val")
    verdict = (f"at least the {MIN_VAL_EXPIRIES} S4's check asks for" if val >= MIN_VAL_EXPIRIES else
               f"**fewer than the {MIN_VAL_EXPIRIES} S4's check asks for**: a finding,\n"
               "recorded in the plan; the rule doesn't change")
    return (
        "# Store pack designs and the v1 split\n\n"
        "Pack design ids for every photo of store sets v1 and v2, named by looking\n"
        "at the photos (drafted by Claude, approved by the owner), and store set v1\n"
        "split by design into training and validation\n"
        "([plan](../docs/m2b-store-and-size.md), \"Design ids\" and \"The split\").\n"
        "The ids and the split stay in `data/store/`: this file holds counts and\n"
        "hashes only, never a photo or a design's name.\n\n"
        f"- **Designs:** {len(designs)} photos, frozen {frozen}\n"
        f"- **SHA-256 of `data/store/designs.json`:** `{store.DESIGNS_SHA256}`\n"
        f"- **Typed against:** v1's labels `{store.TEST_SET_SHA256[:8]}…`, "
        f"v2's `{store.TEST_SET_V2_SHA256[:8]}…`\n"
        f"- **The split:** `store_train.split_v1`, seed {store_train.SEED}, SHA-256 `{digest}`\n\n"
        "## Designs\n\n"
        + markdown_table(design_rows(designs, sets, split), left=("",))
        + "\n\nExpiries in frame are expiry boxes. Seen means trained on: a v2 photo\n"
        "whose design has photos in v1-train.\n\n"
        + markdown_table([{"": v, **concentration(designs, sets[v])} for v in ("v1", "v2")], left=("",))
        + "\n\n## The v1 split\n\n"
        "Each design takes its photos' commonest tag (`other-face` only if every\n"
        "photo is); within each tag, 30% of the designs go to validation.\n\n"
        + markdown_table(by_design, left=("design tag",))
        + "\n\nBy each photo's own tag, the way results are reported:\n\n"
        + markdown_table(by_photo, left=("photo tag",))
        + "\n\nDates are every labelled date box (expiry, mfg and other).\n\n"
        f"Validation holds {val} expiries in frame, {verdict}.\n")


V3_SECTION = "\n## Store set v3\n"


def freeze_v3(today: str) -> None:
    if store_train.DESIGNS_V3_PATH.exists():
        raise SystemExit(f"v3's designs are already frozen: {store_train.DESIGNS_V3_PATH}")
    labels = store.read_frozen(store.V3)["photos"]
    rows = store_train.read_designs_csv(store_train.DESIGNS_V3_CSV)
    problems = store_train.check_designs(rows, sorted(labels), set(store_train.read_designs().values()), "v3")
    for p in problems:
        print(f"  {p}")
    if problems:
        raise SystemExit(f"\n{store_train.DESIGNS_V3_PATH} was not written: fix designs_v3.csv and rerun.")
    content = store_train.freeze(rows, {"v3": store.LABELS_V3_SHA256, "designs": store.DESIGNS_SHA256}, today)
    data = (json.dumps(content, indent=1, sort_keys=True) + "\n").encode()
    store_train.DESIGNS_V3_PATH.write_bytes(data)
    print(f"froze {len(content['designs'])} photos' designs: {store_train.DESIGNS_V3_PATH}")
    print(f'\nnow set, in src/bestbefore/store.py:\nDESIGNS_V3_SHA256 = "{hashlib.sha256(data).hexdigest()}"')


def summary_v3(designs_v3: dict[str, str], labels: dict[str, dict], designs: dict[str, str],
               split_v1: dict[str, str], split: dict[str, str], frozen: str, digest: str) -> str:
    """v3's section: its designs by where they're known from, and its split;
    counts and hashes only."""
    train = {designs[p] for p, part in split_v1.items() if part == "train"}
    in_v1 = {designs[p] for p in split_v1}
    in_v2 = {d for p, d in designs.items() if p.split("/")[0] in store.V2.folders}

    def row(name: str, keep) -> dict[str, str]:
        group = {p: lab for p, lab in labels.items() if keep(designs_v3[p])}
        return {"": name, "designs": f"{len({designs_v3[p] for p in group})}", "photos": f"{len(group)}",
                "expiries in frame": f"{sum(map(expiries, group.values()))}"}

    by_design, by_photo = split_rows(designs_v3, labels, split)
    val = sum(expiries(lab) for p, lab in labels.items() if split[p] == "val")
    return (
        V3_SECTION + "\n"
        "Store set v3's photos ([plan](../docs/m2b-store-and-size.md), S5b), training data\n"
        "beside v1. Their ids are frozen apart from v1's and v2's, which don't change.\n"
        "A design v1 has keeps its side of v1's split; v3's other designs are split\n"
        "among themselves by the same rule.\n\n"
        f"- **Designs:** {len(designs_v3)} photos, frozen {frozen}\n"
        f"- **SHA-256 of `data/store/designs_v3.json`:** `{store.DESIGNS_V3_SHA256}`\n"
        f"- **Typed against:** v3's labels `{store.LABELS_V3_SHA256[:8]}…`, "
        f"v1's and v2's designs `{store.DESIGNS_SHA256[:8]}…`\n"
        f"- **The split:** `store_train.split_v3`, seed {store_train.SEED}, SHA-256 `{digest}`\n\n"
        + markdown_table([row("v3", lambda d: True),
                          row("design in v1-train (trains)", lambda d: d in train),
                          row("design in v1-val only (validates)", lambda d: d in in_v1 - train),
                          row("design in v2, not v1", lambda d: d in in_v2 - in_v1),
                          row("design new to v3", lambda d: d not in in_v1 | in_v2)], left=("",))
        + "\n\nA v3 design that v2 also has becomes seen in training for v2's report.\n\n"
        + markdown_table(by_design, left=("design tag",))
        + "\n\nBy each photo's own tag:\n\n"
        + markdown_table(by_photo, left=("photo tag",))
        + f"\n\nv3's validation part holds {val} expiries in frame.\n")


def report_v3() -> None:
    if store.DESIGNS_V3_SHA256 is None:
        raise SystemExit("v3's designs aren't frozen yet: --version v3 --freeze, then set DESIGNS_V3_SHA256")
    designs_v3 = store_train.read_designs(store_train.DESIGNS_V3_PATH, store.DESIGNS_V3_SHA256)
    labels = store.read_frozen(store.V3)["photos"]
    designs, split_v1 = store_train.read_designs(), store_train.v1_split()
    split = store_train.split_v3(designs_v3, labels, designs, split_v1, store_train.SEED)
    digest = store_train.split_digest(split)
    if store_train.SPLIT_V3_SHA256 not in (None, digest):
        raise SystemExit("v3's split differs from the one recorded: SPLIT_V3_SHA256 in store_train.py")
    frozen = json.loads(store_train.DESIGNS_V3_PATH.read_text())["frozen"]
    v1_part = SUMMARY.read_text().split(V3_SECTION, 1)[0]
    SUMMARY.write_text(v1_part + summary_v3(designs_v3, labels, designs, split_v1, split, frozen, digest))
    parts = Counter(split.values())
    print(f"v3 split, seed {store_train.SEED}: {parts['train']} photos train, {parts['val']} val")
    print(f"wrote {SUMMARY} (v1's part unchanged)")
    if store_train.SPLIT_V3_SHA256 is None:
        print(f'\nnow set, in src/bestbefore/store_train.py:\nSPLIT_V3_SHA256 = "{digest}"')


def report() -> None:
    designs, sets = store_train.read_designs(), frozen_sets()
    split = store_train.split_v1(designs, sets["v1"], store_train.SEED)
    digest = store_train.split_digest(split)
    if store_train.SPLIT_SHA256 not in (None, digest):
        raise SystemExit("v1's split differs from the one recorded: SPLIT_SHA256 in store_train.py")
    frozen = json.loads(store_train.DESIGNS_PATH.read_text())["frozen"]
    SUMMARY.write_text(summary(designs, sets, split, frozen, digest))
    parts = Counter(split.values())
    print(f"v1 split, seed {store_train.SEED}: {parts['train']} photos train, {parts['val']} val")
    print(f"wrote {SUMMARY}")
    if store_train.SPLIT_SHA256 is None:
        print(f'\nnow set, in src/bestbefore/store_train.py:\nSPLIT_SHA256 = "{digest}"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", choices=("v1", "v3"), default="v1",
                        help="v1: v1's and v2's designs (S4); v3: v3's (S5b)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true", help="write the designs' frozen file")
    mode.add_argument("--report", action="store_true", help="split, write results/store_designs.md")
    args = parser.parse_args()
    if args.version == "v3":
        if args.freeze:
            freeze_v3(date.today().isoformat())
        elif args.report:
            report_v3()
        else:
            problems = store_train.check_designs(store_train.read_designs_csv(store_train.DESIGNS_V3_CSV),
                                                 sorted(store.read_frozen(store.V3)["photos"]),
                                                 set(store_train.read_designs().values()), "v3")
            print("\n".join(problems) or "designs_v3.csv: every v3 photo once, no id typed two ways")
    elif args.freeze:
        freeze(date.today().isoformat())
    elif args.report:
        report()
    else:
        sets = frozen_sets()
        check(store_train.read_designs_csv(), sorted(p for labels in sets.values() for p in labels))


if __name__ == "__main__":
    main()
