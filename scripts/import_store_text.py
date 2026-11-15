"""Turn the reviewed store-text project ("Store text (M3 S4)") into
data/store/text.json, checking it first (docs/m3-date-reader.md, S4).

    python scripts/import_store_text.py             # the checks, then data/store/text.json
    python scripts/import_store_text.py --dry-run   # the checks only

Reads Label Studio's full JSON export of the project
(data/store/text/export.json: the JSON export, which keeps region ids),
the OCR proposals the tasks were made from (proposals.json there) and the
frozen files the store export was made from (store_train.export_inputs,
each checked against its hash). Writes data/store/text.json, mode 600,
only when every check passes:

  1. every exported photo (v1 and v3, training and validation) reviewed:
     one submitted annotation, no skip, no draft, no other photo;
  2. every date box an expiry, mfg or other_date box and every life box a
     life box: upright, on the upright photo, inside it;
  3. every date box of the export there once, unchanged, none added, and
     every box (date or life) with one text, not blank, and one turn;
  4. every expiry and mfg text reads as its frozen value, and every
     derived photo's shelf life plus its frozen mfg date gives its frozen
     expiry, or the owner has settled the mismatch in
     data/store/text/settled.json ({key: {"reason": ...}}); a settled key
     that no longer mismatches fails too. Mismatches are listed with their
     reason, never fixed here.

Notes (a turn other than OCR's, a character outside decision 10's set, an
unreadable `#`) are printed, not refused. The file is dated by the
review's last change, so it rebuilds byte for byte from the same export.
Its SHA-256 is printed: pasted into store_text.TEXT_SHA256, it freezes the
file. Everything printed names photos; nothing here is committed. Never v2.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

from bestbefore import store, store_text, store_text_review as review, store_train
from bestbefore.config import store_photos_dir, store_train_dir
from bestbefore.images import upright_size
from bestbefore.tables import markdown_table

SETTLED = "settled.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="run the checks, write nothing")
    args = parser.parse_args()

    folder, root, export_dir = store_text.REVIEW_DIR, store_photos_dir(), store_train_dir()
    export, proposals_path, settled_path = folder / "export.json", folder / "proposals.json", folder / SETTLED
    tasks = json.loads(export.read_text())
    sets, designs, split, sources = store_train.export_inputs()
    labels = {p: lab for given in sets.values() for p, lab in given["labels"].items()}
    plan, planned, values, months, upright = [], {}, {}, {}, {}
    for p in store_train.export_plan(sets, designs, split):
        first = next(iter(store_train.FOLDERS[p.part]))
        p = store_train.without_slips(p, upright_size(export_dir / "images" / first / f"{p.name}.jpg"))
        plan.append(p)
        values[p.photo] = store_text.box_values(p, labels[p.photo])
        dated = [b for b in p.boxes if b.label in store.BOX_LABELS]
        planned[p.photo] = [{"index": v["index"], "label": v["label"], "xyxy": list(b.xyxy)}
                            for b, v in zip(dated, values[p.photo], strict=True)]
        months[p.photo] = store_text.photo_month(root / p.photo)
        upright[p.photo] = upright_size(root / p.photo)
    proposed = json.loads(proposals_path.read_text())
    proposals = {p.photo: {int(k): v for k, v in proposed["photos"][p.name].items()} for p in plan}

    reviewed, read_problems = review.read_review(tasks, planned)
    found = review.mismatches(reviewed, values, labels) if not read_problems else []
    settled = json.loads(settled_path.read_text()) if settled_path.exists() else {}
    left, stale = review.unsettled(found, settled)
    checks = {
        f"1. all {len(plan)} exported photos reviewed: one submitted annotation each, no skip, no draft":
            store_train.check_reviewed(tasks, [p.photo for p in plan], "the store export"),
        "2. date boxes expiry, mfg or other_date, life boxes life: upright, on the upright photo, inside it":
            store_train.check_keyword_boxes(tasks, upright, review.DATE_CONTROL, store.BOX_LABELS)
            + store_train.check_keyword_boxes(tasks, upright, review.LIFE_CONTROL, (store_text.LIFE,)),
        "3. every date box there once and unchanged, none added; one text and one turn on every box":
            read_problems,
        f"4. every expiry and mfg text reads as its frozen value, every shelf life gives its expiry "
        f"({len(found)} mismatches, {len(found) - len(left)} settled)":
            [f"{m['key']} ({m['label']}): {m['text']!r} against {m['value']}: {m['reason']}" for m in left]
            + [f"{k}: settled in {SETTLED} but no longer a mismatch; take it out" for k in stale],
    }
    for name, problems in checks.items():
        print(f"{'✗' if problems else '✓'} {name}" + (f": {len(problems)}" if problems else ""))
        for p in problems:
            print(f"    {p}")
    for m in found:
        if m["key"] in settled:
            print(f"  settled: {m['key']} {m['text']!r} against {m['value']}: {settled[m['key']]['reason']}")
    noted = review.notes(reviewed, planned, proposals) if not read_problems else {}
    for kind, items in noted.items():
        print(f"note, {kind}: {len(items)}")
        for i in items:
            print(f"    {i}")
    if failed := sum(bool(p) for p in checks.values()):
        raise SystemExit(f"\n{failed} checks failed, so {store_text.TEXT_PATH} was not written.")
    if args.dry_run:
        print("\ndry run: nothing written")
        return

    hashes = {"export_sha256": store.sha256(export), "proposals_sha256": store.sha256(proposals_path),
              "settled_sha256": store.sha256(settled_path) if settled_path.exists() else None}
    text = review.text_file(plan, reviewed, values, months, sources, hashes,
                            store_train.reviewed_on(tasks) or date.today().isoformat())
    digest = store_text.write_private(store_text.TEXT_PATH, text)
    life = sum(len(r["life"]) for r in reviewed.values())
    print(f"\nwrote {len(text['photos'])} photos, "
          f"{sum(len(e['boxes']) for e in text['photos'].values())} date boxes and {life} shelf-life boxes "
          f"to {store_text.TEXT_PATH}\nsha256 {digest}  (paste into store_text.TEXT_SHA256 to freeze it)\n")
    counts = review.ocr_help(reviewed, planned, proposals)
    rows = [{"label": label, **{k: f"{counts[label, k]}" for k in
                                ("text kept", "text changed", "turn kept", "turn changed")}}
            for label in store.BOX_LABELS]
    print("OCR's proposals against the reviewed text and turns:\n")
    print(markdown_table(rows, left=("label",)))
    turns = {c: sum(r["turn"] == k for e in reviewed.values() for r in [*e["boxes"].values(), *e["life"]])
             for c, k in store_text.TURNS.items()}
    print("\nturns (date and life boxes): " + ", ".join(f"{c} {n}" for c, n in turns.items()))
    print(f"shelf-life boxes: {life}, on {sum(bool(r['life']) for r in reviewed.values())} photos "
          f"({sum(bool(r['life']) and labels[p]['derived'] for p, r in reviewed.items())} derived)")


if __name__ == "__main__":
    main()
