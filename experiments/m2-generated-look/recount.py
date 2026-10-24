"""Recount S5's look check: which generated blocks lie on the pack?

    python experiments/m2-generated-look/recount.py

on_pack.csv holds one row per generated block on the 50 gen-sample photos
of S5's contact sheet (contact_sheet.py --set gen-sample --n 50, seed 0):
the photo, the block's index in its sidecar line, its place in pixels
(x, y, w, h) and whether it lies mostly on the pack, judged by eye. Each
place is checked against the gen-sample sidecar first, so the labels are
known to belong to the photos on disk. Then, counts only:

- blocks on and off the pack;
- what an edge margin keeps: blocks whose centre is at least 0.15 of the
  photo's short side from every edge (S5's record);
- nearness to the background's labelled dates: blocks whose centre lies
  within 0.2 or 0.15 of the photo's long side of a `date` box (this
  script's reading of S5's measure; see the README).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from bestbefore import m2
from bestbefore.tables import markdown_table

LABELS = Path(__file__).with_name("on_pack.csv")
EDGE = 0.15              # of the short side
NEAR = (0.2, 0.15)       # of the long side


def load() -> list[dict]:
    """Every labelled block: on the pack or not, its edge distance and its
    distance to the nearest labelled date."""
    with m2.sidecar_path("gen_sample").open() as f:
        sidecar = {p["image"]: p for p in map(json.loads, f)}
    with LABELS.open() as f:
        rows = list(csv.DictReader(f))
    photos = {row["image"] for row in rows}
    if len(rows) != sum(len(sidecar[p]["blocks"]) for p in photos):
        raise SystemExit("on_pack.csv doesn't hold every block of its photos")
    blocks = []
    for row in rows:
        photo = sidecar[row["image"]]
        place = [int(row[k]) for k in ("x", "y", "w", "h")]
        if photo["blocks"][int(row["block"])]["place"] != place:
            raise SystemExit(f"{row['image']}, block {row['block']}: not where the sidecar "
                             "puts it, so the labels are for another gen-sample")
        (w, h), (x, y, bw, bh) = photo["size"], place
        cx, cy = x + bw / 2, y + bh / 2
        dates = [(x1 * w, y1 * h, x2 * w, y2 * h)
                 for label, x1, y1, x2, y2 in photo["background"]["boxes"] if label == "date"]
        blocks.append({
            "photo": row["image"], "on": row["on_pack"] == "yes",
            "edge": min(cx, cy, w - cx, h - cy) / min(w, h),
            "near": min((math.hypot(max(x1 - cx, 0, cx - x2), max(y1 - cy, 0, cy - y2)) / max(w, h)
                         for x1, y1, x2, y2 in dates), default=math.inf)})
    return blocks


def row(name: str, blocks: list[dict]) -> dict[str, str]:
    on = sum(b["on"] for b in blocks)
    off = len(blocks) - on
    return {"blocks": name, "n": str(len(blocks)), "on the pack": str(on),
            "off the pack": str(off), "off": f"{100 * off / len(blocks):.0f}%"}


def main() -> None:
    blocks = load()
    rows = [row(f"all, on {len({b['photo'] for b in blocks})} photos", blocks)]
    rows.append(row(f"centre at least {EDGE} of the short side from every edge",
                    [b for b in blocks if b["edge"] >= EDGE]))
    rows += [row(f"centre within {cut} of the long side of a labelled date",
                 [b for b in blocks if b["near"] <= cut]) for cut in NEAR]
    print(markdown_table(rows, left=("blocks",)))
    on = [b for b in blocks if b["on"]]
    lost = sum(b["edge"] < EDGE for b in on)
    print(f"\nThe edge margin would give up {lost} of the {len(on)} on-pack places "
          f"({100 * lost / len(on):.0f}%).")


if __name__ == "__main__":
    main()
