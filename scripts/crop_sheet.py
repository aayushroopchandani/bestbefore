"""A sheet of crops from one pile, to look at by eye (docs/m3-date-reader.md, S3).

    python scripts/crop_sheet.py --pile gen-sample               # 64 crops, seed 0
    python scripts/crop_sheet.py --pile real-val --n 64 --root DIR --out DIR/real-val.png

Each tile is a stored crop as the reader gets it (upright in a training
pile, as it lies in a suite), shown at TILE_H px high with its box in red
and, under it, its text and what it is: class or role, the turn it was
given (t) and the turn still needed to stand it upright (u). The crops are
drawn at random with a fixed seed, shown in key order. Sheets show ExpDate
and generated crops: they go to $GENERATED_DIR/m3/crops/sheets/ by
default, never into git. A store pile's sheet (piles.PRIVATE) shows the
retailer's photos: it goes into the pile's own owner-only folder
(<root>/<pile>/sheets/, mode 600), or under data/store/, nowhere else.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from bestbefore import store
from bestbefore.config import generated_dir
from bestbefore.reader import piles, shards
from bestbefore.sheets import FONT, rows_of

TILE_H = 80             # the crop's height on the sheet
TILE_W_MAX = 420        # a long crop is squeezed to this width
CAPTION_H = 34


def tile(crop: shards.Crop) -> Image.Image:
    image = crop.image()
    scale = TILE_H / image.height
    w = max(40, min(TILE_W_MAX, round(image.width * scale)))
    sx = w / image.width
    shown = image.resize((w, TILE_H), Image.Resampling.LANCZOS)
    x1, y1, x2, y2 = crop.box
    ImageDraw.Draw(shown).rectangle((x1 * sx, y1 * scale, x2 * sx - 1, y2 * scale - 1), outline=(230, 30, 30))
    out = Image.new("RGB", (max(w, 140), TILE_H + CAPTION_H), "white")
    out.paste(shown, (0, 0))
    what = crop.role or crop.cls or "-"
    draw = ImageDraw.Draw(out)
    draw.text((2, TILE_H + 2), crop.text[:40], fill="black", font=FONT)
    draw.text((2, TILE_H + 18), f"{what} t{crop.turn} u{crop.tags.get('upright')} #{crop.key.rsplit('/', 1)[1].lstrip('0') or 0}",
              fill=(90, 90, 90), font=FONT)
    return out


def sheet(folder: Path, pile: str, n: int, seed: int) -> Image.Image:
    env = shards.open_pile(folder)
    try:
        total = shards.count(env)
        picks = sorted(random.Random(seed).sample(range(total), min(n, total)))
        tiles = [tile(shards.get(env, pile, i)) for i in picks]
    finally:
        env.close()
    return rows_of(tiles, width=1800)


def private_out_problem(out: Path, folder: Path) -> str | None:
    """Why a store pile's sheet can't go to `out`, or None: only inside the
    pile's folder or under data/store/."""
    out = out.resolve()
    if out.is_relative_to(folder.resolve()) or out.is_relative_to(store.DATA_DIR.resolve()):
        return None
    return f"{out}: a store pile's sheet goes into {folder} or under {store.DATA_DIR}, nowhere else"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pile", required=True)
    parser.add_argument("--root", type=Path, help="where the piles are (default $GENERATED_DIR/m3/crops)")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="the PNG (default <root>/sheets/<pile>.png)")
    args = parser.parse_args()
    root = args.root or generated_dir() / "m3" / "crops"
    private = args.pile in piles.PRIVATE
    out = args.out or (root / args.pile / "sheets" / f"{args.pile}-{args.seed}.png" if private
                       else root / "sheets" / f"{args.pile}.png")
    if private:
        if why := private_out_problem(out, root / args.pile):
            sys.exit(why)
        os.umask(0o077)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet(root / args.pile, args.pile, args.n, args.seed).save(out)
    print(out)


if __name__ == "__main__":
    main()
