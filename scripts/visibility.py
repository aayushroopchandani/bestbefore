"""M3 S5: how visible each crop's print is (reader/data.py's `visibility`),
measured once per record, and the sheets the cut-off is chosen on
(docs/m3-date-reader.md, S5).

    python scripts/visibility.py measure --pile all --workers 24
    python scripts/visibility.py sheets --pile gen --pile expdate-real --pile expdate-synth
    python scripts/visibility.py check --cut 3.5

measure   every record of each pile, in key order, into <pile>/visibility.npy
          (float32) with <pile>/visibility.json (the measure's code hash, the
          pile's manifest hash, quantiles, timing). `all` is every pile S3
          and S4 made: training piles, suites and the store's (those written
          owner-only, umask 077). Only piles.VISIBILITY_FILTERED is ever
          filtered; a suite's measure is kept so it can be split by it.
sheets    per pile, two sheets: bands of the measure's value, a row each
          with its share of the pile and up to BAND_TILES crops drawn at
          random; and the pile's lowest LOWEST crops. Each tile is the
          stored crop with its box in red, a copy stretched to full
          contrast under it (to look hard), its value and its text.
check     at a cut-off: what each filtered pile loses (over STOP_SHARE
          means stop and look), a sheet of the dropped crops and one of the
          kept crops just above the cut-off (the next ABOVE_SHARE of the
          pile), and check-<cut>.json.

Sheets go to <root>/sheets/visibility/: ExpDate and generated crops only,
never a store pile's (they show the retailer's photos), never into git.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from bestbefore import runs
from bestbefore.config import generated_dir
from bestbefore.reader import data, piles, shards
from bestbefore.sheets import FONT

BANDS = (0, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 15, 1e9)
BAND_TILES = 10
LOWEST = 60
STOP_SHARE = 0.10           # S5: a pile losing more than this is looked at again before training
ABOVE_SHARE = 0.02
QUANTILES = (0.5, 1, 2, 5, 10, 25, 50, 75, 90)
TILE_H = 56
SHEET_W = 2000


def default_root() -> Path:
    return generated_dir() / "m3" / "crops"


def all_piles() -> tuple[str, ...]:
    return piles.JOB_PILES + piles.STORE_PILES


# --- Measuring ------------------------------------------------------------------------

def measure_range(args: tuple[Path, str, int, int]) -> np.ndarray:
    folder, pile, start, stop = args
    env = shards.open_pile(folder)
    try:
        return np.array([data.visibility(np.asarray(c.image()), c.box)
                         for c in shards.records(env, pile, start, stop)], np.float32)
    finally:
        env.close()


def measure(pile: str, root: Path, workers: int) -> dict:
    folder = root / pile
    env = shards.open_pile(folder)
    count = shards.count(env)
    env.close()
    step = max(1, min(5000, -(-count // max(1, workers * 4))))
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        parts = pool.map(measure_range, [(folder, pile, s, min(count, s + step)) for s in range(0, count, step)])
    values = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    seconds = time.perf_counter() - t0
    if len(values) != count:
        raise RuntimeError(f"{pile}: measured {len(values)} of {count}")
    record = {"pile": pile, "count": count, "code": data.visibility_code(),
              "manifest_sha256": piles.sha256_file(folder / "manifest.json"),
              "quantiles": {str(q): round(float(np.percentile(values, q)), 3) for q in QUANTILES},
              "min": round(float(values.min()), 3) if count else None,
              "seconds": round(seconds, 1), "workers": workers, "host": socket.gethostname(),
              "measured": time.strftime("%Y-%m-%d %H:%M:%S"), **runs.git_state()}
    tmp = folder / f".{data.VISIBILITY_FILE}.{os.getpid()}.npy"
    np.save(tmp, values)
    os.replace(tmp, folder / data.VISIBILITY_FILE)
    runs.write_json(folder / data.VISIBILITY_RECORD, record)
    return record


# --- Sheets -------------------------------------------------------------------------

def tile(crop: shards.Crop, value: float) -> Image.Image:
    image = crop.image()
    a = np.asarray(image).astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    stretched = Image.fromarray(np.clip((a - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8))
    scale = TILE_H / image.height
    w = max(40, min(320, round(image.width * scale)))
    shown = image.resize((w, TILE_H), Image.Resampling.LANCZOS)
    x1, y1, x2, y2 = crop.box
    sx = w / image.width
    ImageDraw.Draw(shown).rectangle((x1 * sx, y1 * scale, x2 * sx - 1, y2 * scale - 1), outline=(230, 30, 30))
    out = Image.new("RGB", (max(w, 120), 2 * TILE_H + 30), "white")
    out.paste(shown, (0, 0))
    out.paste(stretched.resize((w, TILE_H), Image.Resampling.LANCZOS), (0, TILE_H))
    draw = ImageDraw.Draw(out)
    draw.text((2, 2 * TILE_H + 1), f"{value:.2f}  #{crop.key.rsplit('/', 1)[1].lstrip('0') or 0}", fill="black", font=FONT)
    draw.text((2, 2 * TILE_H + 15), crop.text[:22], fill=(90, 90, 90), font=FONT)
    return out


def rows_sheet(rows: list[tuple[str, list[Image.Image]]]) -> Image.Image:
    """Rows of tiles, each row headed by its label, wrapped at SHEET_W."""
    label_w, gap, h = 170, 4, 2 * TILE_H + 30
    lines: list[tuple[str, list[Image.Image]]] = []
    for label, tiles in rows:
        line, x, first = [], label_w, True
        for t in tiles:
            if x + t.width > SHEET_W and line:
                lines.append((label if first else "", line))
                line, x, first = [], label_w, False
            line.append(t)
            x += t.width + gap
        lines.append((label if first else "", line))
    sheet = Image.new("RGB", (SHEET_W, len(lines) * (h + gap)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (label, line) in enumerate(lines):
        y = r * (h + gap)
        for k, part in enumerate(label.split("\n")):
            draw.text((4, y + 4 + 15 * k), part, fill="black", font=FONT)
        x = label_w
        for t in line:
            sheet.paste(t, (x, y))
            x += t.width + gap
        draw.line((0, y + h + 1, SHEET_W, y + h + 1), fill=(200, 200, 200))
    return sheet


def tiles_of(env, pile: str, values: np.ndarray, picks) -> list[Image.Image]:
    return [tile(shards.get(env, pile, int(i)), float(values[i])) for i in sorted(picks, key=lambda i: values[i])]


def sheets(pile: str, root: Path, out: Path, seed: int = 0) -> list[Path]:
    if pile in piles.PRIVATE:
        sys.exit(f"{pile}: a store pile is never drawn on these sheets")
    folder = root / pile
    env = shards.open_pile(folder)
    try:
        values = data.read_visibility(folder, shards.count(env))
        rng = random.Random(seed)
        rows = []
        below = 0
        for lo, hi in zip(BANDS, BANDS[1:]):
            idx = np.flatnonzero((values >= lo) & (values < hi))
            below += len(idx)
            top = "and up" if hi > 1e8 else f"to {hi:g}"
            label = (f"{lo:g} {top}\n{len(idx):,} crops, {len(idx) / len(values):.2%}\n"
                     f"under {hi:g}: {below / len(values):.2%}" if hi < 1e8 else
                     f"{lo:g} {top}\n{len(idx):,} crops, {len(idx) / len(values):.2%}")
            picks = rng.sample(list(idx), min(BAND_TILES, len(idx)))
            rows.append((label, tiles_of(env, pile, values, picks)))
        lowest = np.argsort(values, kind="stable")[:LOWEST]
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / f"{pile}-bands.png", out / f"{pile}-lowest.png"]
        rows_sheet(rows).save(paths[0])
        rows_sheet([(f"lowest {len(lowest)}\nof {len(values):,}", tiles_of(env, pile, values, lowest))]).save(paths[1])
    finally:
        env.close()
    return paths


def check(cut: float, pile_names: list[str], root: Path, out: Path, seed: int = 0) -> dict:
    result = {"cut": cut, "piles": {}, "stop": []}
    rng = random.Random(seed)
    for pile in pile_names:
        if pile not in piles.VISIBILITY_FILTERED:
            sys.exit(f"{pile} is never filtered (piles.VISIBILITY_FILTERED)")
        folder = root / pile
        env = shards.open_pile(folder)
        try:
            values = data.read_visibility(folder, shards.count(env))
            dropped = np.flatnonzero(values < cut)
            share = len(dropped) / len(values)
            kept_sorted = np.sort(values[values >= cut])
            upper = kept_sorted[min(len(kept_sorted) - 1, int(ABOVE_SHARE * len(values)))] if len(kept_sorted) else cut
            above = np.flatnonzero((values >= cut) & (values <= upper))
            result["piles"][pile] = {"records": len(values), "dropped": len(dropped), "share": round(share, 5),
                                     "kept_above_band": [cut, round(float(upper), 3)]}
            if share > STOP_SHARE:
                result["stop"].append(pile)
            out.mkdir(parents=True, exist_ok=True)
            for label, idx, name in ((f"dropped\n{len(dropped):,} ({share:.2%})", dropped, "dropped"),
                                     (f"kept, {cut:g} to {upper:.2f}\n(next {ABOVE_SHARE:.0%})", above, "kept-above")):
                picks = rng.sample(list(idx), min(64, len(idx)))
                rows_sheet([(label, tiles_of(env, pile, values, picks))]).save(out / f"{pile}-{name}-{cut:g}.png")
        finally:
            env.close()
    runs.write_json(out / f"check-{cut:g}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("task", choices=("measure", "sheets", "check"))
    parser.add_argument("--pile", action="append", help="a pile, or 'all' (measure); repeatable")
    parser.add_argument("--root", type=Path, help="where the piles are (default $GENERATED_DIR/m3/crops)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--cut", type=float, help="check: the cut-off to look at")
    args = parser.parse_args()
    root = args.root or default_root()
    out = root / "sheets" / "visibility"
    names = args.pile or (["all"] if args.task == "measure" else list(piles.VISIBILITY_FILTERED))
    if "all" in names:
        names = [p for p in all_piles() if (root / p / "data.mdb").exists()]
    if args.task == "measure":
        for pile in names:
            if pile in piles.PRIVATE:
                os.umask(0o077)         # the retailer's photos: owner-only (CLAUDE.md)
            record = measure(pile, root, args.workers)
            print(f"{pile}: {record['count']:,} measured in {record['seconds']} s; "
                  f"quantiles {record['quantiles']}", flush=True)
    elif args.task == "sheets":
        for pile in names:
            for path in sheets(pile, root, out):
                print(path, flush=True)
    else:
        if args.cut is None:
            sys.exit("check needs --cut")
        result = check(args.cut, names, root, out)
        for pile, r in result["piles"].items():
            print(f"{pile}: drops {r['dropped']:,} of {r['records']:,} ({r['share']:.2%})")
        if result["stop"]:
            print(f"STOP: over {STOP_SHARE:.0%} dropped from {', '.join(result['stop'])}: look again before training")


if __name__ == "__main__":
    main()
