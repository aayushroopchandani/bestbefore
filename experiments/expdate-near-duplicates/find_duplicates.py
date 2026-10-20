"""Do the same photos, or the same packs, turn up in two ExpDate splits?

    python experiments/expdate-near-duplicates/find_duplicates.py           # print
    python experiments/expdate-near-duplicates/find_duplicates.py --write   # and update README.md

A photo in validation or test that also sits, nearly unchanged, in a
training split makes that score too optimistic: the model is partly
recognising a photo it has seen. Four checks, strictest first:

  exact       the same file bytes (SHA-256)
  near        the same photo resized or recompressed: a 256-bit difference
              hash (dHash) at most NEAR_BITS bits apart
  reused      a synthetic photo drawn on top of the real one: the same size,
              and most pixels still match after the fake dates and shapes
              were added (a dHash misses the heavily edited ones)
  same date   a different photo that may show the same product: the two
              share a printed date text

Each check gives candidates, not verdicts. The contact sheets written to
results/runs/ (ExpDate images, never committed) are where they are
confirmed by eye, and the README records what the eye check found.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bestbefore import expdate, yolo
from bestbefore.config import expdate_dir
from bestbefore.images import load_upright
from bestbefore.runs import start_run
from bestbefore.samples import DATE_LABELS

README = Path(__file__).with_name("README.md")
HASH = 16           # dHash grid: 16 x 16 = 256 bits
NEAR_BITS = 25      # about 10% of the bits; set before looking at any result
SMALL = 64          # side of the grey thumbnail compared pixel by pixel
PIXEL_TOL = 0.25    # a pixel "matches" within this many standard deviations
REUSED = 0.5        # share of matching pixels that flags a reused photo
SHEET_PAIRS = 24    # pairs per contact sheet

# (name, query group, reference groups). Train vs synthetic isn't checked:
# both are training data, so a photo shared between them leaks nothing.
COMPARISONS = (
    ("validation vs train", "val", ("train",)),
    ("test vs train + validation", "test", ("train", "val")),
    ("validation vs synthetic", "val", ("synth",)),
    ("test vs synthetic", "test", ("synth",)),
)

# Bits set in each byte value, to count differing bits after XOR.
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


def fingerprints(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """SHA-256 of the file, 256-bit dHash, and a 64 x 64 grey thumbnail.

    dHash: shrink to 17 x 16 grey pixels and record, for each pixel,
    whether it is brighter than its right-hand neighbour. Resizing or
    recompressing a photo barely changes these 256 yes/no answers.
    """
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    grey = load_upright(path).convert("L")
    a = np.asarray(grey.resize((HASH + 1, HASH), Image.Resampling.LANCZOS), dtype=np.int16)
    small = np.asarray(grey.resize((SMALL, SMALL), Image.Resampling.BILINEAR), dtype=np.uint8)
    return sha, np.packbits(a[:, 1:] > a[:, :-1]), small


def distances(q: np.ndarray, ref: np.ndarray, chunk: int = 32) -> np.ndarray:
    """Differing bits between every query hash and every reference hash."""
    out = np.empty((len(q), len(ref)), dtype=np.uint16)
    for i in range(0, len(q), chunk):
        x = q[i:i + chunk, None, :] ^ ref[None, :, :]
        out[i:i + chunk] = POPCOUNT[x].sum(-1, dtype=np.uint16)
    return out


def pixel_match(a: np.ndarray, b: np.ndarray) -> float:
    """Share of thumbnail pixels that agree once each photo's brightness
    and contrast are evened out, so a darkened copy still matches."""
    za = (a - a.mean()) / (a.std() + 1e-6)
    zb = (b - b.mean()) / (b.std() + 1e-6)
    return float((np.abs(za - zb) < PIXEL_TOL).mean())


def date_texts(sample) -> set[str]:
    return {re.sub(r"\s+", "", b.text).upper() for b in sample.boxes
            if b.label in DATE_LABELS and b.text}


def load_groups() -> dict[str, list]:
    val_ids = yolo.list_ids(expdate_dir() / "yolo", "val")
    real_train = expdate.load_split("real-train")
    return {
        "train": [s for s in real_train if s.id not in val_ids],
        "val": [s for s in real_train if s.id in val_ids],
        "test": expdate.load_split("real-test"),
        "synth": expdate.load_split("synth"),
    }


def check(name, queries, refs, fp, dist):
    """One comparison: the summary row, and the candidate pairs to draw as
    (score label, query, reference, caption), best first."""
    ref_shas = {fp[r.image][0] for r in refs}
    exact = sum(fp[q.image][0] in ref_shas for q in queries)
    nearest = dist.argmin(1)
    best = dist[np.arange(len(queries)), nearest]
    by_hash = sorted(((int(best[i]), q, refs[nearest[i]]) for i, q in enumerate(queries)),
                     key=lambda t: t[0])
    pairs = {"closest": [(f"{d} bits", q, r, "") for d, q, r in by_hash[:SHEET_PAIRS]]}

    # Reused: same size, then pixels. Synthetic photos keep the size of the
    # photo they were drawn on, so only same-size pairs are compared.
    by_size = defaultdict(list)
    for r in refs:
        by_size[(r.width, r.height)].append(r)
    reused = []
    for q in queries:
        for r in by_size.get((q.width, q.height), ()):
            reused.append((pixel_match(fp[q.image][2], fp[r.image][2]), q, r))
    reused.sort(key=lambda t: -t[0])
    pairs["reused"] = [(f"{m:.0%} pixels", q, r, "") for m, q, r in reused[:SHEET_PAIRS]]

    by_text = defaultdict(list)             # date text -> reference indices
    for j, r in enumerate(refs):
        for t in date_texts(r):
            by_text[t].append(j)
    same_date = []
    for i, q in enumerate(queries):
        cands = {(j, t) for t in date_texts(q) for j in by_text.get(t, ())}
        if cands:
            j, t = min(cands, key=lambda c: (dist[i, c[0]], c[1]))
            same_date.append((int(dist[i, j]), q, refs[j], t))
    same_date.sort(key=lambda t: t[0])
    pairs["same-date"] = [(f"{d} bits", q, r, f"shared date: {t}") for d, q, r, t in same_date]

    row = {
        "check": name,
        "photos checked": len(queries),
        "exact copies": exact,
        f"near copies (≤ {NEAR_BITS} bits)": int((best <= NEAR_BITS).sum()),
        "nearest photo, bits apart: min / median": f"{best.min()} / {int(np.median(best))}",
        f"reused (≥ {REUSED:.0%} pixels match)": len({q.id for m, q, _ in reused if m >= REUSED}),
        "best pixel match": f"{reused[0][0]:.0%}" if reused else "no same-size pair",
        "share a date text": len(same_date),
    }
    return row, pairs


def sheet(pairs, path: Path) -> None:
    """Each pair side by side: query on the left, its match on the right."""
    tw, th, cap = 300, 300, 34
    font = ImageFont.load_default(size=14)
    cols = 3
    rows = -(-len(pairs) // cols)
    out = Image.new("RGB", (cols * (2 * tw + 20), rows * (th + cap)), "white")
    draw = ImageDraw.Draw(out)
    for k, (score, q, r, caption) in enumerate(pairs):
        x0, y0 = (k % cols) * (2 * tw + 20), (k // cols) * (th + cap)
        for side, s in enumerate((q, r)):
            im = load_upright(s.image)
            im.thumbnail((tw - 4, th - 4))
            out.paste(im, (x0 + side * tw + (tw - im.width) // 2, y0))
        draw.text((x0 + 4, y0 + th + 2), f"{score}  {q.id} | {r.id}", fill="black", font=font)
        if caption:
            draw.text((x0 + 4, y0 + th + 17), caption, fill="black", font=font)
    out.save(path, quality=85)


def table(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|---|" + "---:|" * (len(keys) - 1)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="update README.md")
    args = parser.parse_args()

    groups = load_groups()
    run_dir = start_run("expdate-near-duplicates",
                        {"hash": f"dHash {HASH}x{HASH}", "near_bits": NEAR_BITS,
                         "small": SMALL, "pixel_tol": PIXEL_TOL, "reused": REUSED})
    paths = [s.image for g in groups.values() for s in g]
    with Pool() as pool:
        fp = dict(zip(paths, pool.map(fingerprints, paths, chunksize=64)))
    hashes = {g: np.stack([fp[s.image][1] for s in ss]) for g, ss in groups.items()}

    rows = []
    for k, (name, qg, rgs) in enumerate(COMPARISONS, 1):
        refs = [s for g in rgs for s in groups[g]]
        dist = distances(hashes[qg], np.concatenate([hashes[g] for g in rgs]))
        row, pairs = check(name, groups[qg], refs, fp, dist)
        rows.append(row)
        for kind, ps in pairs.items():
            for page in range(0, len(ps), SHEET_PAIRS):
                sheet(ps[page:page + SHEET_PAIRS],
                      run_dir / f"{k}-{kind}-{page // SHEET_PAIRS + 1}.jpg")

    md = table(rows)
    print(md)
    print(run_dir)
    (run_dir / "summary.md").write_text(md + "\n")
    if args.write:
        text = README.read_text()
        start, end = "<!-- results:start -->", "<!-- results:end -->"
        head, rest = text.split(start)
        README.write_text(head + start + "\n" + md + "\n" + end + rest.split(end)[1])


if __name__ == "__main__":
    main()
