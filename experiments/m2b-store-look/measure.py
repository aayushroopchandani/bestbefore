"""What the store's expiry dates look like to the M2 detector, and how big
they are next to generated dates (docs/m2b-store-and-size.md, S3).

    python experiments/m2b-store-look/measure.py

Reads store set v1 (store.V1; never v2), the per-photo predictions that
`scripts/eval_detector.py --set store` kept for M2's three `gen-rot-1280`
seeds in data/store/runs/, real-val, gen-sample and the laptop's 5,000
generated photos (gen50k's tasks 0 to 4). Nothing is trained or scored
again. Each seed's expiries are matched with `store_gallery.match`, the
matching S8's counts use, and checked against that seed's store.json.

Prints markdown tables of counts, medians and percentiles only, never a
photo or a photo name, so the output can go into the README as it is.
Other-face photos are left out: their expiry isn't in frame.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from store_gallery import match                     # noqa: E402  the same matching as S8's counts

from bestbefore import expdate, m2, store             # noqa: E402
from bestbefore.images import load_upright            # noqa: E402
from bestbefore.m2_results import FULL_NAME, cause_of, private_predictions   # noqa: E402
from bestbefore.samples import DATE_LABELS            # noqa: E402
from bestbefore.sets import load_set                  # noqa: E402
from bestbefore.tables import markdown_table          # noqa: E402

SEEDS = ("m2-gen-rot-1280-s0", "m2-gen-rot-1280-s1", "m2-gen-rot-1280-s2")   # s0 is the M2 detector
OUTCOMES = ("kept", "doubted", "never boxed")
TEXT_PX = 32            # each crop resized so its short side is this many pixels, for sharpness
FAINT = 40              # contrast under this is faint print
BANDS = ((0.015, "under 1.5%"), (0.025, "1.5–2.5%"), (np.inf, "2.5% and up"))
GEN_TASKS = range(5)    # the generated photos the laptop holds (M2 S6's trial); task 0 is 1,000 photos


def look(gray: np.ndarray, box) -> dict | None:
    """Contrast (Otsu split of luminance: the bright side's mean minus the
    dark side's, 0-255) and sharpness (std of the Laplacian at a fixed text
    height, over the contrast, so it measures edges, not ink strength)
    inside one 0..1 box."""
    H, W = gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in (box[0] * W, box[1] * H, box[2] * W, box[3] * H))
    crop = gray[max(0, y1):y2, max(0, x1):x2]
    if crop.size < 16:
        return None
    t, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = crop[crop <= t], crop[crop > t]
    contrast = float(hi.mean() - lo.mean()) if lo.size and hi.size else 0.0
    short = min(crop.shape)
    scale = TEXT_PX / short
    small = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    lap = cv2.Laplacian(small.astype(np.float32), cv2.CV_32F)
    return {"contrast": contrast, "sharpness": float(lap.std()) / max(contrast, 1.0), "short_px": short}


def size(box, W: int, H: int) -> float:
    """A box's short side over its photo's long side: the store's small-print
    measure, made turn-neutral (for a sideways date the short side is its
    print height)."""
    return min((box[2] - box[0]) * W, (box[3] - box[1]) * H) / max(W, H)


def gray(sample) -> np.ndarray:
    return np.asarray(load_upright(sample.image).convert("L"), dtype=np.uint8)


def med(values, digits: int = 2) -> str:
    """Median [lower quartile–upper quartile]."""
    v = np.asarray(list(values), dtype=float)
    if not v.size:
        return "–"
    return f"{np.median(v):.{digits}f} [{np.percentile(v, 25):.{digits}f}–{np.percentile(v, 75):.{digits}f}]"


def counts(values) -> str:
    return ", ".join(f"{k} {v}" for k, v in Counter(values).most_common())


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "–"


def band_row(name: str, sizes) -> dict[str, str]:
    v = np.asarray(sizes)
    row = {"dates": name, "n": f"{len(v):,}"}
    lo = 0.0
    for top, label in BANDS:
        row[label] = f"{100 * np.mean((v >= lo) & (v < top)):.1f}%"
        lo = top
    row["median"] = f"{100 * np.median(v):.2f}%"
    return row


# --- The store photos and the three seeds ------------------------------------

def store_dates() -> list[dict]:
    """Every labelled date on store set v1 outside other-face photos, with
    its box, role, size and each seed's outcome and matched confidence."""
    samples = [s for s in store.load_samples(store.V1) if s.tag != "other-face"]
    if any(s.id.split("/")[0] not in store.V1.folders for s in samples):
        raise SystemExit("a photo outside store set v1's folders: stopping")
    labels = store.labels(store.V1)
    seeds = {}
    for run in SEEDS:
        kept = private_predictions(run)
        if kept is None:
            raise SystemExit(f"run scripts/eval_detector.py --set store --runs {run} first")
        seeds[run] = kept
    rows = []
    for s in samples:
        lab = labels[s.id]
        dated = [b for b in s.boxes if b.label in DATE_LABELS]
        truths = [b.xyxy for b in dated]
        per_seed = {}
        for run, (threshold, boxes) in seeds.items():
            preds = boxes.get(s.id, [])
            taken, _ = match(preds, truths)
            per_seed[run] = [(None if i is None else preds[i][4], threshold) for i in taken]
        image = gray(s) if any(b.label == "expiry" for b in dated) else None
        for j, b in enumerate(dated):
            year = {"expiry": lab["expiry"], "mfg": lab["mfg"]}.get(b.label)
            row = {"role": b.label, "tag": s.tag, "format": lab["format"] or "",
                   "cause": cause_of(lab, (s.width, s.height)) if b.label == "expiry" else None,
                   "year": int(year[:4]) if year else None,
                   "size": size(b.xyxy, s.width, s.height),
                   "tall": (b.xyxy[3] - b.xyxy[1]) * s.height > (b.xyxy[2] - b.xyxy[0]) * s.width,
                   "conf": {}, "outcome": {}}
            for run in SEEDS:
                conf, threshold = per_seed[run][j]
                row["conf"][run] = conf
                row["outcome"][run] = ("never boxed" if conf is None else
                                       "kept" if conf >= threshold else "doubted")
            if b.label == "expiry":
                row.update(look(image, b.xyxy) or {})
            rows.append(row)
    return rows


def check_seeds(expiries: list[dict]) -> list[dict]:
    """Each seed's outcome counts, which must add up to its store.json."""
    out = []
    for run in SEEDS:
        c = Counter(r["outcome"][run] for r in expiries)
        scores = m2.read_json(m2.finished_run(run.removeprefix("m2-")) / "store.json")
        found, boxed = scores["iou0.5"]["expiry_found"], scores["any_conf"]["expiry_found"]
        if (c["kept"], c["kept"] + c["doubted"], len(expiries)) != (found, boxed, scores["iou0.5"]["expiry_dates"]):
            raise SystemExit(f"{run}: the matching disagrees with store.json")
        out.append({"seed": run.removeprefix("m2-"), "threshold": f"{scores['threshold']:.3f}",
                    **{o: f"{c[o]}" for o in OUTCOMES}, "boxed at any confidence": f"{c['kept'] + c['doubted']}",
                    "store.json: found, boxed": f"{found}, {boxed}"})
    return out


def reference_looks(name: str) -> list[dict]:
    out = []
    for s in load_set(name):
        dated = [b for b in s.boxes if b.label in DATE_LABELS]
        if dated:
            image = gray(s)
            out += [v for v in (look(image, b.xyxy) for b in dated) if v]
    return out


def generated_sizes() -> tuple[list[float], dict[str, list[float]]]:
    """Generated dates' sizes: task 0's (1,000 photos), and every task's by
    orientation."""
    task0, by = [], defaultdict(list)
    for t in GEN_TASKS:
        for line in open(m2.gen_dir() / "sidecars" / "tasks" / f"{t:02d}.jsonl"):
            s = json.loads(line)
            W, H = s["size"]
            for blk in s["blocks"]:
                for tok in blk["tokens"]:
                    if tok["cls"] == "date" and tok.get("box"):
                        x1, y1, x2, y2 = tok["box"]
                        v = min(x2 - x1, y2 - y1) / max(W, H)
                        by[blk["geometry"]["orientation"]].append(v)
                        if t == 0:
                            task0.append(v)
    return task0, by


def expdate_years() -> Counter:
    """Years in ExpDate real-train's date transcriptions (real-val's photos
    included): the last 4-digit year, else the last 2-digit number."""
    years = Counter()
    for s in expdate.load_split("real-train"):
        for b in s.boxes:
            if b.text:
                m = re.findall(r"(?<!\d)(20\d\d)(?!\d)", b.text) or re.findall(r"(?<!\d)(\d\d)(?!\d)", b.text)[-1:]
                if m:
                    years[int(m[-1]) if len(m[-1]) == 4 else 2000 + int(m[-1])] += 1
    return years


# --- The report ---------------------------------------------------------------

def section(title: str, table: str, *notes: str) -> None:
    print(f"\n### {title}\n\n{table}")
    for n in notes:
        print(f"\n{n}")


def main() -> None:
    dates = store_dates()
    exp = [r for r in dates if r["role"] == "expiry"]
    s0 = SEEDS[0]
    print(f"Store set v1, other-face left out: {len(dates)} labelled dates, {len(exp)} of them expiries.")

    seeds = check_seeds(exp)
    found = [int(r["kept"]) for r in seeds]
    boxed = [int(r["boxed at any confidence"]) for r in seeds]
    section("1. Outcome per seed (store_gallery.match, each seed at its own threshold)",
            markdown_table(seeds, left=("seed",)),
            f"Mean of 3 seeds: kept {np.mean(found):.1f} ({min(found)}–{max(found)}), "
            f"boxed at any confidence {np.mean(boxed):.1f} ({min(boxed)}–{max(boxed)}).")

    kept_sharp = float(np.median([r["sharpness"] for r in exp if r["outcome"][s0] == "kept"]))
    refs = {"real-val": reference_looks("real-val"), "gen-sample": reference_looks("gen-sample")}
    look_rows = [{"expiries (s0's outcome)": o, "n": f"{len(g)}",
                  "contrast": med((r["contrast"] for r in g), 1), "sharpness": med(r["sharpness"] for r in g),
                  "short side, px": med((r["short_px"] for r in g), 1),
                  "short side, % of long side": med(100 * r["size"] for r in g),
                  f"faint (contrast < {FAINT})": f"{sum(r['contrast'] < FAINT for r in g)}"}
                 for o in OUTCOMES for g in [[r for r in exp if r["outcome"][s0] == o]]]
    look_rows += [{"expiries (s0's outcome)": name, "n": f"{len(v)}",
                   "contrast": med((x["contrast"] for x in v), 1), "sharpness": med(x["sharpness"] for x in v),
                   "short side, px": med((x["short_px"] for x in v), 1), "short side, % of long side": "–",
                   f"faint (contrast < {FAINT})": pct(sum(x["contrast"] < FAINT for x in v), len(v))}
                  for name, v in refs.items()]
    section("2. The look inside each date's box: median [quartiles]",
            markdown_table(look_rows, left=("expiries (s0's outcome)",)),
            "real-val and gen-sample rows are over every labelled date; their faint column is a share. "
            f"Soft below means sharpness under the kept expiries' median ({kept_sharp:.2f}); "
            f"real-val is soft for {pct(sum(x['sharpness'] < kept_sharp for x in refs['real-val']), len(refs['real-val']))} "
            f"of its dates, gen-sample for {pct(sum(x['sharpness'] < kept_sharp for x in refs['gen-sample']), len(refs['gen-sample']))}.")

    tag_rows = [{"tag": t, "expiries": f"{len(g)}", "tall boxes": f"{sum(r['tall'] for r in g)}",
                 f"faint (< {FAINT})": f"{sum(r['contrast'] < FAINT for r in g)}",
                 "soft": f"{sum(r['sharpness'] < kept_sharp for r in g)}",
                 "small (under 1.5%)": f"{sum(r['size'] < 0.015 for r in g)}",
                 "kept by s0": f"{sum(r['outcome'][s0] == 'kept' for r in g)}"}
                for t in store.TAGS for g in [[r for r in exp if r["tag"] == t]] if g]
    wide = [r for r in exp if r["tag"] == "sideways" and not r["tall"]]
    best = [max((c for c in r["conf"].values() if c is not None), default=0.0) for r in wide]
    section("3. By tag", markdown_table(tag_rows, left=("tag",)),
            f"Sideways expiries with wide boxes (turned 180°, not 90°): {len(wide)}. "
            f"The highest confidence any seed gives one: {max(best, default=0.0):.3f}. "
            f"At least as sharp as the kept median: {sum(r['sharpness'] >= kept_sharp for r in wide)}; "
            f"2.5% or more of the long side: {sum(r['size'] >= 0.025 for r in wide)}.")

    words = [r for r in exp if "MON" in r["format"].upper()]
    doubted = [r for r in exp if r["outcome"][s0] == "doubted"]
    word_rows = [{"expiries": name, "n": f"{len(g)}",
                  **{o: f"{sum(r['outcome'][s0] == o for r in g)}" for o in OUTCOMES}}
                 for name, g in (("month as a word", words),
                                 ("numbers only", [r for r in exp if "MON" not in r["format"].upper()]))]
    section("4. Month words, by s0's outcome", markdown_table(word_rows, left=("expiries",)),
            f"Of the {len(doubted)} doubted, {sum('MON' in r['format'].upper() for r in doubted)} "
            "have the month as a word.")

    outcome = lambda r: tuple(r["outcome"][run] for run in SEEDS)   # noqa: E731
    full = [r for r in exp if r["cause"] == FULL_NAME]
    seed_full = {run.removeprefix("m2-"): f"{sum(r['outcome'][run] == 'kept' for r in full)} of {len(full)}, "
                 f"median confidence {np.median([r['conf'][run] or 0.0 for r in full]):.3f}"
                 for run in SEEDS}
    agree = [{"expiries": "found by at least one seed", "n": f"{sum('kept' in outcome(r) for r in exp)}"},
             {"expiries": "found by all 3", "n": f"{sum(set(outcome(r)) == {'kept'} for r in exp)}"},
             {"expiries": "boxed by none (never boxed by all 3)",
              "n": f"{sum(set(outcome(r)) == {'never boxed'} for r in exp)}"}]
    section(f"5. The three seeds, over the {len(exp)} expiries", markdown_table(agree, left=("expiries",)),
            "Full month names (upright, M2 S8's cause), kept by each seed; an unboxed one counts as confidence 0: "
            + "; ".join(f"{k} {v}" for k, v in seed_full.items()) + ".")

    groups = defaultdict(list)
    for r in dates:
        band = "–" if r["year"] is None else ("2026 or earlier" if r["year"] <= 2026 else "2027 or later")
        side = "sideways" if r["tag"] == "sideways" else "upright"
        keys = [(r["role"], "all"), (f"{side} {r['role']}", "all")] + ([(r["role"], band)] if r["year"] else [])
        for key in keys:
            groups[key].append(r)
    role_rows = []
    for key in sorted(groups):
        g = groups[key]
        n = len(g) * len(SEEDS)
        role_rows.append({"dates": key[0], "years": key[1], "n": f"{len(g)}",
                          "found (seed-runs)": pct(sum(r["outcome"][run] == "kept" for r in g for run in SEEDS), n),
                          "boxed at any confidence": pct(sum(r["outcome"][run] != "never boxed"
                                                             for r in g for run in SEEDS), n)})
    early = [r for r in exp if r["year"] is not None and r["year"] <= 2026]
    years = expdate_years()
    section("6. Found rates by role and year, over the 3 seeds", markdown_table(role_rows, left=("dates", "years")),
            f"Expiries in 2026 or earlier: {len(early)}; by tag {counts(r['tag'] for r in early)}; "
            f"{sum('MON' not in r['format'].upper() for r in early)} numbers only; "
            f"by format {counts(r['format'] for r in early)}.",
            f"ExpDate real-train's date transcriptions with a year: {sum(years.values()):,}; 2020 to 2023: "
            f"{sum(v for y, v in years.items() if 2020 <= y <= 2023):,}; 2027 or later: "
            f"{sum(v for y, v in years.items() if y >= 2027)}.")

    task0, by = generated_sizes()
    every = [r["size"] for r in dates]
    size_rows = [band_row("store v1, expiry dates", [r["size"] for r in exp]),
                 band_row("store v1, every date", every),
                 band_row("generated (M2's spec, task 0: 1,000 photos)", task0),
                 band_row(f"generated (M2's spec, tasks 0–{GEN_TASKS[-1]}: 5,000 photos)",
                          [v for xs in by.values() for v in xs])]
    size_rows += [band_row(f"generated, {o} (5,000 photos)", by[o]) for o in ("upright", "tilted", "sideways")]
    q = (5, 10, 25, 50, 75, 90, 95)
    section("7. Date sizes: box short side over the photo's long side", markdown_table(size_rows, left=("dates",)),
            "Store v1, every date, percentiles " + ", ".join(f"{p}th {np.percentile(every, p) * 100:.2f}%" for p in q)
            + f"; smallest {min(every) * 100:.2f}%, largest {max(every) * 100:.2f}%.",
            f"Generated median over store median (task 0): {np.median(task0) / np.median(every):.2f} times every "
            f"store date's, {np.median(task0) / np.median([r['size'] for r in exp]):.2f} times the expiries'.")

    print("\nPending S4 (design ids): designs in v1, expiries per design.")


if __name__ == "__main__":
    main()
