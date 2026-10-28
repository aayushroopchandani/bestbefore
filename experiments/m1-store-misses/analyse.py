"""Which store expiry dates does the M1 detector miss: by date format and
by print size?

    python experiments/m1-store-misses/analyse.py

Reads the frozen store test set and the per-photo predictions that
`scripts/eval_detector.py --set store` saved for the three n640 seeds in
data/store/runs/. Prints counts only, never a photo name, so the output
can go into the README.

Sideways dates are left out (no seed finds any; that's M2's job), and so
are other-face photos (no expiry in frame). Each expiry date is counted
once per seed, so the percentages are means over the three seeds.
"""
from __future__ import annotations

import json
from collections import defaultdict

from bestbefore import store
from bestbefore.config import store_photos_dir
from bestbefore.detection_eval import Prediction, match_photo
from bestbefore.images import upright_size
from bestbefore.runs import latest_run
from bestbefore.samples import DATE_LABELS
from bestbefore.store_import import SMALL_PRINT, print_height
from bestbefore.tables import markdown_table

SEEDS = ("n640", "n640-seed1", "n640-seed2")
SIZES = ((SMALL_PRINT, "under 1.5%"), (0.025, "1.5 to 2.5%"), (1.0, "2.5% and up"))


def load_seed(name: str) -> tuple[float, dict]:
    run = latest_run(f"detector-store-{name}", store.RUNS_DIR)
    if run is None:
        raise SystemExit(f"run scripts/eval_detector.py --set store --runs {name} first")
    threshold = json.loads((run / "run.json").read_text())["config"]["threshold"]
    return threshold, json.loads((run / "predictions.json").read_text())


def groups_of(label: dict, size: tuple[int, int]) -> list[str]:
    kind = "month as a word" if "MON" in (label["format"] or "").upper() else "numbers only"
    height = print_height(label, size) or 0.0
    band = next(name for top, name in SIZES if height < top)
    return [f"format: {kind}", f"print: {band}", f"{kind}, print {band}"]


def main() -> None:
    seeds = [load_seed(name) for name in SEEDS]
    root = store_photos_dir()
    counts = defaultdict(lambda: {"n": 0, "found": 0, "boxed": 0})
    formats = defaultdict(lambda: {"month as a word": 0, "numbers only": 0})
    for photo, label in store.labels(store.V1).items():
        if label["tag"] in ("other-face", "sideways"):
            continue
        truths = [b for b in label["boxes"] if b["label"] in DATE_LABELS]
        expiry = [i for i, b in enumerate(truths) if b["label"] == "expiry"]
        if not expiry:
            continue        # a derived expiry: nothing printed to find
        keys = groups_of(label, upright_size(root / photo))
        formats[label["tag"]][keys[0].removeprefix("format: ")] += 1
        for threshold, boxes in seeds:
            preds = [Prediction("date", tuple(b[:4]), b[4]) for b in boxes.get(photo, [])]
            xyxy = [b["xyxy"] for b in truths]
            _, at_threshold = match_photo([p for p in preds if p.conf >= threshold], xyxy, 0.5)
            _, at_any = match_photo(preds, xyxy, 0.5)
            for key in keys:
                counts[key]["n"] += len(expiry)
                counts[key]["found"] += sum(at_threshold[i] is not None for i in expiry)
                counts[key]["boxed"] += sum(at_any[i] is not None for i in expiry)

    rows = [{"expiry dates": key, "n": str(c["n"] // len(seeds)),
             "found at threshold": f"{100 * c['found'] / c['n']:.0f}%",
             "boxed at any confidence": f"{100 * c['boxed'] / c['n']:.0f}%"}
            for key, c in sorted(counts.items())]
    print(markdown_table(rows, left=("expiry dates",)))
    print("\nPhotos with a printed expiry, by failure tag and format:\n")
    print(markdown_table([{"tag": t, **{k: str(v) for k, v in formats[t].items()}}
                          for t in store.TAGS if t in formats], left=("tag",)))


if __name__ == "__main__":
    main()
