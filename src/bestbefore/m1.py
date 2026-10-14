"""M1's experiment grid and decision rules, fixed before any run
(docs/m1-date-detector.md, C1 and C2). scripts/m1_chain.py applies them."""
from __future__ import annotations

# C1: run name -> (weights, input size). Listed cheapest first, which is
# the order the tie-break uses: smaller model first, then smaller input.
C1 = {
    "n640": ("yolo11n", 640),
    "n1280": ("yolo11n", 1280),
    "s640": ("yolo11s", 640),
    "s1280": ("yolo11s", 1280),
}
BOTH = "train_synth,train_real"
REAL = "train_real"
NOISE = 0.01   # 1 point of mAP@0.5: smaller differences are treated as noise


def map50_date(scores: dict) -> float:
    return scores["map50"]["date"]


def decide_c1(scores: dict[str, dict]) -> str:
    """Highest validation mAP@0.5 for dates; within NOISE of the best, the
    cheapest run (the first in C1's order)."""
    best = max(map50_date(s) for s in scores.values())
    return next(n for n in C1 if n in scores and map50_date(scores[n]) >= best - NOISE)


def keep_synthetic(with_synth: dict, real_only: dict) -> bool:
    """Keep synthetic data unless real-only is better by more than NOISE."""
    return not map50_date(real_only) > map50_date(with_synth) + NOISE
