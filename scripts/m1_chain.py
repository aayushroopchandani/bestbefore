"""Every M1 training, back to back, inside the one GPU job (slurm/m1_chain.sh).

    python scripts/m1_chain.py            # what the GPU job runs
    python scripts/m1_chain.py --dry-run  # print the plan, train nothing

Order (docs/m1-date-detector.md, Parts A3, C and D1):
  1. smoke   a 1-epoch run on 2% of the photos: proves the GPU, the data and
             the code work before hours of training are spent
  2. C1      yolo11n / yolo11s at 640 / 1280 px, synthetic + real
  3.         the C1 decision, by the rule written before the runs
  4. C2      the C1 winner's settings, real photos only
  5.         the C2 decision, by the rule written before the runs
  6. D1      the final setup again with seeds 1 and 2 (seed 0 is already done)

It is resumable: a run whose val.json exists is skipped, so if the job's
time limit cuts it short, resubmitting carries on where it stopped. The
decisions go to results/runs/m1-decisions.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from bestbefore.m1 import BOTH, C1, REAL, decide_c1, keep_synthetic, map50_date
from bestbefore.runs import RUNS_DIR, latest_run


def val_scores(name: str) -> dict | None:
    run = latest_run(f"m1-{name}")
    path = run / "val.json" if run else None
    return json.loads(path.read_text()) if path and path.exists() else None


def run(name: str, weights: str, imgsz: int, train: str, seed: int = 0,
        extra: tuple = (), dry: bool = False) -> dict | None:
    if (scores := val_scores(name)) is not None:
        print(f"[skip] {name}: already done", flush=True)
        return scores
    cmd = [sys.executable, "scripts/train_detector.py", "--weights", weights,
           "--imgsz", str(imgsz), "--train", train, "--seed", str(seed),
           "--name", name, *extra]
    print(f"[run]  {name}: {' '.join(cmd[1:])}", flush=True)
    if dry:
        return None
    subprocess.run(cmd, check=True)
    return val_scores(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    dry = parser.parse_args().dry_run

    run("smoke", "yolo11n", 640, BOTH, extra=("--epochs", "1", "--fraction", "0.02"), dry=dry)

    c1 = {name: run(name, w, size, BOTH, dry=dry) for name, (w, size) in C1.items()}
    if dry:
        print("[then] C1 decision, C2 (winner, real only), C2 decision, D1 seeds 1 and 2")
        return
    winner = decide_c1(c1)
    weights, imgsz = C1[winner]

    real_only = run(f"{winner}-real", weights, imgsz, REAL)
    keep_synth = keep_synthetic(c1[winner], real_only)
    final, train = (winner, BOTH) if keep_synth else (f"{winner}-real", REAL)

    decisions = {
        "c1": {n: map50_date(s) for n, s in c1.items()},
        "c1_winner": winner,
        "c2": {"with_synthetic": map50_date(c1[winner]), "real_only": map50_date(real_only)},
        "c2_keep_synthetic": keep_synth,
        "final": final,
        "m1_detector": final + " (seed 0)",
        "m1_threshold": c1[winner]["threshold"] if keep_synth else real_only["threshold"],
    }
    (RUNS_DIR / "m1-decisions.json").write_text(json.dumps(decisions, indent=2))
    print(json.dumps(decisions, indent=2), flush=True)

    for seed in (1, 2):
        run(f"{final}-seed{seed}", weights, imgsz, train, seed=seed)
    print("[done] M1 chain finished", flush=True)


if __name__ == "__main__":
    main()
