"""M3: train a CTC reader (docs/m3-date-reader.md, "CRNN + CTC", S5;
"PP-OCRv6 small", S6), or time its loader.

    python scripts/train_reader.py --reader crnn --arm all --seed 0                 # a run (the chain's, a GPU)
    python scripts/train_reader.py --reader crnn --arm all --seed 0 --smoke --device cpu --threads 22
    python scripts/train_reader.py --reader ppocrv6 --arm all --seed 0              # the fine-tune (S6's PyTorch route)
    python scripts/train_reader.py --reader ppocrv6 --smoke --device cpu --threads 22 --no-amp
    python scripts/train_reader.py --loader-only --arm all --workers 16 --batches 300

A run trains on its arm's mix (reader/data.py), scores m3.EVALS checkpoints
on real-val and synth-val-v2, keeps the best, and ends by exporting it to
ONNX and reading real-val both ways through PyTorch and ONNX Runtime
(onnx_run.PARITY_MIN). Its record goes to results/runs/<time>-<name>/run.json:
config, commit, seed, the crop recipe, each pile's manifest and measure
hashes, PARSE_SHA256, VISIBLE_MIN, crops a second, every eval: counts only,
never a crop, photo or design name. Its checkpoints and ONNX file go to
$CHECKPOINT_DIR/m3/<time>-<name>/.

--smoke is a CPU smoke on `all`, 2 loader workers. The CRNN's (S5): 2,000
steps at batch 64; it passes when the loss falls (the last tenth of the
steps under half the first tenth's) and the ONNX parity holds. PP-OCRv6's
(S6): m3.ROUTE_STEPS steps of its recipe (batch 128, lr 5e-5), 4 evals;
it passes when the CTC loss on m3.PROBE_CROPS crops of the mix is lower
after the steps than before (the route's third condition) and the parity
holds. A pretrained model's training loss needn't halve in 200 steps.

--loader-only times the training stream alone, as the GPU job will run it
(--workers loader workers, no model), after a warm-up, and LMDB's random
reads of the largest pile in this process: the base of the GPU estimate.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from bestbefore import m3, runs
from bestbefore.config import checkpoint_dir, generated_dir
from bestbefore.reader import data, piles, shards, train

SMOKE = {"crnn": {"budget": 128_000, "batch": 64, "workers": 2, "log_every": 50},        # 2,000 steps
         "ppocrv6": {"budget": m3.ROUTE_STEPS * m3.PPOCR_RECIPE["batch"], "workers": 2, "log_every": 20,
                     "evals": 4, "probe": m3.PROBE_CROPS}}
WARMUP_BATCHES = 10


def default_root() -> Path:
    return generated_dir() / "m3" / "crops"


def pile_hashes(root: Path, names) -> dict:
    """Each pile's manifest's SHA-256, and its measure's where it is filtered."""
    out = {}
    for name in names:
        folder = root / name
        out[name] = {"manifest": piles.sha256_file(folder / "manifest.json")}
        if name in piles.VISIBILITY_FILTERED:
            out[name]["visibility"] = piles.sha256_file(folder / data.VISIBILITY_RECORD)
    return out


def recipe_pad() -> tuple[float, dict]:
    decisions = runs.read_json(m3.DECISIONS_PATH)
    if decisions is None:
        sys.exit(f"no {m3.DECISIONS_PATH}: S2's crop recipe (R1) is needed for the padding")
    recipe = m3.crop_recipe_used(decisions)
    return recipe["pad"], recipe


def loader_only(cfg: train.Config, root: Path, batches: int) -> dict:
    mix = data.Mix.for_arm(root, cfg.arm, cfg.visible_min)
    stream = data.Stream(root, mix, cfg.batch, cfg.pad, cfg.seed)
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=cfg.workers,
                                         persistent_workers=cfg.workers > 0,
                                         prefetch_factor=4 if cfg.workers else None,
                                         worker_init_fn=data.worker_init if cfg.workers else None)
    it = iter(loader)
    t0 = time.perf_counter()
    for _ in range(WARMUP_BATCHES):
        next(it)
    t1 = time.perf_counter()
    crops = sum(len(next(it)["lengths"]) for _ in range(batches))
    seconds = time.perf_counter() - t1
    del it, loader
    # LMDB alone: random records of the largest pile, read in this one process, then decoded
    big = max(mix.names, key=lambda n: mix.index[n].count)
    env = shards.open_pile(root / big)
    rng = np.random.default_rng(cfg.seed)
    picks = rng.integers(0, mix.index[big].count, 5000)
    t2 = time.perf_counter()
    got = [shards.get(env, big, int(i)) for i in picks]
    t3 = time.perf_counter()
    for c in got:
        c.image()
    t4 = time.perf_counter()
    env.close()
    return {"workers": cfg.workers, "batch": cfg.batch, "batches": batches, "warmup_seconds": round(t1 - t0, 1),
            "crops_per_second": round(crops / seconds, 1), "per_worker": round(crops / seconds / max(1, cfg.workers), 1),
            "lmdb": {"pile": big, "reads": len(picks), "ms_per_read": round(1000 * (t3 - t2) / len(picks), 3),
                     "kb_per_record": round(sum(len(c.jpeg) for c in got) / len(got) / 1024, 2),
                     "ms_per_decode": round(1000 * (t4 - t3) / len(got), 3)}}


def loss_fell(history: list[dict]) -> bool:
    """The smoke's test: the mean loss over the last tenth of the log under
    half the first tenth's."""
    k = max(1, len(history) // 10)
    first = np.mean([h["loss"] for h in history[:k]])
    last = np.mean([h["loss"] for h in history[-k:]])
    return bool(last < 0.5 * first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reader", default="crnn", choices=train.READERS)
    parser.add_argument("--arm", default="all", choices=tuple(m3.ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="S5's CPU smoke: 2,000 steps at batch 64")
    parser.add_argument("--loader-only", action="store_true")
    parser.add_argument("--batches", type=int, default=300, help="--loader-only: batches timed")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="float32 throughout (if bf16 fails on a CPU)")
    parser.add_argument("--root", type=Path, help="where the piles are (default $GENERATED_DIR/m3/crops)")
    args = parser.parse_args()

    if m3.VISIBLE_MIN is None:
        sys.exit("m3.VISIBLE_MIN isn't set: the cut-off is chosen by eye on the visibility sheets first (S5)")
    root = args.root or default_root()
    pad, recipe = recipe_pad()
    cfg = train.Config.for_reader(args.reader, arm=args.arm, seed=args.seed, device=args.device, pad=pad,
                                  visible_min=m3.VISIBLE_MIN, amp=not args.no_amp, threads=args.threads)
    if args.smoke:
        for k, v in SMOKE[cfg.reader].items():
            setattr(cfg, k, v)
    for k in ("budget", "batch", "workers"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))

    names = m3.ARMS[cfg.arm] + tuple(cfg.suites)
    config = {**asdict(cfg), "steps": cfg.steps, "recipe": recipe, "piles": pile_hashes(root, names),
              "parse_sha256": m3.check_parser(), "visibility_code": data.visibility_code(),
              "root": str(root), "host": os.uname().nodename, "cpus": len(os.sched_getaffinity(0))
              if hasattr(os, "sched_getaffinity") else os.cpu_count(), "smoke": args.smoke}
    if args.loader_only:
        run_dir = runs.start_run(f"m3-loader-{cfg.arm}", config, seed=cfg.seed)
        result = loader_only(cfg, root, args.batches)
        runs.update_run(run_dir, loader=result)
        print(f"loader: {result}", flush=True)
        return

    name = f"m3-{cfg.reader}-smoke-cpu" if args.smoke else f"m3-{cfg.reader}-{cfg.arm}-s{cfg.seed}"
    run_dir = runs.start_run(name, config, seed=cfg.seed)
    checkpoints = checkpoint_dir() / "m3" / run_dir.name
    runs.update_run(run_dir, checkpoints=str(checkpoints))
    print(f"run {run_dir}", flush=True)
    t0 = time.perf_counter()
    try:
        summary = train.run(cfg, root, checkpoints, say=lambda s: print(s, flush=True))
    except train.NonFinite as e:
        report = e.args[0]
        runs.update_run(run_dir, failed={"non_finite": report}, seconds=round(time.perf_counter() - t0, 1))
        print(f"FAILED: not finite at step {report['step']}: loss {report['loss']}, grad norm {report['grad_norm']}; "
              f"weights not finite {report['weights_not_finite']}; gradients not finite in "
              f"{len(report['grads_not_finite'])} tensors, first {report['grads_not_finite'][:3]}, last "
              f"{report['grads_not_finite'][-3:]}; replays {report['replays']}; "
              f"{report['infinite_losses']} infinite crop losses; crops at fault: {report['at_fault']}; "
              f"batch and weights kept in {report['kept']}", flush=True)
        sys.exit(1)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    runs.update_run(run_dir, result=summary)
    threads = cfg.threads or torch.get_num_threads()
    exported = train.export_checked(checkpoints / "best.pt", root, threads)
    runs.update_run(run_dir, export=exported)
    print(f"export: parity {exported['parity']:.4f} on {exported['reads']} reads "
          f"({exported['onnx_ms_per_read']} ms a read in ONNX Runtime, {threads} threads)", flush=True)
    ok = exported["passed"]
    if args.smoke:
        if cfg.reader == "ppocrv6":
            probe = summary["probe_loss"]
            fell = probe["after"] < probe["before"]
        else:
            fell = loss_fell(summary["history"])
        ok = ok and fell
        runs.update_run(run_dir, smoke={"loss_fell": fell, "parity_passed": exported["passed"], "passed": ok})
        print(f"smoke: loss fell {fell}, parity passed {exported['passed']}: {'PASSED' if ok else 'FAILED'}",
              flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
