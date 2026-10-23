"""M2's data layout, arms, recipe and decision rules, fixed before any run
(docs/m2-synthetic-generator.md). scripts/train_detector.py trains an arm
with the recipe, scripts/m2_chain.py applies the rules, and S7's
experiments/m2-*/summarise.py re-check them."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import runs
from .config import expdate_dir, generated_dir
from .derived import read_list
from .images import upright_size
from .yolo import split_repeats

# The 14 synthetic training photos drawn on top of validation photo
# img_00277.jpg, found by experiments/expdate-near-duplicates and checked
# by eye. M2 leaves them out of training, so validation is fully clean
# (decision 18).
REUSED_VAL_COPIES = frozenset({
    "img_00948.jpg", "img_02041.jpg", "img_02301.jpg", "img_02734.jpg",
    "img_03871.jpg", "img_03967.jpg", "img_04028.jpg", "img_04501.jpg",
    "img_04759.jpg", "img_05924.jpg", "img_05985.jpg", "img_08164.jpg",
    "img_08236.jpg", "img_08250.jpg",
})


# $GENERATED_DIR/m2/
#   derived/images/<set>/, derived/labels/<set>/   turned copies and grids
#   gen/images/<list>/, gen/labels/<list>/         generated photos (scripts/generate.py)
#   gen/sidecars/<list>.jsonl                      one line per generated photo, in photo order
#   lists/<name>.txt                               image lists, absolute paths
#   manifests/<name>.json                          what made each part, and counts
def root() -> Path:
    return generated_dir() / "m2"


def derived_dir() -> Path:
    return root() / "derived"


def gen_dir() -> Path:
    return root() / "gen"


def sidecar_path(name: str) -> Path:
    return gen_dir() / "sidecars" / f"{name}.jsonl"


# Generated sets, by set name: their list (and folder) names. gen-sample is
# S5's 500 photos to look at; S6 makes the training photos (gen, listed as
# gen50k, and gen10k for its first 10,000) and synth-val.
GENERATED_SETS = {"gen-sample": "gen_sample", "gen": "gen50k", "synth-val": "synth_val"}


def list_path(name: str) -> Path:
    return root() / "lists" / f"{name}.txt"


def manifest_path(name: str) -> Path:
    return root() / "manifests" / f"{name}.json"


# --- Lists ------------------------------------------------------------------

# ExpDate's own lists, in $EXPDATE_DIR/yolo/ (M0 A4). Every other list is M2's.
EXPDATE_LISTS = frozenset({"train_real", "val"})

# The manifest that vouches for each of M2's lists, and the step that writes it.
MANIFEST_OF = {
    "train_synth_clean": "derived", "real_rot": "derived", "synth_rot": "derived",
    "real_val_rot": "derived", "real_val_small": "derived",
    "gen50k": "gen", "gen10k": "gen", "synth_val": "synth-val",
}
MADE_BY = {"derived": "S2's derive_sets.py", "gen": "S6's generation",
           "synth-val": "S6's generation"}


def list_file(name: str) -> Path:
    """A list by name: ExpDate's own (train_real, val) or one of M2's."""
    return expdate_dir() / "yolo" / f"{name}.txt" if name in EXPDATE_LISTS else list_path(name)


def list_missing(name: str) -> str | None:
    """Why a list can't be used yet, or None if it can. Its file must exist
    and, for M2's lists, the manifest that made it must vouch for it: S2's
    names its sets, S6's says "complete" once every photo is checked."""
    manifest = MANIFEST_OF.get(name)
    if not list_file(name).exists():
        return f"not made yet ({MADE_BY[manifest]})" if manifest else "missing from $EXPDATE_DIR/yolo"
    if manifest is None:
        return None
    record = read_json(manifest_path(manifest)) or {}
    vouched = (name in record.get("sets", {}) if manifest == "derived"
               else record.get("complete") is True)
    return None if vouched else f"not yet vouched for by manifests/{manifest}.json ({MADE_BY[manifest]})"


def missing(reasons: Mapping[str, str | None]) -> str | None:
    """The names that aren't ready, grouped by reason, e.g. "real_rot,
    synth_rot not made yet (S2's derive_sets.py)"; None if all are ready."""
    grouped: dict[str, list[str]] = {}
    for name, why in reasons.items():
        if why:
            grouped.setdefault(why, []).append(name)
    return "; ".join(f"{', '.join(names)} {why}" for why, names in grouped.items()) or None


def manifest_hashes(names: Sequence[str]) -> dict[str, str]:
    """SHA-256 of each manifest behind these lists, so a run's record names
    exactly the data it saw."""
    found = sorted({MANIFEST_OF[n] for n in names if n in MANIFEST_OF})
    return {m: hashlib.sha256(manifest_path(m).read_bytes()).hexdigest() for m in found}


def count_listed(name: str) -> int:
    """How many photos a list names (repeats count)."""
    with list_file(name).open() as f:
        return sum(1 for line in f if line.strip())


def listed_images(lists: Sequence[str]) -> list[Path]:
    """Every photo a dataset file's train lists name, repeats included."""
    images: list[Path] = []
    for spec in lists:
        name, k = split_repeats(spec)
        images += read_list(list_file(name)) * k
    return images


def list_spec(spec: str) -> str:
    """A list as detector.dataset_yaml takes it: ExpDate's own by name (it
    is in the dataset file's folder), M2's by absolute path; repeats kept."""
    name, k = split_repeats(spec)
    path = name if name in EXPDATE_LISTS else str(list_path(name))
    return f"{path}*{k}" if k > 1 else path


# --- The arms and the training budget (decisions 6 to 8, 13, 14) -----------

BUDGET = 2_560_000     # images per run: twice M1's n640 (100 epochs x 12,795)
REAL_SHARE = 0.07      # M1's share of real photos (937 of 12,795)
SEEDS = (0, 1, 2)
WEIGHTS = "yolo11n"    # M1's detector (n640): M2 changes the data and input size only
PILOT_SHARE = 0.1      # the pilot trains on 10% of the budget


@dataclass(frozen=True)
class Arm:
    """What an arm trains on. The real lists are repeated together to about
    REAL_SHARE of the mix (decision 8); the others are listed as written,
    where `gen10k*5` lists gen10k five times (decision 14)."""
    real: tuple[str, ...]
    other: tuple[str, ...]
    imgsz: int = 640

    def names(self) -> tuple[str, ...]:
        """Every list the arm reads, once each."""
        return (*self.real, *(split_repeats(spec)[0] for spec in self.other))


# R = ExpDate real train (937), E = ExpDate synthetic without the 14 copies
# (11,844), -rot = one turned copy of each, G50k / G10k = generated photos.
ARMS = {
    "base": Arm(("train_real",), ("train_synth_clean",)),                     # R, E
    "rot": Arm(("train_real", "real_rot"), ("train_synth_clean", "synth_rot")),
    "gen": Arm(("train_real",), ("train_synth_clean", "gen50k")),
    "gen-rot": Arm(("train_real", "real_rot"), ("train_synth_clean", "synth_rot", "gen50k")),
    "gen10k": Arm(("train_real",), ("train_synth_clean", "gen10k*5")),
    "gen-rot-1280": Arm(("train_real", "real_rot"),
                        ("train_synth_clean", "synth_rot", "gen50k"), imgsz=1280),
}


@dataclass(frozen=True)
class Mix:
    """An arm's training set: the lists as its dataset file names them
    (repeats as name*k), how often the real photos are listed, the total."""
    lists: tuple[str, ...]
    real_repeats: int
    n_listed: int


def real_repeats(n_real: int, n_other: int) -> int:
    """How often to list the real photos so they're about REAL_SHARE of the mix."""
    return max(1, round(REAL_SHARE * n_other / ((1 - REAL_SHARE) * n_real)))


def mix(arm: Arm, counts: Mapping[str, int]) -> Mix:
    """The arm's lists with the real photos' repeats, from each list's size."""
    n_real = sum(counts[name] for name in arm.real)
    n_other = sum(counts[name] * k for name, k in map(split_repeats, arm.other))
    r = real_repeats(n_real, n_other)
    lists = tuple(f"{name}*{r}" if r > 1 else name for name in arm.real) + arm.other
    return Mix(lists, r, r * n_real + n_other)


def schedule(n_listed: int, share: float = 1.0, epochs: int | None = None) -> dict:
    """Epochs for the budget, or `epochs` (smoke and speed runs); warmup and
    mosaic-off keep M1's 3% and 10% of them. Recomputing both from a given
    `epochs` keeps mosaic on in a 3-epoch speed run's epoch 2."""
    epochs = epochs or max(1, round(share * BUDGET / n_listed))
    return dict(epochs=epochs, warmup_epochs=round(0.03 * epochs, 2),
                close_mosaic=max(1, round(0.10 * epochs)))


def expected_images(n_listed: int, fraction: float = 1.0) -> int:
    """How many listed photos Ultralytics trains on: with a fraction it
    keeps the first round(n x fraction) of the sorted list (data/base.py)."""
    return n_listed if fraction >= 1 else max(1, round(n_listed * fraction))


# --- The recipe (decisions 9 and 10) ----------------------------------------

# Everything else is Ultralytics' default, as in M1: mosaic, HSV, +-10%
# shift, +-50% zoom; no rotation, no flips.
RECIPE = dict(batch=16, fliplr=0.0, deterministic=True,     # as M1
              optimizer="MuSGD", lr0=0.01, momentum=0.937,   # what n640 trained with (decision 9)
              warmup_bias_lr=0.0,                            # auto sets this too
              patience=0, val=False)                         # no early stopping; validate at the end


def recipe_problems(applied: dict, wanted: dict, images: int) -> list[str]:
    """How what Ultralytics trains with (detector.applied) differs from what
    was asked: an argument it changed, another optimiser, or another number
    of training photos (repeats lost, or photos it couldn't read). Empty
    when everything is as asked."""
    args = applied["args"]
    problems = [f"{k}: asked {v!r}, Ultralytics has {args.get(k)!r}"
                for k, v in wanted.items() if args.get(k) != v]
    if "optimizer" in wanted and applied["optimizer"] != wanted["optimizer"]:
        problems.append(f"optimizer: asked {wanted['optimizer']}, built {applied['optimizer']}")
    if applied["train_images"] != images:
        problems.append(f"training photos: expected {images}, "
                        f"Ultralytics has {applied['train_images']}")
    return problems


# --- Loader settings (decision 15) -----------------------------------------

# The speed test's settings, simplest first.
LOADERS = {"a": dict(workers=8, cache=False),
           "b": dict(workers=24, cache=False),
           "c": dict(workers=24, cache="ram")}
LOADER_TIE = 0.10      # within 10% of the fastest is as fast
RAM_SHARE = 0.60       # the RAM cache may fill at most 60% of the job's memory


def loader_choice(speed: dict[str, float]) -> str:
    """Images per second in epoch 2 of each speed run: a = 8 workers,
    b = 24 workers, c = 24 workers + RAM cache. The fastest; within 10%
    of it, the simplest (a, then b, then c)."""
    fastest = max(speed.values())
    return next(k for k in LOADERS if k in speed and speed[k] >= (1 - LOADER_TIE) * fastest)


def epoch_seconds(results_csv: Path, epoch: int = 2) -> float | None:
    """How long one epoch took, from the cumulative `time` column of
    Ultralytics' results.csv; None until that epoch has finished.
    Epoch 2 is timed: epoch 1 includes the workers' start-up and the first
    read of every photo, the last epoch the final validation."""
    with results_csv.open() as f:
        times = [float(row["time"]) for row in csv.DictReader(f)]
    if len(times) < epoch:
        return None
    return times[epoch - 1] - (times[epoch - 2] if epoch > 1 else 0.0)


def cached_bytes(width: int, height: int, imgsz: int) -> int:
    """One photo in Ultralytics' RAM cache: its long side resized to imgsz,
    up or down (data/base.py, load_image), 3 bytes a pixel."""
    r = imgsz / max(width, height)
    if r != 1:
        width, height = min(math.ceil(width * r), imgsz), min(math.ceil(height * r), imgsz)
    return width * height * 3


def ram_check(images: Sequence[Path], imgsz: int, mem_mb: str | None = None,
              sample: int = 30, seed: int = 0) -> dict:
    """Decision 15's guard for the RAM cache: 30 listed photos, scaled to
    the whole list (a repeated photo is cached once per listing), must need
    under RAM_SHARE of the job's memory, SLURM_MEM_PER_NODE (MB).
    Ultralytics checks the node's free memory instead, which on a shared
    node can be far more than the job may use."""
    mem_mb = os.environ.get("SLURM_MEM_PER_NODE") if mem_mb is None else mem_mb
    rng = random.Random(seed)
    picks = [images[rng.randrange(len(images))] for _ in range(min(sample, len(images)))]
    need = float(np.mean([cached_bytes(*upright_size(p), imgsz) for p in picks])) * len(images)
    out = {"need_gb": round(need / 2**30, 1), "photos": len(images), "sampled": len(picks)}
    if not mem_mb:
        return {**out, "fits": False, "reason": "no SLURM_MEM_PER_NODE: not a Slurm job with --mem"}
    limit = RAM_SHARE * int(mem_mb) * 2**20
    return {**out, "limit_gb": round(limit / 2**30, 1), "fits": need < limit}


# --- The validation suite and the decision rules (decisions 11 and 12) -----

# M2's hard validation sets, made from ExpDate validation (S2): set -> list.
DERIVED_SETS = {"real-val-rot": "real_val_rot", "real-val-small": "real_val_small"}
# Every run is scored on these ("The validation suite"); synth-val comes in S6.
SUITE = {"real-val": "val", **DERIVED_SETS, "synth-val": "synth_val"}
# The threshold: best F2 over the real sets pooled. Synthetic photos don't
# set it: their confidences may not carry over to real ones.
THRESHOLD_SETS = ("real-val", "real-val-rot", "real-val-small")
THRESHOLD_BETA = 2.0


def suite_missing(name: str) -> str | None:
    """Why a suite set can't be scored yet (synth-val until S6), or None."""
    return list_missing(SUITE[name])


PILOT_RECALL = 0.5


def pilot_passes(suite: dict) -> bool:
    """The generated labels are learnable: a short run on `gen` boxes at
    least half of synth-val's targets at any confidence (IoU 0.5)."""
    synth = suite["synth-val"]
    if "missing" in synth:
        raise ValueError(f"the pilot gate needs synth-val: {synth['missing']}")
    return synth["any_conf"]["expiry_recall"] >= PILOT_RECALL


NOISE = 0.01      # 1 point of mAP@0.5: smaller differences are noise (as M1)
CANDIDATES = ("base", "rot", "gen", "gen-rot", "gen-rot-1280")   # cheapest first
HARD = ("real-val-rot", "real-val-small", "synth-val")


def map50_date(suites: Sequence[dict], sets: Sequence[str]) -> float:
    """Mean mAP@0.5 for dates over the seeds' suites and the given sets."""
    return float(np.mean([s[name]["map50"]["date"] for s in suites for name in sets]))


def decide_detector(suites: dict[str, list[dict]]) -> str:
    """Eligible: mean mAP@0.5 for dates on real-val no more than NOISE
    under base's. Among them, the highest mean mAP@0.5 over the HARD sets;
    any candidate within NOISE of it goes to the cheapest."""
    def at_least(value: float, floor: float) -> bool:
        return value >= floor - 1e-9       # float noise must not break a tie

    floor = map50_date(suites["base"], ("real-val",)) - NOISE
    eligible = [a for a in CANDIDATES if at_least(map50_date(suites[a], ("real-val",)), floor)]
    best = max(map50_date(suites[a], HARD) for a in eligible)
    return next(a for a in eligible if at_least(map50_date(suites[a], HARD), best - NOISE))


# --- Run records ----------------------------------------------------------

def run_dirs(name: str, root: Path | None = None) -> list[Path]:
    """Every attempt at M2 run `name`, oldest first: <runs>/<stamp>-m2-<name>/.
    The name must match whole: base-s0 is not gen-base-s0."""
    base = root or runs.RUNS_DIR
    return sorted(d for d in base.glob(f"*-m2-{name}") if d.name[16:] == f"m2-{name}")


def finished_run(name: str, root: Path | None = None) -> Path | None:
    """The newest attempt at this run whose run.json says it finished."""
    return next((d for d in reversed(run_dirs(name, root))
                 if (read_json(d / "run.json") or {}).get("status") == "finished"), None)


def read_json(path: Path) -> dict | None:
    """A JSON file, or None if it doesn't exist (yet). Records are written
    atomically (runs.write_json), so one that exists is whole."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
