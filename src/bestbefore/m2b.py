"""M2b's layout, model sizes, arms and the no-training control, fixed before
any run (docs/m2b-store-and-size.md). Step 1 is M2's gen-rot-1280 arm
trained from other COCO weights: only the weights change (decision 1). n's
step 1 is M2's own three runs, kept as they are and recorded as the control
(decision 12). Step 2 fine-tunes each step-1 run on the store lines (S6)
and a replay sample, with one of two recipes (S10).
scripts/train_detector.py trains an arm and scripts/m2b_chain.py runs them
all."""
from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import derived, m2, runs
from .config import checkpoint_dir, generated_dir, store_train_dir
from .runs import read_json
from .store import sha256
from .yolo import split_repeats


# $GENERATED_DIR/m2b/
#   gen/, derived/      gen-v2 and the 180° copies (S8, S9)
#   lists/<name>.txt    image lists, absolute paths
#   manifests/<name>.json
# apart from M2's $GENERATED_DIR/m2/, which doesn't change.
def root() -> Path:
    return generated_dir() / "m2b"


def derived_dir() -> Path:
    return root() / "derived"


def list_path(name: str) -> Path:
    return root() / "lists" / f"{name}.txt"


def gen_dir() -> Path:
    return root() / "gen"


def sidecar_path(name: str) -> Path:
    return gen_dir() / "sidecars" / f"{name}.jsonl"


# gen-v2's sets, by set name: their list (and folder) names, under gen/ as
# M2's are under m2/gen/. gen-v2-sample is S8's 500 photos to look at; S9
# makes gen-v2 (step 2's replay) and synth-val-v2 (8 strata of 300).
GENERATED_SETS = {"gen-v2-sample": "gen_v2_sample", "gen-v2": "gen_v2", "synth-val-v2": "synth_val_v2"}
# M2b's derived validation set (S9): ExpDate's 165 validation photos turned
# 180°, by set name: its list (and folder) name. Its training twins,
# real_rot180 and synth_rot180, are step 2's (STEP2_LISTS).
DERIVED_SETS = {"real-val-rot180": "real_val_rot180"}


def manifest_path(name: str) -> Path:
    return root() / "manifests" / f"{name}.json"


# --- What step 2 waits for, made in S4 to S9 ---------------------------------------

# M2b's own lists for step 2's replay (S9), and the manifest that vouches for each.
STEP2_LISTS = ("gen_v2", "real_rot180", "synth_rot180")
MANIFEST_OF = {"gen_v2": "gen-v2", "real_rot180": "derived", "synth_rot180": "derived",
               "synth_val_v2": "synth-val-v2", "real_val_rot180": "derived"}    # the last two: the M2b suite's (S10)
MADE_BY = {"gen-v2": "S9's generation", "synth-val-v2": "S9's generation",
           "derived": "S9's derive_sets.py --only rot180"}


def list_missing(name: str) -> str | None:
    """Why one of M2b's lists can't be used yet, or None, as m2.list_missing:
    its file must exist and its manifest vouch for it (derived.json by
    naming it, gen-v2.json by saying complete)."""
    manifest = MANIFEST_OF[name]
    if not list_path(name).exists():
        return f"not made yet ({MADE_BY[manifest]})"
    record = read_json(manifest_path(manifest)) or {}
    vouched = (name in record.get("sets", {}) if manifest == "derived"
               else record.get("complete") is True)
    return None if vouched else f"not yet vouched for by m2b/manifests/{manifest}.json ({MADE_BY[manifest]})"


def store_export_missing() -> str | None:
    """Why store set v1's training export can't be used yet, or None: S6
    writes it into $STORE_TRAIN_DIR, with a manifest."""
    try:
        folder = store_train_dir()
    except RuntimeError:
        return "store set v1's export ($STORE_TRAIN_DIR unset; S6)"
    if not (folder / "manifest.json").exists():
        return "store set v1's export ($STORE_TRAIN_DIR/manifest.json; S6)"
    return None


# --- The store lines (decision 7, S6) -----------------------------------------------

def store_pool(folder: Path | None = None) -> tuple[list[Path], dict[str, str]]:
    """The export's training images (every photo in all 4 turns) and each
    exported photo's design, from its train.txt and manifest (S6)."""
    folder = folder or store_train_dir()
    manifest = read_json(folder / "manifest.json")
    return (derived.read_list(folder / manifest["settings"]["lists"]["train"]),
            {name: p["design"] for name, p in manifest["photos"].items() if p["part"] == "train"})


def store_lines(images: Sequence[Path], designs: Mapping[str, str], count: int, seed: int) -> list[Path]:
    """One run's store lines, design-balanced (decision 7): every design
    among `images` gets the same number of lines, count / designs rounded
    half up, spread as evenly as can be over its photos and their turns;
    the copies that get one line more are drawn with `seed`. `images` is
    the export's training list; `designs` maps an exported photo's name
    (v1-0001, its file's stem) to its design (d01), as store_pool reads them."""
    by_design: dict[str, list[Path]] = {}
    for image in images:
        by_design.setdefault(designs[image.stem], []).append(image)
    n = len(by_design)
    per = (2 * count + n) // (2 * n)
    rng, lines = random.Random(seed), []
    for design in sorted(by_design):
        copies = sorted(by_design[design])
        each, extra = divmod(per, len(copies))
        more = set(rng.sample(range(len(copies)), extra))
        lines += [c for k, c in enumerate(copies) for _ in range(each + (k in more))]
    return lines


# --- Model size (decisions 1 and 2) -------------------------------------------

SIZES = {"n": "yolo11n", "s": "yolo11s", "m": "yolo11m"}     # COCO-pretrained, by size
SEEDS = m2.SEEDS
LOADER = m2.LOADERS["a"]       # M2's loader decision, kept: 8 workers, no RAM cache


@dataclass(frozen=True)
class StepOne:
    """A step-1 arm: one of M2's arms trained from another size's COCO
    weights. Its lists, mix, budget, schedule, recipe and seeds are that
    arm's; only the weights differ."""
    size: str                       # n, s or m
    m2_arm: str = "gen-rot-1280"

    @property
    def weights(self) -> str:
        return SIZES[self.size]

    @property
    def arm(self) -> m2.Arm:
        return m2.ARMS[self.m2_arm]


STEP1 = {"s1280": StepOne("s"), "m1280": StepOne("m")}

# The no-training control: n1280 is M2's gen-rot-1280, its three seeds as
# they are. Never trained again, copied or changed.
CONTROL = "n1280"
CONTROL_FROM = "gen-rot-1280"


# --- Step 2: store fine-tuning (decisions 7 to 11) -----------------------------

STORE_SHARE = 0.20            # store photos' share of every epoch (decision 7)
EPOCH_PHOTOS = 25_000         # photos listed per epoch: the replay sample plus the store lines
BUDGET_FT = 256_000           # images per fine-tuning run: 10% of M2's BUDGET (decision 9)
FT_RECIPE = {**m2.RECIPE, "lr0": 0.001}
FREEZE = 11                   # two-stage: layers 0 to 10, YOLO11's backbone, frozen in stage 1
FT2_LR0 = 0.0005              # two-stage: stage 2's rate, half FT_RECIPE's (decision 10)


@dataclass(frozen=True)
class Stage:
    """One Ultralytics training within a fine-tuning run: its share of
    BUDGET_FT, and what it changes in FT_RECIPE."""
    share: float
    freeze: int | None = None       # Ultralytics' freeze: the first k layers keep their weights
    lr0: float = FT_RECIPE["lr0"]

    def recipe(self) -> dict:
        return {**FT_RECIPE, "lr0": self.lr0, "freeze": self.freeze}


RECIPES = ("ft", "ft2")           # simplest first (R2)
STAGES = {"ft": (Stage(1.0),),                                        # the whole network
          "ft2": (Stage(0.5, freeze=FREEZE), Stage(0.5, lr0=FT2_LR0))}   # the head, then all


@dataclass(frozen=True)
class StepTwo:
    """A step-2 arm: a size's step-1 runs, seed for seed, fine-tuned with one
    recipe on the store lines and a replay sample."""
    size: str                       # n, s or m
    recipe: str                     # ft or ft2
    generated: str = "gen_v2"       # the replay's generated photos: gen-v2, or M2's gen50k (decision 11)

    @property
    def start(self) -> str:
        """The step-1 arm it starts from: n1280 (the control, M2's weights), s1280 or m1280."""
        return f"{self.size}1280"

    @property
    def stages(self) -> tuple[Stage, ...]:
        return STAGES[self.recipe]

    @property
    def imgsz(self) -> int:
        return m2.ARMS[CONTROL_FROM].imgsz


STEP2 = {**{f"{size}1280-{recipe}": StepTwo(size, recipe) for size in SIZES for recipe in RECIPES},
         "n1280-ft-gen50k": StepTwo("n", "ft", "gen50k")}


def schedule(n_listed: int, stage: Stage, epochs: int | None = None) -> dict:
    """A stage's epochs for its share of BUDGET_FT, or `epochs` (smoke runs),
    with M2's warmup and mosaic-off shares (m2.schedule)."""
    return m2.schedule(n_listed, stage.share * BUDGET_FT / m2.BUDGET, epochs)


# The replay (decision 8): gen-rot-1280's mix without the store, pool by pool.
# Its lists' sizes, as M2's runs recorded them (and step 1's, which trained on them).
GEN_ROT_1280_COUNTS = {"train_real": 937, "real_rot": 937, "train_synth_clean": 11_844,
                       "synth_rot": 11_844, "gen50k": 50_000}
# Each pool and its lists. The turned pools add ExpDate's 180° copies (S9) to
# the 90° and 270° ones gen-rot-1280 had, so the turned share stays as it was.
REPLAY = {"real": ("train_real",), "real_turned": ("real_rot", "real_rot180"),
          "synthetic": ("train_synth_clean",), "synthetic_turned": ("synth_rot", "synth_rot180"),
          "generated": ("gen_v2",)}


def replay_shares() -> dict[str, float]:
    """Each replay pool's share of gen-rot-1280's mix: the photos its first
    list gave there (real ones repeated, as that arm lists them), over all."""
    mix = m2.mix(m2.ARMS[CONTROL_FROM], GEN_ROT_1280_COUNTS)
    listed = {name: GEN_ROT_1280_COUNTS[name] * k for name, k in map(split_repeats, mix.lists)}
    first = {"real": "train_real", "real_turned": "real_rot", "synthetic": "train_synth_clean",
             "synthetic_turned": "synth_rot", "generated": "gen50k"}
    return {pool: listed[name] / mix.n_listed for pool, name in first.items()}


REPLAY_SHARES = replay_shares()


def replay_lists(arm: StepTwo) -> dict[str, tuple[str, ...]]:
    """The arm's replay pools and their lists: REPLAY, with its generated photos."""
    return {**REPLAY, "generated": (arm.generated,)}


def list_file(name: str) -> Path:
    """Any list step 2 reads: one of M2b's (MANIFEST_OF), else ExpDate's or M2's."""
    return list_path(name) if name in MANIFEST_OF else m2.list_file(name)


def any_list_missing(name: str) -> str | None:
    """Why a list step 2 reads can't be used yet, or None: M2b's own, or M2's."""
    return list_missing(name) if name in MANIFEST_OF else m2.list_missing(name)


def replay_pools(arm: StepTwo) -> dict[str, list[Path]]:
    """Every photo of each of the arm's replay pools, from its lists."""
    return {pool: [p for name in names for p in derived.read_list(list_file(name))]
            for pool, names in replay_lists(arm).items()}


def manifest_hashes(names: Sequence[str]) -> dict[str, str]:
    """SHA-256 of every manifest behind these lists: M2's, and M2b's as m2b/<name>."""
    found = m2.manifest_hashes([n for n in names if n not in MANIFEST_OF])
    for name in sorted({MANIFEST_OF[n] for n in names if n in MANIFEST_OF}):
        found[f"m2b/{name}"] = sha256(manifest_path(name))
    return found


def mix(pools: Mapping[str, Sequence[Path]], designs: Mapping[str, str], seed: int,
        epoch_photos: int = EPOCH_PHOTOS) -> dict[str, list[Path]]:
    """One run's lists (decisions 7 and 8): "store", the store lines,
    design-balanced over v1-train's designs and its 4 turns (STORE_SHARE of
    the epoch, store_lines), and a sample of each replay pool, its share of
    gen-rot-1280's mix (REPLAY_SHARES) of the rest, drawn with `seed`, each
    photo at most once. `pools` holds the export's training images under
    "store" and each replay pool's photos under its name (replay_pools);
    `designs` maps an exported photo to its design (store_pool). The same
    seed gives the same lists, in whatever order the pools come."""
    store_count = round(STORE_SHARE * epoch_photos)
    lists = {"store": store_lines(pools["store"], designs, store_count, seed)}
    rest = epoch_photos - store_count
    for pool, share in REPLAY_SHARES.items():
        photos, n = sorted(pools[pool]), round(share * rest)
        if n > len(photos):
            raise ValueError(f"the {pool} pool holds {len(photos)} photos; the mix draws {n}")
        lists[pool] = sorted(random.Random(f"{seed}/{pool}").sample(photos, n))
    return lists


def list_text(paths: Sequence[Path]) -> str:
    """A list file's text: one absolute path a line."""
    return "".join(f"{p}\n" for p in paths)


def list_record(paths: Sequence[Path]) -> dict:
    """What a run records of one of its lists: its lines, distinct photos and SHA-256."""
    return {"photos": len(paths), "distinct": len(set(paths)),
            "sha256": hashlib.sha256(list_text(paths).encode()).hexdigest()}


def start_weights(arm: StepTwo, seed: int, root: Path | None = None) -> tuple[Path, str, str]:
    """A step-2 run's starting weights, from its step-1 run's record, seed
    for seed (decision 9): the last.pt its run.json names, that file's
    SHA-256, and the run's folder name. The control's record holds the hash
    of M2's last.pt, and a file that no longer matches it is refused."""
    name = f"{arm.start}-s{seed}"
    run_dir = finished_run(name, root)
    if run_dir is None:
        raise FileNotFoundError(f"m2b-{name} hasn't finished: no weights to start from")
    record = read_json(run_dir / "run.json")
    weights = Path(record["checkpoint"])
    if not weights.exists():
        raise FileNotFoundError(f"{weights}, m2b-{name}'s weights, isn't on this machine")
    digest = sha256(weights)
    if record.get("checkpoint_sha256", digest) != digest:
        raise RuntimeError(f"{weights} has changed since m2b-{name} recorded it")
    return weights, digest, run_dir.name


def start_missing(arm: StepTwo, seed: int, root: Path | None = None) -> str | None:
    """What a step-2 run waits for before it can start, or None: its step-1 run."""
    name = f"{arm.start}-s{seed}"
    return None if finished_run(name, root) else f"m2b-{name} to finish"


def frozen_drift(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray],
                 layers: int = FREEZE) -> dict:
    """How far a frozen stage moved each kind of tensor, from the weights it
    started from (`before`) to its last.pt (`after`), both as
    detector.weights_state reads them: in the frozen layers (model.0 to
    model.<layers - 1>) their convolution weights, batch-norm weights and
    biases, and batch-norm running means and variances; and every tensor of
    the layers it trained. Per kind: tensors, values, values changed, and
    the largest and the mean absolute change. Batch counts aren't compared."""
    frozen = tuple(f"model.{i}." for i in range(layers))
    kinds: dict[str, dict] = {}
    for key in sorted(before):
        if key.endswith("num_batches_tracked"):
            continue
        if not key.startswith(frozen):
            kind = "trained"
        elif key.endswith(("running_mean", "running_var")):
            kind = "bn_" + key.rsplit(".", 1)[1]
        else:
            kind = "bn_affine" if ".bn." in key else "conv"
        diff = np.abs(np.asarray(after[key], np.float64) - np.asarray(before[key], np.float64))
        k = kinds.setdefault(kind, {"tensors": 0, "values": 0, "changed": 0, "max_abs": 0.0, "sum_abs": 0.0})
        k["tensors"] += 1
        k["values"] += diff.size
        k["changed"] += int((diff > 0).sum())
        k["max_abs"] = max(k["max_abs"], float(diff.max(initial=0.0)))
        k["sum_abs"] += float(diff.sum())
    for k in kinds.values():
        k["mean_abs"] = k.pop("sum_abs") / k["values"] if k["values"] else 0.0
    return {"layers": layers, **dict(sorted(kinds.items()))}


def frozen_problems(drift: dict) -> list[str]:
    """What a frozen stage must not do: move a frozen weight (a convolution's,
    or a batch-norm's weight or bias). Batch-norm statistics are only
    measured (docs/m2b-store-and-size.md, Step 2)."""
    return [f"{kind}: {drift[kind]['changed']} of {drift[kind]['values']} values changed, "
            f"by up to {drift[kind]['max_abs']:.3g}"
            for kind in ("conv", "bn_affine") if drift.get(kind, {}).get("changed")]


# --- The M2b suite (S10) ---------------------------------------------------------

# Every run is scored on these ("The validation suite, M2b"): M2's four sets,
# real-val-rot180, synth-val-v2 (per stratum) and v1-val, the store set:
# store set v1's validation part with v3's (S5b), as the store export holds it.
STORE_SET = "v1-val"
SUITE = (*m2.SUITE, *DERIVED_SETS, "synth-val-v2", STORE_SET)
SUITE_FILE = "suite-m2b.json"
# Cause 4's size bands: a date's box short side over its photo's long side.
SIZE_BANDS = (("under 1.5%", 0.0, 0.015), ("1.5-2.5%", 0.015, 0.025), ("2.5% and up", 0.025, float("inf")))


def suite_list(name: str) -> Path:
    """The image list behind an M2b suite set."""
    if name == STORE_SET:
        folder = store_train_dir()
        return folder / read_json(folder / "manifest.json")["settings"]["lists"]["val"]
    if name in m2.SUITE:
        return m2.list_file(m2.SUITE[name])
    return list_path({**DERIVED_SETS, **GENERATED_SETS}[name])


def suite_missing(name: str) -> str | None:
    """Why an M2b suite set can't be scored yet, or None."""
    if name == STORE_SET:
        return store_export_missing()
    if name in m2.SUITE:
        return m2.suite_missing(name)
    return list_missing({**DERIVED_SETS, **GENERATED_SETS}[name])


def load_v1_val(folder: Path | None = None) -> list:
    """v1-val as Samples: the export's validation photos, numbered (v1-0004,
    v3-0002, ...), each box labelled as the manifest names it in its label
    file's order (expiry, mfg, other_date, or a keyword's class), so expiry
    dates count apart, and each photo with its failure tag. A label file
    whose classes don't follow its manifest is refused."""
    from .store_train import YOLO_LABEL

    folder = folder or store_train_dir()
    manifest = read_json(folder / "manifest.json")
    samples = []
    for image in derived.read_list(folder / manifest["settings"]["lists"]["val"]):
        sample, photo = derived.load_sample(image), manifest["photos"][image.stem]
        if [YOLO_LABEL[label] for label in photo["labels"]] != [b.label for b in sample.boxes]:
            raise ValueError(f"{image.name}: its label file doesn't follow its labels in the manifest")
        samples.append(replace(sample, tag=photo["tag"], boxes=tuple(
            replace(b, label=label) for b, label in zip(sample.boxes, photo["labels"]))))
    return samples


def date_size(box, width: int, height: int) -> float:
    """A date's size as the store measures it (Cause 4): its box's short
    side over its photo's long side, so a sideways date counts its height."""
    x1, y1, x2, y2 = box.xyxy
    return min((x2 - x1) * width, (y2 - y1) * height) / max(width, height)


def found_by_size(samples: Sequence, at_05, threshold: float, threshold_m2: float) -> dict:
    """The labelled dates of `samples` by size band: per band, how many,
    how many found at IoU 0.5 at `threshold` and at `threshold_m2`, and how
    many boxed at any confidence. `at_05` is detection_eval.evaluate's
    matching of these samples, which lists their dates in this same order."""
    from .samples import DATE_LABELS

    sizes = [date_size(b, s.width, s.height) for s in samples for b in s.boxes if b.label in DATE_LABELS]
    if len(sizes) != len(at_05.truth_conf):
        raise ValueError(f"{len(sizes)} dates, but {len(at_05.truth_conf)} matched")
    out = {}
    for band, low, high in SIZE_BANDS:
        conf = [c for c, size in zip(at_05.truth_conf, sizes) if low <= size < high]
        out[band] = {"dates": len(conf), "found": sum(c >= threshold for c in conf),
                     "found_at_threshold_m2": sum(c >= threshold_m2 for c in conf),
                     "boxed_at_any_confidence": sum(c >= 0.0 for c in conf)}
    return out


# --- Run records ----------------------------------------------------------------

def run_dirs(name: str, root: Path | None = None) -> list[Path]:
    """Every attempt at M2b run `name`, oldest first: <runs>/<stamp>-m2b-<name>/.
    The name must match whole, as M2's (m2.run_dirs)."""
    base = root or runs.RUNS_DIR
    return sorted(d for d in base.glob(f"*-m2b-{name}") if d.name[16:] == f"m2b-{name}")


def finished_run(name: str, root: Path | None = None) -> Path | None:
    """The newest attempt at this run whose run.json says it finished."""
    return next((d for d in reversed(run_dirs(name, root))
                 if (read_json(d / "run.json") or {}).get("status") == "finished"), None)


def decisions_path(root: Path | None = None) -> Path:
    """Where the chain records its decisions."""
    return (root or runs.RUNS_DIR) / "m2b-decisions.json"


def suite_of(run: str, root: Path | None = None) -> dict | None:
    """A finished run's M2b suite (suite-m2b.json); None until it is scored."""
    run_dir = finished_run(run, root)
    return read_json(run_dir / SUITE_FILE) if run_dir else None


def arm_suites(arm: str, root: Path | None = None) -> list[dict | None]:
    """An arm's M2b suite for every seed, in SEEDS order; None where a seed
    isn't scored yet. n1280's are the control's."""
    return [suite_of(f"{arm}-s{s}", root) for s in SEEDS]


# --- The decision rules (written before any M2b run, "Decision rules") ------------

NOISE = 0.01               # real-val and synth-val, as M2
NOISE_STORE_MIN = 0.03     # v1-val: one date of about 60 is about 1.7 points
GUARDED = ("real-val", "synth-val")


def v1_val_map50(suites: Sequence[dict]) -> float:
    """Mean mAP@0.5 for dates on v1-val over the seeds' M2b suites."""
    return m2.map50_date(suites, (STORE_SET,))


def noise_store(control: Sequence[dict]) -> float:
    """The noise band on v1-val: the spread (highest minus lowest) of
    mAP@0.5 for dates over the control's three seeds (M2's n1280, no
    store training), at least NOISE_STORE_MIN. For means of 3 seeds, the
    range is about a two-sigma band. Fixed once the control is scored,
    before any M2b run is."""
    values = [s[STORE_SET]["map50"]["date"] for s in control]
    return max(max(values) - min(values), NOISE_STORE_MIN)


def guard(suites: Sequence[dict], control: Sequence[dict]) -> bool:
    """No forgetting: mean mAP@0.5 for dates on real-val and on synth-val,
    each no more than NOISE under the control's."""
    return all(m2.at_least(m2.map50_date(suites, (name,)), m2.map50_date(control, (name,)) - NOISE)
               for name in GUARDED)


def probe_m(m: dict, s: Sequence[dict], control: Sequence[dict]) -> bool:
    """R1. m1280 earns 2 more seeds and step 2 only if seed 0's v1-val
    mAP@0.5 beats every s1280 seed's, and it passes the guard."""
    beats = all(m[STORE_SET]["map50"]["date"] > x[STORE_SET]["map50"]["date"] for x in s)
    return beats and guard([m], control)


def decide_recipe(n_runs: Mapping[str, Sequence[dict]], band: float, control: Sequence[dict]) -> str | None:
    """R2, on n. Eligible: the guard. Then the highest mean v1-val
    mAP@0.5; any recipe within `band` of it goes to the simplest. None if
    no recipe passes the guard (S10): step 2 on s and m then waits for a
    person, and so does R3."""
    ok = [r for r in RECIPES if guard(n_runs[r], control)]
    if not ok:
        return None
    best = max(v1_val_map50(n_runs[r]) for r in ok)
    return next(r for r in ok if m2.at_least(v1_val_map50(n_runs[r]), best - band))


CANDIDATES = ("n1280", "n1280-ft", "s1280", "s1280-ft", "m1280", "m1280-ft")
#             cheapest first: inference cost, then training cost;
#             "-ft" stands for R2's recipe (ft or ft2)


def decide_detector(suites: Mapping[str, Sequence[dict | None]], band: float,
                    control: Sequence[dict]) -> str:
    """R3. Only candidates with all 3 seeds scored take part (without R1,
    m1280 has one). Eligible: the guard. Then the highest mean v1-val
    mAP@0.5; any candidate within `band` of it goes to the cheapest. The
    M2b detector is that arm's seed 0, at its threshold_store, unless R4
    adopts the ensemble. `suites` is keyed by CANDIDATES' names; n1280's
    are the control's, so n1280 always takes part."""
    scored = [c for c in CANDIDATES if c in suites and len(suites[c]) == len(SEEDS)
              and None not in suites[c]]
    ok = [c for c in scored if guard(suites[c], control)]
    best = max(v1_val_map50(suites[c]) for c in ok)
    return next(c for c in ok if m2.at_least(v1_val_map50(suites[c]), best - band))


def candidate_runs(recipe: str, with_m: bool) -> dict[str, str]:
    """R3's candidates and the arm each stands for: "-ft" is R2's recipe;
    m1280 and m1280-ft take part only once R1 has given m its seeds."""
    sizes = ("n", "s", "m") if with_m else ("n", "s")
    return {c: c.replace("-ft", f"-{recipe}") for c in CANDIDATES if c[0] in sizes}


def decisions(root: Path | None = None) -> dict:
    """Every decision whose inputs are complete, from the runs' M2b suites:
    the noise band (the control's three seeds), R1 (m1280-s0 and s1280's
    seeds), R2 (n1280-ft and -ft2) and R3 (every candidate, once R1 and R2
    are known and each candidate's seeds are scored). The chain writes them
    to results/runs/m2b-decisions.json; the experiments' summarise.py work
    them out again with this same function and compare."""
    out: dict = {}
    control = arm_suites(CONTROL, root)
    if None in control:
        return out
    band = noise_store(control)
    values = [s[STORE_SET]["map50"]["date"] for s in control]
    out["noise_store"] = {"band": band, "spread": round(max(values) - min(values), 4),
                          "control_v1_val_map50": values}
    s1280, m1280 = arm_suites("s1280", root), suite_of("m1280-s0", root)
    if None not in s1280 and m1280 is not None:
        out["R1"] = {"passes": probe_m(m1280, s1280, control), "guard": guard([m1280], control),
                     "m1280_s0_v1_val_map50": m1280[STORE_SET]["map50"]["date"],
                     "s1280_v1_val_map50": [s[STORE_SET]["map50"]["date"] for s in s1280]}
    n_runs = {r: arm_suites(f"{CONTROL}-{r}", root) for r in RECIPES}
    if all(None not in seeds for seeds in n_runs.values()):
        out["R2"] = {"recipe": decide_recipe(n_runs, band, control),
                     "guard": {r: guard(n_runs[r], control) for r in RECIPES},
                     "v1_val_map50": {r: round(v1_val_map50(n_runs[r]), 4) for r in RECIPES},
                     "band": band}
    if "R1" in out and (out.get("R2") or {}).get("recipe"):
        arms = candidate_runs(out["R2"]["recipe"], out["R1"]["passes"])
        suites = {c: arm_suites(arm, root) for c, arm in arms.items()}
        if all(None not in seeds for seeds in suites.values()):
            chosen = arms[decide_detector(suites, band, control)]
            out["R3"] = {"arm": chosen, "run": f"{chosen}-s0",
                         "threshold": suite_of(f"{chosen}-s0", root)["threshold"],
                         "eligible": [arms[c] for c in suites if guard(suites[c], control)],
                         "v1_val_map50": {arms[c]: round(v1_val_map50(suites[c]), 4) for c in suites},
                         "band": band}
    return out


# --- The control (decision 12) ----------------------------------------------------

def m2_weights(run_dir: Path, record: dict) -> Path:
    """An M2 run's last.pt: where its run.json says, if it is there (the
    cluster), else the laptop's copy, as scripts/score_suite.py finds it."""
    recorded = Path(record["checkpoint"])
    return recorded if recorded.exists() else checkpoint_dir() / "m2" / run_dir.name / "weights" / "last.pt"


def control_missing(seed: int, root: Path | None = None) -> str | None:
    """Why the control's record for this seed can't be written yet, or None:
    M2's run must have finished, and its last.pt be on this machine."""
    source = f"{CONTROL_FROM}-s{seed}"
    run_dir = m2.finished_run(source, root)
    if run_dir is None:
        return f"M2's {source} to finish"
    if not m2_weights(run_dir, read_json(run_dir / "run.json")).exists():
        return f"M2's {source} last.pt on this machine"
    return None


def write_control(seed: int, root: Path | None = None) -> Path:
    """The control's record, m2b-n1280-s<seed>: M2's gen-rot-1280-s<seed> as
    it is. Its run.json names M2's run and points at M2's last.pt, with that
    file's SHA-256, and says finished. Nothing is trained or copied, and
    M2's run folder and weights are only read."""
    if why := control_missing(seed, root):
        raise FileNotFoundError(f"no control record for seed {seed} without {why}")
    source = m2.finished_run(f"{CONTROL_FROM}-s{seed}", root)
    record = read_json(source / "run.json")
    weights = m2_weights(source, record)
    config = {"milestone": "m2b", "arm": CONTROL, "size": "n", "weights": record["config"]["weights"],
              "imgsz": record["config"]["imgsz"], "seed": seed, "name": f"{CONTROL}-s{seed}",
              "trained": False, "weights_from": source.name}
    run_dir = runs.start_run(f"m2b-{CONTROL}-s{seed}", config, seed=seed, root=root)
    runs.update_run(run_dir, status="finished", weights_from=source.name, checkpoint=str(weights),
                    checkpoint_sha256=sha256(weights))
    return run_dir
