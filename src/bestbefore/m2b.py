"""M2b's layout, model sizes, step-1 arms and the no-training control, fixed
before any run (docs/m2b-store-and-size.md). Step 1 is M2's gen-rot-1280
arm trained from other COCO weights: only the weights change (decision 1).
n's step 1 is M2's own three runs, kept as they are and recorded as the
control (decision 12). scripts/train_detector.py trains a step-1 arm and
scripts/m2b_chain.py runs them all; S10 adds step 2 and the decision
rules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import m2, runs
from .config import checkpoint_dir, generated_dir, store_train_dir
from .runs import read_json
from .store import sha256


# $GENERATED_DIR/m2b/
#   gen/, derived/      gen-v2 and the 180° copies (S8, S9)
#   lists/<name>.txt    image lists, absolute paths
#   manifests/<name>.json
# apart from M2's $GENERATED_DIR/m2/, which doesn't change.
def root() -> Path:
    return generated_dir() / "m2b"


def list_path(name: str) -> Path:
    return root() / "lists" / f"{name}.txt"


def manifest_path(name: str) -> Path:
    return root() / "manifests" / f"{name}.json"


# --- What step 2 waits for, made in S4 to S9 ---------------------------------------

# M2b's own lists for step 2's replay (S9), and the manifest that vouches for each.
STEP2_LISTS = ("gen_v2", "real_rot180", "synth_rot180")
MANIFEST_OF = {"gen_v2": "gen-v2", "real_rot180": "derived", "synth_rot180": "derived"}
MADE_BY = {"gen-v2": "S9's generation", "derived": "S9's derive_sets.py --only rot180"}


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


def decisions(root: Path | None = None) -> dict:
    """Every decision whose inputs are complete: none yet. S10 adds the
    noise band and rules R1 to R3, which need the M2b suite."""
    return {}


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
