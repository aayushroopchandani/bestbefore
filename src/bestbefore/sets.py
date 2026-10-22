"""Load any dataset by name, so every script can take `--set <name>`."""
from __future__ import annotations

from pathlib import Path

from . import derived, expdate, m2, store, yolo
from .config import expdate_dir
from .samples import Sample

# real-val: the 165 validation photos, which the YOLO conversion took out
# of real-train (seed 0). Its val.txt is the one record of which they are.
# real-val-rot, real-val-small: the same photos turned 90 or 270 degrees,
# and in 2 x 2 grids (M2's hard validation sets, scripts/derive_sets.py).
# store: the retailer's photos, test only (store.py).
DERIVED = {"real-val-rot": "real_val_rot", "real-val-small": "real_val_small"}
NAMES = (*expdate.SPLITS, "real-val", *DERIVED, "store")


def load_set(name: str) -> list[Sample]:
    if name in expdate.SPLITS:
        return expdate.load_split(name)
    if name == "real-val":
        ids = yolo.list_ids(expdate_dir() / "yolo", "val")
        return [s for s in expdate.load_split("real-train") if s.id in ids]
    if name in DERIVED:
        return derived.load_yolo_set(m2.list_path(DERIVED[name]))
    if name == "store":
        return store.load_samples()
    raise ValueError(f"unknown set {name!r}; one of {', '.join(NAMES)}")


def runs_root(name: str) -> Path | None:
    """Where a run on this set keeps its files: data/store/runs/ for the
    store photos, which never go into results/; the default otherwise."""
    return store.RUNS_DIR if name == "store" else None
