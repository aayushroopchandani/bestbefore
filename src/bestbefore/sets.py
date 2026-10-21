"""Load any dataset by name, so every script can take `--set <name>`."""
from __future__ import annotations

from . import expdate, store, yolo
from .config import expdate_dir
from .samples import Sample

# real-val: the 165 validation photos, which the YOLO conversion took out
# of real-train (seed 0). Its val.txt is the one record of which they are.
# store: the retailer's photos, test only (store.py).
NAMES = (*expdate.SPLITS, "real-val", "store")


def load_set(name: str) -> list[Sample]:
    if name in expdate.SPLITS:
        return expdate.load_split(name)
    if name == "real-val":
        ids = yolo.list_ids(expdate_dir() / "yolo", "val")
        return [s for s in expdate.load_split("real-train") if s.id in ids]
    if name == "store":
        return store.load_samples()
    raise ValueError(f"unknown set {name!r}; one of {', '.join(NAMES)}")
