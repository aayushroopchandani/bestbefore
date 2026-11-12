"""Load any dataset by name, so every script can take `--set <name>`."""
from __future__ import annotations

from pathlib import Path

from . import derived, expdate, m2, m2b, store, yolo
from .config import expdate_dir
from .samples import Sample

# real-val: the 165 validation photos, which the YOLO conversion took out
# of real-train (seed 0). Its val.txt is the one record of which they are.
# real-val-rot, real-val-small: the same photos turned 90 or 270 degrees,
# and in 2 x 2 grids (M2's hard validation sets, scripts/derive_sets.py).
# gen-sample, gen, synth-val: generated photos (scripts/generate.py), read
# with their sidecars, so synth-val's targets load as exp and its strata as
# tags; gen-v2-sample, gen-v2, synth-val-v2: M2b's, under $GENERATED_DIR/m2b/,
# with real-val-rot180 (the validation photos turned 180°).
# store: store test set v1 (store.V1), the retailer's photos that M0 to
# M2 were tested on. v1-val: its validation part from M2b on, with v3's
# (S5b), as the store export in $STORE_TRAIN_DIR holds it: numbered copies,
# the M2b suite's store set (m2b.load_v1_val).
DERIVED = {**m2.DERIVED_SETS, **m2b.DERIVED_SETS}
GENERATED = {**m2.GENERATED_SETS, **m2b.GENERATED_SETS}
NAMES = (*expdate.SPLITS, "real-val", *DERIVED, *GENERATED, "store", m2b.STORE_SET)


def load_set(name: str) -> list[Sample]:
    if name in expdate.SPLITS:
        return expdate.load_split(name)
    if name == "real-val":
        ids = yolo.list_ids(expdate_dir() / "yolo", "val")
        return [s for s in expdate.load_split("real-train") if s.id in ids]
    if name in DERIVED:
        home = m2b if name in m2b.DERIVED_SETS else m2
        return derived.load_yolo_set(home.list_path(DERIVED[name]))
    if name in GENERATED:
        home = m2b if name in m2b.GENERATED_SETS else m2
        return derived.load_yolo_set(home.list_path(GENERATED[name]), home.sidecar_path(GENERATED[name]))
    if name == "store":
        return store.load_samples(store.V1)
    if name == m2b.STORE_SET:
        return m2b.load_v1_val()
    raise ValueError(f"unknown set {name!r}; one of {', '.join(NAMES)}")


def runs_root(name: str) -> Path | None:
    """Where a run on this set keeps its files: data/store/runs/ for the
    store photos (v1-val's too), which never go into results/; the default
    otherwise."""
    return store.RUNS_DIR if name in ("store", m2b.STORE_SET) else None
