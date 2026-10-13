"""Load any dataset by name, so every script can take `--set <name>`."""
from __future__ import annotations

from . import expdate
from .samples import Sample

NAMES = tuple(expdate.SPLITS)


def load_set(name: str) -> list[Sample]:
    if name in expdate.SPLITS:
        return expdate.load_split(name)
    raise ValueError(f"unknown set {name!r}; one of {', '.join(NAMES)}")
