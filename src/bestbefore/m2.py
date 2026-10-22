"""M2's data layout and fixed facts (docs/m2-synthetic-generator.md). The
arms, the training budget and the decision rules join in S3."""
from __future__ import annotations

from pathlib import Path

from .config import generated_dir

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
#   lists/<name>.txt                               image lists, absolute paths
#   manifests/<name>.json                          what made each part, and counts
def root() -> Path:
    return generated_dir() / "m2"


def derived_dir() -> Path:
    return root() / "derived"


def list_path(name: str) -> Path:
    return root() / "lists" / f"{name}.txt"


def manifest_path(name: str) -> Path:
    return root() / "manifests" / f"{name}.json"
