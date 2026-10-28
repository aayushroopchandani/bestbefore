"""The store test sets: the retailer's photos, labelled by hand
(docs/m0-data-and-test-set.md: Part B for v1, Part D for v2).

Each set is a `StoreSet`: its photo folders under `$STORE_PHOTOS_DIR`,
where its files live and the hashes they were frozen with. Anything that
lists or loads store photos takes one, so it always says which set it
means. v1 (`V1`) is the test set M0 to M2 were measured on; v2 (`V2`),
new photos, is the test set once store photos are trained on.

The photos and their labels belong to the retailer. They stay in
`$STORE_PHOTOS_DIR` and `data/store/` (gitignored); only counts and the
frozen file's hash go into git (decision 7).

    data/store/labels.json     written by scripts/import_store_labels.py (B2)
    data/store/test_set.json   the same labels plus each photo's SHA-256,
                               written once by scripts/freeze_store_test_set.py (B3)
    data/store/v2/             v2's files, the same names, plus photos.json:
                               its photo list, frozen before labelling (D3)

labels.json and test_set.json map a photo's path relative to
`$STORE_PHOTOS_DIR` to its label:

    expiry      "YYYY-MM-DD", or "YYYY-MM" when the pack prints no day;
                None when this photo doesn't show a readable expiry
    derived     True when the expiry was worked out from mfg + shelf life
    mfg         the manufacture date, same format, or None
    format      how the expiry is printed, e.g. "DD/MM/YYYY"
    tag         the failure tag, one of TAGS
    unreadable  True when the expiry is in frame but can't be read, even zoomed in
    boxes       [{"label": one of BOX_LABELS, "xyxy": [x1, y1, x2, y2]}], 0..1
                of the upright photo
    note        free text
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import REPO_ROOT, store_photos_dir
from .images import upright_size
from .samples import Box, Expiry, Sample

DATA_DIR = REPO_ROOT / "data" / "store"
LABELS_PATH = DATA_DIR / "labels.json"
TEST_SET_PATH = DATA_DIR / "test_set.json"
# Anything drawn from the store photos (contact sheets, galleries of misses)
# goes here, never into results/.
RUNS_DIR = DATA_DIR / "runs"

# In the order they're tried: a photo gets the first that applies (decision 5).
TAGS = ("other-face", "sideways", "curved", "dot-matrix", "small-print", "clean")
BOX_LABELS = ("expiry", "mfg", "other_date")

# SHA-256 of data/store/test_set.json, pasted in when it was frozen (B3).
# v1, frozen 2026-10-21: results/store_test_set.md.
TEST_SET_SHA256: str | None = "7b9a4043886c9f4fcb37cba339962261acaef19e2993f7fba365a75f49b588f4"
# v2's, pasted in by hand as each is frozen: its photo list before labelling
# (D3), then its labels (D6). results/store_test_set_v2.md.
PHOTOS_V2_SHA256: str | None = "8be1f9e9cb256625259c17b88a070a9a30e204c94a85ff4ad1cd498be1d6080c"
TEST_SET_V2_SHA256: str | None = None


@dataclass(frozen=True)
class StoreSet:
    """One store test set: its photo folders under `$STORE_PHOTOS_DIR`, the
    folder holding its files, and the hash its test-set file was frozen with."""
    version: str
    folders: tuple[str, ...]
    data_dir: Path
    test_set_sha256: str | None
    # From v2 on, the photo list is frozen before labelling (Part D,
    # decision 3): photos.json's hash, None until then.
    photo_list: bool = False
    photos_sha256: str | None = None
    # Import check 4: Scanify's ground truth covers v1's photos only.
    scanify_check: bool = False

    @property
    def export_path(self) -> Path:
        return self.data_dir / "label_studio_export.json"

    @property
    def labels_path(self) -> Path:
        return self.data_dir / "labels.json"

    @property
    def test_set_path(self) -> Path:
        return self.data_dir / "test_set.json"

    @property
    def photos_path(self) -> Path:
        return self.data_dir / "photos.json"


# v1's files are the ones above, as they were before there was a v2.
V1 = StoreSet("v1", ("samples1", "samples2", "samples3"), DATA_DIR, TEST_SET_SHA256,
              scanify_check=True)
V2 = StoreSet("v2", ("testset2",), DATA_DIR / "v2", TEST_SET_V2_SHA256,
              photo_list=True, photos_sha256=PHOTOS_V2_SHA256)
SETS = {s.version: s for s in (V1, V2)}

_DATE = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?")


def parse_date(text: str) -> Expiry:
    """'2026-11-20', or '2026-11' when no day is printed."""
    match = _DATE.fullmatch(text.strip())
    if not match:
        raise ValueError(f"{text!r} is not YYYY-MM-DD or YYYY-MM")
    year, month, day = match.groups()
    return Expiry(int(year), int(month), int(day) if day else None)


def photos_on_disk(store_set: StoreSet, root: Path | None = None) -> list[str]:
    """Every photo in one set's folders, as a path relative to `root`, sorted."""
    root = root or store_photos_dir()
    missing = [f for f in store_set.folders if not (root / f).is_dir()]
    if missing:
        raise FileNotFoundError(f"store test set {store_set.version}: no folder "
                                f"{', '.join(missing)} in {root}")
    return sorted(p.relative_to(root).as_posix()
                  for folder in store_set.folders for p in (root / folder).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg"))


def sha256(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def to_samples(labels: dict[str, dict], root: Path | None = None) -> list[Sample]:
    root = root or store_photos_dir()
    samples = []
    for photo in sorted(labels):
        label, path = labels[photo], root / photo
        w, h = upright_size(path)
        samples.append(Sample(
            id=photo, image=path, width=w, height=h,
            boxes=tuple(Box(b["label"], tuple(b["xyxy"])) for b in label["boxes"]),
            expiry=parse_date(label["expiry"]) if label["expiry"] else None,
            tag=label["tag"]))
    return samples


def freeze(labels: dict[str, dict], version: str, date: str,
           root: Path | None = None) -> dict:
    """The test-set file's content: every label plus its photo's SHA-256."""
    root = root or store_photos_dir()
    return {"version": version, "frozen": date,
            "photos": {p: {**labels[p], "sha256": sha256(root / p)}
                       for p in sorted(labels)}}


def read_test_set(path: Path = TEST_SET_PATH, root: Path | None = None,
                  expected: str | None = None) -> dict:
    """The frozen test set, once its file and every photo are shown unchanged."""
    expected = expected or TEST_SET_SHA256
    if expected is None:
        raise RuntimeError("the store test set isn't frozen yet: "
                           "run scripts/freeze_store_test_set.py")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(
            f"{path} has changed since it was frozen. A label change is a new "
            "version (docs/m0-data-and-test-set.md, B3).")
    test_set = json.loads(data)
    root = root or store_photos_dir()
    changed = [p for p, label in test_set["photos"].items()
               if sha256(root / p) != label["sha256"]]
    if changed:
        raise RuntimeError(f"{len(changed)} store photos have changed since "
                           f"freezing: {', '.join(changed)}")
    return test_set


def freeze_photos(store_set: StoreSet, date: str, root: Path | None = None) -> dict:
    """The photo-list file's content: every photo in the set's folders and
    its SHA-256, frozen before any labelling."""
    root = root or store_photos_dir()
    return {"version": store_set.version, "frozen": date, "folders": list(store_set.folders),
            "photos": {p: sha256(root / p) for p in photos_on_disk(store_set, root)}}


def read_photo_list(store_set: StoreSet, root: Path | None = None) -> dict:
    """A set's frozen photo list, once the file and the set's folders are
    shown unchanged: no photo added, removed or changed."""
    v = store_set.version
    if not store_set.photo_list:
        raise ValueError(f"store test set {v} has no photo list of its own")
    if store_set.photos_sha256 is None:
        raise RuntimeError(f"store test set {v}'s photo list isn't frozen yet: "
                           f"run scripts/freeze_store_test_set.py --version {v} --photos")
    data = store_set.photos_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != store_set.photos_sha256:
        raise RuntimeError(f"{store_set.photos_path} has changed since it was frozen")
    photo_list = json.loads(data)
    root = root or store_photos_dir()
    listed, on_disk = photo_list["photos"], set(photos_on_disk(store_set, root))
    differ = {"added": sorted(on_disk - set(listed)),
              "removed": sorted(set(listed) - on_disk),
              "changed": sorted(p for p in on_disk & set(listed) if sha256(root / p) != listed[p])}
    if any(differ.values()):
        raise RuntimeError(f"store test set {v}'s photos differ from its frozen list: "
                           + "; ".join(f"{what}: {', '.join(ps)}" for what, ps in differ.items() if ps))
    return photo_list


def read_frozen(store_set: StoreSet, root: Path | None = None) -> dict:
    """A set's frozen test set, once its file and every photo are shown
    unchanged and, for a set with a photo list, its photos are the list's."""
    v = store_set.version
    if store_set.test_set_sha256 is None:
        raise RuntimeError(f"store test set {v} isn't frozen yet: "
                           f"run scripts/freeze_store_test_set.py --version {v}")
    test_set = read_test_set(store_set.test_set_path, root, expected=store_set.test_set_sha256)
    if test_set["version"] != v:
        raise RuntimeError(f"{store_set.test_set_path} holds {test_set['version']}, not {v}")
    if store_set.photo_list:
        listed = read_photo_list(store_set, root)["photos"]
        if {p: label["sha256"] for p, label in test_set["photos"].items()} != listed:
            raise RuntimeError(f"store test set {v}'s labelled photos aren't its frozen photo list")
    return test_set


def read_labels(path: Path = LABELS_PATH) -> dict[str, dict]:
    """labels.json as imported, not yet frozen."""
    return json.loads(path.read_text())


def labels(store_set: StoreSet) -> dict[str, dict]:
    """One set's frozen labels. Before they're frozen (B3), its labels.json."""
    if store_set.test_set_sha256 is None:
        print(f"note: store test set {store_set.version} isn't frozen yet; "
              f"reading {store_set.labels_path}", file=sys.stderr)
        return read_labels(store_set.labels_path)
    return read_frozen(store_set)["photos"]


def load_samples(store_set: StoreSet) -> list[Sample]:
    return to_samples(labels(store_set))


def tag_counts(labels: dict[str, dict]) -> list[dict[str, str]]:
    """One row per tag, then the total: what the test set holds."""
    def row(name: str, group: list[dict]) -> dict[str, str]:
        expiry = [lab["expiry"] for lab in group if lab["expiry"]]
        return {"tag": name, "photos": f"{len(group)}",
                "with an expiry": f"{len(expiry)}",
                "to the day": f"{sum(len(e) == 10 for e in expiry)}",
                "to the month": f"{sum(len(e) == 7 for e in expiry)}",
                "derived": f"{sum(lab['derived'] for lab in group)}",
                "unreadable": f"{sum(lab['unreadable'] for lab in group)}"}

    rows = [row(t, [lab for lab in labels.values() if lab["tag"] == t]) for t in TAGS]
    return rows + [row("**all**", list(labels.values()))]
