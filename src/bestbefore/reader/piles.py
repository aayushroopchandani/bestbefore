"""What each crop pile holds and where its crops come from
(docs/m3-date-reader.md, "Piles and roles", S3).

A pile is read photo by photo: each `Photo` names an image and the crops to
cut from it (`Item`s), every box in 0..1 of the upright photo, every text as
its source gives it. scripts/make_crops.py cuts them (reader/crops.py) and
stores them (reader/shards.py).

Training piles:
  expdate-real    ExpDate real-train without real-val: its dates
  expdate-synth   Products-Synth without the 14 copies of a validation
                  photo (m2.REUSED_VAL_COPIES): its dates
  gen             gen-v2: dates, codes, and class-less tokens of at least
                  MIN_CLASSLESS characters (decision 9), each turned
                  upright by its block's quarter
Suites, cut as the photo shows them (the orientation step is tested on them):
  real-val          ExpDate's 165 validation photos: their dates
  real-val-rot      the same photos turned 90 or 270 degrees (M2's)
  real-val-rot180   the same photos turned 180 degrees (M2b's)
  synth-val         M2's 2,100 generated photos: each one's target date
  synth-val-v2      M2b's 2,400: each one's target date
And gen-sample, gen-v2-sample's 500 photos as the gen pile: the laptop's trial.

The store's (S4), from the numbered store export ($STORE_TRAIN_DIR: its
upright copies train_0 and val, their label files, manifest.json and S4's
text.json, which must come from the frozen store text,
store_text.TEXT_SHA256):
  store           v1-train and v3-train: every date box and every
                  shelf-life phrase (class-less), turned upright by the
                  owner's turn
  store-train     the same photos' dates, cut as they lie, the owner's turn
                  in tags["upright"]: a memorisation check only
  store-val       v1-val and v3-val: their dates, cut as they lie
They show the retailer's photos: make_crops.py writes them owner-only
(PRIVATE), and their crops are never visibility-filtered (S5's filter
takes VISIBILITY_FILTERED only). A store crop's source is its numbered
photo and line, never a photo's name; an expiry's or mfg's parts are its
frozen value, never parsed; an other date's are None.

Never ExpDate test (decision 20: it is cut once, at S13). Never store set
v2: it isn't in the export.

Where a text comes from: ExpDate's transcription; a generated token's own
text from its sidecar. A date's parts: a generated date's from its sidecar,
never parsed from its text (the frozen parser reads gen-v2's DD.MM.YY
year first, S2); an ExpDate date's from the frozen parser on its
transcription (ExpDate's train and synth splits mark no parts; the parser
is ExpDate's rule), None where it finds no date.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .. import derived, expdate, m2, m2b, store, store_text, store_train, yolo
from ..config import expdate_dir, store_train_dir
from ..runs import read_json
from . import parse

MIN_CLASSLESS = 3       # decision 9: class-less tokens of at least 3 characters
TRAINING = ("expdate-real", "expdate-synth", "gen")
SUITES = ("real-val", "real-val-rot", "real-val-rot180", "synth-val", "synth-val-v2")
JOB_PILES = TRAINING + SUITES       # what S3's cluster job made
STORE_TRAINING = ("store",)
STORE_SUITES = ("store-train", "store-val")
STORE_PILES = STORE_TRAINING + STORE_SUITES     # S4's job
PRIVATE = STORE_PILES               # show the retailer's photos: owner-only folders and files
VISIBILITY_FILTERED = TRAINING      # S5's filter: never a suite, never the store's crops
TRIAL = ("gen-sample",)             # the laptop's trial, beside real-val
PILES = JOB_PILES + STORE_PILES + TRIAL
STORE_ROLE = {"expiry": "expiry", "mfg": "mfg", "other_date": "other"}

XYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class Item:
    """One crop to cut. `turn`: the quarter turns (anticlockwise) to give it
    when cut; tags["upright"]: those still needed afterwards to stand it
    upright (shards.Crop)."""
    xyxy: XYXY
    text: str
    cls: str | None
    source: str
    role: str | None = None
    parts: tuple[int, int, int | None] | None = None
    turn: int = 0
    tags: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Photo:
    image: Path
    items: tuple[Item, ...]


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# --- ExpDate ---------------------------------------------------------------------------

def val_ids() -> frozenset[str]:
    """real-val's photos: the YOLO conversion's val.txt (sets.load_set's rule)."""
    return frozenset(yolo.list_ids(expdate_dir() / "yolo", "val"))


def expdate_parts(text: str) -> tuple[int, int, int | None] | None:
    """The frozen parser's one date in an ExpDate transcription (S2's truth
    for real-val), as (year, month, day or None); None if it finds none."""
    date = parse.crop_date(parse.read_dates(text))
    return None if date is None else (date.year, date.month, date.day)


def _expdate(split: str, keep, tag: dict) -> Iterator[Photo]:
    """Every dated photo of an ExpDate split that `keep(name)` admits, its
    dates with text as crops, upright as the photo shows them."""
    if split not in ("real-train", "synth"):
        raise ValueError(f"ExpDate's {split!r} is never cut here (only real-train and synth)")
    folder = expdate.split_dir(split)
    ann = expdate.read_annotations(split)
    for name in sorted(ann):
        if not keep(name):
            continue
        entry = ann[name]
        items = []
        for k, a in enumerate(entry["ann"]):
            if a["cls"] != "date":
                continue
            text = (a.get("transcription") or "").strip()
            xyxy = expdate.normalise(a["bbox"], entry["width"], entry["height"])
            items.append(Item(xyxy, text, "date", f"{split}/{name}#{k}", parts=expdate_parts(text),
                              tags={**tag, "upright": 0, "parts_from": "parser"}))
        if items:
            yield Photo(folder / "images" / name, tuple(items))


def _turned_val(home, list_name: str) -> Iterator[Photo]:
    """real-val's dates cut from a turned copy, as it lies. A copy's label
    file keeps the original's boxes in order (zero-area ones dropped, as
    yolo.label_lines does), so each date's text is found by its place; the
    turn is found by matching every box, and must be one turn for all."""
    ann = expdate.read_annotations("real-train")
    for path in derived.read_list(home.list_path(list_name)):
        entry = ann[path.name]
        w, h = entry["width"], entry["height"]
        kept = [(k, a, expdate.normalise(a["bbox"], w, h)) for k, a in enumerate(entry["ann"])]
        kept = [(k, a, xy) for k, a, xy in kept if xy[2] > xy[0] and xy[3] > xy[1]]
        turned = derived.read_boxes(path)
        if len(turned) != len(kept):
            raise ValueError(f"{path}: {len(turned)} labels, {len(kept)} boxes in the annotations")
        quarter = _the_turn([xy for _, _, xy in kept], [b.xyxy for b in turned], path)
        items = []
        for (k, a, _), b in zip(kept, turned):
            if yolo.CLASSES[yolo.LABEL_TO_CLASS[a["cls"]]] != b.label:
                raise ValueError(f"{path}: label {b.label} where the annotations have {a['cls']}")
            if a["cls"] != "date":
                continue
            text = (a.get("transcription") or "").strip()
            items.append(Item(b.xyxy, text, "date", f"{list_name}/{path.name}#{k}", parts=expdate_parts(text),
                              tags={"upright": (4 - quarter) % 4, "photo_turn": quarter, "parts_from": "parser"}))
        if items:
            yield Photo(path, tuple(items))


def _the_turn(original: list[XYXY], turned: list[XYXY], path: Path, tol: float = 2e-5) -> int:
    """The one quarter turn (anticlockwise) that takes every original box
    to its turned label."""
    for q in (1, 2, 3):
        if all(max(abs(p - t) for p, t in zip(derived.turn_box(o, q), b)) <= tol
               for o, b in zip(original, turned)):
            return q
    raise ValueError(f"{path}: no quarter turn takes its photo's boxes to its labels")


# --- Generated ---------------------------------------------------------------------------

def gen_wanted(token: dict) -> bool:
    """Decision 9: a training pile takes dates, codes and class-less tokens
    of at least MIN_CLASSLESS characters; never keyword tokens (due, prod)."""
    if token["cls"] in ("date", "code"):
        return True
    return token["cls"] is None and len(token["text"].strip()) >= MIN_CLASSLESS


def target(token: dict) -> bool:
    """A validation photo's target date (synth-val, synth-val-v2)."""
    return token["cls"] == "date" and token.get("role") == "target"


def sidecar_lines(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def generated_items(line: dict, list_name: str, wanted, as_cut: bool) -> tuple[list[Item], int]:
    """A sidecar line's crops, and how many wanted tokens have no box (no
    ink left): each turned upright by its block's quarter, or cut as it
    lies (`as_cut`), its truth in tags["upright"]."""
    w, h = line["size"]
    items, no_box = [], 0
    for bi, block in enumerate(line["blocks"]):
        g = block["geometry"]
        upright = (4 - g["quarter"]) % 4
        for ti, t in enumerate(block["tokens"]):
            if not wanted(t):
                continue
            if t["box"] is None:
                no_box += 1
                continue
            x1, y1, x2, y2 = t["box"]
            tags = {"style": block["style"], "kind": block["kind"], "orientation": g["orientation"],
                    "surface": g["surface"], "band": block["band"],
                    "upright": upright if as_cut else 0}
            if "stratum" in line:
                tags["stratum"] = line["stratum"]
            date = t["cls"] == "date"
            if date:
                tags |= {"format": t["format"], "parts_from": "sidecar"}
            items.append(Item((x1 / w, y1 / h, x2 / w, y2 / h), t["text"].strip(), t["cls"],
                              f"{list_name}/{line['image']}#b{bi}t{ti}",
                              role=t["role"] if date else None,
                              parts=tuple(t["parts"]) if date else None,
                              turn=0 if as_cut else upright, tags=tags))
    return items, no_box


def _generated(home, list_name: str, wanted, as_cut: bool, no_box: list[int]) -> Iterator[Photo]:
    """A generated set's photos in list order, each with its sidecar line
    (written in photo order). Tokens wanted but without a box are counted
    into no_box[0]."""
    images = derived.read_list(home.list_path(list_name))
    lines = sidecar_lines(home.sidecar_path(list_name))
    for path in images:
        line = next(lines)
        if line["image"] != path.name:
            raise ValueError(f"{list_name}: the sidecar's {line['image']} where the list has {path.name}")
        items, missing = generated_items(line, list_name, wanted, as_cut)
        no_box[0] += missing
        if items:
            yield Photo(path, tuple(items))
    if next(lines, None) is not None:
        raise ValueError(f"{list_name}: the sidecar has more lines than the list has photos")


# --- The store (S4) -----------------------------------------------------------------------

def store_text_path() -> Path:
    return store_train_dir() / store_text.NAME


def store_items(name: str, entry: dict, meta: dict, lines: list[XYXY], as_cut: bool) -> list[Item]:
    """One numbered photo's crops: each date box (`lines`: its label file's
    boxes, upright, in order) and, unless cut as they lie (a suite), each
    shelf-life box. Training crops are turned upright by the owner's turn;
    a suite's lie as the photo shows them, the turn in tags["upright"]."""
    tags = {"design": meta["design"], "part": entry["part"], "tag": meta["tag"]}
    items = []
    for b in entry["boxes"]:
        if meta["labels"][b["index"]] != b["label"]:
            raise ValueError(f"{name}: line {b['index']} is {meta['labels'][b['index']]}, text.json says {b['label']}")
        parts = None
        if b["label"] in store_text.VALUE_LABELS and b["value"]:
            d = store.parse_date(b["value"])
            parts = (d.year, d.month, d.day)
        items.append(Item(lines[b["index"]], b["text"], "date", f"{name}#{b['index']}", role=STORE_ROLE[b["label"]],
                          parts=parts, turn=0 if as_cut else b["turn"],
                          tags={**tags, "upright": b["turn"] if as_cut else 0,
                                "parts_from": "frozen" if parts else "none"}))
    if not as_cut:
        for k, b in enumerate(entry["life"]):
            items.append(Item(tuple(b["xyxy"]), b["text"], None, f"{name}#life{k}", turn=b["turn"],
                              tags={**tags, "upright": 0, "kind": "shelf-life"}))
    return items


def _store(part: str, as_cut: bool) -> Iterator[Photo]:
    """The export's photos of one part (train or val), in numbered order,
    from its upright copies."""
    folder = store_train_dir()
    text = json.loads(store_text_path().read_text())
    manifest = json.loads((folder / store_train.MANIFEST).read_text())
    upright = next(f for f, q in store_train.FOLDERS[part].items() if q == 0)
    for name in sorted(text["photos"]):
        entry = text["photos"][name]
        if entry["part"] != part:
            continue
        image = folder / "images" / upright / f"{name}.jpg"
        lines = [b.xyxy for b in derived.read_boxes(image)]
        if len(lines) != len(manifest["photos"][name]["labels"]):
            raise ValueError(f"{name}: {len(lines)} label lines, the manifest lists "
                             f"{len(manifest['photos'][name]['labels'])}")
        if items := store_items(name, entry, manifest["photos"][name], lines, as_cut):
            yield Photo(image, tuple(items))


# --- The piles -----------------------------------------------------------------------

def photos(pile: str, no_box: list[int] | None = None) -> Iterator[Photo]:
    """A pile's photos, in a fixed order, each with its crops."""
    no_box = no_box if no_box is not None else [0]
    if pile == "expdate-real":
        val = val_ids()
        return _expdate("real-train", lambda n: n not in val, {"split": "real-train"})
    if pile == "expdate-synth":
        return _expdate("synth", lambda n: n not in m2.REUSED_VAL_COPIES, {"split": "synth"})
    if pile == "real-val":
        val = val_ids()
        return _expdate("real-train", lambda n: n in val, {"split": "real-train"})
    if pile == "real-val-rot":
        return _turned_val(m2, "real_val_rot")
    if pile == "real-val-rot180":
        return _turned_val(m2b, "real_val_rot180")
    if pile == "synth-val":
        return _generated(m2, "synth_val", target, True, no_box)
    if pile == "synth-val-v2":
        return _generated(m2b, "synth_val_v2", target, True, no_box)
    if pile == "gen":
        return _generated(m2b, "gen_v2", gen_wanted, False, no_box)
    if pile == "gen-sample":
        return _generated(m2b, "gen_v2_sample", gen_wanted, False, no_box)
    if pile == "store":
        return _store("train", as_cut=False)
    if pile == "store-train":
        return _store("train", as_cut=True)
    if pile == "store-val":
        return _store("val", as_cut=True)
    raise ValueError(f"unknown pile {pile!r}; one of {', '.join(PILES)}")


# Each pile's generated set or derived list, its home (m2 or m2b) and manifest.
_GENERATED = {"gen": (m2b, "gen_v2", "gen-v2"), "gen-sample": (m2b, "gen_v2_sample", "gen-v2-sample"),
              "synth-val": (m2, "synth_val", "synth-val"), "synth-val-v2": (m2b, "synth_val_v2", "synth-val-v2")}
_DERIVED = {"real-val-rot": (m2, "real_val_rot"), "real-val-rot180": (m2b, "real_val_rot180")}


def missing(pile: str) -> str | None:
    """Why a pile can't be cut yet, or None: its sources must exist and,
    for derived and generated ones, their manifests vouch for them (a
    generated set's sidecar the very one its manifest checked)."""
    if pile in _DERIVED:
        home, name = _DERIVED[pile]
        return home.list_missing(name)
    if pile in _GENERATED:
        home, name, manifest = _GENERATED[pile]
        if not home.list_path(name).exists() or not home.sidecar_path(name).exists():
            return f"{name}: its list or sidecar isn't there"
        record = read_json(home.manifest_path(manifest))
        if record is None:
            return f"{name}: no manifests/{manifest}.json"
        if pile != "gen-sample":        # a sample's manifest vouches for nothing more
            if record.get("complete") is not True:
                return f"{name}: manifests/{manifest}.json doesn't say complete"
            if sha256_file(home.sidecar_path(name)) != record.get("sidecar_sha256"):
                return f"{name}: its sidecar isn't the one manifests/{manifest}.json checked"
        return None
    if pile in STORE_PILES:
        return store_missing()
    if pile in PILES:
        return None if (expdate_dir() / "yolo" / "val.txt").exists() else "ExpDate's yolo/val.txt isn't there"
    raise ValueError(f"unknown pile {pile!r}")


def store_missing() -> str | None:
    """Why the store's piles can't be cut, or None: the export's text.json
    must be S4's (version "text"), made from the frozen data/store/text.json
    (store_text.TEXT_SHA256) for this very export (its manifest's hash)."""
    path, manifest = store_text_path(), store_train_dir() / store_train.MANIFEST
    if not path.exists() or not manifest.exists():
        return f"{path} or the export's manifest isn't there"
    text = json.loads(path.read_text())
    if text.get("version") != "text":
        return f"{path} is version {text.get('version')!r}, not S4's 'text'"
    if store_text.TEXT_SHA256 is None or text.get("text_sha256") != store_text.TEXT_SHA256:
        return f"{path} wasn't made from the frozen store text (store_text.TEXT_SHA256)"
    if text.get("export_manifest_sha256") != sha256_file(manifest):
        return f"{path} was made for another export (manifest hashes differ)"
    return None


def sources(pile: str) -> dict[str, Path]:
    """The files a pile is cut from (images aside), whose SHA-256 its
    manifest keeps."""
    out: dict[str, Path] = {}
    if pile in ("expdate-real", "real-val", "real-val-rot", "real-val-rot180"):
        out["real-train/annotations.json"] = expdate.split_dir("real-train") / "annotations.json"
        out["yolo/val.txt"] = expdate_dir() / "yolo" / "val.txt"
    if pile == "expdate-synth":
        out["synth/annotations.json"] = expdate.split_dir("synth") / "annotations.json"
    if pile in _DERIVED:
        home, name = _DERIVED[pile]
        out[f"{home.__name__.rsplit('.', 1)[-1]}/lists/{name}.txt"] = home.list_path(name)
    if pile in _GENERATED:
        home, name, _ = _GENERATED[pile]
        short = home.__name__.rsplit(".", 1)[-1]
        out[f"{short}/lists/{name}.txt"] = home.list_path(name)
        out[f"{short}/sidecars/{name}.jsonl"] = home.sidecar_path(name)
    if pile in STORE_PILES:
        out["store/manifest.json"] = store_train_dir() / store_train.MANIFEST
        out["store/text.json"] = store_text_path()
    return out


def upstream(pile: str) -> dict:
    """What the manifests of the piles' sources say they hold, counted by
    the code that made them, not by this module: dates for ExpDate's
    piles (M2's and M2b's turned copies hold the same dates as their
    originals), generated dates for gen, targets (one a photo) for the
    validation sets. scripts/make_crops.py --check compares the pile with it."""
    if pile in ("expdate-real", "expdate-synth", "real-val", "real-val-rot"):
        sets = (read_json(m2.manifest_path("derived")) or {}).get("sets", {})
        name = {"expdate-real": "real_rot", "expdate-synth": "synth_rot",
                "real-val": "real_val_rot", "real-val-rot": "real_val_rot"}[pile]
        return {"date": sets[name]["dates"], "from": f"m2/manifests/derived.json, {name}"} if name in sets else {}
    if pile in STORE_PILES:
        part = "val" if pile == "store-val" else "train"
        listed = json.loads((store_train_dir() / store_train.MANIFEST).read_text())["photos"].values()
        return {"date": sum(label in store.BOX_LABELS for p in listed if p["part"] == part for label in p["labels"]),
                "from": f"the store export's manifest.json, its {part} photos' date labels"}
    if pile == "real-val-rot180":
        sets = (read_json(m2b.manifest_path("derived")) or {}).get("sets", {})
        return ({"date": sets["real_val_rot180"]["dates"], "from": "m2b/manifests/derived.json, real_val_rot180"}
                if "real_val_rot180" in sets else {})
    home, _, manifest = _GENERATED[pile]
    record = read_json(home.manifest_path(manifest)) or {}
    where = f"{home.__name__.rsplit('.', 1)[-1]}/manifests/{manifest}.json"
    if pile in ("synth-val", "synth-val-v2"):
        return {"date": record.get("photos"), "from": f"{where}, photos (one target each)"}
    dates = record.get("counts", {}).get("generated_dates", record.get("generated_dates"))
    return {"date": dates, "from": f"{where}, generated_dates"}
