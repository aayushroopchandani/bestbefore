"""M3's training stream (docs/m3-date-reader.md, "The mix", "Augmentation",
S5), and the visibility measure that keeps empty crops out of it.

The stream, in each loader worker, forever:

  draw     a pile by weight (m3.PILE_WEIGHT over the arm's piles), then a
           record: uniformly among those a pile may give (a filtered
           pile's crops at or over m3.VISIBLE_MIN); the store pile picks a
           pack design first, then one of its crops (Mix)
  view     the crop's box with S2's padding, each edge jittered, cut from
           the stored crop's margin, augmented (augmented_view)
  bucket   POOL batches' worth of views sorted by width, cut into batches,
           handed out in random order: a batch pads little
  batch    onnx_run.preprocess (the one preprocessing), the texts as
           the reader's classes (its onnx_run.Spec: M3's charset for the
           CRNN, PP-OCRv6's dictionary for it, S6)

Seeded per run and per worker. Suites are read with plain_view: the same
cut, no jitter, no augmentation.

The visibility measure (S5, `visibility`): S3's sheets showed about 4 or 5
of every 64 generated crops with no print a person could read, and
ExpDate's black synthetic photos give blank crops; each still carries its
text, and a reader trained on them learns to answer a date from nothing.
The measure is computed once per record (scripts/visibility.py) and kept
beside the pile's LMDB; training never draws a crop of
piles.VISIBILITY_FILTERED under m3.VISIBLE_MIN.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import m3
from . import crops, onnx_run, piles, shards

# --- The visibility measure -----------------------------------------------------------

VIS_SCALE = 12.0        # the box's short side, in px, the measure looks at: strokes about 1.5 px
VIS_FLOOR = 1.0         # noise is never taken under one grey level
VIS_SPREAD = (5, 95)    # the percentiles whose distance is the print's spread
VIS_NOISE_Q = 30        # the Laplacian's percentile the noise is read from
_LAPLACIAN = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)   # its response to noise sigma: 6 sigma
_HALF_NORMAL_Q30 = 0.3853       # |N(0, 1)|'s 30th percentile
_WHITE_NOISE_SPREAD = 3.29      # N(0, 1)'s 5th to 95th percentile
VISIBILITY_FILE, VISIBILITY_RECORD = "visibility.npy", "visibility.json"


def visibility(rgb: np.ndarray, box) -> float:
    """How far the print inside `box` stands out from the crop's own noise.

    The crop is shrunk so the box's short side is VIS_SCALE px (never
    enlarged): about the scale a reader sees print at, and where noise from
    the camera and JPEG is nearer independent from pixel to pixel. There:
      - noise: a robust estimate over the whole crop, from the 3x3
        Laplacian's 30th percentile (robust to the print's own edges), at
        least VIS_FLOOR grey levels;
      - spread: the box's colours, with slow shading taken out (minus a
        blur as wide as the box's short side), projected on their main
        colour axis, so print that differs in colour only counts; the
        distance between their 5th and 95th percentiles.
    The result is the spread over the spread white noise of that level
    would give (3.29 sigmas): white noise alone scores about 1, clear print
    tens. Blotchy noise (JPEG of a noisy photo) is undercounted, so a crop
    of pure noise scores 2.5 to 3. Known limit: ink nearly the colour of a
    textured surface scores like faint readable print."""
    x1, y1, x2, y2 = box
    image = rgb.astype(np.float32)
    f = min(1.0, VIS_SCALE / max(min(x2 - x1, y2 - y1), 1e-6))
    if f < 1.0:
        w, h = max(3, round(image.shape[1] * f)), max(3, round(image.shape[0] * f))
        fx, fy = w / image.shape[1], h / image.shape[0]
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        x1, x2, y1, y2 = x1 * fx, x2 * fx, y1 * fy, y2 * fy
    h, w = image.shape[:2]
    laplacian = np.abs(cv2.filter2D(image, -1, _LAPLACIAN, borderType=cv2.BORDER_REFLECT))
    noise = max(VIS_FLOOR, float(np.percentile(laplacian, VIS_NOISE_Q)) / (_HALF_NORMAL_Q30 * 6.0))
    detail = image - cv2.GaussianBlur(image, (0, 0), max(1.0, min(x2 - x1, y2 - y1)),
                                      borderType=cv2.BORDER_REFLECT)
    bx1, by1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    bx2, by2 = min(w, max(bx1 + 1, math.ceil(x2))), min(h, max(by1 + 1, math.ceil(y2)))
    colours = detail[by1:by2, bx1:bx2].reshape(-1, 3).astype(np.float64)
    colours -= colours.mean(axis=0)
    axis = np.linalg.eigh(colours.T @ colours)[1][:, -1]       # the main colour axis
    lo, hi = np.percentile(colours @ axis, VIS_SPREAD)
    return float(hi - lo) / (_WHITE_NOISE_SPREAD * noise)


def visibility_code() -> str:
    """The measure's identity: its source and its constants. A pile
    measured by any other version is refused (the cut-off was chosen on
    this one)."""
    constants = (VIS_SCALE, VIS_FLOOR, VIS_SPREAD, VIS_NOISE_Q, _LAPLACIAN.tolist(), _HALF_NORMAL_Q30,
                 _WHITE_NOISE_SPREAD)
    return hashlib.sha256((inspect.getsource(visibility) + repr(constants)).encode()).hexdigest()


def read_visibility(folder: Path, count: int) -> np.ndarray:
    """A pile's measure, one value a record in key order, after checking
    that it was measured by this code on this very pile."""
    record_path = folder / VISIBILITY_RECORD
    if not record_path.exists():
        raise FileNotFoundError(f"{folder} has no {VISIBILITY_RECORD}: run scripts/visibility.py measure")
    record = json.loads(record_path.read_text())
    if record.get("code") != visibility_code():
        raise RuntimeError(f"{folder} was measured by another version of data.visibility: measure it again")
    if record.get("manifest_sha256") != piles.sha256_file(folder / "manifest.json"):
        raise RuntimeError(f"{folder}'s pile changed since it was measured: measure it again")
    values = np.load(folder / VISIBILITY_FILE)
    if len(values) != count:
        raise RuntimeError(f"{folder}: {len(values)} values for {count} records")
    return values


# --- What may be drawn ---------------------------------------------------------------

@dataclass(frozen=True)
class PileIndex:
    """The records of one pile training may draw: `kept` (a filtered pile's
    visible crops, else all), or for the store, `designs` (each pack
    design's records)."""
    name: str
    count: int
    kept: np.ndarray
    designs: tuple[np.ndarray, ...] | None = None


def record_number(key: str) -> int:
    return int(key.rsplit("/", 1)[1])


def pile_index(root: Path, name: str, visible_min: float | None) -> PileIndex:
    folder = root / name
    env = shards.open_pile(folder)
    try:
        count = shards.count(env)
        designs = None
        if name in piles.VISIBILITY_FILTERED:
            if visible_min is None:
                raise RuntimeError("m3.VISIBLE_MIN isn't set: the cut-off is chosen by eye before any training (S5)")
            kept = np.flatnonzero(read_visibility(folder, count) >= visible_min)
        else:
            kept = np.arange(count)
        if name in piles.STORE_TRAINING:        # design first (M2b decision 7): ids, never names
            by: dict[str, list[int]] = {}
            for crop in shards.records(env, name):
                by.setdefault(crop.tags["design"], []).append(record_number(crop.key))
            designs = tuple(np.array(v) for _, v in sorted(by.items()))
    finally:
        env.close()
    if not len(kept):
        raise RuntimeError(f"{name}: nothing to draw")
    return PileIndex(name, count, kept, designs)


class Mix:
    """Draws (pile, record): a pile by its weight, normalised over the
    given piles; then a record uniformly among those kept, or for the
    store a design and then one of its records."""

    def __init__(self, index: dict[str, PileIndex], weights: dict[str, float] = m3.PILE_WEIGHT):
        self.names = list(index)
        self.index = index
        w = np.array([weights[n] for n in self.names], np.float64)
        self.shares = w / w.sum()
        self._cumulative = np.cumsum(self.shares)

    @classmethod
    def for_arm(cls, root: Path, arm: str, visible_min: float | None) -> Mix:
        return cls({p: pile_index(root, p, visible_min) for p in m3.ARMS[arm]})

    def draw(self, rng: np.random.Generator) -> tuple[int, int]:
        """(the pile's number in self.names, the record's number)."""
        p = min(int(np.searchsorted(self._cumulative, rng.random(), side="right")), len(self.names) - 1)
        ix = self.index[self.names[p]]
        if ix.designs is not None:
            design = ix.designs[rng.integers(len(ix.designs))]
            return p, int(design[rng.integers(len(design))])
        return p, int(ix.kept[rng.integers(len(ix.kept))])

    def describe(self) -> dict:
        """Counts for a run record: no design names (they are ids, but the
        record needs only how many)."""
        return {n: {"share": round(float(s), 6), "records": self.index[n].count, "drawn_from": len(self.index[n].kept),
                    "designs": None if self.index[n].designs is None else len(self.index[n].designs)}
                for n, s in zip(self.names, self.shares)}


# --- Views and augmentation (fixed, not tuned) -------------------------------------------

@dataclass(frozen=True)
class Augment:
    """docs/m3-date-reader.md, "Augmentation", with the numbers S5 fixed
    where the plan left them open."""
    jitter: float = m3.EDGE_JITTER     # each edge, x the box's short side
    turn: float = 4.0                  # degrees, either way
    p_turn: float = 0.3
    shrink: tuple[float, float] = (0.35, 1.0)
    p_shrink: float = 0.4
    blur: tuple[float, float] = (0.3, 1.5)
    p_blur: float = 0.3
    jpeg: tuple[int, int] = (25, 90)
    p_jpeg: float = 0.4
    gain: tuple[float, float] = (0.7, 1.3)        # brightness
    contrast: tuple[float, float] = (0.7, 1.3)    # about the mean
    cast: tuple[float, float] = (0.9, 1.1)        # a gain a channel
    p_colour: float = 0.5
    p_invert: float = 0.05


def plain_view(rgb: np.ndarray, box, pad: float) -> np.ndarray:
    """What a reader is given to read: the box with `pad`, cut from the
    stored crop (suites, scoring)."""
    x0, y0, x1, y1 = crops.view_rect(box, (rgb.shape[1], rgb.shape[0]), pad)
    return np.ascontiguousarray(rgb[y0:y1, x0:x1])


def _turned(rgb: np.ndarray, box, degrees: float) -> tuple[np.ndarray, tuple]:
    """The crop turned about its box's centre (the margin fills the corners,
    the border repeated past it) and the upright box round the turned one,
    as a detector boxes tilted print."""
    x1, y1, x2, y2 = box
    m = cv2.getRotationMatrix2D(((x1 + x2) / 2, (y1 + y2) / 2), degrees, 1.0)
    turned = cv2.warpAffine(rgb, m, (rgb.shape[1], rgb.shape[0]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], np.float64) @ m.T
    return turned, (corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max())


def augmented_view(rgb: np.ndarray, box, pad: float, rng: np.random.Generator, aug: Augment) -> np.ndarray:
    """A training view: turned (sometimes), cut with each edge jittered,
    then shrunk and back, blurred, recoloured, inverted, JPEG'd, each with
    its probability, in that order (the camera, then the file)."""
    if rng.random() < aug.p_turn:
        rgb, box = _turned(rgb, box, rng.uniform(-aug.turn, aug.turn))
    jitter = tuple(rng.uniform(-aug.jitter, aug.jitter, 4))
    x0, y0, x1, y1 = crops.view_rect(box, (rgb.shape[1], rgb.shape[0]), pad, jitter)
    view = rgb[y0:y1, x0:x1]
    h, w = view.shape[:2]
    if rng.random() < aug.p_shrink:
        f = rng.uniform(*aug.shrink)
        small = cv2.resize(view, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
        view = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < aug.p_blur:
        view = cv2.GaussianBlur(view, (0, 0), rng.uniform(*aug.blur))
    if rng.random() < aug.p_colour:
        x = view.astype(np.float32)
        mean = x.mean()
        x = ((x - mean) * rng.uniform(*aug.contrast) + mean) * rng.uniform(*aug.gain)
        x = x * rng.uniform(*aug.cast, 3).astype(np.float32)
        view = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < aug.p_invert:
        view = 255 - view
    if rng.random() < aug.p_jpeg:
        quality = int(rng.integers(aug.jpeg[0], aug.jpeg[1] + 1))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(view, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        view = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(view)


def padded_share(views: list[np.ndarray], batches: list[list[int]], spec: onnx_run.Spec = onnx_run.M3_SPEC) -> float:
    """The share of a batch's pixels that are padding, over all batches."""
    total = pad = 0
    for b in batches:
        widths = [onnx_run.resized_width(views[i].shape[1], views[i].shape[0], spec.max_width) for i in b]
        padded = onnx_run.padded_width(widths, spec)
        total += padded * len(b)
        pad += sum(padded - w for w in widths)
    return pad / total


def width_buckets(views: list[np.ndarray], batch: int, rng: np.random.Generator,
                  max_width: int = onnx_run.MAX_WIDTH) -> list[list[int]]:
    """Views sorted by width, cut into batches, the batches in random order."""
    order = sorted(range(len(views)),
                   key=lambda i: onnx_run.resized_width(views[i].shape[1], views[i].shape[0], max_width))
    batches = [order[i:i + batch] for i in range(0, len(order), batch)]
    return [batches[k] for k in rng.permutation(len(batches))]


# --- The stream ---------------------------------------------------------------------

POOL = 16               # batches' worth of views a worker sorts by width at a time


def collate(views: list[np.ndarray], texts: list[str], piles_drawn: list[int],
            records: list[int] | None = None, spec: onnx_run.Spec = onnx_run.M3_SPEC) -> dict[str, torch.Tensor]:
    """A batch: the images, each crop's own width, the texts as the
    reader's classes (concatenated) with their lengths, and each crop's
    pile (its number in the mix) and record number, so a bad step can name
    its crops."""
    x, widths = onnx_run.preprocess(views, spec)
    encoded = [spec.encode(t) for t in texts]
    return {"x": torch.from_numpy(x), "widths": torch.tensor(widths),
            "targets": torch.tensor([k for e in encoded for k in e], dtype=torch.long),
            "lengths": torch.tensor([len(e) for e in encoded]), "piles": torch.tensor(piles_drawn),
            "records": torch.tensor(records if records is not None else [-1] * len(views))}


class Stream(torch.utils.data.IterableDataset):
    """Endless training batches from `mix`, read from the piles under
    `root`. Each loader worker opens its own LMDB environments (never
    shared across a fork) and its own generator, seeded by (seed, worker)."""

    def __init__(self, root: Path, mix: Mix, batch: int, pad: float, seed: int,
                 augment: Augment | None = Augment(), pool: int = POOL, spec: onnx_run.Spec = onnx_run.M3_SPEC):
        self.root, self.mix, self.batch, self.pad, self.seed = root, mix, batch, pad, seed
        self.augment, self.pool, self.spec = augment, pool, spec

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        envs: dict[str, object] = {}
        try:
            while True:
                drawn = [self._sample(rng, envs) for _ in range(self.batch * self.pool)]
                views = [d[0] for d in drawn]
                for b in width_buckets(views, self.batch, rng, self.spec.max_width):
                    yield collate([views[i] for i in b], [drawn[i][1] for i in b], [drawn[i][2] for i in b],
                                  [drawn[i][3] for i in b], self.spec)
        finally:
            for env in envs.values():
                env.close()

    def _sample(self, rng: np.random.Generator, envs: dict) -> tuple[np.ndarray, str, int, int]:
        p, i = self.mix.draw(rng)
        name = self.mix.names[p]
        if name not in envs:
            envs[name] = shards.open_pile(self.root / name)
        crop = shards.get(envs[name], name, i)
        rgb = np.asarray(crop.image())
        view = (augmented_view(rgb, crop.box, self.pad, rng, self.augment) if self.augment
                else plain_view(rgb, crop.box, self.pad))
        return view, crop.text, p, i


def worker_init(_: int) -> None:
    """One thread a loader worker: the workers are the parallelism."""
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def suite(root: Path, name: str, pad: float) -> tuple[list[np.ndarray], list[str]]:
    """A suite's crops as a reader is given them (plain views) and their texts."""
    env = shards.open_pile(root / name)
    try:
        crops_ = list(shards.records(env, name))
    finally:
        env.close()
    return [plain_view(np.asarray(c.image()), c.box, pad) for c in crops_], [c.text for c in crops_]
