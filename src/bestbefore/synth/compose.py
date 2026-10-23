"""Generated text laid onto real pack photos (docs/m2-synthetic-generator.md,
"Placement and blending").

  backgrounds  ExpDate training photos (GenSpec.backgrounds), long side at
               least 640 px (1,000 when a block is small print), shrunk to
               at most 1,600 px; their own labelled boxes stay
  placement    inside the photo, clear of every labelled box and every
               block already placed by half a line; of up to 20 free places
               the plainest (lowest mean gradient) 70% of the time
  ink          dark ink on light surfaces, light ink on dark ones; now and
               then colourful or low in contrast; opacity 75 to 100%
  defects      uneven density, a slight spread, now and then a scratch, all
               on the drawn ink at 4x, so the boxes come from what is drawn
  blending     dark ink multiplies the surface; light ink is laid over it,
               darkening where the surface does; a label's sticker is laid
               over and its text multiplied; embossed text darkens the
               surface in its shade and lightens it where lit

Photos are float32 RGB arrays, 0 to 255, blended in place. Every random
choice comes from the numpy Generator passed in.
"""
from __future__ import annotations

import colorsys
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .. import derived, m2
from ..config import expdate_dir
from ..images import upright_size
from ..samples import Box
from ..yolo import list_ids
from . import render
from .render import Layer, Sprite
from .spec import COMPOSE, SIZE, ComposeSpec, Shares, SizeSpec
from .text import choice, pick

LUMA = np.array([0.299, 0.587, 0.114])      # Rec. 601, as OpenCV's grey
Rect = tuple[float, float, float, float]     # x1, y1, x2, y2 in photo pixels
_LIGHT_SCALE = 8                             # the local-brightness map is kept at 1/8 size


# --- Backgrounds ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Background:
    list: str                   # the m2 list it comes from, e.g. train_real
    path: Path
    width: int                  # upright
    height: int
    boxes: tuple[Box, ...]      # its own labels, 0..1

    @property
    def long_side(self) -> int:
        return max(self.width, self.height)


class Backgrounds:
    """The photos a set draws its backgrounds from, by list, in list order."""

    def __init__(self, by_list: dict[str, tuple[Background, ...]]):
        self.by_list = by_list
        self._eligible: dict[tuple[str, int], tuple[Background, ...]] = {}

    def eligible(self, name: str, min_side: int) -> tuple[Background, ...]:
        key = (name, min_side)
        if key not in self._eligible:
            self._eligible[key] = tuple(b for b in self.by_list[name] if b.long_side >= min_side)
        return self._eligible[key]

    def pick(self, rng: np.random.Generator, shares: Shares, min_side: int) -> Background:
        """A list by its share, then any of its photos at least min_side long."""
        found = self.eligible(pick(rng, shares), min_side)
        return found[int(rng.integers(len(found)))]


def load_backgrounds(shares: Shares, min_side: int) -> Backgrounds:
    """Every photo of the lists in `shares` with a long side of at least
    min_side, with its size and labelled boxes. Refuses a list holding a
    test photo, one of the 14 copies of a validation photo, or a
    validation photo outside "val" (synth-val's own list, S6)."""
    by_list = {}
    for name, _ in shares:
        paths = derived.read_list(m2.list_file(name))
        if problems := forbidden(name, paths):
            raise ValueError(f"{name}: never a background: {', '.join(problems[:5])}"
                             f"{' ...' if len(problems) > 5 else ''}")
        found = []
        for p in paths:
            w, h = upright_size(p)
            if max(w, h) >= min_side:
                found.append(Background(name, p, w, h, derived.read_boxes(p)))
        by_list[name] = tuple(found)
    return Backgrounds(by_list)


def forbidden(name: str, paths: Sequence[Path]) -> list[str]:
    """The photos of list `name` that no background may come from: test
    photos, the 14 copies, and validation photos unless the list is val."""
    val = list_ids(expdate_dir() / "yolo", "val") if name != "val" else set()
    return [str(p) for p in paths
            if p.parent.name == "real-test"
            or (p.parent.name == "synth" and p.name in m2.REUSED_VAL_COPIES)
            or (p.parent.name == "real-train" and p.name in val)]


def load_photo(bg: Background, max_side: int) -> np.ndarray:
    """The background as float32 RGB, its long side shrunk to at most
    max_side with an area filter (PIL's BOX). A JPEG much bigger than that
    is first decoded at 1/2, 1/4 or 1/8 size by libjpeg itself (`draft`),
    which saves most of the decoding."""
    scale = min(1.0, max_side / bg.long_side)
    size = (round(bg.width * scale), round(bg.height * scale))
    with Image.open(bg.path) as im:
        im.draft("RGB", size if im.size == (bg.width, bg.height) else size[::-1])   # an EXIF turn swaps them
        im = ImageOps.exif_transpose(im).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.Resampling.BOX)
    return np.asarray(im, dtype=np.float32)


def box_rects(boxes: Sequence[Box], width: int, height: int) -> list[Rect]:
    """Boxes (0..1) as photo pixels."""
    return [(b.xyxy[0] * width, b.xyxy[1] * height, b.xyxy[2] * width, b.xyxy[3] * height) for b in boxes]


# --- Size --------------------------------------------------------------------------------------

def print_height(rng: np.random.Generator, spec: SizeSpec = SIZE) -> tuple[str, float]:
    """A size band, and a print height in it drawn log-uniformly: the
    digits' height over the photo's long side."""
    band = pick(rng, spec.bands)
    lo, hi = getattr(spec, band)
    return band, float(np.exp(rng.uniform(math.log(lo), math.log(hi))))


# --- The surface ---------------------------------------------------------------------------------

class Surface:
    """The background, read once: its luminance, how busy it is (the
    gradient, at half size, as an integral image) and, when first needed,
    its local brightness."""

    def __init__(self, photo: np.ndarray, spec: ComposeSpec = COMPOSE):
        self.height, self.width = photo.shape[:2]
        self.spec = spec
        self.grey = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        half = cv2.resize(self.grey, (max(1, self.width // 2), max(1, self.height // 2)),
                          interpolation=cv2.INTER_AREA)
        gradient = np.abs(cv2.Sobel(half, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(half, cv2.CV_32F, 0, 1))
        self._detail = cv2.integral(gradient, sdepth=cv2.CV_64F)

    def luminance(self, x: int, y: int, w: int, h: int) -> float:
        """The mean luminance under a rect, 0..1."""
        return float(self.grey[y:y + h, x:x + w].mean()) / 255

    def plainness(self, x: int, y: int, w: int, h: int) -> float:
        """The mean gradient under a rect: lower is plainer."""
        x0, y0 = min(x // 2, self._detail.shape[1] - 2), min(y // 2, self._detail.shape[0] - 2)
        x1 = min(max(x0 + 1, -(-(x + w) // 2)), self._detail.shape[1] - 1)
        y1 = min(max(y0 + 1, -(-(y + h) // 2)), self._detail.shape[0] - 1)
        s = self._detail
        return float(s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]) / ((x1 - x0) * (y1 - y0))

    @cached_property
    def _light(self) -> np.ndarray:
        """The light falling on the surface, at 1/8 size. A photo can't tell
        a shadow from dark print, but light varies slowly and print has
        edges: so the luminance's brightest within spec.shading_window of the
        long side (dark print narrower than that is taken for print), then
        blurred over spec.shading_sigma."""
        f, long = _LIGHT_SCALE, max(self.width, self.height) / _LIGHT_SCALE
        small = cv2.resize(self.grey, (max(1, self.width // f), max(1, self.height // f)),
                           interpolation=cv2.INTER_AREA)
        k = 2 * round(self.spec.shading_window * long / 2) + 1
        envelope = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return cv2.GaussianBlur(envelope, (0, 0), self.spec.shading_sigma * long)

    def shading(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """The light over a rect (h x w), relative to the brightest under it:
        where a shadow crosses a block, light ink and a sticker darken with
        the surface (down to spec.shading_floor)."""
        f, light = _LIGHT_SCALE, self._light
        xs = (np.arange(x, x + w, dtype=np.float32) + 0.5) / f - 0.5
        ys = (np.arange(y, y + h, dtype=np.float32) + 0.5) / f - 0.5
        local = cv2.remap(light, np.broadcast_to(xs[None, :], (h, w)).astype(np.float32),
                          np.broadcast_to(ys[:, None], (h, w)).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return np.clip(local / max(float(local.max()), 1.0), self.spec.shading_floor, 1.0)


# --- Placement ------------------------------------------------------------------------------------

def place(w: int, h: int, margin: float, taken: Sequence[Rect], surface: Surface,
          rng: np.random.Generator, spec: ComposeSpec = COMPOSE) -> tuple[int, int] | None:
    """A top-left for a w x h block inside the photo, at least `margin`
    pixels clear of every rect in `taken`. Up to spec.candidates free
    places are found in spec.tries tries; the plainest is taken with share
    spec.plainest, else a random one. None when no try was free."""
    if w > surface.width or h > surface.height:
        return None
    rects = np.asarray(taken, dtype=np.float64).reshape(-1, 4)
    free: list[tuple[int, int]] = []
    for _ in range(spec.tries):
        x, y = int(rng.integers(surface.width - w + 1)), int(rng.integers(surface.height - h + 1))
        if not overlaps(rects, (x - margin, y - margin, x + w + margin, y + h + margin)):
            free.append((x, y))
            if len(free) == spec.candidates:
                break
    if not free:
        return None
    if rng.random() < spec.plainest:
        return min(free, key=lambda p: surface.plainness(*p, w, h))
    return free[int(rng.integers(len(free)))]


def overlaps(rects: np.ndarray, r: Rect) -> bool:
    """Whether rect r overlaps any of rects (n x 4); touching edges don't."""
    return bool(np.any((rects[:, 0] < r[2]) & (r[0] < rects[:, 2]) & (rects[:, 1] < r[3]) & (r[1] < rects[:, 3])))


# --- Ink --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Ink:
    """How a block is laid on the photo."""
    mode: str                                 # dark (multiplied), light (laid over), label, relief
    colour: tuple[float, float, float]        # RGB 0..255; a label's text black, a relief's highlight white
    opacity: float = 1.0
    colourful: bool = False
    low_contrast: float | None = None         # the luminance gap asked for, as blended

    def record(self) -> dict:
        return {"mode": self.mode, "colour": [round(c) for c in self.colour],
                "opacity": round(self.opacity, 3), "colourful": self.colourful,
                "low_contrast": None if self.low_contrast is None else round(self.low_contrast, 3)}


def luminance(rgb) -> float:
    """Rec. 601 luminance of an RGB colour (0..255), as 0..1."""
    return float(np.dot(LUMA, rgb)) / 255


def choose_ink(rng: np.random.Generator, style: str, surface: float,
               spec: ComposeSpec = COMPOSE) -> Ink:
    """The ink for a block on a surface of mean luminance `surface` (0..1).
    An embossed block has none (its light and shade); a label's text is
    black on its sticker. Otherwise dark ink on a light surface, light ink
    on a dark one, colourful with share spec.colourful; with share
    spec.low_contrast its luminance then moves to just `gap` from the
    surface's, as blended at its opacity."""
    if style == "embossed":
        return Ink("relief", (255.0, 255.0, 255.0))
    opacity = float(rng.uniform(*spec.opacity))
    if style == "label":
        return Ink("label", tuple(float(c) for c in spec.dark_inks[0]), opacity)
    mode = "dark" if surface >= spec.light_surface else "light"
    colourful = bool(rng.random() < spec.colourful)
    if colourful:
        value = spec.colourful_dark if mode == "dark" else spec.colourful_light
        colour = np.array(colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(*spec.colourful_saturation),
                                              rng.uniform(*value))) * 255
    else:
        colour = np.array(choice(rng, spec.dark_inks if mode == "dark" else spec.light_inks), dtype=float)
    gap = None
    if rng.random() < spec.low_contrast:
        gap = float(rng.uniform(*spec.low_contrast_gap))
        # dark: surface x (1 - o + o y) = surface - gap;  light: surface (1 - o) + o y = surface + gap
        target = 1 - gap / (opacity * surface) if mode == "dark" else surface + gap / opacity
        colour = at_luminance(colour, float(np.clip(target, 0, 1)))
    return Ink(mode, tuple(float(c) for c in colour), opacity, colourful, gap)


def at_luminance(colour: np.ndarray, target: float) -> np.ndarray:
    """The colour with luminance `target` (0..1), its hue kept: scaled down
    towards black, or mixed up towards white."""
    y = luminance(colour)
    if y >= target:
        return colour * (target / max(y, 1e-6))
    return 255 - (255 - colour) * ((1 - target) / (1 - y))


# --- Ink defects (at 4x, before shrinking) ---------------------------------------------------------------

def defects(layer: Layer, style: str, boxed: Sequence[bool], rng: np.random.Generator,
            spec: ComposeSpec = COMPOSE) -> tuple[Layer, dict]:
    """Ink defects on a block drawn at 4x: uneven density (lighter in
    places, about a character apart), a slight spread (not on a label's
    crisp thermal print), and now and then a scratch through one boxed
    token. They change the drawn ink, so the boxes, made after, include
    them. An embossed block has no ink, and none."""
    if style == "embossed":
        return layer, {}
    k = layer.scale
    char = layer.params["size"]                                # a digit's height, final pixels
    density = float(rng.uniform(*spec.density))
    field = _smooth_field(rng, layer.width // k, layer.height // k, char)
    spread = 0.0 if style == "label" else float(rng.uniform(*spec.spread))
    scratched = None
    if rng.random() < spec.scratch:
        targets = [i for i, (s, b) in enumerate(zip(layer.tokens, boxed)) if b and s is not None]
        scratched = int(choice(rng, targets)) if targets else None
    tokens: list[Sprite | None] = []
    for i, s in enumerate(layer.tokens):
        if s is None:
            tokens.append(None)
            continue
        h, w = s.alpha.shape
        local = cv2.resize(field[s.y // k:(s.y + h) // k, s.x // k:(s.x + w) // k], (w, h),
                           interpolation=cv2.INTER_LINEAR)
        alpha = s.alpha * (1 - density * local)
        if i == scratched:
            alpha = _scratch(alpha, char * k, rng.uniform(*spec.scratch_width) * k, rng)
        if spread * k >= 0.25:
            m = math.ceil(3 * spread * k) + 1
            alpha = cv2.GaussianBlur(np.pad(alpha, m), (0, 0), spread * k, borderType=cv2.BORDER_CONSTANT)
            tokens.append(render.aligned(Sprite(s.x - m, s.y - m, alpha), k))
        else:
            tokens.append(Sprite(s.x, s.y, alpha))
    record = {"density": round(density, 3), "spread": round(spread, 3), "scratch": scratched}
    return render.assemble(tokens, layer.backing, k, layer.params), record


def _smooth_field(rng: np.random.Generator, w: int, h: int, cell: float) -> np.ndarray:
    """A smooth random field (h x w, 0..1) varying about every `cell` pixels."""
    gw, gh = math.ceil(w / cell) + 2, math.ceil(h / cell) + 2
    coarse = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(coarse, (max(1, w), max(1, h)), interpolation=cv2.INTER_LINEAR)


def _scratch(alpha: np.ndarray, length: float, width: float, rng: np.random.Generator) -> np.ndarray:
    """The ink with a straight scratch through it, `length` long, through a
    point of solid ink, at any angle."""
    ys, xs = np.nonzero(alpha >= 0.5)
    if not len(ys):
        return alpha
    j, angle = int(rng.integers(len(ys))), rng.uniform(0, math.pi)
    dx, dy = 0.5 * length * math.cos(angle), 0.5 * length * math.sin(angle)
    mask = np.zeros(alpha.shape, np.uint8)
    cv2.line(mask, (round(xs[j] - dx), round(ys[j] - dy)), (round(xs[j] + dx), round(ys[j] + dy)), 255,
             thickness=max(1, round(width)), lineType=cv2.LINE_AA)
    return alpha * (1 - mask.astype(np.float32) / 255)


# --- Blending -----------------------------------------------------------------------------------------------

def blend(photo: np.ndarray, layer: Layer, x: int, y: int, ink: Ink, surface: Surface) -> None:
    """Lay a shrunk block onto the photo at (x, y), in place. Dark ink
    multiplies the surface, so its texture and shading show through. Light
    ink is laid over, scaled by the surface's local brightness, and so is a
    label's sticker, whose black text then multiplies it. An embossed
    block's shade multiplies the surface towards black and its light is laid
    over in white."""
    h, w = layer.height, layer.width
    region = photo[y:y + h, x:x + w]
    shade = surface.shading(x, y, w, h)[..., None] if ink.mode != "dark" else None
    if layer.backing is not None:
        sticker = np.asarray(layer.params["label"]["colour"], np.float32)
        a = _canvas(layer.backing, h, w)[..., None]
        region += a * (sticker * shade - region)
    a = (render.ink(layer) * ink.opacity)[..., None]
    colour = np.asarray(ink.colour, np.float32)
    if ink.mode == "relief":
        lit = render.tone(layer)[..., None]
        region *= 1 - a * (1 - lit)
        region += a * lit * (colour * shade - region)
    elif ink.mode == "light":
        region += a * (colour * shade - region)
    else:                                                    # dark ink, a label's text
        region *= 1 - a * (1 - colour / 255)


def ink_change(before: np.ndarray, after: np.ndarray, s: Sprite) -> float:
    """How much a token's ink changed the photo: the colour distance between
    before and after under its sprite (RGB, scaled so that a grey change is
    its change in luminance, 0..1), weighted by its alpha, so its strongest
    ink counts most. Colour counts as well as light: strong blue on dark
    brown is close in luminance, yet a camera sees it. `before` and `after`
    are the block's region, as the sprite's own coordinates are."""
    h, w = s.alpha.shape
    diff = after[s.y:s.y + h, s.x:s.x + w] - before[s.y:s.y + h, s.x:s.x + w]
    change = np.sqrt((diff * diff).mean(axis=-1)) / 255
    return float((change * s.alpha).sum() / max(float(s.alpha.sum()), 1e-6))


def _canvas(s: Sprite, h: int, w: int) -> np.ndarray:
    """One sprite's alpha on an h x w canvas."""
    out = np.zeros((h, w), np.float32)
    sh, sw = s.alpha.shape
    out[s.y:s.y + sh, s.x:s.x + sw] = s.alpha
    return out
