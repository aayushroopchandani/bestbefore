"""Text blocks drawn as pixels, in the four print styles, with an exact box
for every token (docs/m2-synthetic-generator.md, "Print styles").

A block is drawn at 4x its final size (spec.supersample) and shrunk with an
area filter, so small print gets the soft edges a camera gives it. Each
token is drawn on its own sprite: its ink alone, styled, at its place on
the block. A token's box is the tight rectangle around the pixels of its
sprite that are ink, at whatever size the layer is, and **ink is alpha >=
INK**, here and everywhere: boxes come from the ink drawn, never from font
metrics. Sprites sit on the 4x grid (positions and sizes are multiples of
4), so shrinking a sprite gives exactly the pixels that shrinking the whole
block would.

  solid       the font's glyphs, regular or bold
  dot-matrix  dotify: the bold glyphs sampled on a grid of round dots
  label       crisp black text on a white or cream sticker; the sticker is
              no token's ink and gets no box
  embossed    no ink: the glyphs raise or press the surface, lit from one
              side, and the ink is the light and the shade on the relief's
              slopes (`tone` says which)

Every random choice comes from the numpy Generator passed in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import fonts
from .fonts import Family
from .spec import STYLE, StyleSpec
from .text import Block, choice, pick

INK = 0.25          # a pixel is ink when at least a quarter of it is covered
STYLES = ("solid", "dot-matrix", "label", "embossed")
_GLYPH_PAD = 2      # blank pixels around each glyph's bitmap, so none is cut off


@dataclass(frozen=True)
class Sprite:
    """A patch of a layer: `alpha` (h x w, float32 0..1) with its top-left
    at (x, y) of the layer."""
    x: int
    y: int
    alpha: np.ndarray
    tone: np.ndarray | None = None      # embossed: 1 where the relief is lit, 0 in its shade


@dataclass(frozen=True)
class Layer:
    """A drawn block: one sprite per token, in Block.tokens order (None for
    a token with no ink at all), a label's sticker, and how it was drawn."""
    width: int
    height: int
    scale: int                          # pixels per final pixel: the supersample, or 1 once shrunk
    tokens: tuple[Sprite | None, ...]
    backing: Sprite | None = None
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Glyph:
    bitmap: np.ndarray | None           # coverage, float32 0..1; None for a space
    left: int                           # the bitmap's top-left, from the pen on the baseline
    top: int
    advance: float


def random_style(rng: np.random.Generator, spec: StyleSpec = STYLE) -> str:
    return pick(rng, spec.styles)


# --- Drawing -------------------------------------------------------------------

def render(block: Block, style: str, family: Family, size: float,
           rng: np.random.Generator, spec: StyleSpec = STYLE) -> Layer:
    """The block in a print style and a font family, its digits `size`
    final pixels tall, drawn at spec.supersample times that size. Every
    line starts at the same left edge, as a coding printer prints."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; one of {', '.join(STYLES)}")
    k = spec.supersample
    bold = style in ("dot-matrix", "embossed") or bool(rng.random() < spec.bold)
    face = family.face(bold)
    px = max(1, round(k * size / fonts.digit_height(face)))
    font = fonts.load(face, px)
    tracking = rng.uniform(*spec.letter_spacing) * px
    line_pitch = rng.uniform(*spec.line_spacing) * px
    params = {"style": style, "family": family.name,
              "weight": "bold" if face is family.bold else "regular",
              "size": round(size, 3), "px": px, "letter_spacing": round(tracking / px, 4),
              "line_spacing": round(line_pitch / px, 4)}
    if style == "dot-matrix":
        digit_top = fonts.digit_band(face)[0] * px            # the grid's rows span the digits
        dots, params["dots"] = dot_params(rng, spec, fonts.digit_height(face) * px)
    elif style == "embossed":
        relief = {"soft": rng.uniform(*spec.emboss_soft) * px,
                  "strength": rng.uniform(*spec.emboss_strength),
                  "light": rng.uniform(0, 2 * math.pi), "raised": bool(rng.random() < 0.5)}
        params["relief"] = {"soft": round(relief["soft"] / px, 4),
                            "strength": round(relief["strength"], 3),
                            "light_deg": round(math.degrees(relief["light"]), 1),
                            "raised": relief["raised"]}

    glyphs: dict[str, _Glyph] = {}
    space = font.getlength(" ")
    sprites: list[Sprite | None] = []
    for i, line in enumerate(block.lines):
        baseline = round(i * line_pitch)
        x = 0.0
        for j, token in enumerate(line):
            if j and token.space_before:
                x += space + tracking
            placed = []
            for ch in token.text:
                if ch not in glyphs:
                    glyphs[ch] = _glyph(font, ch)
                placed.append((glyphs[ch], round(x)))
                x += glyphs[ch].advance + tracking
            ink = _stamp(placed, baseline)
            if ink is not None and style == "dot-matrix":
                ink = dotify(ink, baseline + digit_top, dots, rng)
            elif ink is not None and style == "embossed":
                ink = emboss(ink, **relief)
            sprites.append(None if ink is None else _aligned(ink, k))

    backing = None
    if style == "label":
        backing, params["label"] = _sticker([s for s in sprites if s], px, spec, rng)
        backing = _aligned(backing, k)
    return _placed(sprites, backing, k, params)


def _glyph(font, ch: str) -> _Glyph:
    left, top, right, bottom = font.getbbox(ch, anchor="ls")
    advance = font.getlength(ch)
    if right <= left or bottom <= top:
        return _Glyph(None, 0, 0, advance)
    pad = _GLYPH_PAD
    im = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad))
    ImageDraw.Draw(im).text((pad - left, pad - top), ch, font=font, fill=255, anchor="ls")
    return _Glyph(np.asarray(im, dtype=np.float32) / 255, left - pad, top - pad, advance)


def _stamp(placed: list[tuple[_Glyph, int]], baseline: int) -> Sprite | None:
    """One token's coverage: its glyphs at their pens on the baseline."""
    at = [(g, x + g.left, baseline + g.top) for g, x in placed if g.bitmap is not None]
    if not at:
        return None
    x0, y0 = min(x for _, x, _ in at), min(y for _, _, y in at)
    x1 = max(x + g.bitmap.shape[1] for g, x, _ in at)
    y1 = max(y + g.bitmap.shape[0] for g, _, y in at)
    alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
    for g, x, y in at:
        h, w = g.bitmap.shape
        window = alpha[y - y0:y - y0 + h, x - x0:x - x0 + w]
        np.maximum(window, g.bitmap, out=window)
    return Sprite(x0, y0, alpha)


def _aligned(s: Sprite, k: int) -> Sprite:
    """The sprite with at least one blank pixel around it, grown to a
    position and size that are multiples of k."""
    h, w = s.alpha.shape
    x0, y0 = (s.x - 1) // k * k, (s.y - 1) // k * k
    x1, y1 = -(-(s.x + w + 1) // k) * k, -(-(s.y + h + 1) // k) * k
    pad = ((s.y - y0, y1 - s.y - h), (s.x - x0, x1 - s.x - w))
    return Sprite(x0, y0, np.pad(s.alpha, pad),
                  None if s.tone is None else np.pad(s.tone, pad))


def _placed(sprites: list[Sprite | None], backing: Sprite | None, k: int, params: dict) -> Layer:
    """The sprites moved so the block's top-left is (0, 0), on the k grid."""
    every = [s for s in (*sprites, backing) if s is not None]
    if not every:
        raise ValueError("the block has no ink")
    x0, y0 = min(s.x for s in every), min(s.y for s in every)
    x1 = max(s.x + s.alpha.shape[1] for s in every)
    y1 = max(s.y + s.alpha.shape[0] for s in every)

    def moved(s: Sprite | None) -> Sprite | None:
        return None if s is None else replace(s, x=s.x - x0, y=s.y - y0)
    return Layer(x1 - x0, y1 - y0, k, tuple(moved(s) for s in sprites), moved(backing), params)


# --- Dot-matrix: dotify ----------------------------------------------------------

def dot_params(rng: np.random.Generator, spec: StyleSpec, digit_px: float) -> tuple[dict, dict]:
    """One block's dot grid and dots, and their record for the sidecar.
    Lengths are in pixels at the drawn size; the pitch is the vertical one,
    a digit's height over the rows."""
    rows = int(choice(rng, spec.dot_rows))
    pitch = digit_px / rows
    aspect, radius = rng.uniform(*spec.dot_aspect), rng.uniform(*spec.dot_radius)
    drop, blur = rng.uniform(*spec.dot_drop), rng.uniform(*spec.dot_blur)
    dots = {"pitch_x": pitch * aspect, "pitch_y": pitch, "cover": spec.dot_cover,
            "drop": drop, "jitter": spec.dot_jitter * pitch,
            "kernel": _dot_kernel(radius * pitch, blur * pitch)}
    return dots, {"rows": rows, "aspect": round(aspect, 3), "radius": round(radius, 3),
                  "drop": round(drop, 4), "blur": round(blur, 3)}


def _dot_kernel(radius: float, blur: float) -> np.ndarray:
    """One dot: an anti-aliased disc, blurred (the inkjet look)."""
    half = math.ceil(radius + 4 * blur) + 1
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    disc = np.clip(radius + 0.5 - np.hypot(xx, yy), 0, 1).astype(np.float32)
    if blur <= 0:
        return disc
    return cv2.GaussianBlur(disc, (0, 0), blur, borderType=cv2.BORDER_CONSTANT)


def dotify(cov: Sprite, grid_top: float, p: dict, rng: np.random.Generator) -> Sprite | None:
    """The token's text as dots. The grid starts at the line's left edge
    (x = 0) and at the top of its digits; a dot goes on every cell the text
    covers by at least p["cover"], unless dropped, and lands a little off
    its grid point."""
    px_, py_ = p["pitch_x"], p["pitch_y"]
    h, w = cov.alpha.shape
    j0, j1 = math.floor(cov.x / px_), math.ceil((cov.x + w) / px_)
    i0, i1 = math.floor((cov.y - grid_top) / py_), math.ceil((cov.y + h - grid_top) / py_)
    # How much of each cell the text covers, from the integral image.
    xs = np.clip(np.round(np.arange(j0, j1 + 1) * px_ - cov.x).astype(int), 0, w)
    ys = np.clip(np.round(grid_top + np.arange(i0, i1 + 1) * py_ - cov.y).astype(int), 0, h)
    total = cv2.integral(cov.alpha, sdepth=cv2.CV_64F)[np.ix_(ys, xs)]
    cells = (total[1:, 1:] - total[:-1, 1:] - total[1:, :-1] + total[:-1, :-1]) / (px_ * py_)
    on = (cells >= p["cover"]) & (rng.random(cells.shape) >= p["drop"])
    rows, cols = np.nonzero(on)
    if not len(rows):
        return None
    cx = np.round((j0 + cols + 0.5) * px_ + rng.normal(0, p["jitter"], len(cols))).astype(int)
    cy = np.round(grid_top + (i0 + rows + 0.5) * py_
                  + rng.normal(0, p["jitter"], len(rows))).astype(int)
    kernel = p["kernel"]
    half = kernel.shape[0] // 2
    x0, y0 = cx.min() - half, cy.min() - half
    alpha = np.zeros((cy.max() + half + 1 - y0, cx.max() + half + 1 - x0), np.float32)
    np.add.at(alpha, (cy - y0, cx - x0), 1.0)
    alpha = cv2.filter2D(alpha, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return Sprite(int(x0), int(y0), np.clip(alpha, 0, 1))


# --- Embossed ------------------------------------------------------------------

def emboss(cov: Sprite, soft: float, strength: float, light: float, raised: bool) -> Sprite:
    """The light and shade of the text pressed into, or raised from, the
    surface. The relief is the coverage blurred by `soft`; a slope facing
    the light (at angle `light`) is lit, one facing away is in shade. The
    shading is scaled so a straight edge's steepest point has `strength`."""
    m = math.ceil(4 * soft) + 2
    height = cv2.GaussianBlur(np.pad(cov.alpha, m), (0, 0), soft,
                              borderType=cv2.BORDER_CONSTANT)     # flat surface all round
    gy, gx = np.gradient(height)
    shade = -(gx * math.cos(light) + gy * math.sin(light)) * (soft * math.sqrt(2 * math.pi))
    if not raised:
        shade = -shade
    return Sprite(cov.x - m, cov.y - m, np.clip(np.abs(shade) * strength, 0, 1).astype(np.float32),
                  (shade > 0).astype(np.float32))


# --- Label ----------------------------------------------------------------------

def _sticker(sprites: list[Sprite], px: int, spec: StyleSpec,
             rng: np.random.Generator) -> tuple[Sprite, dict]:
    """A white or cream sticker under all the text, with a margin."""
    pad = rng.uniform(*spec.label_pad) * px
    corner = rng.uniform(*spec.label_corner) * px
    colour = choice(rng, spec.label_colours)
    x0 = math.floor(min(s.x for s in sprites) - pad)
    y0 = math.floor(min(s.y for s in sprites) - pad)
    x1 = math.ceil(max(s.x + s.alpha.shape[1] for s in sprites) + pad)
    y1 = math.ceil(max(s.y + s.alpha.shape[0] for s in sprites) + pad)
    im = Image.new("L", (x1 - x0, y1 - y0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, x1 - x0 - 1, y1 - y0 - 1), radius=corner, fill=255)
    sticker = Sprite(x0, y0, np.asarray(im, dtype=np.float32) / 255)
    return sticker, {"pad": round(pad / px, 3), "corner": round(corner / px, 3),
                     "colour": list(colour)}


# --- Shrinking, ink and boxes ------------------------------------------------------

def _area(a: np.ndarray, k: int) -> np.ndarray:
    """Shrink by a whole factor k: every output pixel the mean of k x k."""
    h, w = a.shape
    return cv2.resize(a, (w // k, h // k), interpolation=cv2.INTER_AREA)


def shrink(layer: Layer) -> Layer:
    """The layer at its final size: every sprite shrunk with an area filter
    (an embossed tone weighted by the alpha it shades)."""
    k = layer.scale
    if k == 1:
        return layer

    def small(s: Sprite | None) -> Sprite | None:
        if s is None:
            return None
        alpha = _area(s.alpha, k)
        tone = None if s.tone is None else _area(s.alpha * s.tone, k) / np.maximum(alpha, 1e-6)
        return Sprite(s.x // k, s.y // k, alpha, tone)

    return replace(layer, width=layer.width // k, height=layer.height // k, scale=1,
                   tokens=tuple(small(s) for s in layer.tokens), backing=small(layer.backing))


def ink(layer: Layer) -> np.ndarray:
    """All the tokens' ink as one alpha image (height x width); where two
    sprites overlap, the stronger."""
    out = np.zeros((layer.height, layer.width), np.float32)
    for s in layer.tokens:
        if s is not None:
            h, w = s.alpha.shape
            window = out[s.y:s.y + h, s.x:s.x + w]
            np.maximum(window, s.alpha, out=window)
    return out


def tone(layer: Layer) -> np.ndarray | None:
    """An embossed layer's light (1) and shade (0) per pixel; None otherwise."""
    if not any(s is not None and s.tone is not None for s in layer.tokens):
        return None
    alpha = np.zeros((layer.height, layer.width), np.float32)
    out = np.zeros_like(alpha)
    for s in layer.tokens:
        if s is not None:
            h, w = s.alpha.shape
            stronger = s.alpha > alpha[s.y:s.y + h, s.x:s.x + w]
            alpha[s.y:s.y + h, s.x:s.x + w][stronger] = s.alpha[stronger]
            out[s.y:s.y + h, s.x:s.x + w][stronger] = s.tone[stronger]
    return out


def ink_box(s: Sprite | None) -> tuple[int, int, int, int] | None:
    """The tight box (x1, y1, x2, y2, ends exclusive) around a sprite's ink
    in layer pixels; None if it has none at this size."""
    if s is None:
        return None
    on = s.alpha >= INK
    rows, cols = np.flatnonzero(on.any(axis=1)), np.flatnonzero(on.any(axis=0))
    if not len(rows):
        return None
    return (s.x + int(cols[0]), s.y + int(rows[0]), s.x + int(cols[-1]) + 1, s.y + int(rows[-1]) + 1)


def boxes(layer: Layer) -> tuple[tuple[int, int, int, int] | None, ...]:
    """Every token's box, in Block.tokens order, at the layer's size."""
    return tuple(ink_box(s) for s in layer.tokens)
