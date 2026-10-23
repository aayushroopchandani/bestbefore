"""One generated photo from its index alone (docs/m2-synthetic-generator.md,
"One generated photo").

Photo i of a set comes from numpy.random.default_rng([set_seed, i]) and
nothing else, so it is the same however the work is split. make_one returns
the JPEG's bytes (the save is the camera's last step), the YOLO label file
and the sidecar line; write_one saves the first two where Ultralytics looks,
and the caller writes the sidecar.

The label lines come in the sidecar's order: every boxed token, block by
block, then the background's own boxes. So the two always match.
"""
from __future__ import annotations

import io
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .. import derived, yolo
from ..samples import Box, Sample
from . import camera, compose, fonts, render, text, warp
from .compose import Backgrounds, Surface
from .render import Layer
from .spec import ComposeSpec, GenSpec
from .text import Block, Token
from .warp import Geometry


@dataclass(frozen=True)
class Drawn:
    """A block at its final size, boxed from its ink."""
    layer: Layer
    boxes: tuple            # per token, in layer pixels; None for a token with no ink
    defects: dict
    redraws: int            # earlier draws that lost a boxed token's ink
    scale: float = 1.0      # of its size: spec.compose.shrink once no place was free


@dataclass(frozen=True)
class Made:
    name: str           # the image's file name
    jpeg: bytes
    labels: str         # the label file's text
    sidecar: dict


class Clock:
    """Seconds per stage, summed: where a photo's time goes."""

    def __init__(self):
        self.seconds: Counter = Counter()

    @contextmanager
    def __call__(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[stage] += time.perf_counter() - start


def image_name(i: int) -> str:
    return f"{i:06d}.jpg"


def make_one(i: int, spec: GenSpec, pool: Backgrounds, set_seed: int, clock: Clock | None = None,
             stratum: str | None = None) -> Made:
    """Photo i of the set with seed `set_seed`: 1 or 2 blocks drawn onto a
    background, through the camera, saved as a JPEG. A block that finds no
    place, or whose ink stays too faint, is skipped; with
    spec.compose.block_tries above 1 a fresh block is drawn in its place
    (synth-val, whose every photo needs its target). `stratum` (synth-val's)
    goes into the sidecar."""
    clock = clock or Clock()
    rng = np.random.default_rng([set_seed, i])
    c = spec.compose
    with clock("background"):
        # Sizes first: a photo with small print gets a background of at least min_side_small.
        heights = [compose.print_height(rng, spec.size) for _ in range(2 if rng.random() < c.two_blocks else 1)]
        small = any(band == "small" for band, _ in heights)
        bg = pool.pick(rng, spec.backgrounds, c.min_side_small if small else c.min_side)
        photo = compose.load_photo(bg, c.max_side)
    height, width = photo.shape[:2]
    with clock("surface"):
        surface = Surface(photo, c)
    taken = compose.box_rects(bg.boxes, width, height)
    blocks, skipped = [], []
    for band, fraction in heights:
        for _ in range(c.block_tries):
            placed, skip = _block(band, fraction * max(width, height), photo, surface, taken, rng, spec, clock)
            if placed is None:
                skipped.append(skip)
                continue
            x, y, w, h = placed["place"]
            taken.append((x, y, x + w, y + h))
            blocks.append(placed)
            break
    with clock("camera"):
        effects = camera.draw(rng, width, height, [tuple(_rect(b["place"])) for b in blocks], spec.camera)
        photo = camera.apply(photo, effects, rng)
        quality = int(rng.integers(spec.camera.jpeg_quality[0], spec.camera.jpeg_quality[1] + 1))
    with clock("jpeg"):
        jpeg = encode(photo, quality)
    name = image_name(i)
    sidecar = {"image": name, "i": i, "seed": [set_seed, i], "size": [width, height],
               "background": {"list": bg.list, "image": bg.path.name, "size": [bg.width, bg.height],
                              "boxes": [[b.label, *(round(v, 6) for v in b.xyxy)] for b in bg.boxes]},
               "wanted": len(heights), "blocks": blocks, "skipped": skipped, "camera": effects,
               "jpeg_quality": quality}
    if stratum is not None:
        sidecar["stratum"] = stratum
    return Made(name, jpeg, label_text(sidecar), sidecar)


def _block(band: str, size: float, photo: np.ndarray, surface: Surface, taken: list, rng: np.random.Generator,
           spec: GenSpec, clock: Clock) -> tuple[dict | None, dict | None]:
    """One block drawn, placed and inked onto the photo: its sidecar record,
    or None and why it was skipped (then the photo is left as it was)."""
    c = spec.compose
    block = text.random_block(rng, spec.text)
    style = render.random_style(rng, spec.style)
    family = fonts.random_family(rng, spec.fonts)
    geometry = warp.random_geometry(rng, spec.geometry)
    drawn, spot = _place(block, style, family, size, geometry, surface, taken, rng, spec, clock)
    skip = {"kind": block.kind, "style": style, "band": band, "size": round(size, 2)}
    if spot is None:
        return None, skip | {"why": "no free place" if drawn else "a boxed token kept no ink"}
    (x, y), layer = spot, drawn.layer
    with clock("blend"):
        inks, shown = ink_block(photo, layer, block, style, x, y, surface, rng, c)
    if shown < c.visible:
        return None, skip | {"why": "its ink stayed too faint", "inks": [i.record() for i in inks]}
    height, width = photo.shape[:2]
    return {
        "kind": block.kind, "style": style, "family": family.name, "weight": layer.params["weight"],
        "band": band, "size": round(size * drawn.scale, 2),
        "print_height": round(size * drawn.scale / max(width, height), 5),
        "shrunk": drawn.scale < 1, "redraws": drawn.redraws,
        "render": {k: v for k, v in layer.params.items() if k not in ("style", "family", "weight", "size")},
        "geometry": geometry.record(), "defects": drawn.defects, "ink": inks[-1].record(),
        "faint_inks": [i.record() for i in inks[:-1]], "shown": round(shown, 4),
        "place": [x, y, layer.width, layer.height],
        "tokens": [_token(t, None if b is None else [x + b[0], y + b[1], x + b[2], y + b[3]])
                   for t, b in zip(block.tokens, drawn.boxes)]}, None


def _place(block: Block, style: str, family: fonts.Family, size: float, geometry: Geometry,
           surface: Surface, taken: list, rng: np.random.Generator, spec: GenSpec,
           clock: Clock) -> tuple[Drawn | None, tuple[int, int] | None]:
    """The block drawn, and a free place for it. With none free, it is
    drawn once more at spec.compose.shrink of its size; a place of None
    then means it is skipped (a Drawn of None: it kept no ink to box)."""
    c = spec.compose
    drawn = None
    for scale in (1.0, c.shrink):
        with clock("draw"):
            drawn = _draw(block, style, family, size, scale, geometry, rng, spec)
        if drawn is None:
            return None, None
        layer = drawn.layer
        line = layer.params["line_spacing"] * layer.params["px"] / spec.style.supersample
        with clock("place"):
            spot = compose.place(layer.width, layer.height, c.margin * line, taken, surface, rng, c)
        if spot is not None:
            return drawn, spot
    return drawn, None


def _draw(block: Block, style: str, family: fonts.Family, size: float, scale: float, geometry: Geometry,
          rng: np.random.Generator, spec: GenSpec) -> Drawn | None:
    """The block drawn at 4x, warped, its ink defects added, shrunk, and
    boxed from its ink. A boxed token whose ink is all gone at this size
    has no box, so then the block is drawn again, spec.compose.redraws
    times in all; None if every draw lost a box. A date is never written
    without its box."""
    boxed = [t.cls is not None for t in block.tokens]
    for attempt in range(spec.compose.redraws):
        layer = warp.apply(render.render(block, style, family, size * scale, rng, spec.style), geometry)
        layer, defects = compose.defects(layer, style, boxed, rng, spec.compose)
        layer = render.shrink(layer)
        boxes = render.boxes(layer)
        if all(box is not None for box, b in zip(boxes, boxed) if b):
            return Drawn(layer, boxes, defects, attempt, scale)
    return None


def ink_block(photo: np.ndarray, layer: Layer, block: Block, style: str, x: int, y: int, surface: Surface,
              rng: np.random.Generator, c: ComposeSpec) -> tuple[list[compose.Ink], float]:
    """Lay the block onto the photo at (x, y), and check that every boxed
    token shows: its ink must change the photo by at least c.visible
    (compose.ink_change), or the block is taken off again and gets a fresh
    ink, c.inks in all. An embossed block's one "ink" is its relief, so it gets no
    other. Returns the inks drawn, the last on the photo, and how much the
    faintest boxed token shows; the photo is left as it was if that is
    still under c.visible."""
    h, w = layer.height, layer.width
    before = photo[y:y + h, x:x + w].copy()
    boxed = [s for s, t in zip(layer.tokens, block.tokens) if t.cls is not None and s is not None]
    luminance = surface.luminance(x, y, w, h)
    inks, shown = [], 0.0
    for _ in range(1 if style == "embossed" else c.inks):
        inks.append(compose.choose_ink(rng, style, luminance, c))
        compose.blend(photo, layer, x, y, inks[-1], surface)
        shown = min((compose.ink_change(before, photo[y:y + h, x:x + w], s) for s in boxed), default=1.0)
        if shown >= c.visible:
            break
        photo[y:y + h, x:x + w] = before
    return inks, shown


def _token(t: Token, box: list[int] | None) -> dict:
    """A token for the sidecar; its box in photo pixels (ends exclusive),
    or None when it has no ink (a distractor may lose its)."""
    out = {"text": t.text, "cls": t.cls, "box": box}
    if t.cls == "date":
        out |= {"role": t.role, "format": t.format, "parts": list(t.parts)}
    return out


def _rect(place: Sequence[int]) -> tuple[int, int, int, int]:
    x, y, w, h = place
    return x, y, x + w, y + h


def encode(photo: np.ndarray, quality: int) -> bytes:
    """The photo (float32 RGB) as JPEG bytes, rounded and clipped to 0..255."""
    buf = io.BytesIO()
    Image.fromarray(np.clip(photo + 0.5, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def label_boxes(sidecar: dict) -> list[Box]:
    """The label file's boxes, from the sidecar alone and in its order:
    every boxed token, block by block, then the background's boxes."""
    w, h = sidecar["size"]
    boxes = [Box(t["cls"], (t["box"][0] / w, t["box"][1] / h, t["box"][2] / w, t["box"][3] / h))
             for b in sidecar["blocks"] for t in b["tokens"] if t["cls"]]
    return boxes + [Box(label, tuple(xyxy)) for label, *xyxy in sidecar["background"]["boxes"]]


def label_text(sidecar: dict) -> str:
    """The YOLO label file for a sidecar line."""
    w, h = sidecar["size"]
    sample = Sample(id=sidecar["image"], image=Path(sidecar["image"]), width=w, height=h,
                    boxes=tuple(label_boxes(sidecar)))
    lines, dropped = yolo.label_lines(sample)
    if dropped:
        raise ValueError(f"{sidecar['image']}: {dropped} boxes with no area: the labels would not match the sidecar")
    return "".join(f"{line}\n" for line in lines)


def write_one(made: Made, images: Path) -> Path:
    """The JPEG into `images` and its label file where Ultralytics looks
    for it (both folders must exist); returns the image's path."""
    path = images / made.name
    path.write_bytes(made.jpeg)
    derived.label_path(path).write_text(made.labels)
    return path


# --- The spec, measured on what was made -----------------------------------------------------------

def shares(sidecars: Sequence[dict], spec: GenSpec) -> list[dict]:
    """Every share the spec sets, as drawn in these photos: rows of
    {what, spec, measured, n}. Text, style and size count every block drawn,
    skipped ones too; geometry counts the placed ones, and ink every ink
    drawn; month words count placed dates outside ranges, as the text spec
    draws them."""
    rows: list[dict] = []

    def row(what: str, expected: float, hits: Sequence[bool]) -> None:
        rows.append({"what": what, "spec": round(expected, 4),
                     "measured": round(float(np.mean(hits)), 4) if len(hits) else None, "n": len(hits)})

    placed = [b for s in sidecars for b in s["blocks"]]
    skipped = [k for s in sidecars for k in s["skipped"]]
    drawn = placed + skipped
    row("photos with 2 blocks", spec.compose.two_blocks, [s["wanted"] == 2 for s in sidecars])
    for name, share in spec.backgrounds:
        row(f"background from {name}", share, [s["background"]["list"] == name for s in sidecars])
    for field, table in (("kind", spec.text.kinds), ("style", spec.style.styles), ("band", spec.size.bands)):
        for name, share in table:
            row(f"{field}: {name}", share, [b[field] == name for b in drawn])
    g = spec.geometry
    for name, share in g.orientation:
        row(f"orientation: {name}", share, [b["geometry"]["orientation"] == name for b in placed])
    for name, share in g.surface:
        row(f"surface: {name}", share, [b["geometry"]["surface"] == name for b in placed])
    row("perspective", g.perspective, [b["geometry"]["perspective"] is not None for b in placed])
    # Every ink drawn, those replaced for showing too little included: the
    # spec draws them all alike, then the floor on faint ink
    # (spec.compose.visible) takes some low-contrast ones off again.
    inked = ("solid", "dot-matrix")
    inks = [i for b in placed if b["style"] in inked for i in (*b["faint_inks"], b["ink"])]
    inks += [i for k in skipped if k["style"] in inked for i in k.get("inks", ())]
    row("inks drawn: colourful (solid, dot-matrix)", spec.compose.colourful, [i["colourful"] for i in inks])
    row("inks drawn: low contrast (solid, dot-matrix)", spec.compose.low_contrast,
        [i["low_contrast"] is not None for i in inks])
    row("ink defects: a scratch", spec.compose.scratch, [b["defects"]["scratch"] is not None for b in placed
                                                          if b["style"] != "embossed"])
    dates = [(text.FAMILY_OF[d["format"]], d) for b in placed if b["kind"] != "range"
             for d in b["tokens"] if d["cls"] == "date"]
    words = [(family, d) for family, d in dates if family.startswith("word")]
    numbers = [(family, d) for family, d in dates if not family.startswith("word")]
    ts = spec.text
    row("dates: month as a word (ranges aside)", ts.month_word, [f.startswith("word") for f, _ in dates])
    row("month words: full name", ts.full_name, [f == "word-full" for f, _ in words])
    row("numeric dates: a day printed", ts.day_numbers, [d["parts"][2] is not None for _, d in numbers])
    row("month-word dates: a day printed", ts.day_words, [d["parts"][2] is not None for _, d in words])
    row("month words with a day: an ordinal", ts.ordinal,
        ["DDth" in d["format"] for _, d in words if d["parts"][2] is not None])
    cam = spec.camera
    for name in ("uneven", "dim", "glare", "defocus", "motion", "cast"):
        row(f"camera: {name}", getattr(cam, name), [name in s["camera"] for s in sidecars])
    row("camera: noise (and every dim photo)", cam.noise + (1 - cam.noise) * cam.dim,
        ["noise" in s["camera"] for s in sidecars])
    return rows
