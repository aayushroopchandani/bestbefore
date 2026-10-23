"""Geometry for text layers: turned, tilted, seen at an angle, printed on a
can or a cap (docs/m2-synthetic-generator.md, "Geometry").

A layer is warped at its drawn size (4x), before render.shrink(), sprite by
sprite: each sprite is resampled onto a new patch big enough for all of it,
so nothing is cut off and no canvas the size of the whole block is made.
The patches are then padded back onto the 4x grid (render.aligned), so
shrinking a warped layer still gives exactly the pixels that shrinking it
whole would, and every box is the tight rectangle around its token's warped
ink (render.boxes after render.shrink), never a transformed rectangle.

The steps, in the order a pack is printed, turned and photographed:

  surface   flat; a cylinder (a can or bottle: the text compresses towards
            its edges); or a bend (a cap or jar: the lines arc)
  turn      a quarter turn for sideways text: numpy.rot90, exact
  view      the angle left over, and the perspective: one homography

A sprite's pixel (u, v) is the point (x + u, y + v) of the layer. Patches
are resampled bilinearly with zeros outside them (outside a sprite is flat
surface); an embossed sprite's tone is resampled weighted by its alpha, as
render.shrink does. Angles are in degrees, and a positive one turns
anticlockwise on screen, as derived.turn does.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import cv2
import numpy as np

from . import render
from .render import Layer, Sprite
from .spec import GEOMETRY, GeometrySpec
from .text import pick

_MARGIN = 2         # pixels around a new patch beyond its sprite's reach

Points = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Geometry:
    """How one block is turned and bent."""
    orientation: str                # upright, tilted or sideways
    quarter: int                    # quarter turns anticlockwise: 1 or 3 when sideways, else 0
    angle: float                    # the turn after the quarter turns
    surface: str = "flat"           # flat, cylinder or bend
    arc: float = 0.0                # cylinder, bend: how much of a circle the block's width covers
    centre: float = 0.0             # cylinder: where the block's middle faces, from the camera
    arch: str = "up"                # bend: "up" (the top edge of a cap) or "down"
    tilt: float = 0.0               # perspective: the block's plane turned from the camera,
    tilt_axis: float = 0.0          #   about this direction in it,
    distance: float = 0.0           #   seen from this many block diagonals; 0: no perspective

    def record(self) -> dict:
        """For the sidecar: what was drawn, without the unused fields."""
        out = {"orientation": self.orientation, "quarter": self.quarter,
               "angle": round(self.angle, 2), "surface": self.surface}
        if self.surface == "cylinder":
            out |= {"arc": round(self.arc, 1), "centre": round(self.centre, 1)}
        elif self.surface == "bend":
            out |= {"arc": round(self.arc, 1), "arch": self.arch}
        out["perspective"] = ({"tilt": round(self.tilt, 1), "axis": round(self.tilt_axis, 1),
                               "distance": round(self.distance, 2)} if self.distance else None)
        return out


def random_geometry(rng: np.random.Generator, spec: GeometrySpec = GEOMETRY) -> Geometry:
    """An orientation, a surface and maybe a perspective, each by its share."""
    orientation = pick(rng, spec.orientation)
    quarter = 0
    if orientation == "upright":
        angle = rng.uniform(-spec.upright_angle, spec.upright_angle)
    elif orientation == "tilted":
        angle = rng.uniform(*spec.tilt_angle) * (1 if rng.random() < 0.5 else -1)
    else:
        quarter = 1 if rng.random() < 0.5 else 3
        angle = rng.uniform(-spec.sideways_angle, spec.sideways_angle)
    g = Geometry(orientation, quarter, float(angle), pick(rng, spec.surface))
    if g.surface == "cylinder":
        arc = rng.uniform(*spec.cylinder_arc)
        reach = spec.cylinder_edge - arc / 2               # both ends stay within cylinder_edge
        g = replace(g, arc=float(arc), centre=float(rng.uniform(-reach, reach)))
    elif g.surface == "bend":
        g = replace(g, arc=float(rng.uniform(*spec.bend_arc)),
                    arch="up" if rng.random() < 0.5 else "down")
    if rng.random() < spec.perspective:
        g = replace(g, tilt=float(rng.uniform(*spec.perspective_tilt)),
                    tilt_axis=float(rng.uniform(0, 180)),
                    distance=float(rng.uniform(*spec.perspective_distance)))
    return g


def apply(layer: Layer, g: Geometry) -> Layer:
    """The layer on its surface, turned, then seen at its angle."""
    return view(turn(curve(layer, g), g.quarter), g)


# --- Turn: exact quarter turns --------------------------------------------------------

def turn(layer: Layer, quarter: int) -> Layer:
    """The layer turned `quarter` x 90 degrees anticlockwise. Exact: pixels
    move, none is resampled, and a layer on the grid stays on it."""
    q = quarter % 4
    if q == 0:
        return layer
    w0, h0 = layer.width, layer.height

    def turned(s: Sprite | None) -> Sprite | None:
        if s is None:
            return None
        h, w = s.alpha.shape
        # Anticlockwise, the right edge becomes the top: (x, y) -> (y, w0 - x).
        x, y = {1: (s.y, w0 - s.x - w), 2: (w0 - s.x - w, h0 - s.y - h), 3: (h0 - s.y - h, s.x)}[q]
        tone = None if s.tone is None else np.ascontiguousarray(np.rot90(s.tone, q))
        return Sprite(x, y, np.ascontiguousarray(np.rot90(s.alpha, q)), tone)

    width, height = (h0, w0) if q % 2 else (w0, h0)
    return replace(layer, width=width, height=height, tokens=tuple(turned(s) for s in layer.tokens),
                   backing=turned(layer.backing))


# --- Surface: a cylinder or a bend ------------------------------------------------------

def curve(layer: Layer, g: Geometry) -> Layer:
    """The layer printed on its surface: flat leaves it as it is."""
    if g.surface == "flat":
        return layer
    if g.surface == "cylinder":
        forward, inverse = _cylinder(layer.width, math.radians(g.arc), math.radians(g.centre))
    elif g.surface == "bend":
        forward, inverse = _bend(layer.width, layer.height, math.radians(g.arc), g.arch)
    else:
        raise ValueError(f"unknown surface {g.surface!r}")
    return _resampled(layer, lambda a, x, y: _remap(a, x, y, forward, inverse))


def _cylinder(width: float, arc: float, centre: float) -> tuple[Points, Points]:
    """Text round a cylinder whose axis runs across its lines, seen square
    on: a point `t` radians round from facing the camera lands at R sin(t).
    The block's width covers `arc` radians, its middle at `centre`, and its
    middle stays where it was."""
    r, cx, s = width / arc, width / 2, math.sin(centre)

    def forward(x, y):
        return cx + r * (np.sin(centre + (x - cx) / r) - s), y

    def inverse(x, y):
        return cx + r * (np.arcsin(np.clip((x - cx) / r + s, -1, 1)) - centre), y
    return forward, inverse


def _bend(width: float, height: float, arc: float, arch: str) -> tuple[Points, Points]:
    """The lines follow circles round one centre, below the block when it
    arches up and above when down. The middle line keeps its length, `arc`
    radians of a circle of radius R; R is at least the block's height, so
    no line folds over."""
    r0, cx, cy = max(width / arc, height), width / 2, height / 2
    up = 1 if arch == "up" else -1
    oy = cy + up * r0                                   # the circles' centre

    def forward(x, y):
        t, r = (x - cx) / r0, r0 - up * (y - cy)
        return cx + r * np.sin(t), oy - up * r * np.cos(t)

    def inverse(x, y):
        dx, dy = x - cx, up * (oy - y)
        return cx + r0 * np.arctan2(dx, dy), cy + up * (r0 - np.hypot(dx, dy))
    return forward, inverse


def _remap(a: np.ndarray, x: int, y: int, forward: Points, inverse: Points
           ) -> tuple[np.ndarray, int, int]:
    """A patch with its top-left at (x, y) through a curved map: its new
    place from where its outline goes (one pixel beyond its outer pixels),
    each new pixel read from where `inverse` says it came from."""
    h, w = a.shape[:2]
    us, vs = np.arange(x - 1, x + w + 1, dtype=np.float64), np.arange(y - 1, y + h + 1, dtype=np.float64)
    ox = np.concatenate([us, us, np.full_like(vs, us[0]), np.full_like(vs, us[-1])])
    oy = np.concatenate([np.full_like(us, vs[0]), np.full_like(us, vs[-1]), vs, vs])
    fx, fy = forward(ox, oy)
    x0, y0 = math.floor(fx.min()) - _MARGIN, math.floor(fy.min()) - _MARGIN
    x1, y1 = math.ceil(fx.max()) + _MARGIN + 1, math.ceil(fy.max()) + _MARGIN + 1
    # A row of x and a column of y, broadcast: float32 is exact to about
    # 0.001 px here, at a fraction of float64's cost.
    sx, sy = inverse(np.arange(x0, x1, dtype=np.float32)[None, :], np.arange(y0, y1, dtype=np.float32)[:, None])
    shape = (y1 - y0, x1 - x0)
    out = cv2.remap(a, np.broadcast_to(sx - x, shape).astype(np.float32),
                    np.broadcast_to(sy - y, shape).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, x0, y0


# --- View: the angle left over and the perspective ------------------------------------------

def view(layer: Layer, g: Geometry) -> Layer:
    """The layer turned by g.angle about its centre and, with a
    perspective, seen at an angle: one homography, one resampling."""
    m = view_matrix(layer.width, layer.height, g)
    return _resampled(layer, lambda a, x, y: _warp(a, x, y, m))


def view_matrix(width: int, height: int, g: Geometry) -> np.ndarray:
    """The homography (3 x 3, layer pixels to new pixels). The perspective
    turns the block's plane by g.tilt about an axis in it, at g.tilt_axis,
    and looks at it from g.distance block diagonals: its near side grows,
    its far side shrinks, its middle stays."""
    cx, cy = width / 2, height / 2
    c, s = math.cos(math.radians(g.angle)), math.sin(math.radians(g.angle))
    # Anticlockwise on screen, where y points down: (1, 0) from the centre goes to (c, -s).
    m = np.array([[c, s, cx - c * cx - s * cy], [-s, c, cy + s * cx - c * cy], [0.0, 0.0, 1.0]])
    if not g.distance:
        return m
    corners = _points(m, np.array([[0, 0], [width, 0], [width, height], [0, height]], float))
    p = np.column_stack([corners - (cx, cy), np.zeros(4)])
    a, t = math.radians(g.tilt_axis), math.radians(g.tilt)
    k = np.array([math.cos(a), math.sin(a), 0.0])
    p = p * math.cos(t) + np.cross(k, p) * math.sin(t) + np.outer(p @ k, k) * (1 - math.cos(t))
    d = g.distance * math.hypot(width, height)
    seen = p[:, :2] * (d / (d + p[:, 2:])) + (cx, cy)
    return cv2.getPerspectiveTransform(corners.astype(np.float32), seen.astype(np.float32)) @ m


def _points(m: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Points (n x 2) through a homography."""
    p = np.column_stack([xy, np.ones(len(xy))]) @ m.T
    return p[:, :2] / p[:, 2:]


def _warp(a: np.ndarray, x: int, y: int, m: np.ndarray) -> tuple[np.ndarray, int, int]:
    """A patch with its top-left at (x, y) through homography m; its new
    place from where its corners go (one pixel beyond its outer pixels)."""
    h, w = a.shape[:2]
    moved = _points(m, np.array([[x - 1, y - 1], [x + w, y - 1], [x + w, y + h], [x - 1, y + h]], float))
    x0, y0 = math.floor(moved[:, 0].min()) - _MARGIN, math.floor(moved[:, 1].min()) - _MARGIN
    x1, y1 = math.ceil(moved[:, 0].max()) + _MARGIN + 1, math.ceil(moved[:, 1].max()) + _MARGIN + 1
    local = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ m @ np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
    size, flags = (x1 - x0, y1 - y0), dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
    if np.allclose(local[2], (0, 0, 1)):
        return cv2.warpAffine(a, local[:2], size, **flags), x0, y0
    return cv2.warpPerspective(a, local, size, **flags), x0, y0


# --- Every sprite through one map -----------------------------------------------------------

def _resampled(layer: Layer, fn: Callable[[np.ndarray, int, int], tuple[np.ndarray, int, int]]) -> Layer:
    """Every sprite through fn(patch, x, y) -> (new patch, new x, new y),
    padded back onto the grid, the layer's top-left moved to (0, 0)."""
    k = layer.scale

    def one(s: Sprite | None) -> Sprite | None:
        if s is None:
            return None
        if s.tone is None:
            alpha, x, y = fn(s.alpha, s.x, s.y)
            return render.aligned(Sprite(x, y, alpha), k)
        both, x, y = fn(np.dstack([s.alpha, s.alpha * s.tone]), s.x, s.y)
        alpha = np.ascontiguousarray(both[..., 0])
        tone = np.clip(both[..., 1] / np.maximum(alpha, 1e-6), 0, 1)
        return render.aligned(Sprite(x, y, alpha, tone), k)

    return render.assemble([one(s) for s in layer.tokens], one(layer.backing), k, layer.params)
