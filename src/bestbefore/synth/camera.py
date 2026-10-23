"""Camera effects on the whole finished photo (docs/m2-synthetic-generator.md,
"Camera effects"). The new text and the background get one blur, one
noise, one light and one compression together, never the text alone
(realism rule 2). Mild on purpose: heavy damage is M4's readability
checker's, which can reuse this module with a stronger CameraSpec.

Applied in the order light reaches the photo: uneven and dim light, glare
(the sensor saturates there), defocus and motion blur, sensor noise, a
colour cast. The JPEG save comes last, in make.py. Dim light brings noise
with it, so a dim photo always gets noise.

`draw` makes every random choice but the noise's pixels and returns the
record that goes into the sidecar; `apply` follows it.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from .spec import CAMERA, CameraSpec

Rect = tuple[float, float, float, float]      # x1, y1, x2, y2 in photo pixels


def draw(rng: np.random.Generator, width: int, height: int, near: Sequence[Rect],
         spec: CameraSpec = CAMERA) -> dict:
    """Which effects a photo gets, each by its share, and how strong. A
    glare is centred on one of the `near` rects (the placed blocks) with
    share spec.glare_near_block, anywhere otherwise."""
    effects: dict = {}
    if rng.random() < spec.uneven:
        ramp = rng.uniform(*spec.uneven_ramp) * (1 if rng.random() < 0.5 else -1)
        if rng.random() < spec.uneven_radial:          # brightest (or darkest) round a point
            effects["uneven"] = {"kind": "radial", "ramp": round(ramp, 4),
                                 "centre": [round(rng.uniform(0, width), 1), round(rng.uniform(0, height), 1)]}
        else:                                          # brighter towards one side
            effects["uneven"] = {"kind": "linear", "ramp": round(ramp, 4), "angle": round(rng.uniform(0, 360), 1)}
    if rng.random() < spec.dim:
        effects["dim"] = round(rng.uniform(*spec.dim_level), 4)
    if rng.random() < spec.glare:
        glares = []
        for _ in range(int(rng.integers(spec.glare_count[0], spec.glare_count[1] + 1))):
            if near and rng.random() < spec.glare_near_block:
                x1, y1, x2, y2 = near[int(rng.integers(len(near)))]
                cx, cy = rng.uniform(x1, x2), rng.uniform(y1, y2)
            else:
                cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            glares.append({"centre": [round(cx, 1), round(cy, 1)],
                           "radius": round(rng.uniform(*spec.glare_size) * min(width, height), 1),
                           "aspect": round(rng.uniform(*spec.glare_aspect), 3),
                           "angle": round(rng.uniform(0, 180), 1),
                           "strength": round(rng.uniform(*spec.glare_strength), 3)})
        effects["glare"] = glares
    if rng.random() < spec.defocus:
        effects["defocus"] = round(rng.uniform(*spec.defocus_sigma), 3)
    if rng.random() < spec.motion:
        effects["motion"] = {"length": int(rng.integers(spec.motion_length[0], spec.motion_length[1] + 1)),
                             "angle": round(rng.uniform(0, 180), 1)}
    if rng.random() < spec.noise or "dim" in effects:
        effects["noise"] = round(rng.uniform(*spec.noise_sigma), 3)
    if rng.random() < spec.cast:
        effects["cast"] = [round(rng.uniform(-spec.cast_shift, spec.cast_shift), 4) for _ in range(3)]
    return effects


def apply(photo: np.ndarray, effects: dict, rng: np.random.Generator) -> np.ndarray:
    """The photo (float32 RGB, 0..255) through `effects`, as `draw` recorded
    them; the noise's pixels come from rng. Changes `photo` in place where
    it can, and returns the result."""
    h, w = photo.shape[:2]
    if "uneven" in effects:
        photo *= ramp(effects["uneven"], w, h)[..., None]
    if "dim" in effects:
        photo *= effects["dim"]
    for g in effects.get("glare", ()):
        glare(photo, g)
    np.minimum(photo, 255, out=photo)                      # the sensor saturates
    if "defocus" in effects:
        photo = cv2.GaussianBlur(photo, (0, 0), effects["defocus"], borderType=cv2.BORDER_REFLECT)
    if "motion" in effects:
        photo = cv2.filter2D(photo, -1, motion_kernel(**effects["motion"]), borderType=cv2.BORDER_REFLECT)
    if "noise" in effects:
        photo += rng.standard_normal(photo.shape, dtype=np.float32) * np.float32(effects["noise"])
    if "cast" in effects:
        photo *= 1 + np.asarray(effects["cast"], np.float32)
    return photo


def ramp(u: dict, w: int, h: int) -> np.ndarray:
    """Brightness factors (h x w) from 1 - ramp to 1 + ramp: linear, along a
    direction across the whole photo; radial, from a point (1 + ramp) to the
    corner farthest from it (1 - ramp)."""
    xs, ys = np.arange(w, dtype=np.float32)[None, :], np.arange(h, dtype=np.float32)[:, None]
    if u["kind"] == "linear":
        a = math.radians(u["angle"])
        c, s = math.cos(a), math.sin(a)
        reach = (abs(c) * w + abs(s) * h) / 2               # from the middle to the farthest edge
        t = ((xs - w / 2) * c + (ys - h / 2) * s) / reach
    else:
        cx, cy = u["centre"]
        far = max(math.hypot(cx - x, cy - y) for x in (0, w) for y in (0, h))
        t = 1 - 2 * np.hypot(xs - cx, ys - cy) / far
    return (1 + np.float32(u["ramp"]) * t).astype(np.float32)


def glare(photo: np.ndarray, g: dict) -> None:
    """A soft bright ellipse, in place: towards white by g["strength"] at its
    centre, falling off as a Gaussian whose long radius is g["radius"]."""
    h, w = photo.shape[:2]
    cx, cy = g["centre"]
    sx, sy = g["radius"] / 2, g["radius"] * g["aspect"] / 2          # the Gaussian's sigmas
    reach = 3 * sx                                                    # beyond it, under 1%
    x0, x1 = max(0, math.floor(cx - reach)), min(w, math.ceil(cx + reach) + 1)
    y0, y1 = max(0, math.floor(cy - reach)), min(h, math.ceil(cy + reach) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    a = math.radians(g["angle"])
    dx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
    dy = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
    u, v = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
    p = (np.float32(g["strength"]) * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2)))[..., None]
    region = photo[y0:y1, x0:x1]
    region += p * (255 - region)


def motion_kernel(length: int, angle: float) -> np.ndarray:
    """A straight streak `length` pixels long at `angle` degrees, summing to 1."""
    size = length if length % 2 else length + 1
    k = np.zeros((size, size), np.float32)
    c, a = size // 2, math.radians(angle)
    for t in np.linspace(-(length - 1) / 2, (length - 1) / 2, 8 * length):   # splat a fine line
        x, y = c + t * math.cos(a), c + t * math.sin(a)
        i, j = math.floor(y), math.floor(x)
        fy, fx = y - i, x - j
        for di, dj, wt in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
            if 0 <= i + di < size and 0 <= j + dj < size:
                k[i + di, j + dj] += wt
    return k / k.sum()
