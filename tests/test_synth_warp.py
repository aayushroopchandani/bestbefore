"""Geometry for text layers (docs/m2-synthetic-generator.md, "Geometry").

A quarter turn must move every box exactly as it moves the pixels. After a
tilt, a perspective, a cylinder or a bend, every box must still hold all of
its token's ink and touch it on all four sides. As in test_synth_render.py,
the ink is found a second way: every warped 4x sprite on one canvas, shrunk
by a plain mean, thresholded at INK. That also catches a patch knocked off
the 4x grid or cut off at its edge.

Tests that draw text need the fonts (python scripts/fetch_fonts.py).
"""
from dataclasses import replace

import numpy as np
import pytest

from bestbefore.synth import fonts, render as R, text as T, warp as W
from bestbefore.synth.spec import GEOMETRY
from bestbefore.synth.warp import Geometry

needs_fonts = pytest.mark.skipif(not fonts.fetched(), reason="fonts not fetched: python scripts/fetch_fonts.py")

GEOMETRIES = {
    "tilted": Geometry("tilted", 0, 31.0),
    "tilted the other way": Geometry("tilted", 0, -17.5),
    "sideways": Geometry("sideways", 1, 6.0),
    "sideways at 270": Geometry("sideways", 3, -8.0),
    "perspective": Geometry("upright", 0, 2.5, tilt=30.0, tilt_axis=70.0, distance=2.0),
    "cylinder": Geometry("upright", 0, 0.0, "cylinder", arc=100.0, centre=20.0),
    "bend up": Geometry("upright", 0, 1.0, "bend", arc=70.0, arch="up"),
    "bend down": Geometry("upright", 0, -4.0, "bend", arc=70.0, arch="down"),
    "everything": Geometry("sideways", 3, 7.0, "cylinder", arc=80.0, centre=-25.0,
                           tilt=25.0, tilt_axis=135.0, distance=2.5),
}


def canvas_ink(layer4: R.Layer) -> np.ndarray:
    """The block's ink, the other way: every 4x sprite on one canvas, shrunk
    by a plain mean over k x k, thresholded at INK."""
    k = layer4.scale
    full = R.ink(layer4)
    h, w = full.shape
    return full.reshape(h // k, k, w // k, k).mean(axis=(1, 3)) >= R.INK


def assert_boxes_hug_the_ink(block: T.Block, layer4: R.Layer) -> None:
    ink = canvas_ink(layer4)
    boxed = np.zeros_like(ink)
    for token, box in zip(block.tokens, R.boxes(R.shrink(layer4))):
        if box is None:     # no ink left at this size (make.py redraws the block); any would be unboxed
            continue
        x1, y1, x2, y2 = box
        ys, xs = np.nonzero(ink[y1:y2, x1:x2])
        assert len(ys), f"{token.text!r}: no ink inside its box"
        assert xs.min() <= 1 and ys.min() <= 1, f"{token.text!r}: box loose at the left or top"
        assert xs.max() >= x2 - x1 - 2 and ys.max() >= y2 - y1 - 2, \
            f"{token.text!r}: box loose at the right or bottom"
        boxed[max(0, y1 - 1):y2 + 1, max(0, x1 - 1):x2 + 1] = True
    assert not (ink & ~boxed).any(), "ink outside every box"


def assert_on_the_grid_and_whole(layer: R.Layer) -> None:
    """Every patch on the 4x grid, inside the layer, with blank edges. (A
    patch cut too small would still have them, padded back onto the grid:
    test_a_warp_needs_no_more_room_than_its_patch is what catches that.)"""
    assert layer.width % 4 == 0 and layer.height % 4 == 0
    for s in (*layer.tokens, layer.backing):
        if s is None:
            continue
        h, w = s.alpha.shape
        assert s.x % 4 == 0 and s.y % 4 == 0 and h % 4 == 0 and w % 4 == 0
        assert 0 <= s.x and s.x + w <= layer.width and 0 <= s.y and s.y + h <= layer.height
        edge = np.concatenate([s.alpha[0], s.alpha[-1], s.alpha[:, 0], s.alpha[:, -1]])
        assert edge.max() < 1 / 255, "ink cut off at a patch's edge"


def _bars(rects, size) -> R.Layer:
    """A 4x layer of `size` (w, h) whose tokens are filled rectangles
    (x1, y1, x2, y2), each on its own patch on the grid."""
    sprites = tuple(R.aligned(R.Sprite(x1, y1, np.ones((y2 - y1, x2 - x1), np.float32)), 4)
                    for x1, y1, x2, y2 in rects)
    return R.Layer(size[0], size[1], 4, sprites)


def _roomy(layer: R.Layer, p: int = 16) -> R.Layer:
    """The same layer, in the same frame, with p blank pixels more round every patch."""
    def pad(s):
        return None if s is None else R.Sprite(s.x - p, s.y - p, np.pad(s.alpha, p),
                                               None if s.tone is None else np.pad(s.tone, p))
    return replace(layer, tokens=tuple(pad(s) for s in layer.tokens), backing=pad(layer.backing))


def _ink_box(a: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(a >= R.INK)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _turned_box(box, size, quarter):
    """A pixel box (ends exclusive) after `quarter` anticlockwise turns of a size (w, h) image."""
    (x1, y1, x2, y2), (w, h) = box, size
    for _ in range(quarter):
        (x1, y1, x2, y2), (w, h) = (y1, w - x2, y2, w - x1), (h, w)
    return (x1, y1, x2, y2)


# --- Quarter turns: exact ------------------------------------------------------------------

def test_a_quarter_turn_is_anticlockwise_and_moves_pixels_exactly():
    layer = _bars([(40, 8, 60, 16)], (64, 32))           # a patch at the top right
    turned = W.turn(layer, 1)
    assert (turned.width, turned.height) == (32, 64)
    assert _ink_box(R.ink(turned)) == (8, 4, 16, 24)     # the right edge is now the top
    assert np.array_equal(R.ink(W.turn(turned, 3)), R.ink(layer))
    assert np.array_equal(R.ink(W.turn(W.turn(layer, 2), 2)), R.ink(layer))


@needs_fonts
@pytest.mark.parametrize("quarter", [1, 2, 3])
@pytest.mark.parametrize("style", R.STYLES)
def test_a_quarter_turn_moves_every_box_exactly(quarter, style):
    rng = np.random.default_rng(quarter)
    for _ in range(6):
        block = T.random_block(rng)
        layer = R.render(block, style, fonts.random_family(rng, "train"), float(rng.choice([7, 16])), rng)
        small = R.shrink(layer)
        turned = R.shrink(W.turn(layer, quarter))
        size = (small.width, small.height)
        assert R.boxes(turned) == tuple(None if b is None else _turned_box(b, size, quarter)
                                        for b in R.boxes(small))
        assert np.array_equal(R.ink(W.turn(layer, quarter)), np.rot90(R.ink(layer), quarter))
        # Shrunk, only the order each 4 x 4 is summed in differs: float rounding.
        assert np.allclose(R.ink(turned), np.rot90(R.ink(small), quarter), atol=1e-6)
        assert_on_the_grid_and_whole(W.turn(layer, quarter))


# --- Tilt, perspective, cylinder, bend: boxes from the warped ink -----------------------------

@needs_fonts
@pytest.mark.parametrize("name", GEOMETRIES)
@pytest.mark.parametrize("style", R.STYLES)
def test_every_box_holds_all_its_ink_after_a_warp(name, style):
    rng = np.random.default_rng(len(name) * 7 + R.STYLES.index(style))
    for _ in range(8):
        block = T.random_block(rng)
        layer = R.render(block, style, fonts.random_family(rng, "train"), float(rng.choice([6, 11, 24])), rng)
        warped = W.apply(layer, GEOMETRIES[name])
        assert_on_the_grid_and_whole(warped)
        assert_boxes_hug_the_ink(block, warped)


@needs_fonts
@pytest.mark.parametrize("name", GEOMETRIES)
def test_a_warp_needs_no_more_room_than_its_patch(name):
    """Each step, warping every patch onto its own new patch, keeps all the
    ink that the same step keeps with plenty of room round each patch:
    nothing is cut off. (Step by step: a step's transform depends on its
    layer's frame, which a roomier layer's next step would not share.)"""
    g, rng = GEOMETRIES[name], np.random.default_rng(len(name))
    for i in range(12):
        block = T.random_block(rng)
        layer = R.render(block, R.STYLES[i % 4], fonts.random_family(rng, "train"),
                         float(rng.choice([6, 12, 30])), rng)
        for step, before in ((W.curve, layer), (W.view, W.turn(W.curve(layer, g), g.quarter))):
            tight, roomy = step(before, g), step(_roomy(before), g)
            for a, b in zip((*tight.tokens, tight.backing), (*roomy.tokens, roomy.backing)):
                if a is not None:
                    assert a.alpha.sum() == pytest.approx(b.alpha.sum(), rel=2e-4, abs=0.05), (name, step)


@needs_fonts
def test_random_geometries_keep_every_box_on_its_ink():
    rng = np.random.default_rng(11)
    for i in range(60):
        block = T.random_block(rng)
        layer = R.render(block, R.STYLES[i % 4], fonts.random_family(rng, "train"),
                         float(rng.uniform(5, 40)), rng)
        warped = W.apply(layer, W.random_geometry(rng))
        assert_on_the_grid_and_whole(warped)
        assert_boxes_hug_the_ink(block, warped)


@needs_fonts
def test_an_embossed_tone_is_resampled_with_its_light_and_shade():
    block = T.Block(((T.Token("EXP", "due"), T.Token("08/2027", "date")),), "single")
    layer = R.render(block, "embossed", fonts.families("train")[0], 24, np.random.default_rng(0))
    same = W.view(layer, Geometry("upright", 0, 0.0))            # no turn: the same pixels, padded
    (x0, y0, x1, y1), (u0, v0, u1, v1) = (R.ink_box(R.Sprite(0, 0, R.ink(a))) for a in (layer, same))
    on = R.ink(layer)[y0:y1, x0:x1] >= R.INK
    assert np.allclose(R.tone(same)[v0:v1, u0:u1][on], R.tone(layer)[y0:y1, x0:x1][on], atol=1e-5)
    turned = W.view(layer, Geometry("tilted", 0, 20.0))
    tone, on = R.tone(turned), R.ink(turned) >= R.INK
    assert np.mean(np.minimum(tone[on], 1 - tone[on]) < 0.2) > 0.9     # lit or shaded, not grey


def test_turning_loses_no_ink():
    layer = _bars([(8, 8, 200, 40), (8, 56, 120, 88)], (208, 96))
    total = R.ink(layer).sum()
    for g in (Geometry("tilted", 0, 33.0), Geometry("tilted", 0, -5.0), Geometry("sideways", 3, -8.0)):
        warped = W.apply(layer, g)
        assert_on_the_grid_and_whole(warped)
        assert abs(R.ink(warped).sum() / total - 1) < 0.01, g


def test_a_positive_angle_turns_anticlockwise():
    layer = _bars([(8, 40, 240, 56)], (248, 96))             # a long bar
    a = R.ink(W.view(layer, Geometry("tilted", 0, 20.0)))
    ys, xs = np.nonzero(a >= 0.5)
    left, right = ys[xs < np.percentile(xs, 10)].mean(), ys[xs > np.percentile(xs, 90)].mean()
    assert right < left - 40                                 # its right end went up


def test_a_perspective_makes_the_near_side_bigger():
    layer = _bars([(8, 8, 40, 200), (216, 8, 248, 200)], (256, 208))   # two tall bars, left and right
    # The plane turned about the vertical axis by 30 degrees: one side comes closer.
    near_far = [_ink_box(s.alpha) for s in W.view(layer, Geometry("upright", 0, 0.0, tilt=30.0,
                                                                   tilt_axis=90.0, distance=2.0)).tokens]
    heights = [y2 - y1 for _, y1, _, y2 in near_far]
    assert max(heights) > 1.1 * min(heights)


def test_a_cylinder_squeezes_the_text_towards_its_ends():
    squares = [(8 + 40 * i, 8, 40 + 40 * i, 40) for i in range(9)]     # nine 32 px squares in a row
    layer = W.curve(_bars(squares, (376, 48)), Geometry("upright", 0, 0.0, "cylinder", arc=100.0))
    widths = [x2 - x1 for x1, _, x2, _ in (_ink_box(s.alpha) for s in layer.tokens)]
    heights = {y2 - y1 for _, y1, _, y2 in (_ink_box(s.alpha) for s in layer.tokens)}
    assert widths[4] > widths[2] > widths[0] and widths[4] > widths[6] > widths[8]
    assert widths[0] < 0.8 * widths[4] and heights == {32}
    # Turned so its right end faces away: the right end is squeezed more than the left.
    turned = W.curve(_bars(squares, (376, 48)), Geometry("upright", 0, 0.0, "cylinder", arc=80.0, centre=30.0))
    widths = [x2 - x1 for x1, _, x2, _ in (_ink_box(s.alpha) for s in turned.tokens)]
    assert widths[8] < widths[0] - 4 and widths == sorted(widths, reverse=True)


@pytest.mark.parametrize("arch", ["up", "down"])
def test_a_bend_arcs_the_line(arch):
    squares = [(8 + 40 * i, 8, 40 + 40 * i, 40) for i in range(9)]
    layer = W.curve(_bars(squares, (376, 48)), Geometry("upright", 0, 0.0, "bend", arc=60.0, arch=arch))
    tops = [s.y + _ink_box(s.alpha)[1] for s in layer.tokens]
    middle_higher = tops[4] < min(tops[0], tops[8]) - 10
    assert middle_higher if arch == "up" else tops[4] > max(tops[0], tops[8]) + 10


# --- Drawing a geometry -------------------------------------------------------------------------

def test_random_geometry_follows_the_spec():
    rng = np.random.default_rng(0)
    drawn = [W.random_geometry(rng) for _ in range(10_000)]
    for name, share in GEOMETRY.orientation:
        assert abs(np.mean([g.orientation == name for g in drawn]) - share) < 0.03, name
    for name, share in GEOMETRY.surface:
        assert abs(np.mean([g.surface == name for g in drawn]) - share) < 0.03, name
    assert abs(np.mean([g.distance > 0 for g in drawn]) - GEOMETRY.perspective) < 0.03
    sideways = [g for g in drawn if g.orientation == "sideways"]
    assert abs(np.mean([g.quarter == 1 for g in sideways]) - 0.5) < 0.03
    assert {g.quarter for g in sideways} == {1, 3} and {g.quarter for g in drawn if g not in sideways} == {0}
    assert all(abs(g.angle) <= 5 for g in drawn if g.orientation == "upright")
    assert all(5 <= abs(g.angle) <= 35 for g in drawn if g.orientation == "tilted")
    assert all(abs(g.angle) <= 8 for g in sideways)
    for g in drawn:
        if g.surface == "cylinder":               # both ends of the block within 70 degrees of the camera
            assert abs(g.centre) + g.arc / 2 <= GEOMETRY.cylinder_edge + 1e-9
