"""Composing generated photos (docs/m2-synthetic-generator.md, "Placement and
blending", "Labels and the sidecar").

The backgrounds here are made-up textured photos in a temporary folder, so
nothing needs ExpDate; the tests that draw text need the fonts
(python scripts/fetch_fonts.py).
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bestbefore import derived
from bestbefore.samples import Box
from bestbefore.synth import compose, fonts, make, render as R, text as T, warp as W
from bestbefore.synth.compose import Background, Backgrounds, Surface
from bestbefore.synth.spec import COMPOSE, GEN, SIZE

needs_fonts = pytest.mark.skipif(not fonts.fetched(), reason="fonts not fetched: python scripts/fetch_fonts.py")


def _texture(w: int, h: int, seed: int) -> np.ndarray:
    """A photo-like surface: smooth light and dark patches, some fine grain."""
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(40, 230, (6, 8, 3)).astype(np.float32)
    im = np.asarray(Image.fromarray(coarse.astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC), np.float32)
    return np.clip(im + rng.normal(0, 6, im.shape), 0, 255).astype(np.float32)


@pytest.fixture(scope="module")
def pool(tmp_path_factory) -> Backgrounds:
    """Three backgrounds: 1,200 and 1,800 px (both fit small print; the
    second is shrunk to 1,600) and 700 px (normal print only), with boxes."""
    root = tmp_path_factory.mktemp("backgrounds")
    made = []
    for name, (w, h), boxes, seed in (
            ("train_real", (1200, 900), (Box("date", (0.1, 0.1, 0.3, 0.14)), Box("due", (0.6, 0.7, 0.7, 0.75))), 1),
            ("train_synth_clean", (1300, 1800), (Box("code", (0.4, 0.4, 0.6, 0.43)),), 2),
            ("train_synth_clean", (700, 500), (), 3)):
        path = root / f"{seed}.jpg"
        Image.fromarray(_texture(w, h, seed).astype(np.uint8)).save(path, quality=92)
        made.append(Background(name, path, w, h, boxes))
    return Backgrounds({"train_real": (made[0],), "train_synth_clean": tuple(made[1:])})


def _photos(pool: Backgrounds, n: int, seed: int = 3) -> list[make.Made]:
    return [make.make_one(i, GEN, pool, seed) for i in range(n)]


# --- Backgrounds ------------------------------------------------------------------------

def test_a_background_is_shrunk_to_1600_with_its_shape(pool):
    big = pool.by_list["train_synth_clean"][0]
    photo = compose.load_photo(big, COMPOSE.max_side)
    assert photo.shape == (1600, round(1300 * 1600 / 1800), 3) and photo.dtype == np.float32
    small = pool.by_list["train_synth_clean"][1]
    assert compose.load_photo(small, COMPOSE.max_side).shape == (500, 700, 3)       # never enlarged


def test_small_print_gets_a_background_of_at_least_1000_px(pool):
    rng = np.random.default_rng(0)
    picked = {pool.pick(rng, GEN.backgrounds, COMPOSE.min_side_small).path.name for _ in range(200)}
    assert picked == {"1.jpg", "2.jpg"}
    picked = {pool.pick(rng, GEN.backgrounds, COMPOSE.min_side).path.name for _ in range(200)}
    assert picked == {"1.jpg", "2.jpg", "3.jpg"}


def test_no_list_may_hold_a_test_photo_a_copy_or_a_validation_photo(monkeypatch):
    monkeypatch.setattr(compose, "list_ids", lambda folder, name: {"img_00010.jpg"})
    paths = [Path("/x/images/real-train/img_00010.jpg"), Path("/x/images/real-train/img_00011.jpg"),
             Path("/x/images/synth/img_00948.jpg"), Path("/x/images/synth/img_00010.jpg"),
             Path("/x/images/real-test/img_00001.jpg")]
    assert compose.forbidden("train_real", paths) == [str(paths[i]) for i in (0, 2, 4)]
    assert compose.forbidden("val", paths) == [str(paths[i]) for i in (2, 4)]        # synth-val's own photos


# --- Size and placement -------------------------------------------------------------------------

def test_print_heights_follow_their_bands():
    rng = np.random.default_rng(0)
    drawn = [compose.print_height(rng) for _ in range(10_000)]
    assert abs(np.mean([band == "small" for band, _ in drawn]) - 0.30) < 0.03
    for band, h in drawn:
        lo, hi = getattr(SIZE, band)
        assert lo <= h <= hi
    normal = np.log([h for band, h in drawn if band == "normal"])          # log-uniform: flat in log
    assert abs(np.median(normal) - np.log(np.sqrt(0.015 * 0.045))) < 0.05


def test_a_placed_block_is_inside_the_photo_and_clear_of_every_box():
    surface = Surface(_texture(800, 600, 0))
    rng = np.random.default_rng(1)
    for _ in range(300):
        taken = [tuple(sorted(rng.uniform(0, 800, 2))) + tuple(sorted(rng.uniform(0, 600, 2)))
                 for _ in range(int(rng.integers(0, 6)))]
        taken = [(x1, y1, x2, y2) for x1, x2, y1, y2 in taken]
        w, h, margin = int(rng.integers(20, 300)), int(rng.integers(10, 150)), float(rng.uniform(0, 20))
        spot = compose.place(w, h, margin, taken, surface, rng)
        if spot is None:
            continue
        x, y = spot
        assert 0 <= x and x + w <= 800 and 0 <= y and y + h <= 600
        for x1, y1, x2, y2 in taken:
            assert x + w + margin <= x1 or x2 <= x - margin or y + h + margin <= y1 or y2 <= y - margin


def test_placement_prefers_the_plainest_surface():
    photo = _texture(800, 400, 5)
    photo[:, 400:] = 128                                      # the right half is flat
    surface = Surface(photo)
    rng = np.random.default_rng(2)
    spots = [compose.place(100, 50, 5, [], surface, rng) for _ in range(400)]
    on_flat = np.mean([x >= 400 for x, _ in spots])
    assert 0.70 <= on_flat <= 0.95       # the plainest of 20 (70%), else random (30%, half of them flat)


def test_a_block_too_big_for_the_photo_has_no_place():
    assert compose.place(900, 50, 0, [], Surface(_texture(800, 600, 0)), np.random.default_rng(0)) is None


# --- Ink and blending ------------------------------------------------------------------------------

def test_dark_ink_on_light_surfaces_and_light_ink_on_dark_ones():
    rng = np.random.default_rng(0)
    for surface, mode in ((0.8, "dark"), (0.5, "dark"), (0.3, "light")):
        inks = [compose.choose_ink(rng, "solid", surface) for _ in range(500)]
        assert {i.mode for i in inks} == {mode}
        assert all(0.75 <= i.opacity <= 1.0 for i in inks)
        assert abs(np.mean([i.colourful for i in inks]) - 0.15) < 0.05
        assert abs(np.mean([i.low_contrast is not None for i in inks]) - 0.20) < 0.05
    assert compose.choose_ink(rng, "label", 0.1).mode == "label"           # black, on its sticker
    assert compose.choose_ink(rng, "embossed", 0.9).mode == "relief"


def _flat_layer(w: int = 40, h: int = 20) -> R.Layer:
    return R.Layer(w, h, 1, (R.Sprite(0, 0, np.ones((h, w), np.float32)),), None, {})


@pytest.mark.parametrize("grey", [60.0, 200.0])
def test_low_contrast_ink_lands_its_gap_from_the_surface(grey):
    rng = np.random.default_rng(int(grey))
    for _ in range(50):
        ink = compose.choose_ink(rng, "dot-matrix", grey / 255)
        if ink.low_contrast is None:
            continue
        photo = np.full((20, 40, 3), grey, np.float32)
        compose.blend(photo, _flat_layer(), 0, 0, ink, Surface(photo.copy()))
        gap = abs(compose.luminance(photo[10, 20]) - grey / 255)
        assert gap == pytest.approx(ink.low_contrast, abs=0.02)


def test_dark_ink_keeps_the_texture_and_light_ink_follows_a_shadow():
    photo = np.full((60, 120, 3), 200.0, np.float32)
    photo[:, ::2] = 170.0                                     # fine texture
    before = photo.copy()
    compose.blend(photo, _flat_layer(120, 60), 0, 0, compose.Ink("dark", (40.0, 40.0, 40.0)), Surface(before))
    assert np.allclose(photo[:, ::2] / photo[:, 1::2], 170 / 200)        # multiplied: texture shows through

    lit = np.full((200, 400, 3), 60.0, np.float32)
    lit[:, 200:] *= 0.5                                      # a shadow over the right half
    surface = Surface(lit.copy())
    compose.blend(lit, _flat_layer(400, 200), 0, 0, compose.Ink("light", (240.0, 240.0, 240.0)), surface)
    assert lit[100, 380, 0] < 0.7 * lit[100, 20, 0]          # the ink darkens with the surface


def test_the_light_follows_a_shadow_but_not_dark_print():
    printed = np.full((400, 800, 3), 220.0, np.float32)
    for x in range(0, 800, 24):
        printed[:, x:x + 6] = 40.0                         # dark print, narrower than 5% of the long side
    assert Surface(printed).shading(100, 100, 400, 200).min() > 0.97
    shadowed = printed.copy()
    shadowed[:, 400:] *= 0.5                               # a shadow over half the photo
    shade = Surface(shadowed).shading(100, 100, 600, 200)
    assert shade[:, :150].min() > 0.97 and shade[:, 450:].max() < 0.65
    assert shade.min() >= COMPOSE.shading_floor


# --- Ink defects ------------------------------------------------------------------------------------

@needs_fonts
@pytest.mark.parametrize("style", R.STYLES)
def test_boxes_hold_all_the_ink_after_the_defects(style):
    rng = np.random.default_rng(R.STYLES.index(style))
    spec = replace(COMPOSE, scratch=1.0, density=(0.4, 0.4), spread=(0.4, 0.4))     # every defect, at its most
    for _ in range(15):
        block = T.random_block(rng)
        layer = W.apply(R.render(block, style, fonts.random_family(rng, "train"), float(rng.choice([7, 18])), rng),
                        W.random_geometry(rng))
        layer, record = compose.defects(layer, style, [t.cls is not None for t in block.tokens], rng, spec)
        ink = R.ink(R.shrink(layer)) >= R.INK
        boxed = np.zeros_like(ink)
        for box in R.boxes(R.shrink(layer)):
            if box:
                x1, y1, x2, y2 = box
                boxed[y1:y2, x1:x2] = True
        assert not (ink & ~boxed).any()
        assert record == {} if style == "embossed" else record["scratch"] is not None


def test_spreading_ink_loses_none_of_it():
    """Ink one pixel from its patch's edge spreads past it: the patch grows first."""
    bar = R.Sprite(0, 0, np.pad(np.ones((22, 46), np.float32), 1))      # 1 blank pixel all round
    layer = R.Layer(48, 24, 4, (bar,), None, {"size": 6.0})
    spec = replace(COMPOSE, density=(0.0, 0.0), spread=(0.4, 0.4), scratch=0.0)
    spread, record = compose.defects(layer, "solid", [True], np.random.default_rng(0), spec)
    assert record["spread"] == 0.4
    assert R.ink(spread).sum() == pytest.approx(R.ink(layer).sum(), rel=1e-4)
    assert spread.width > layer.width                                  # it has grown on the grid


@needs_fonts
def test_uneven_density_and_a_scratch_take_ink_away():
    block = T.Block(((T.Token("EXP", "due"), T.Token("08/2027", "date")),), "single")
    layer = R.render(block, "solid", fonts.families("train")[2], 30, np.random.default_rng(0))
    uneven = replace(COMPOSE, density=(0.4, 0.4), spread=(0.0, 0.0), scratch=0.0)
    lighter, _ = compose.defects(layer, "solid", [True, True], np.random.default_rng(1), uneven)
    ratio = R.ink(lighter)[R.ink(layer) > 0.99] / R.ink(layer)[R.ink(layer) > 0.99]
    assert 0.6 - 1e-6 <= ratio.min() and ratio.max() <= 1 + 1e-6 and ratio.max() - ratio.min() > 0.1
    scratched = replace(COMPOSE, density=(0.0, 0.0), spread=(0.0, 0.0), scratch=1.0)
    after, record = compose.defects(layer, "solid", [True, True], np.random.default_rng(2), scratched)
    i = record["scratch"]
    assert after.tokens[i].alpha.sum() < 0.995 * layer.tokens[i].alpha.sum()      # a thin line through it
    assert after.tokens[1 - i].alpha.sum() == pytest.approx(layer.tokens[1 - i].alpha.sum())


# --- Whole photos ---------------------------------------------------------------------------------------

@needs_fonts
def test_the_same_index_gives_the_same_jpeg_bytes(pool):
    for i in range(3):
        first, again = make.make_one(i, GEN, pool, 3), make.make_one(i, GEN, pool, 3)
        assert first.jpeg == again.jpeg and first.labels == again.labels and first.sidecar == again.sidecar
    assert make.make_one(1, GEN, pool, 3).jpeg != make.make_one(2, GEN, pool, 3).jpeg
    assert make.make_one(2, GEN, pool, 4).jpeg != make.make_one(2, GEN, pool, 3).jpeg   # another set's seed


@needs_fonts
def test_a_block_whose_ink_vanishes_is_skipped_never_written_without_its_boxes(pool):
    tiny = replace(GEN, size=replace(SIZE, bands=(("small", 1.0),), small=(0.0004, 0.0005)))   # under 1 px
    for i in range(3):
        s = make.make_one(i, tiny, pool, 3).sidecar
        assert s["blocks"] == [] and {k["why"] for k in s["skipped"]} == {"a boxed token kept no ink"}
        assert len(make.label_boxes(s)) == len(s["background"]["boxes"])


@needs_fonts
def test_every_box_is_inside_the_photo_and_every_date_has_one(pool):
    for made in _photos(pool, 25):
        s = made.sidecar
        w, h = s["size"]
        for b in make.label_boxes(s):
            x1, y1, x2, y2 = b.xyxy
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
        for block in s["blocks"]:
            x, y, bw, bh = block["place"]
            assert 0 <= x and x + bw <= w and 0 <= y and y + bh <= h
            for t in block["tokens"]:
                assert t["cls"] is None or t["box"] is not None, t
                if t["box"]:
                    x1, y1, x2, y2 = t["box"]
                    assert x <= x1 < x2 <= x + bw and y <= y1 < y2 <= y + bh
        if any(b["band"] == "small" for b in s["blocks"] + s["skipped"]):
            assert max(s["background"]["size"]) >= COMPOSE.min_side_small


@needs_fonts
def test_blocks_never_overlap_each_other_or_the_backgrounds_boxes(pool):
    for made in _photos(pool, 25, seed=5):
        s = made.sidecar
        w, h = s["size"]
        rects = [(x, y, x + bw, y + bh) for x, y, bw, bh in (b["place"] for b in s["blocks"])]
        theirs = [(x1 * w, y1 * h, x2 * w, y2 * h) for _, x1, y1, x2, y2 in s["background"]["boxes"]]
        for i, r in enumerate(rects):
            others = np.asarray(rects[:i] + rects[i + 1:] + theirs, float).reshape(-1, 4)
            assert not compose.overlaps(others, r)


@needs_fonts
def test_label_lines_round_trip_in_the_sidecars_order(pool, tmp_path):
    images = tmp_path / "images" / "set"
    for folder in (images, tmp_path / "labels" / "set"):
        folder.mkdir(parents=True)
    for made in _photos(pool, 10):
        path = make.write_one(made, images)
        s = made.sidecar
        assert Image.open(path).size == tuple(s["size"])
        back = derived.load_sample(path)
        meant = make.label_boxes(s)
        boxed = [t for b in s["blocks"] for t in b["tokens"] if t["cls"]]
        assert len(meant) == len(boxed) + len(s["background"]["boxes"]) == len(back.boxes)
        w, h = s["size"]
        for m, b in zip(meant, back.boxes):
            assert m.label == b.label
            assert np.allclose(m.xyxy, b.xyxy, atol=2e-6)
        for t, m in zip(boxed, meant):                      # the sidecar's tokens, in its order
            assert t["cls"] == m.label
            assert np.allclose(m.xyxy, (t["box"][0] / w, t["box"][1] / h, t["box"][2] / w, t["box"][3] / h))
        assert derived.label_path(path).read_text() == make.label_text(s)
