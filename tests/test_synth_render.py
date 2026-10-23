"""The renderer (docs/m2-synthetic-generator.md, "Print styles").

The box tests compare each token's box, made from its own shrunk sprite,
with the block's ink found another way: all sprites on one 4x canvas,
shrunk with a plain numpy mean, then thresholded at INK. Every box must hold
all of its token's ink and touch it on all four sides (within 1 px), in
every style, including dot-matrix (dots at the edges) and embossed (no ink
colour, only light and shade).

Most tests need the fonts (python scripts/fetch_fonts.py) and are skipped
without them.
"""
from dataclasses import replace

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from bestbefore.synth import fonts, render as R, text as T
from bestbefore.synth.spec import STYLE

FAMILY = {f.name: f for f in fonts.FAMILIES}
needs_fonts = pytest.mark.skipif(not fonts.fetched(), reason="fonts not fetched: python scripts/fetch_fonts.py")


def _block(*lines) -> T.Block:
    """A block from lines of (text, class) pairs."""
    return T.Block(tuple(tuple(T.Token(text, cls) for text, cls in line) for line in lines), "test")


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
        if box is None:
            assert token.cls is None, f"{token.text!r} ({token.cls}) has no ink to box"
            continue
        x1, y1, x2, y2 = box
        ys, xs = np.nonzero(ink[y1:y2, x1:x2])
        assert len(ys), f"{token.text!r}: no ink inside its box"
        assert xs.min() <= 1 and ys.min() <= 1, f"{token.text!r}: box loose at the left or top"
        assert xs.max() >= x2 - x1 - 2 and ys.max() >= y2 - y1 - 2, \
            f"{token.text!r}: box loose at the right or bottom"
        boxed[max(0, y1 - 1):y2 + 1, max(0, x1 - 1):x2 + 1] = True
    assert not (ink & ~boxed).any(), "ink outside every box"


# --- Size --------------------------------------------------------------------------------

@needs_fonts
@pytest.mark.parametrize("size", [8, 20, 45])
def test_size_is_the_height_of_the_digits(size):
    block = _block([("2031", "date")])
    for family in fonts.families("train"):
        (box,) = R.boxes(R.shrink(R.render(block, "solid", family, size, np.random.default_rng(0))))
        assert size - 1 <= box[3] - box[1] <= size + 2, family.name


# --- Boxes from the ink -----------------------------------------------------------------

@needs_fonts
@pytest.mark.parametrize("style", R.STYLES)
@pytest.mark.parametrize("name", ["Roboto", "Merriweather", "Oswald", "Source Code Pro", "VT323"])
def test_a_lone_token_is_boxed_to_its_ink(style, name):
    block = _block([("15 AUG 2027", "date")])
    for size in (6, 11, 30):
        layer = R.render(block, style, FAMILY[name], size, np.random.default_rng(size))
        assert_boxes_hug_the_ink(block, layer)
        ys, xs = np.nonzero(canvas_ink(layer))                  # one token: its box is the ink's
        (box,) = R.boxes(R.shrink(layer))
        assert np.allclose(box, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), atol=1)


@needs_fonts
@pytest.mark.parametrize("style", R.STYLES)
def test_every_box_holds_all_its_ink_and_touches_it(style):
    rng = np.random.default_rng(R.STYLES.index(style))
    for i in range(40):
        block = T.random_block(rng)
        family = fonts.random_family(rng, "train")
        size = float(rng.choice([6, 9, 14, 24, 40]))
        assert_boxes_hug_the_ink(block, R.render(block, style, family, size, rng))


@needs_fonts
def test_sprites_sit_on_the_grid_and_lose_nothing_at_their_edges():
    rng = np.random.default_rng(1)
    for i in range(60):
        block, style = T.random_block(rng), R.STYLES[i % 4]
        layer = R.render(block, style, fonts.random_family(rng, "train"), float(rng.uniform(5, 50)), rng)
        assert layer.width % 4 == 0 and layer.height % 4 == 0
        for s in (*layer.tokens, layer.backing):
            if s is None:
                continue
            h, w = s.alpha.shape
            assert s.x % 4 == 0 and s.y % 4 == 0 and h % 4 == 0 and w % 4 == 0
            assert 0 <= s.x and s.x + w <= layer.width and 0 <= s.y and s.y + h <= layer.height
            border = np.concatenate([s.alpha[0], s.alpha[-1], s.alpha[:, 0], s.alpha[:, -1]])
            assert border.max() < 1 / 255, f"{style}: ink cut off at a sprite's edge"


def test_a_box_is_exactly_the_pixels_at_or_above_ink():
    alpha = np.zeros((10, 12), np.float32)
    alpha[3, 4] = R.INK                        # just ink
    alpha[6, 9] = 1.0
    alpha[8, 2] = R.INK - 0.01                 # just not
    assert R.ink_box(R.Sprite(100, 50, alpha)) == (104, 53, 110, 57)     # ends exclusive
    assert R.ink_box(R.Sprite(0, 0, np.full((4, 4), R.INK - 0.01, np.float32))) is None


def test_shrinking_is_a_mean_over_each_4_by_4():
    a = np.random.default_rng(0).random((48, 64), dtype=np.float32)
    layer = R.Layer(64, 48, 4, (R.Sprite(0, 0, a),))
    (small,) = R.shrink(layer).tokens
    assert np.allclose(small.alpha, a.reshape(12, 4, 16, 4).mean(axis=(1, 3)), atol=1e-6)
    assert (small.x, small.y, R.shrink(layer).width) == (0, 0, 16)


# --- Dot-matrix ------------------------------------------------------------------------------

EXACT_DOTS = replace(STYLE, dot_rows=(7,), dot_aspect=(1.0, 1.0), dot_radius=(0.3, 0.3),
                     dot_drop=(0.0, 0.0), dot_blur=(0.0, 0.0), dot_jitter=0.0)


def test_dotify_draws_dots_on_a_character_and_none_on_a_blank():
    rng = np.random.default_rng(0)
    dots, record = R.dot_params(rng, EXACT_DOTS, digit_px=70)       # a pitch of 10 px
    assert record["rows"] == 7
    bar = np.zeros((90, 60), np.float32)
    bar[10:80, 20:50] = 1.0                                        # 7 cells tall, 3 wide
    drawn = R.dotify(R.Sprite(0, 0, bar), grid_top=10, p=dots, rng=rng)
    n, _ = cv2.connectedComponents((drawn.alpha >= R.INK).astype(np.uint8))
    assert n - 1 == 21
    assert drawn.x >= 20 - 10 and drawn.x + drawn.alpha.shape[1] <= 50 + 10
    assert R.dotify(R.Sprite(0, 0, np.zeros_like(bar)), 10, dots, rng) is None
    all_dropped = {**dots, "drop": 1.0}
    assert R.dotify(R.Sprite(0, 0, bar), 10, all_dropped, rng) is None


def test_a_dot_fits_inside_its_kernel():
    widest = replace(STYLE, dot_rows=(min(STYLE.dot_rows),), dot_radius=(0.48, 0.48),
                     dot_blur=(0.35, 0.35))                     # the biggest dots drawn
    for digit_px in (20, 80, 300):
        kernel = R.dot_params(np.random.default_rng(0), widest, digit_px)[0]["kernel"]
        border = np.concatenate([kernel[0], kernel[-1], kernel[:, 0], kernel[:, -1]])
        assert border.max() < 1 / 255


@needs_fonts
def test_dotify_draws_a_glyph_as_separate_dots():
    block = _block([("8", "date")])
    layer = R.render(block, "dot-matrix", FAMILY["Roboto"], 30, np.random.default_rng(0), EXACT_DOTS)
    (sprite,) = layer.tokens
    n, _ = cv2.connectedComponents((sprite.alpha >= R.INK).astype(np.uint8))
    assert 10 <= n - 1 <= 35        # an 8 in a 7-row grid: about 20 dots, each on its own


# --- Embossed and label --------------------------------------------------------------------

def test_emboss_is_light_and_shade_at_the_edges_and_none_on_the_flat():
    square = np.zeros((120, 120), np.float32)
    square[30:90, 30:90] = 1.0
    lit_from_left = R.emboss(R.Sprite(0, 0, square), soft=4, strength=0.8, light=np.pi, raised=True)
    m = -lit_from_left.x                                           # the relief's margin
    alpha, tone = lit_from_left.alpha, lit_from_left.tone
    row = 60 + m
    assert alpha[row, 60 + m] < 0.01                               # the flat top: no ink
    assert alpha[row, 30 + m] >= R.INK and tone[row, 30 + m] == 1  # left edge faces the light
    assert alpha[row, 89 + m] >= R.INK and tone[row, 89 + m] == 0  # right edge in shade
    pressed = R.emboss(R.Sprite(0, 0, square), soft=4, strength=0.8, light=np.pi, raised=False)
    assert pressed.tone[row, 30 + m] == 0 and pressed.tone[row, 89 + m] == 1
    assert np.max(alpha) <= 0.8 + 1e-6


def test_a_relief_needs_no_more_room_than_its_sprite():
    """Embossing a token on its own sprite gives what embossing it in the
    middle of a much larger surface gives: no slope is cut off."""
    cov = np.zeros((40, 70), np.float32)
    cov[5:35, 10:20] = cov[5:12, 10:60] = 1.0                    # an L-ish glyph
    for soft in (1.5, 4.0):
        alone = R.emboss(R.Sprite(0, 0, cov), soft, 0.8, 1.0, True)
        wide = R.emboss(R.Sprite(-100, -100, np.pad(cov, 100)), soft, 0.8, 1.0, True)
        h, w = alone.alpha.shape
        dy, dx = alone.y - wide.y, alone.x - wide.x
        assert np.allclose(alone.alpha, wide.alpha[dy:dy + h, dx:dx + w], atol=1e-4)
        assert wide.alpha.sum() == pytest.approx(alone.alpha.sum(), rel=1e-4)


@needs_fonts
def test_embossed_layers_have_tone_and_the_others_none():
    block = _block([("EXP", "due"), ("08/2027", "date")])
    for style in R.STYLES:
        layer = R.shrink(R.render(block, style, FAMILY["Oswald"], 30, np.random.default_rng(0)))
        tone = R.tone(layer)
        if style == "embossed":
            on = R.ink(layer) >= R.INK
            assert set(np.unique(np.round(tone[on]))) == {0.0, 1.0}
        else:
            assert tone is None


@needs_fonts
def test_the_label_sticker_lies_under_all_the_text_and_is_no_ones_ink():
    block = _block([("MFG:", "prod"), ("12/2025", "date")], [("EXP:", "due"), ("11/2027", "date")])
    layer = R.shrink(R.render(block, "label", FAMILY["Roboto Mono"], 20, np.random.default_rng(0)))
    s = layer.backing
    for x1, y1, x2, y2 in R.boxes(layer):
        assert s.x <= x1 and s.y <= y1 and x2 <= s.x + s.alpha.shape[1] and y2 <= s.y + s.alpha.shape[0]
    assert layer.params["label"]["colour"] in [list(c) for c in STYLE.label_colours]
    assert R.ink(layer)[s.y + 1, s.x + 1] == 0                      # the sticker's corner isn't ink


# --- Same seed, same pixels; layout as Pillow's ------------------------------------------------

@needs_fonts
@pytest.mark.parametrize("style", R.STYLES)
def test_the_same_seed_gives_the_same_pixels(style):
    block = T.random_block(np.random.default_rng(5))

    def draw(seed: int) -> R.Layer:
        return R.shrink(R.render(block, style, FAMILY["Work Sans"], 18, np.random.default_rng(seed)))

    first = draw(3)
    fonts.load.cache_clear()                    # a cold font cache changes nothing
    again, other = draw(3), draw(4)
    assert again.params == first.params
    assert np.array_equal(R.ink(again), R.ink(first))
    assert not np.array_equal(R.ink(other), R.ink(first)) or other.params != first.params


@needs_fonts
def test_glyphs_are_placed_as_pillow_places_them():
    """With no letter spacing, a token drawn glyph by glyph matches Pillow
    drawing the whole string in the same font."""
    flat = replace(STYLE, letter_spacing=(0.0, 0.0), bold=0.0)
    text = "15 AUG 2027"
    for name in ("Roboto", "Merriweather", "VT323"):
        layer = R.render(_block([(text, "date")]), "solid", FAMILY[name], 25,
                         np.random.default_rng(0), flat)
        font = fonts.load(FAMILY[name].regular, layer.params["px"])
        left, top, right, bottom = font.getbbox(text, anchor="ls")
        im = Image.new("L", (right - left, bottom - top))
        ImageDraw.Draw(im).text((-left, -top), text, font=font, fill=255, anchor="ls")
        theirs = np.asarray(im, dtype=np.float32) / 255
        (ours,) = layer.tokens
        x1, y1, x2, y2 = R.ink_box(R.Sprite(0, 0, ours.alpha))
        u1, v1, u2, v2 = R.ink_box(R.Sprite(0, 0, theirs))
        assert abs((x2 - x1) - (u2 - u1)) <= 2 and abs((y2 - y1) - (v2 - v1)) <= 1, name
        assert abs(ours.alpha.sum() / theirs.sum() - 1) < 0.02, name     # no glyph cut off


def test_an_unknown_style_is_refused():
    with pytest.raises(ValueError, match="unknown style"):
        R.render(_block([("1", "date")]), "stencil", FAMILY["Roboto"], 10, np.random.default_rng(0))
