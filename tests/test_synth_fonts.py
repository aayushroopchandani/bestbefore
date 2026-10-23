"""The generator's fonts (docs/m2-synthetic-generator.md, "Fonts"): the
plan's families, SIL OFL or Apache 2.0, every file hashed, regular and bold
instances as asked, and the held-out families never drawing training text.

Tests that open the font files need them fetched (python
scripts/fetch_fonts.py) and are skipped without them.
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from bestbefore.synth import fonts, text as T

FAMILY = {f.name: f for f in fonts.FAMILIES}
needs_fonts = pytest.mark.skipif(not fonts.fetched(), reason="fonts not fetched: python scripts/fetch_fonts.py")

TRAIN = ("Roboto", "Open Sans", "Lato", "Montserrat", "Source Sans 3", "Noto Sans", "Work Sans",
         "Oswald", "Roboto Condensed", "Barlow Condensed", "Noto Serif", "Merriweather",
         "Roboto Mono", "Source Code Pro", "Share Tech Mono", "VT323")
HELD_OUT = ("IBM Plex Sans", "Archivo Narrow", "PT Serif", "IBM Plex Mono")


# --- The font table (no files needed) -------------------------------------------------

def test_the_font_table_is_the_plans():
    assert tuple(f.name for f in fonts.families("train")) == TRAIN
    assert tuple(f.name for f in fonts.families("held-out")) == HELD_OUT
    assert {f.licence for f in fonts.FAMILIES} <= set(fonts.LICENCES)
    files = {face.file for f in fonts.FAMILIES for face in fonts.faces(f)}
    assert files == set(fonts.SHA256)
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in fonts.SHA256.values())
    assert [f.name for f in fonts.FAMILIES if f.bold is None] == ["Share Tech Mono", "VT323"]
    for f in fonts.FAMILIES:                           # regular 400, bold 700, on purpose
        if f.regular.axes:
            assert dict(f.regular.axes)["wght"] == 400 and dict(f.bold.axes)["wght"] == 700


def test_training_text_never_draws_a_held_out_family():
    rng = np.random.default_rng(0)
    drawn = {fonts.random_family(rng, "train").name for _ in range(5000)}
    assert drawn == set(TRAIN)
    assert {fonts.random_family(rng, "held-out").name for _ in range(500)} == set(HELD_OUT)
    with pytest.raises(ValueError):
        fonts.families("test")


# --- The fonts themselves ----------------------------------------------------------------

@needs_fonts
@pytest.mark.parametrize("family", fonts.FAMILIES, ids=lambda f: f.name)
def test_every_face_has_its_hash_characters_axes_and_licence(family):
    for face in fonts.faces(family):
        assert fonts.problems(face, T.CHARSET) == []


def _ink_area(font: ImageFont.FreeTypeFont) -> int:
    left, top, right, bottom = font.getbbox(fonts.DIGITS, anchor="ls")
    im = Image.new("L", (right - left, bottom - top))
    ImageDraw.Draw(im).text((-left, -top), fonts.DIGITS, font=font, fill=255, anchor="ls")
    return int(np.asarray(im, dtype=np.int64).sum())


@needs_fonts
def test_regular_and_bold_are_the_weights_asked_for():
    for f in fonts.FAMILIES:
        if f.bold:
            assert _ink_area(fonts.load(f.bold, 200)) > 1.15 * _ink_area(fonts.load(f.regular, 200)), f.name
    # Montserrat's default instance is Thin: without its axes set, digits come out hairline.
    face = FAMILY["Montserrat"].regular
    default = ImageFont.truetype(str(fonts.path(face)), 200, layout_engine=ImageFont.Layout.BASIC)
    assert _ink_area(fonts.load(face, 200)) > 1.5 * _ink_area(default)
