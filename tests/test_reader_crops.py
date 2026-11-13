"""Cutting date boxes with padding (reader/crops.py)."""
from bestbefore.reader import crops


def test_to_pixels():
    assert crops.to_pixels((0.1, 0.2, 0.5, 0.4), (1000, 500)) == (100.0, 100.0, 500.0, 200.0)


def test_padding_is_a_share_of_the_short_side():
    box = (100.0, 50.0, 300.0, 70.0)               # 200 x 20: short side 20
    assert crops.padded(box, 0.0, (1000, 1000)) == (100, 50, 300, 70)
    assert crops.padded(box, 0.15, (1000, 1000)) == (97, 47, 303, 73)
    assert crops.padded(box, 0.4, (1000, 1000)) == (92, 42, 308, 78)
    tall = (50.0, 100.0, 70.0, 300.0)              # a sideways date: the same margin
    assert crops.padded(tall, 0.4, (1000, 1000)) == (42, 92, 78, 308)


def test_padding_rounds_out_and_clips():
    assert crops.padded((10.4, 20.6, 50.5, 30.2), 0.0, (100, 100)) == (10, 20, 51, 31)
    assert crops.padded((2.0, 3.0, 98.0, 99.0), 0.4, (100, 100)) == (0, 0, 100, 100)


def test_shape_and_turns():
    assert not crops.is_tall((0, 0, 20, 10)) and not crops.is_tall((0, 0, 10, 10))
    assert crops.is_tall((0, 0, 10, 20))
    assert crops.turns(False) == (0, 2) and crops.turns(True) == (1, 3)


# --- Stored crops (S3): a margin, shrunk, turned ----------------------------------------

import numpy as np                      # noqa: E402
import pytest                           # noqa: E402
from PIL import Image                   # noqa: E402

from bestbefore import derived          # noqa: E402

_ROT = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}


def _marked(size=(300, 200), box=(60, 80, 180, 110)) -> Image.Image:
    """A grey photo with a dark bar where `box` is and a white notch in the
    bar's top-left corner, so every turn of it looks different."""
    a = np.full((size[1], size[0], 3), 180, np.uint8)
    x1, y1, x2, y2 = box
    a[y1:y2, x1:x2] = 20
    a[y1:y1 + 6, x1:x1 + 6] = 255
    return Image.fromarray(a)


def _dark_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    a = np.asarray(image.convert("L")) < 100
    ys, xs = np.nonzero(a)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def test_turn_box_follows_the_pixels():
    image = _marked()
    for q in (1, 2, 3):
        turned = image.transpose(_ROT[q])
        assert crops.turn_box((60, 80, 180, 110), image.size, q) == _dark_bbox(turned)


def test_cut_keeps_half_the_short_side_round_the_box():
    image = _marked()
    crop, box = crops.cut(image, (60.0, 80.0, 180.0, 110.0))
    assert crop.size == (120 + 30, 30 + 30)           # 15 px (half of 30) on every side
    assert box == (15.0, 15.0, 135.0, 45.0)
    assert _dark_bbox(crop) == box


def test_cut_clips_the_margin_at_the_photo_edge():
    image = _marked(box=(2, 80, 120, 110))
    crop, box = crops.cut(image, (2.0, 80.0, 120.0, 110.0))
    assert crop.size == (120 + 15, 60) and box[0] == 2.0


def test_cut_shrinks_a_big_box_to_max_short_and_never_enlarges():
    image = _marked(size=(1200, 800), box=(100, 100, 900, 260))     # short side 160
    crop, box = crops.cut(image, (100.0, 100.0, 900.0, 260.0))
    assert min(box[2] - box[0], box[3] - box[1]) == pytest.approx(crops.MAX_SHORT, abs=0.5)
    assert crop.height == round(320 * crops.MAX_SHORT / 160)
    small = _marked()
    crop, box = crops.cut(small, (60.0, 80.0, 180.0, 110.0))
    assert crop.size == (150, 60)                       # 30 px short side: kept as it is


def test_cut_turns_a_turned_photo_back():
    """A box cut from a photo turned q quarters, turned (4 - q) % 4 when cut,
    is the crop of the unturned photo: the rule gen's and the suites' turns use."""
    image = _marked()
    want, want_box = crops.cut(image, (60.0, 80.0, 180.0, 110.0))
    for q in (1, 2, 3):
        turned = image.transpose(_ROT[q])
        xyxy = derived.turn_box((60 / 300, 80 / 200, 180 / 300, 110 / 200), q)
        box_px = crops.to_pixels(xyxy, turned.size)
        crop, box = crops.cut(turned, box_px, (4 - q) % 4)
        assert crop.size == want.size and np.array_equal(np.asarray(crop), np.asarray(want))
        assert box == pytest.approx(want_box)


def test_cut_refuses_a_box_outside_the_photo():
    with pytest.raises(ValueError):
        crops.cut(_marked(), (400.0, 10.0, 450.0, 30.0))
