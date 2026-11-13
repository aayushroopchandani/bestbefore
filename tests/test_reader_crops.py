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
