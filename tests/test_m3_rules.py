"""M3's decision rules on made-up records (m3.py). So far R1 (S2) and the
parser's freeze."""
import pytest

from bestbefore import m3


def rows(spec):
    """spec: (reader, mode, pad, shape, outcome, count) -> that many rows."""
    return [{"reader": r, "mode": m, "pad": p, "shape": s, "outcome": o}
            for r, m, p, s, o, k in spec for _ in range(k)]


def test_pad_is_the_most_right_alone_turned_over_both_readers():
    got = m3.crop_recipe(rows([
        ("rapidocr", "turned", 0.0, "wide", "right", 5), ("ppocrv6", "turned", 0.0, "wide", "right", 5),
        ("rapidocr", "turned", 0.15, "wide", "right", 4), ("ppocrv6", "turned", 0.15, "wide", "right", 7),
        ("rapidocr", "turned", 0.4, "wide", "right", 9),
        # the full pipeline and as-cut reads don't count toward the pad
        ("rapidocr", "full", 0.4, "wide", "right", 50), ("ppocrv6", "alone", 0.0, "wide", "right", 50),
    ]))
    assert got["pad"] == 0.15
    assert got["right_alone_turned_by_pad"] == {"0.0": 10, "0.15": 11, "0.4": 9}


def test_a_tie_goes_to_the_smaller_pad():
    got = m3.crop_recipe(rows([("rapidocr", "turned", p, "wide", "right", 3) for p in (0.4, 0.15)]))
    assert got["pad"] == 0.15


@pytest.mark.parametrize("alone, turned, keep", [
    ({"right": 2, "wrong": 1}, {"right": 5, "wrong": 1}, True),
    ({"right": 2, "wrong": 1}, {"right": 5, "wrong": 2}, False),     # adds a wrong read
    ({"right": 2, "wrong": 1}, {"right": 2, "wrong": 0}, False),     # adds no right read
])
def test_turning_judged_on_tall_crops_at_the_chosen_pad(alone, turned, keep):
    spec = [("ppocrv6", "turned", 0.0, "wide", "right", 10)]          # pad 0 wins
    for mode, counts in (("alone", alone), ("turned", turned)):
        spec += [("rapidocr", mode, 0.0, "tall", o, k) for o, k in counts.items()]
        spec += [("rapidocr", mode, 0.4, "tall", "wrong", 9)]        # another pad: ignored
    spec += [("ppocrv6", "alone", 0.0, "wide", "wrong", 9)]           # wide crops: ignored
    got = m3.crop_recipe(rows(spec))
    assert got["pad"] == 0.0 and got["turn"] is keep


def test_unknown_readers_and_untruthed_crops_count_nothing():
    got = m3.crop_recipe(rows([("other", "turned", 0.4, "wide", "right", 9),
                               ("rapidocr", "turned", 0.4, "wide", None, 9),
                               ("rapidocr", "turned", 0.15, "wide", "right", 1)]))
    assert got["pad"] == 0.15


def test_the_parser_is_frozen():
    assert m3.PARSE_SHA256 is not None
    assert m3.check_parser() == m3.PARSE_SHA256
