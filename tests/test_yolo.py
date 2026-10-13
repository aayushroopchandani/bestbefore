from pathlib import Path

import pytest

from bestbefore.samples import Box, Sample
from bestbefore.yolo import from_yolo, label_lines, read_label_file, to_yolo


@pytest.mark.parametrize("xyxy", [
    (0.1, 0.4, 0.6, 0.6),
    (0.0, 0.0, 1.0, 1.0),
    (0.913, 0.271, 0.9871, 0.2893),
])
def test_round_trip(xyxy):
    assert from_yolo(to_yolo(xyxy)) == pytest.approx(xyxy)


def test_label_lines():
    sample = Sample(
        id="a.jpg", image=Path("a.jpg"), width=200, height=100,
        boxes=(Box("exp", (0.1, 0.4, 0.6, 0.6)),       # expiry -> date
               Box("code", (0.5, 0.5, 0.5, 0.7)),      # zero width: dropped
               Box("due", (0.0, 0.4, 0.09, 0.6))))
    lines, dropped = label_lines(sample)
    assert dropped == 1
    assert lines == ["0 0.350000 0.500000 0.500000 0.200000",
                     "1 0.045000 0.500000 0.090000 0.200000"]
    [(cls, xyxy), _] = read_label_file("\n".join(lines))
    assert cls == 0 and xyxy == pytest.approx((0.1, 0.4, 0.6, 0.6))


def test_empty_label_file():
    assert read_label_file("") == []
