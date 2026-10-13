import json

import pytest
from PIL import Image

from bestbefore import expdate
from bestbefore.samples import Expiry, parse_month, parse_year

# One test image in the documented format: an expiry with its day / month /
# year parts, an "EXP" mark, and a box that runs past the right edge.
TEST_JSON = {
    "test_00001.jpg": {
        "height": 100, "width": 200,
        "ann": [
            {"cls": "exp", "bbox": [20, 40, 120, 60], "transcription": "2021.08.03",
             "dmy_ann": [
                 {"cls": "year", "bbox": [20, 40, 60, 60], "transcription": "2021"},
                 {"cls": "month", "bbox": [65, 40, 85, 60], "transcription": "08"},
                 {"cls": "day", "bbox": [90, 40, 120, 60], "transcription": "03"}]},
            {"cls": "due", "bbox": [0, 40, 18, 60]},
            {"cls": "code", "bbox": [150, 80, 210, 95], "transcription": "L123"},
        ],
    }
}


@pytest.fixture
def root(tmp_path):
    split = tmp_path / expdate.SPLITS["real-test"]
    (split / "images").mkdir(parents=True)
    (split / "annotations.json").write_text(json.dumps(TEST_JSON))
    Image.new("RGB", (200, 100)).save(split / "images" / "test_00001.jpg")
    return tmp_path


def test_load_split(root):
    [s] = expdate.load_split("real-test", root)
    assert (s.id, s.width, s.height) == ("test_00001.jpg", 200, 100)
    assert [b.label for b in s.boxes] == ["exp", "due", "code"]
    assert s.boxes[0].xyxy == (0.1, 0.4, 0.6, 0.6)
    assert s.boxes[2].xyxy[2] == 1.0                # clipped to the edge
    assert s.expiry == Expiry(2021, 8, 3)


def test_missing_image_is_an_error(root):
    (root / expdate.SPLITS["real-test"] / "images" / "test_00001.jpg").unlink()
    with pytest.raises(ValueError, match="1 annotated images missing"):
        expdate.load_split("real-test", root)


@pytest.mark.parametrize("parts, expected", [
    ({"year": "2027", "month": "08", "day": "15"}, Expiry(2027, 8, 15)),
    ({"year": "27", "month": "AUG", "day": "15"}, Expiry(2027, 8, 15)),
    ({"year": "2031", "month": "04"}, Expiry(2031, 4)),     # month only
    ({"month": "08", "day": "15"}, None),                    # no year printed
    ({"year": "2027", "month": "30", "day": "08"}, None),    # bad month
    ({"year": "2027", "month": "02", "day": "30"}, None),    # no such day
    ({"year": "2028", "month": "02", "day": "29"}, Expiry(2028, 2, 29)),
])
def test_expiry_from_parts(parts, expected):
    dmy = [{"cls": k, "transcription": v} for k, v in parts.items()]
    assert expdate.expiry_from_parts(dmy) == expected


def test_parsers():
    assert parse_month("August") == 8
    assert parse_month("13") is None
    assert parse_year("31") == 2031
    assert parse_year("202") is None
    assert str(Expiry(2031, 4)) == "2031-04"
    assert str(Expiry(2026, 11, 20)) == "2026-11-20"
