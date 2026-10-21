import json

import pytest
from PIL import Image

from bestbefore import store
from bestbefore.samples import Expiry

LABEL = {"expiry": "2027-08", "derived": False, "mfg": None, "format": "MM/YYYY",
         "tag": "clean", "unreadable": False, "note": "",
         "boxes": [{"label": "expiry", "xyxy": [0.1, 0.5, 0.4, 0.55]}]}


@pytest.fixture
def root(tmp_path):
    (tmp_path / "samples9").mkdir()
    Image.new("RGB", (40, 20)).save(tmp_path / "samples9" / "a.jpg")
    (tmp_path / "samples9" / "notes.txt").write_text("not a photo")
    return tmp_path


@pytest.fixture
def frozen(root, tmp_path_factory):
    path = tmp_path_factory.mktemp("store") / "test_set.json"
    path.write_text(json.dumps(store.freeze({"samples9/a.jpg": LABEL}, "v1", "2026-10-21", root)))
    return path, store.sha256(path)


def test_parse_date():
    assert store.parse_date("2026-11-20") == Expiry(2026, 11, 20)
    assert store.parse_date(" 2031-04 ") == Expiry(2031, 4)
    with pytest.raises(ValueError):
        store.parse_date("04/2031")


def test_photos_on_disk(root):
    assert store.photos_on_disk(root) == ["samples9/a.jpg"]


def test_to_samples(root):
    [s] = store.to_samples({"samples9/a.jpg": LABEL}, root)
    assert (s.id, s.width, s.height, s.tag) == ("samples9/a.jpg", 40, 20, "clean")
    assert s.expiry == Expiry(2027, 8)
    assert s.boxes[0].label == "expiry" and s.boxes[0].xyxy == (0.1, 0.5, 0.4, 0.55)


def test_read_test_set(root, frozen):
    path, digest = frozen
    test_set = store.read_test_set(path, root, expected=digest)
    assert test_set["version"] == "v1"
    assert test_set["photos"]["samples9/a.jpg"]["tag"] == "clean"


def test_a_changed_label_is_caught(root, frozen):
    path, digest = frozen
    path.write_text(path.read_text().replace('"clean"', '"small-print"'))
    with pytest.raises(RuntimeError, match="has changed since it was frozen"):
        store.read_test_set(path, root, expected=digest)


def test_a_changed_photo_is_caught(root, frozen):
    path, digest = frozen
    Image.new("RGB", (40, 20), "white").save(root / "samples9" / "a.jpg")
    with pytest.raises(RuntimeError, match="1 store photos have changed"):
        store.read_test_set(path, root, expected=digest)


def test_tag_counts():
    labels = {"a": LABEL, "b": {**LABEL, "tag": "other-face", "expiry": None}}
    rows = {r["tag"]: r for r in store.tag_counts(labels)}
    assert rows["clean"]["to the month"] == "1"
    assert rows["other-face"]["with an expiry"] == "0"
    assert rows["**all**"]["photos"] == "2"
