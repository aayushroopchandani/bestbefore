import json
from pathlib import Path

import pytest
from PIL import Image

from bestbefore import store
from bestbefore.samples import Expiry

LABEL = {"expiry": "2027-08", "derived": False, "mfg": None, "format": "MM/YYYY",
         "tag": "clean", "unreadable": False, "note": "",
         "boxes": [{"label": "expiry", "xyxy": [0.1, 0.5, 0.4, 0.55]}]}


# Two made-up sets side by side in one photo folder, as v1 and v2 are.
OLD = store.StoreSet("v8", ("samples9",), Path("unused"), None)
NEW = store.StoreSet("v9", ("testset9",), Path("unused"), None)


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
    assert store.photos_on_disk(OLD, root) == ["samples9/a.jpg"]


def test_each_set_lists_only_its_own_photos(root):
    (root / "testset9").mkdir()
    Image.new("RGB", (40, 20)).save(root / "testset9" / "b.jpeg")
    Image.new("RGB", (40, 20)).save(root / "testset9" / "c.jpg")
    assert store.photos_on_disk(OLD, root) == ["samples9/a.jpg"]
    assert store.photos_on_disk(NEW, root) == ["testset9/b.jpeg", "testset9/c.jpg"]


def test_a_missing_folder_is_an_error(root):
    with pytest.raises(FileNotFoundError, match="v9: no folder testset9"):
        store.photos_on_disk(NEW, root)


def test_v1_is_as_it_was_frozen():
    """v1's folders, files and hash, exactly as before there was a v2."""
    assert store.V1.folders == ("samples1", "samples2", "samples3")
    assert store.V1.labels_path == store.LABELS_PATH == store.DATA_DIR / "labels.json"
    assert store.V1.test_set_path == store.TEST_SET_PATH == store.DATA_DIR / "test_set.json"
    assert store.V1.export_path == store.DATA_DIR / "label_studio_export.json"
    assert store.V1.test_set_sha256 == store.TEST_SET_SHA256 == (
        "7b9a4043886c9f4fcb37cba339962261acaef19e2993f7fba365a75f49b588f4")


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


def test_labels_of_a_set_not_frozen_yet(tmp_path, capsys):
    unfrozen = store.StoreSet("v9", ("testset9",), tmp_path, None)
    unfrozen.labels_path.write_text(json.dumps({"testset9/b.jpeg": LABEL}))
    assert store.labels(unfrozen) == {"testset9/b.jpeg": LABEL}
    assert "v9 isn't frozen yet" in capsys.readouterr().err


def test_labels_of_a_frozen_set(root, frozen, monkeypatch):
    monkeypatch.setenv("STORE_PHOTOS_DIR", str(root))
    path, digest = frozen
    frozen_set = store.StoreSet("v8", ("samples9",), path.parent, digest)
    assert store.labels(frozen_set) == {"samples9/a.jpg": {**LABEL, "sha256": store.sha256(root / "samples9" / "a.jpg")}}


def test_tag_counts():
    labels = {"a": LABEL, "b": {**LABEL, "tag": "other-face", "expiry": None}}
    rows = {r["tag"]: r for r in store.tag_counts(labels)}
    assert rows["clean"]["to the month"] == "1"
    assert rows["other-face"]["with an expiry"] == "0"
    assert rows["**all**"]["photos"] == "2"
