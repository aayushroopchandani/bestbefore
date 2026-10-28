import json
from dataclasses import replace
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
    frozen_set = store.StoreSet("v1", ("samples9",), path.parent, digest)
    assert store.labels(frozen_set) == {"samples9/a.jpg": {**LABEL, "sha256": store.sha256(root / "samples9" / "a.jpg")}}


def test_tag_counts():
    labels = {"a": LABEL, "b": {**LABEL, "tag": "other-face", "expiry": None}}
    rows = {r["tag"]: r for r in store.tag_counts(labels)}
    assert rows["clean"]["to the month"] == "1"
    assert rows["other-face"]["with an expiry"] == "0"
    assert rows["**all**"]["photos"] == "2"


def test_v2_is_kept_apart_from_v1():
    assert set(store.SETS) == {"v1", "v2"}
    assert not set(store.V1.folders) & set(store.V2.folders)
    v1_files = {store.V1.export_path, store.V1.labels_path, store.V1.test_set_path}
    v2_files = {store.V2.export_path, store.V2.labels_path, store.V2.test_set_path, store.V2.photos_path}
    assert not v1_files & v2_files and len(v2_files) == 4
    assert (store.V1.photo_list, store.V1.scanify_check) == (False, True)
    assert (store.V2.photo_list, store.V2.scanify_check) == (True, False)


@pytest.fixture
def listed(root, tmp_path_factory):
    """A made-up set like v2: two photos in testset9/, its list frozen."""
    (root / "testset9").mkdir()
    for name, colour in (("b.jpeg", "red"), ("c.jpg", "blue")):
        Image.new("RGB", (40, 20), colour).save(root / "testset9" / name)
    unfrozen = store.StoreSet("v9", ("testset9",), tmp_path_factory.mktemp("v9"), None, photo_list=True)
    unfrozen.photos_path.write_text(json.dumps(store.freeze_photos(unfrozen, "2026-10-28", root)))
    return replace(unfrozen, photos_sha256=store.sha256(unfrozen.photos_path))


def test_freeze_photos_lists_only_the_sets_folders(root, listed):
    photo_list = store.read_photo_list(listed, root)
    assert (photo_list["version"], photo_list["folders"]) == ("v9", ["testset9"])
    assert sorted(photo_list["photos"]) == ["testset9/b.jpeg", "testset9/c.jpg"]
    assert photo_list["photos"]["testset9/c.jpg"] == store.sha256(root / "testset9" / "c.jpg")


@pytest.mark.parametrize("change, message", [
    (lambda d: Image.new("RGB", (40, 20)).save(d / "d.jpg"), "added: testset9/d.jpg"),
    (lambda d: (d / "b.jpeg").unlink(), "removed: testset9/b.jpeg"),
    (lambda d: Image.new("RGB", (40, 20), "white").save(d / "c.jpg"), "changed: testset9/c.jpg"),
])
def test_the_photo_list_catches_a_photo_added_removed_or_changed(root, listed, change, message):
    change(root / "testset9")
    with pytest.raises(RuntimeError, match=message):
        store.read_photo_list(listed, root)


def test_the_photo_list_catches_a_changed_list_file(root, listed):
    listed.photos_path.write_text(listed.photos_path.read_text().replace("2026-10-28", "2026-10-29"))
    with pytest.raises(RuntimeError, match="has changed since it was frozen"):
        store.read_photo_list(listed, root)


def test_a_photo_list_not_frozen_yet(root, listed):
    with pytest.raises(RuntimeError, match="v9's photo list isn't frozen yet"):
        store.read_photo_list(replace(listed, photos_sha256=None), root)
    with pytest.raises(ValueError, match="has no photo list"):
        store.read_photo_list(OLD, root)


def _frozen_labels(store_set, photos, root, version=None):
    store_set.test_set_path.write_text(json.dumps(
        store.freeze({p: LABEL for p in photos}, version or store_set.version, "2026-11-01", root)))
    return replace(store_set, test_set_sha256=store.sha256(store_set.test_set_path))


def test_read_frozen_checks_the_photo_list(root, listed):
    both = _frozen_labels(listed, ["testset9/b.jpeg", "testset9/c.jpg"], root)
    assert sorted(store.read_frozen(both, root)["photos"]) == ["testset9/b.jpeg", "testset9/c.jpg"]
    one = _frozen_labels(listed, ["testset9/b.jpeg"], root)
    with pytest.raises(RuntimeError, match="labelled photos aren't its frozen photo list"):
        store.read_frozen(one, root)


def test_read_frozen_refuses_another_sets_file(root, listed):
    other = _frozen_labels(listed, ["testset9/b.jpeg", "testset9/c.jpg"], root, version="v8")
    with pytest.raises(RuntimeError, match="holds v8, not v9"):
        store.read_frozen(other, root)


def test_freeze_summary_holds_counts_and_hashes_only(load_script, listed, root):
    freeze = load_script("freeze_store_test_set")
    v2 = replace(store.V2, photos_sha256=listed.photos_sha256)
    text = freeze.summary(v2, ["- **v2 photo list** (2026-10-28): 2 photos."],
                          store.read_photo_list(listed, root))
    assert "- **Photo list:** 2 photos, frozen 2026-10-28\n" in text
    assert f"`data/store/v2/photos.json`:** `{listed.photos_sha256}`" in text
    assert "- **Labels:** not frozen yet" in text
    assert "testset9" not in text and ".jpeg" not in text and ".jpg" not in text
