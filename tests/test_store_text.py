"""text.json, the store dates' values by numbered photo (store_text.py)."""
import json
import stat

from PIL import Image

from bestbefore import store_text
from bestbefore.samples import Box
from bestbefore.store_train import ExportPhoto


def photo(labels, part="train"):
    return ExportPhoto("v1-0001", "samples1/x.jpg", "0" * 64, part, "d01", "a design", "clean",
                       tuple(Box(label, (0.1, 0.1, 0.2, 0.2)) for label in labels))


LABEL = {"expiry": "2027-03-15", "mfg": "2025-03-15", "derived": False}


def test_box_values_by_label_file_line():
    p = photo(["expiry", "mfg", "other_date", "due", "prod", "code"])
    assert store_text.box_values(p, LABEL) == [
        {"index": 0, "label": "expiry", "value": "2027-03-15"},
        {"index": 1, "label": "mfg", "value": "2025-03-15"},
        {"index": 2, "label": "other_date", "value": None}]


def test_an_expiry_box_on_a_derived_photo_gets_no_value():
    p = photo(["mfg", "expiry"])
    assert [b["value"] for b in store_text.box_values(p, {**LABEL, "derived": True})] == ["2025-03-15", None]


def make(labels=("expiry", "mfg", "due"), part="train"):
    manifest = {"photos": {"v1-0001": {"part": part, "labels": list(labels)}}}
    boxes = store_text.box_values(photo(labels, part), LABEL)
    text = store_text.values_file({"v1-0001": {"part": part, "month": "2026-08", "boxes": boxes}},
                                  "abc", {}, "2026-11-13", {})
    return text, manifest


def test_check_passes_its_own_file():
    text, manifest = make()
    assert store_text.check(text, manifest, "abc") == []


def test_check_finds_problems():
    text, manifest = make()
    assert store_text.check(text, manifest, "xyz")                       # another export
    bad = json.loads(json.dumps(text))
    bad["photos"]["v1-0001"]["boxes"][0]["label"] = "mfg"
    assert any("line 0" in p for p in store_text.check(bad, manifest, "abc"))
    bad = json.loads(json.dumps(text))
    bad["photos"]["v1-0001"]["boxes"][1]["value"] = "2025-13"
    assert any("isn't a date" in p for p in store_text.check(bad, manifest, "abc"))
    _, other = make(labels=("due", "expiry", "mfg"))
    assert store_text.check(text, other, "abc")                          # lines moved
    assert store_text.check(text, {"photos": {}}, "abc")                 # not in the export


def test_photo_month(tmp_path):
    with_date, without = tmp_path / "a.jpg", tmp_path / "b.jpg"
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9003] = "2026:08:14 10:21:05"
    Image.new("RGB", (8, 8)).save(with_date, exif=exif)
    Image.new("RGB", (8, 8)).save(without)
    assert store_text.photo_month(with_date) == "2026-08"
    assert store_text.photo_month(without) is None


def test_written_for_its_owner_only(tmp_path):
    path = tmp_path / "text.json"
    digest = store_text.write_private(path, {"a": 1})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(digest) == 64 and json.loads(path.read_text()) == {"a": 1}
