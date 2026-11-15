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


# --- S4: turns, the frozen file and its numbered copy ------------------------------------

import numpy as np
import pytest

from bestbefore.reader import crops

START, TOPS = 255, 128      # the line's first character; the tops of its characters


def _upright() -> np.ndarray:
    """A made-up line of print, 20 x 60: its first character at the left,
    its characters' tops along the top edge."""
    a = np.zeros((20, 60), np.uint8)
    a[:, :6] = START
    a[:3, :] = TOPS
    return a


def _as_printed(choice: str) -> np.ndarray:
    """The same line as a photo shows it, drawn by hand from the words of
    each choice, not by turning: 'reads bottom to top' starts at the bottom
    with its tops to the left (tilt your head left to read it)."""
    if choice == "upright":
        return _upright()
    if choice == "reads bottom to top":
        a = np.zeros((60, 20), np.uint8)
        a[-6:, :] = START
        a[:, :3] = TOPS
    elif choice == "reads top to bottom":
        a = np.zeros((60, 20), np.uint8)
        a[:6, :] = START
        a[:, -3:] = TOPS
    else:                                   # upside down: starts at the right, tops at the bottom
        a = np.zeros((20, 60), np.uint8)
        a[:, -6:] = START
        a[-3:, :] = TOPS
    return a


@pytest.mark.parametrize("choice", list(store_text.TURNS))
def test_each_choice_maps_to_the_quarter_turns_that_stand_it_upright(choice):
    k = store_text.TURNS[choice]
    printed = _as_printed(choice)
    assert np.array_equal(np.rot90(printed, k), _upright())                     # np.rot90's k
    im = Image.fromarray(printed)
    turned = im.transpose(crops._TRANSPOSE[k]) if k else im                     # PIL's ROTATE_90 ...
    assert np.array_equal(np.asarray(turned), _upright())
    cut, box = crops.cut(im, (0, 0, im.width, im.height), turn=k, margin=0)     # what a crop record holds
    assert np.array_equal(np.asarray(cut), _upright()) and box == (0, 0, 60, 20)


def test_no_choice_is_backwards():
    """The two sideways choices are each other's inverse, never the same turn."""
    assert sorted(store_text.TURNS.values()) == [0, 1, 2, 3]
    assert (store_text.TURNS["reads bottom to top"] + store_text.TURNS["reads top to bottom"]) % 4 == 0


TEXT = {"version": "text", "reviewed": "2026-11-15", "sources": {"v1": {"labels": "x"}}, "review": {},
        "photos": {"samples1/x.jpg": {
            "name": "v1-0001", "part": "train", "month": "2026-08",
            "boxes": [{"index": 0, "label": "expiry", "xyxy": [0.1, 0.1, 0.2, 0.2], "value": "2027-03-15",
                       "text": "15/03/2027", "turn": 0},
                      {"index": 1, "label": "mfg", "xyxy": [0.1, 0.1, 0.2, 0.2], "value": "2025-03-15",
                       "text": "MFG 15/03/2025", "turn": 3}],
            "life": [{"xyxy": [0.3, 0.3, 0.6, 0.35], "text": "24 months from Mfg.", "turn": 0}]}}}


def test_numbered_file_names_no_photo_and_passes_the_check():
    numbered = store_text.numbered_file(TEXT, "f" * 64, "abc", "2026-11-15", {})
    assert "samples1" not in json.dumps(numbered) and numbered["text_sha256"] == "f" * 64
    entry = numbered["photos"]["v1-0001"]
    assert entry["boxes"][1] == {"index": 1, "label": "mfg", "value": "2025-03-15", "text": "MFG 15/03/2025", "turn": 3}
    assert entry["life"] == [{"xyxy": [0.3, 0.3, 0.6, 0.35], "text": "24 months from Mfg.", "turn": 0}]
    manifest = {"photos": {"v1-0001": {"part": "train", "labels": ["expiry", "mfg", "due"]}}}
    assert store_text.check(numbered, manifest, "abc") == []


def test_check_finds_a_box_without_its_text_or_turn():
    numbered = store_text.numbered_file(TEXT, "f" * 64, "abc", "2026-11-15", {})
    manifest = {"photos": {"v1-0001": {"part": "train", "labels": ["expiry", "mfg"]}}}
    bad = json.loads(json.dumps(numbered))
    bad["photos"]["v1-0001"]["boxes"][0]["text"] = " "
    bad["photos"]["v1-0001"]["boxes"][1]["turn"] = 4
    bad["photos"]["v1-0001"]["life"][0]["xyxy"] = [0.5, 0.3, 0.4, 0.35]
    found = store_text.check(bad, manifest, "abc")
    assert len(found) == 3 and any("no text" in p for p in found) and any("turn is 4" in p for p in found)


def test_read_text_refuses_a_changed_file(tmp_path):
    path = tmp_path / "text.json"
    digest = store_text.write_private(path, TEXT)
    assert store_text.read_text(path, digest) == TEXT
    path.write_text(path.read_text().replace("15/03/2027", "16/03/2027"))
    with pytest.raises(RuntimeError, match="changed"):
        store_text.read_text(path, digest)
