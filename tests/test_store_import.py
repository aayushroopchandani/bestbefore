import pytest

from bestbefore import store_import as imp
from bestbefore.samples import Expiry

# A made-up export in Label Studio's JSON-MIN shape: no store data here.
EXPORT = [
    {"image": "/data/local-files/?d=samples9/pack%20a.jpg", "id": 1, "tag": "clean",
     "expiry": "2027-08-15", "mfg": "2025-08", "format": "DD/MM/YYYY", "note": "",
     "box": [{"x": 10.0, "y": 50.0, "width": 30.0, "height": 2.5, "rotation": 0,
              "rectanglelabels": ["expiry"], "original_width": 400, "original_height": 800},
             {"x": 90.0, "y": 60.0, "width": 15.0, "height": 2.0, "rotation": 0,
              "rectanglelabels": ["mfg"], "original_width": 400, "original_height": 800}]},
    {"image": "/data/local-files/?d=samples9/pack%20b.jpg", "id": 2, "tag": "other-face"},
]


@pytest.fixture
def labels():
    return imp.from_export(EXPORT)


def test_from_export(labels):
    a = labels["samples9/pack a.jpg"]
    assert (a["expiry"], a["mfg"], a["tag"], a["derived"]) == ("2027-08-15", "2025-08", "clean", False)
    assert a["boxes"][0] == {"label": "expiry", "xyxy": [0.1, 0.5, 0.4, 0.525]}
    assert a["boxes"][1]["xyxy"][2] == 1.0                  # clipped to the edge
    b = labels["samples9/pack b.jpg"]
    assert (b["expiry"], b["boxes"], b["unreadable"]) == (None, [], False)


def test_drawn_sizes():
    assert imp.drawn_sizes(EXPORT) == {"samples9/pack a.jpg": (400, 800)}


def test_a_clean_export_passes(labels):
    upright = {"samples9/pack a.jpg": (400, 800), "samples9/pack b.jpg": (600, 400)}
    assert imp.check_export(EXPORT) == []
    assert imp.check_photos(list(labels), list(upright)) == []
    assert imp.check_dates(labels) == []
    assert imp.check_consistency(labels) == []
    assert imp.check_orientation(imp.drawn_sizes(EXPORT), upright) == []


def test_check_export_catches_rotated_and_unknown_boxes():
    task = {**EXPORT[0], "tag": "blurry",
            "box": [{**EXPORT[0]["box"][0], "rotation": 90, "rectanglelabels": ["date"]}]}
    problems = imp.check_export([task])
    assert len(problems) == 3       # the tag, the label and the rotation


def test_check_photos():
    problems = imp.check_photos(["a.jpg", "a.jpg", "x.jpg"], ["a.jpg", "b.jpg"])
    assert problems == ["a.jpg: labelled 2 times", "b.jpg: not labelled",
                        "x.jpg: labelled, but not in $STORE_PHOTOS_DIR"]


@pytest.mark.parametrize("expiry, mfg, ok", [
    ("2027-08-15", "2025-08", True),
    ("2027-08", "2027-08-01", True),        # same month, no day on one side
    ("2027-13", None, False),               # no such month
    ("2027-02-30", None, False),            # no such day
    ("27-08-15", None, False),              # not YYYY-MM-DD
    ("2045-01", None, False),               # implausible year
    ("2026-01", "2026-03-01", False),       # mfg after expiry
])
def test_check_dates(expiry, mfg, ok):
    labels = {"p.jpg": {"expiry": expiry, "mfg": mfg}}
    assert (imp.check_dates(labels) == []) is ok


def _label(**fields):
    return {"expiry": None, "derived": False, "mfg": None, "tag": "clean",
            "unreadable": False, "boxes": [], **fields}


def _boxes(*kinds):
    return [{"label": k, "xyxy": [0.1, 0.1, 0.3, 0.12]} for k in kinds]


@pytest.mark.parametrize("label, ok", [
    (_label(expiry="2027-08", boxes=_boxes("expiry")), True),
    (_label(expiry="2027-08"), False),                                      # no box
    (_label(tag="other-face", boxes=_boxes("mfg")), True),
    (_label(tag="other-face", boxes=_boxes("expiry")), False),
    (_label(), False),                                                      # nothing typed
    (_label(unreadable=True), True),
    (_label(derived=True, expiry="2027-08", mfg="2025-08", boxes=_boxes("mfg")), True),
    (_label(derived=True, expiry="2027-08", mfg="2025-08", boxes=_boxes("mfg", "expiry")), False),
    (_label(derived=True, expiry="2027-08", boxes=_boxes("mfg")), False),   # no mfg date
])
def test_check_consistency(label, ok):
    assert (imp.check_consistency({"p.jpg": label}) == []) is ok


def test_read_scanify_truth(tmp_path):
    path = tmp_path / "ground_truth.py"
    path.write_text('X = 1\nEXPIRY_GROUND_TRUTH = {\n    "sample 1.jpg": "30/04/31",\n'
                    '    "sample 2.jpg": None,\n}\n')
    assert imp.read_scanify_truth(path) == {"samples2/sample 1.jpg": Expiry(2031, 4, 30),
                                            "samples2/sample 2.jpg": None}


@pytest.mark.parametrize("ours, theirs, same", [
    (Expiry(2031, 4), Expiry(2031, 4, 30), True),       # Scanify's last-day rule
    (Expiry(2031, 4, 12), Expiry(2031, 4, 30), False),
    (Expiry(2031, 4), Expiry(2021, 4, 30), False),
    (None, None, True),
    (Expiry(2031, 4), None, False),
])
def test_same_expiry(ours, theirs, same):
    assert imp.same_expiry(ours, theirs) is same


def test_check_scanify():
    labels = {"samples2/sample 1.jpg": _label(expiry="2031-04")}
    truth = {"samples2/sample 1.jpg": Expiry(2031, 4, 30), "samples2/sample 2.jpg": None}
    assert imp.check_scanify(labels, truth) == [
        "samples2/sample 2.jpg: in Scanify's ground truth, but not labelled"]


def test_check_orientation_spots_the_raw_frame():
    problems = imp.check_orientation({"p.jpg": (800, 400)}, {"p.jpg": (400, 800)})
    assert problems == ["p.jpg: drawn on 800x400, upright is 400x800 (swapped: drawn on the raw frame)"]


def test_apply_print_rule():
    # A 1000 px tall photo: a box 0.01 tall is 10 px, 1% of the long side;
    # one 0.02 tall is 2%. The rule's line is 1.5%.
    def boxed(height, kind="expiry"):
        boxes = _boxes(kind)
        boxes[0]["xyxy"] = [0.1, 0.5, 0.3, 0.5 + height]
        return boxes
    labels = {
        "a.jpg": _label(expiry="2027-08", boxes=boxed(0.01)),                    # -> small-print
        "b.jpg": _label(tag="small-print", expiry="2027-08", boxes=boxed(0.02)), # -> clean
        "c.jpg": _label(tag="small-print", expiry="2027-08", boxes=boxed(0.01)), # stays
        "d.jpg": _label(tag="dot-matrix", expiry="2027-08", boxes=boxed(0.02)),  # earlier tag: stays
        "e.jpg": _label(derived=True, expiry="2027-08", mfg="2025-08",
                        boxes=boxed(0.01, "mfg")),                               # mfg box -> small-print
    }
    upright = {p: (600, 1000) for p in labels}
    out, moved = imp.apply_print_rule(labels, upright)
    assert {p: out[p]["tag"] for p in out} == {
        "a.jpg": "small-print", "b.jpg": "clean", "c.jpg": "small-print",
        "d.jpg": "dot-matrix", "e.jpg": "small-print"}
    assert moved == {"clean -> small-print": 2, "small-print -> clean": 1}
    assert labels["a.jpg"]["tag"] == "clean"        # the input is left alone
