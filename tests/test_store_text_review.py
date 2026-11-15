"""S4's Label Studio review of the store text (store_text_review.py)."""
import copy

import pytest

from bestbefore import store_text, store_text_review as review
from bestbefore.samples import Box
from bestbefore.store_train import ExportPhoto

PHOTO = "samples1/a b.jpg"
SIZE = (1200, 1600)
BOXES = [{"index": 0, "label": "expiry", "xyxy": [0.1, 0.2, 0.4, 0.25]},
         {"index": 1, "label": "mfg", "xyxy": [0.6, 0.3, 0.63, 0.5]}]
PROPOSALS = {0: {"text": "15/03/2O27", "turn": 0}, 1: {"text": "15/03/2025", "turn": 3}}


def made_task(derived=False):
    return review.task(PHOTO, SIZE, BOXES, PROPOSALS, derived)


def submit(t, result=None):
    """The task as Label Studio's full JSON export gives it once submitted
    unchanged (or with `result`)."""
    t = copy.deepcopy(t)
    t["annotations"] = [{"result": result if result is not None else t["predictions"][0]["result"],
                         "was_cancelled": False, "updated_at": "2026-11-15T10:00:00Z"}]
    return t


def by(result, rid, kind):
    return next(r for r in result if r["id"] == rid and r["type"] == kind)


def test_a_task_carries_each_box_with_its_text_and_turn_under_one_region_id():
    t = made_task(derived=True)
    assert t["data"]["image"] == "/data/local-files/?d=samples1/a%20b.jpg"
    assert "shelf life" in t["data"]["note"] and "2 date boxes" in t["data"]["note"]
    result = t["predictions"][0]["result"]
    assert {r["id"] for r in result} == {"date0", "date1"} and len(result) == 6
    assert by(result, "date1", "rectanglelabels")["value"]["rectanglelabels"] == ["mfg"]
    assert by(result, "date0", "textarea")["value"]["text"] == ["15/03/2O27"]
    assert by(result, "date1", "choices")["value"]["choices"] == ["reads bottom to top"]


def test_the_review_reads_back_texts_turns_and_life_boxes():
    t = made_task()
    result = copy.deepcopy(t["predictions"][0]["result"])
    by(result, "date0", "textarea")["value"]["text"] = ["  15/03/2027 "]       # the owner's fix
    by(result, "date1", "choices")["value"]["choices"] = ["reads top to bottom"]
    life = review._regions("L1", review.LIFE_CONTROL, "life", [0.2, 0.6, 0.5, 0.63], SIZE,
                           "24 months from Mfg.", 0)
    reviewed, problems = review.read_review([submit(t, result + life)], {PHOTO: BOXES})
    assert problems == []
    got = reviewed[PHOTO]
    assert got["boxes"][0]["text"] == "15/03/2027" and got["boxes"][0]["turn"] == 0
    assert got["boxes"][1]["turn"] == store_text.TURNS["reads top to bottom"] == 1
    assert got["life"] == [{"xyxy": [0.2, 0.6, 0.5, 0.63], "text": "24 months from Mfg.", "turn": 0, "id": "L1"}]


def test_the_review_finds_moved_lost_added_and_unfinished_boxes():
    t = made_task()
    result = copy.deepcopy(t["predictions"][0]["result"])
    for r in result:
        if r["id"] == "date0":
            r["value"]["x"] += 1                                  # moved
    result = [r for r in result if not (r["id"] == "date1" and r["type"] == "textarea")]   # no text
    result += review._regions("N1", review.DATE_CONTROL, "other_date", [0.7, 0.7, 0.8, 0.72], SIZE, "x", 0)
    result += review._regions("L1", review.LIFE_CONTROL, "life", [0.2, 0.6, 0.5, 0.63], SIZE, "12 months", None)
    _, problems = review.read_review([submit(t, result)], {PHOTO: BOXES})
    text = "\n".join(problems)
    assert "region date0: a expiry box that isn't one" in text            # moved
    assert "region N1: a other_date box" in text                           # added
    assert "date lines [0] lost" in text
    assert "region date1 (mfg): 0 texts" in text
    assert "region L1 (life): turn []" in text


@pytest.mark.parametrize("text, value, reason", [
    ("15/03/2027", "2027-03-15", None),
    ("EXP 15/03/2027", "2027-03-15", None),
    ("MAR 2027", "2027-03", None),
    ("15.03.27", "2027-03-15", "S2's known trap"),
    ("15/03/2027", "2027-03", "a day in the text"),
    ("MAR 2027", "2027-03-15", "no day in the text"),
    ("03/15/2027", "2027-03-15", None),            # the parser reads month first when the day can't be a month
    ("04/03/2027", "2027-04-03", "day and month swapped"),
    ("Best before", "2027-03-15", "finds no date"),
    ("15/03/2021", "2027-03-15", "another date"),
    ("03/2025-03/2027", "2027-03", "another of its dates"),
])
def test_value_reason(text, value, reason):
    got = review.value_reason(text, value)
    assert got is None if reason is None else reason in got


def test_life_reason():
    assert review.life_reason(["24 months from Mfg."], "2025-01-10", "2027-01-10") is None
    assert review.life_reason(["Best Before 1 Year from Mfg. Dt."], "2024-12", "2025-12") is None
    assert "the frozen expiry is" in review.life_reason(["18 months from mfg"], "2025-01", "2027-01")
    assert "reads no shelf life" in review.life_reason(["best before one year"], "2025-01", "2026-01")
    assert "no life box" in review.life_reason([], "2025-01", "2026-01")


def test_mismatches_and_what_the_owner_settled():
    texts = {PHOTO: {"boxes": {0: {"text": "15.03.27", "turn": 0}, 1: {"text": "15/03/2025", "turn": 3}},
                     "life": []}}
    values = {PHOTO: [{"index": 0, "label": "expiry", "value": "2027-03-15"},
                      {"index": 1, "label": "mfg", "value": "2025-03-15"}]}
    labels = {PHOTO: {"derived": False, "mfg": "2025-03-15", "expiry": "2027-03-15"}}
    found = review.mismatches(texts, values, labels)
    assert [m["key"] for m in found] == [f"{PHOTO}#0"]
    left, stale = review.unsettled(found, {f"{PHOTO}#0": {"reason": "day first"}, "old#1": {}})
    assert left == [] and stale == ["old#1"]
    derived = {PHOTO: {**labels[PHOTO], "derived": True}}
    assert any(m["key"] == f"{PHOTO}#life" for m in review.mismatches(texts, values, derived))


def test_notes_flag_a_turn_that_does_not_fit_its_box():
    texts = {PHOTO: {"boxes": {0: {"text": "15/03/2027", "turn": 1}, 1: {"text": "15/03/2025", "turn": 3}},
                     "life": [{"text": "24 months from Mfg. ±", "turn": 0}]}}
    found = review.notes(texts, {PHOTO: BOXES}, {PHOTO: PROPOSALS})
    assert len(found["turn and shape differ (the photo's aspect aside)"]) == 1      # the wide box read sideways
    assert len(found["turn other than OCR's"]) == 1
    assert len(found["characters outside decision 10's set"]) == 1


def test_ocr_help_counts_kept_and_changed_texts_and_turns():
    texts = {PHOTO: {"boxes": {0: {"text": "15/03/2027", "turn": 0}, 1: {"text": "15/03/2025", "turn": 1}},
                     "life": []}}
    got = review.ocr_help(texts, {PHOTO: BOXES}, {PHOTO: PROPOSALS})
    assert got == {("expiry", "text changed"): 1, ("expiry", "turn kept"): 1,
                   ("mfg", "text kept"): 1, ("mfg", "turn changed"): 1}


def test_text_file_from_a_review():
    p = ExportPhoto("v1-0001", PHOTO, "0" * 64, "train", "d01", "a design", "clean",
                    tuple(Box(b["label"], tuple(b["xyxy"])) for b in BOXES) + (Box("due", (0, 0, 0.1, 0.1)),))
    reviewed = {PHOTO: {"boxes": {0: {"text": "15/03/2027", "turn": 0}, 1: {"text": "15/03/2025", "turn": 3}},
                        "life": []}}
    values = {PHOTO: store_text.box_values(p, {"expiry": "2027-03-15", "mfg": "2025-03-15", "derived": False})}
    out = review.text_file([p], reviewed, values, {PHOTO: "2026-08"}, {"v1": {}}, {"export_sha256": "e"},
                           "2026-11-15")
    entry = out["photos"][PHOTO]
    assert entry["name"] == "v1-0001" and entry["life"] == []
    assert entry["boxes"][1] == {"index": 1, "label": "mfg", "xyxy": [0.6, 0.3, 0.63, 0.5], "value": "2025-03-15",
                                 "text": "15/03/2025", "turn": 3}
    numbered = store_text.numbered_file(out, "f" * 64, "abc", "2026-11-15", {})
    manifest = {"photos": {"v1-0001": {"part": "train", "labels": ["expiry", "mfg", "due"]}}}
    assert store_text.check(numbered, manifest, "abc") == []
