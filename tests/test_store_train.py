import inspect
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bestbefore import derived, store, store_train


def label(tag: str, expiries: int = 1) -> dict:
    return {"expiry": "2027-08" if expiries else None, "tag": tag,
            "boxes": [{"label": "expiry", "xyxy": [0.1, 0.5, 0.4, 0.55]}] * expiries
                     + [{"label": "mfg", "xyxy": [0.1, 0.6, 0.4, 0.65]}]}


# Made-up v1: 11 designs, 25 photos in v1's folders; and 3 v2 photos.
V1_TAGS = {"pack01": ["clean", "clean"], "pack02": ["clean", "small-print", "clean"],
           "pack03": ["clean", "clean"], "pack04": ["clean", "clean", "clean"], "pack05": ["clean", "other-face"],
           "pack06": ["dot-matrix", "dot-matrix"], "pack07": ["dot-matrix", "clean", "dot-matrix"],
           "pack08": ["sideways", "sideways"], "pack09": ["sideways", "other-face"],
           "pack10": ["other-face", "other-face"], "pack11": ["small-print", "small-print"]}
DESIGNS = {f"samples{1 + k % 3}/{d}-{k}.jpg": d
           for d, tags in V1_TAGS.items() for k in range(len(tags))}
LABELS = {p: label(V1_TAGS[d][int(p.split("-")[1][0])]) for p, d in DESIGNS.items()}
V2 = {"testset2/x-0.jpeg": "pack01", "testset2/y-0.jpeg": "pack06", "testset2/z-0.jpg": "pack99"}


def rows(designs: dict[str, str]) -> list[list[str]]:
    return [[p, d] for p, d in designs.items()]


def test_the_made_up_set_is_what_it_says():
    assert len(set(DESIGNS.values())) == 11 and len(DESIGNS) == 25
    assert all(p.split("/")[0] in store.V1.folders for p in DESIGNS)


def test_a_complete_file_passes():
    assert store_train.check_designs(rows({**DESIGNS, **V2}), sorted({**DESIGNS, **V2})) == []


def test_a_photo_missing_or_listed_twice_fails_the_freeze():
    photos = sorted({**DESIGNS, **V2})
    dropped, *missing = rows({**DESIGNS, **V2})
    assert store_train.check_designs(missing, photos) == [f"not listed: {dropped[0]}"]
    twice = rows({**DESIGNS, **V2}) + [["testset2/x-0.jpeg", "pack01"]]
    assert store_train.check_designs(twice, photos) == ["listed 2 times: testset2/x-0.jpeg"]
    extra = rows({**DESIGNS, **V2}) + [["samples1/elsewhere.jpg", "pack01"]]
    assert store_train.check_designs(extra, photos) == [
        f"line {len(extra) + 1}: not a photo of v1 or v2: samples1/elsewhere.jpg"]


def test_an_empty_id_or_a_stray_comma_fails():
    photos = sorted({**DESIGNS, **V2})
    bad = rows({**DESIGNS, **V2})
    bad[0][1], bad[1] = "", bad[1] + ["more"]
    assert store_train.check_designs(bad, photos) == [
        f"line 2: no design for {bad[0][0]}", "line 3: 3 fields, not 2 (a comma in a design?)",
        f"not listed: {bad[1][0]}"]


def test_one_design_typed_two_ways_fails():
    typed = rows({**DESIGNS, **V2})
    typed[-1][1] = " Pack 01"
    problems = store_train.check_designs([[p, d.strip()] for p, d in typed], sorted({**DESIGNS, **V2}))
    assert problems == ["one design typed 2 ways? Pack 01 / pack01"]


def test_the_csv_is_read_as_typed(tmp_path):
    path = tmp_path / "designs.csv"
    path.write_text("﻿photo,design\r\nsamples1/a.jpg , Pouch A\r\n\r\ntestset2/b.jpg,\r\n",
                    encoding="utf-8")
    assert store_train.read_designs_csv(path) == [["samples1/a.jpg", "Pouch A"], ["testset2/b.jpg", ""]]
    path.write_text("photo;design\n")
    with pytest.raises(ValueError, match="header"):
        store_train.read_designs_csv(path)


def test_the_frozen_file_is_checked(tmp_path):
    path = tmp_path / "designs.json"
    path.write_text(json.dumps(store_train.freeze(rows(DESIGNS), {"v1": "a", "v2": "b"}, "2026-11-03")))
    digest = store.sha256(path)
    assert store_train.read_designs(path, digest) == dict(sorted(DESIGNS.items()))
    path.write_text(path.read_text().replace("pack01", "pack12"))
    with pytest.raises(RuntimeError, match="changed"):
        store_train.read_designs(path, digest)


def test_a_design_takes_its_commonest_tag():
    assert store_train.design_tag(["clean", "small-print", "clean"]) == "clean"
    assert store_train.design_tag(["other-face", "other-face", "clean"]) == "clean"
    assert store_train.design_tag(["other-face", "other-face"]) == "other-face"
    assert store_train.design_tag(["clean", "sideways"]) == "sideways"     # a tie: store.TAGS' order


def test_validation_takes_30_percent_of_a_tags_designs_rounded_half_up():
    assert [store_train.val_count(n) for n in (0, 1, 2, 3, 4, 5, 10, 15, 35)] == [0, 0, 1, 1, 1, 2, 3, 5, 11]


def test_no_design_is_on_both_sides():
    split = store_train.split_v1(DESIGNS, LABELS)
    assert set(split) == set(LABELS)
    for d in set(DESIGNS.values()):
        assert len({split[p] for p, e in DESIGNS.items() if e == d}) == 1


def test_the_split_is_the_same_for_the_same_input():
    split = store_train.split_v1(DESIGNS, LABELS)
    backwards = dict(reversed(list(LABELS.items())))
    assert store_train.split_v1(dict(reversed(list(DESIGNS.items()))), backwards) == split
    assert store_train.split_digest(store_train.split_v1(DESIGNS, LABELS, seed=0)) == \
        store_train.split_digest(split)
    others = {store_train.split_digest(store_train.split_v1(DESIGNS, LABELS, seed=s)) for s in range(1, 6)}
    assert others - {store_train.split_digest(split)}     # the seed does choose


def test_each_tag_with_two_or_more_designs_has_one_in_validation():
    split = store_train.split_v1(DESIGNS, LABELS)
    by_tag: dict[str, set[str]] = {}
    for d in set(DESIGNS.values()):
        tags = [LABELS[p]["tag"] for p, e in DESIGNS.items() if e == d]
        by_tag.setdefault(store_train.design_tag(tags), set()).add(d)
    val = {DESIGNS[p] for p, part in split.items() if part == "val"}
    # clean: pack01 to pack05 (5 designs, 2 to validation); dot-matrix 2; sideways 2; other-face 1;
    # small-print 1
    assert {t: len(ds) for t, ds in by_tag.items()} == {
        "clean": 5, "dot-matrix": 2, "sideways": 2, "other-face": 1, "small-print": 1}
    assert {t: len(ds & val) for t, ds in by_tag.items()} == {
        "clean": 2, "dot-matrix": 1, "sideways": 1, "other-face": 0, "small-print": 0}


def test_a_v2_photo_cant_enter_the_split():
    with pytest.raises(ValueError, match="only store set v1 is split"):
        store_train.split_v1({**DESIGNS, **V2}, {**LABELS, "testset2/x-0.jpeg": label("clean")})
    with pytest.raises(ValueError, match="have no design"):
        store_train.split_v1(V2, LABELS)


def test_v1s_split_is_checked_against_its_hash(monkeypatch):
    monkeypatch.setattr(store_train, "read_designs", lambda: DESIGNS)
    monkeypatch.setattr(store, "labels", lambda store_set: LABELS)
    digest = store_train.split_digest(store_train.split_v1(DESIGNS, LABELS))
    monkeypatch.setattr(store_train, "SPLIT_SHA256", digest)
    assert store_train.split_digest(store_train.v1_split()) == digest
    monkeypatch.setattr(store_train, "SPLIT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="differs"):
        store_train.v1_split()


def test_the_freeze_writes_once_and_only_a_complete_file(load_script, tmp_path, monkeypatch, capsys):
    script = load_script("store_designs")
    sets = {"v1": LABELS, "v2": {p: label("clean") for p in V2}}
    monkeypatch.setattr(script, "frozen_sets", lambda: sets)
    monkeypatch.setattr(store_train, "DESIGNS_CSV", tmp_path / "designs.csv")
    monkeypatch.setattr(store_train, "DESIGNS_PATH", tmp_path / "designs.json")
    lines = [[p, d] for p, d in {**DESIGNS, **V2}.items()]
    (tmp_path / "designs.csv").write_text("photo,design\n" + "".join(f"{p},{d}\n" for p, d in lines[1:]))
    with pytest.raises(SystemExit, match="was not written"):
        script.freeze("2026-11-03")
    assert not (tmp_path / "designs.json").exists()
    (tmp_path / "designs.csv").write_text("photo,design\n" + "".join(f"{p},{d}\n" for p, d in lines))
    script.freeze("2026-11-03")
    digest = store.sha256(tmp_path / "designs.json")
    assert f'DESIGNS_SHA256 = "{digest}"' in capsys.readouterr().out
    assert store_train.read_designs(expected=digest) == dict(sorted({**DESIGNS, **V2}.items()))
    with pytest.raises(SystemExit, match="already frozen"):
        script.freeze("2026-11-04")


def test_the_summary_names_no_photo_or_design(load_script, monkeypatch):
    script = load_script("store_designs")
    monkeypatch.setattr(store, "DESIGNS_SHA256", "f" * 64)
    designs = {**DESIGNS, **V2}
    sets = {"v1": LABELS, "v2": {p: label("clean") for p in V2}}
    split = store_train.split_v1(designs, LABELS)
    text = script.summary(designs, sets, split, "2026-11-03", store_train.split_digest(split))
    names = {*designs, *designs.values(), *(p.split("/")[1] for p in designs)}
    assert not [n for n in names if n in text]
    # v2: pack01 and pack06 are in v1, pack99 isn't; every made-up photo has one expiry box
    assert "| v2, design not in v1 (unseen) | 1 | 1 | 1 |" in text
    assert "| v1 | 11 | 25 | 25 |" in text


# --- Keyword boxes (S5) ---------------------------------------------------------

def keywords(text: str) -> list[tuple[str, str]]:
    return [(label, text[a:b]) for label, a, b in store_train.find_keywords(text)]


def test_a_keyword_glued_to_its_date_is_boxed_alone():
    assert keywords("EXP.FEB 2027") == [("due", "EXP.")]
    # a 12-character line 120 px long: EXP. is its first 40 px, the date stays out
    assert store_train.span_box([(0, 0), (120, 0), (120, 10), (0, 10)], 0, 4, 12) == (0, 0, 40, 10)


@pytest.mark.parametrize("line, found", [
    ("Best before:03-03-2027", [("due", "Best before:")]),
    ("Expiry Date : FEB-2029", [("due", "Expiry Date :")]),
    ("USE BY 12/2026", [("due", "USE BY")]),
    ("MFD(Date): April 2026", [("prod", "MFD(Date):")]),
    ("Mfg. Date: 05/12/2024", [("prod", "Mfg. Date:")]),
    ("Month and year of manufacture: 19/06/2023", [("prod", "Month and year of manufacture:")]),
    ("MFG:FEB 2025 | IMPORT: MAR 2025", [("prod", "MFG:"), ("prod", "IMPORT:")]),
    ("MFG28/11/25;EXP22/05/27", [("prod", "MFG"), ("due", "EXP")]),
    ("Batch No.: K2311A", [("code", "Batch No.: K2311A")]),
    ("B.NO: 2307B", [("code", "B.NO: 2307B")]),
    ("LOT A12B3 14:32", [("code", "LOT A12B3"), ("code", "14:32")]),
    ("Lot/Batch No - as printed on bag", [("code", "Lot/Batch No -")]),
    ("Best Before 12 months from Mfg.", [("due", "Best Before")]),
    ("BEST BEFORE 24 MONTHS FROM DATE OF MANUFACTURE", [("due", "BEST BEFORE")]),
])
def test_keywords_are_boxed_as_printed_with_their_colon(line, found):
    assert keywords(line) == found


@pytest.mark.parametrize("line", [
    "Manufactured By: Some Petfood Ltd", "Imported & Marketed by:", "Importer: Some Traders",
    "Mfd. By : See On Pack", "Mfg. Lic. No. 10012345", "MRP: ₹450.00", "USP Rs. 1.50/g",
    "8901234567890", "Net Wt. 2.5 kg", "Customer care 1800 123 4567", "Plot No. 12, Some Road",
    "Our experts export worldwide", "05/12/2024",
])
def test_no_box_for_prices_quantities_addresses_or_dates(line):
    assert keywords(line) == []


def test_a_turned_or_flipped_line_is_cut_from_its_reading_end():
    tall = [(0, 0), (10, 0), (10, 120), (0, 120)]          # read top to bottom
    assert store_train.span_box(tall, 0, 4, 12, turned=True) == (0, 0, 10, 40)
    assert store_train.span_box(tall, 0, 4, 12, turned=True, flipped=True) == pytest.approx((0, 80, 10, 120))
    wide = [(0, 0), (120, 0), (120, 10), (0, 10)]           # upside down: reads right to left
    assert store_train.span_box(wide, 0, 4, 12, flipped=True) == pytest.approx((80, 0, 120, 10))


def test_proposals_are_numbered_and_in_the_photos_frame():
    lines = [{"quad": [(100, 200), (220, 200), (220, 210), (100, 210)], "text": "EXP.FEB 2027",
              "turned": False, "flipped": False}]
    assert store_train.propose(lines, (1000, 750)) == [
        {"id": "kw1", "label": "due", "xyxy": [0.1, 0.266667, 0.14, 0.28], "text": "EXP.", "line": "EXP.FEB 2027"}]


def test_a_task_names_its_photo_as_v1s_project_does_and_reads_back():
    from bestbefore.store_import import photo_id, rect_box
    proposal = {"id": "kw1", "label": "due", "xyxy": [0.1, 0.25, 0.14, 0.28]}
    task = store_train.keyword_task("samples2/made up 1.jpg", (3024, 4032), [proposal], label("clean"))
    assert task["data"]["image"] == "/data/local-files/?d=samples2/made%20up%201.jpg"
    assert photo_id(task["data"]["image"]) == "samples2/made up 1.jpg"
    (prediction,) = task["predictions"]
    keyword, *dates = prediction["result"]
    assert rect_box(keyword["value"]) == {"label": "due", "xyxy": [0.1, 0.25, 0.14, 0.28]}
    assert (keyword["original_width"], keyword["original_height"]) == (3024, 4032)
    assert [rect_box(d["value"]) for d in dates] == label("clean")["boxes"]
    assert all(d["readonly"] and d["from_name"] == "date" for d in dates) and "readonly" not in keyword


# A made-up reviewed export: three v1 photos, 1000x750 upright.
PHOTOS = ["samples1/a.jpg", "samples2/b.jpg", "samples3/c.jpg"]
SIZE = (1000, 750)
PROPOSED = {"samples1/a.jpg": [{"id": "kw1", "label": "due", "xyxy": [0.0, 0.5, 0.09, 0.55]},
                               {"id": "kw2", "label": "prod", "xyxy": [0.0, 0.6, 0.09, 0.65]}],
            "samples2/b.jpg": [{"id": "kw1", "label": "code", "xyxy": [0.5, 0.1, 0.6, 0.12]}],
            "samples3/c.jpg": []}


def reviewed(photo: str, boxes: list[dict], **annotation) -> dict:
    result = [store_train.ls_region(b["id"], "keyword", b, SIZE) for b in boxes]
    result += [store_train.ls_region(f"date{k}", "date", b, SIZE, readonly=True)
               for k, b in enumerate(label("clean")["boxes"], start=1)]
    return {"data": {"image": f"/data/local-files/?d={photo}"}, "drafts": [],
            "annotations": [{"result": result, "was_cancelled": False, **annotation}]}


def export(**changed) -> list[dict]:
    """Every photo reviewed: a's prod box moved, b's code box deleted, a due box added on c."""
    final = {"samples1/a.jpg": [PROPOSED["samples1/a.jpg"][0], {**PROPOSED["samples1/a.jpg"][1],
                                                                "xyxy": [0.0, 0.6, 0.08, 0.65]}],
             "samples2/b.jpg": [], "samples3/c.jpg": [{"id": "Xy12Ab", "label": "due", "xyxy": [0.2, 0.2, 0.3, 0.24]}]}
    return [changed.get(p) or reviewed(p, final[p]) for p in PHOTOS]


def checks(tasks: list[dict]) -> dict[str, list[str]]:
    labels = {p: label("clean") for p in PHOTOS}
    boxes = store_train.keyword_boxes(tasks)
    return {"reviewed": store_train.check_reviewed(tasks, PHOTOS),
            "boxes": store_train.check_keyword_boxes(tasks, {p: SIZE for p in PHOTOS}),
            "off dates": store_train.check_off_dates(boxes, labels),
            "reference": store_train.check_reference(tasks, labels)}


def test_a_reviewed_export_passes_and_ocrs_help_is_counted():
    tasks = export()
    assert all(not p for p in checks(tasks).values())
    counts = store_train.ocr_help(PROPOSED, store_train.keyword_boxes(tasks))
    assert counts == {("kept", "due"): 1, ("moved", "prod"): 1, ("relabelled", "prod"): 0,
                      ("deleted", "code"): 1, ("added", "due"): 1}


def test_a_relabelled_proposal_counts_as_moved():
    final = {"samples2/b.jpg": [{**PROPOSED["samples2/b.jpg"][0], "label": "due"}]}
    counts = store_train.ocr_help({"samples2/b.jpg": PROPOSED["samples2/b.jpg"]}, final)
    assert counts == {("moved", "code"): 1, ("relabelled", "code"): 1}


def test_a_photo_not_reviewed_fails():
    unsubmitted = {**reviewed("samples2/b.jpg", []), "annotations": []}
    assert checks(export(**{"samples2/b.jpg": unsubmitted}))["reviewed"] == [
        "samples2/b.jpg: 0 submitted and 0 skipped annotations, not 1 submitted"]
    skipped = reviewed("samples2/b.jpg", [], was_cancelled=True)
    assert checks(export(**{"samples2/b.jpg": skipped}))["reviewed"] == [
        "samples2/b.jpg: 0 submitted and 1 skipped annotations, not 1 submitted"]
    draft = {**reviewed("samples2/b.jpg", []), "drafts": [{"result": []}]}
    assert checks(export(**{"samples2/b.jpg": draft}))["reviewed"] == ["samples2/b.jpg: 1 drafts not submitted"]
    missing = export()[:2]
    assert store_train.check_reviewed(missing, PHOTOS) == ["samples3/c.jpg: not labelled"]


def test_a_photo_from_outside_v1_fails():
    tasks = export() + [reviewed("testset2/x.jpeg", [])]
    assert store_train.check_reviewed(tasks, PHOTOS) == ["testset2/x.jpeg: labelled, but not in store set v1"]


def test_a_box_outside_the_photo_or_off_its_frame_fails():
    outside = reviewed("samples3/c.jpg", [{"id": "k", "label": "due", "xyxy": [0.95, 0.2, 1.02, 0.24]}])
    assert checks(export(**{"samples3/c.jpg": outside}))["boxes"] == [
        "samples3/c.jpg: a keyword box reaches outside the photo"]
    raw = reviewed("samples3/c.jpg", [{"id": "k", "label": "due", "xyxy": [0.2, 0.2, 0.3, 0.24]}])
    raw["annotations"][0]["result"][0].update(original_width=750, original_height=1000)
    assert checks(export(**{"samples3/c.jpg": raw}))["boxes"] == [
        "samples3/c.jpg: a keyword box drawn on 750x1000, not the upright photo's (1000, 750)"]
    wrong = reviewed("samples3/c.jpg", [{"id": "k", "label": "expiry", "xyxy": [0.2, 0.2, 0.3, 0.24]}])
    assert checks(export(**{"samples3/c.jpg": wrong}))["boxes"] == [
        "samples3/c.jpg: a keyword box labelled ['expiry']"]


def test_a_keyword_box_over_a_date_box_fails():
    date_box = label("clean")["boxes"][0]["xyxy"]          # the expiry, [0.1, 0.5, 0.4, 0.55]
    over = reviewed("samples3/c.jpg", [{"id": "k", "label": "due", "xyxy": [0.12, 0.5, 0.4, 0.56]}])
    assert checks(export(**{"samples3/c.jpg": over}))["off dates"] == [
        "samples3/c.jpg: a due box over its expiry box (IoU 0.79)"]
    beside = reviewed("samples3/c.jpg", [{"id": "k", "label": "due", "xyxy": [0.02, 0.5, date_box[0], 0.55]}])
    assert checks(export(**{"samples3/c.jpg": beside}))["off dates"] == []


def test_a_changed_reference_date_box_fails():
    moved = reviewed("samples3/c.jpg", [])
    moved["annotations"][0]["result"][0]["value"]["x"] += 1
    assert checks(export(**{"samples3/c.jpg": moved}))["reference"] == [
        "samples3/c.jpg: 1 of its frozen date boxes moved or deleted"]
    deleted = reviewed("samples3/c.jpg", [])
    del deleted["annotations"][0]["result"][0]
    assert checks(export(**{"samples3/c.jpg": deleted}))["reference"] == [
        "samples3/c.jpg: 1 of its frozen date boxes moved or deleted"]


def test_a_date_v1_missed_is_kept_apart_from_its_frozen_labels():
    added = reviewed("samples3/c.jpg", [])
    added["annotations"][0]["result"].append(store_train.ls_region(
        "Ab12", "date", {"label": "other_date", "xyxy": [0.5, 0.7, 0.7, 0.74]}, SIZE))
    tasks = export(**{"samples3/c.jpg": added})
    labels = {p: label("clean") for p in PHOTOS}
    assert checks(tasks)["reference"] == []
    assert store_train.added_dates(tasks, labels) == {
        "samples3/c.jpg": [{"label": "other_date", "xyxy": [0.5, 0.7, 0.7, 0.74]}]}
    assert store_train.added_dates(export(), labels) == {}


def test_the_keyword_file_holds_every_photo_and_no_region_id():
    content = store_train.keywords_file(store_train.keyword_boxes(export()), {"export_sha256": "e"}, "2026-11-03")
    assert sorted(content["photos"]) == PHOTOS and content["export_sha256"] == "e" and content["added_dates"] == {}
    assert content["photos"]["samples3/c.jpg"] == [{"label": "due", "xyxy": [0.2, 0.2, 0.3, 0.24]}]


def test_the_import_writes_only_a_checked_export(load_script, tmp_path, monkeypatch, capsys):
    from PIL import Image
    script = load_script("import_store_keywords")
    for photo in PHOTOS:
        (tmp_path / photo).parent.mkdir(exist_ok=True)
        Image.new("RGB", SIZE).save(tmp_path / photo)
    monkeypatch.setattr(script, "store_photos_dir", lambda: tmp_path)
    monkeypatch.setattr(store, "labels", lambda s: {p: label("clean") for p in PHOTOS})
    proposals = {"test_set_sha256": store.TEST_SET_SHA256, "photos": {p: {"boxes": b} for p, b in PROPOSED.items()}}
    (tmp_path / "proposals.json").write_text(json.dumps(proposals))
    out = tmp_path / "v1_keywords.json"
    args = ["--proposals", str(tmp_path / "proposals.json"), "--export", str(tmp_path / "export.json"),
            "--out", str(out)]
    monkeypatch.setattr("sys.argv", ["import_store_keywords.py", *args])

    (tmp_path / "export.json").write_text(json.dumps(export()[:2]))
    with pytest.raises(SystemExit, match="1 checks failed"):
        script.main()
    assert not out.exists()

    (tmp_path / "export.json").write_text(json.dumps(export()))
    script.main()
    printed = capsys.readouterr().out
    assert sorted(store_train.read_keywords(out)) == PHOTOS
    assert "| kept | 1 | 0 | 0 | 1 |" in printed and "| added | 1 | 0 | 0 | 1 |" in printed
    assert "kept or moved, by IoU with the final box: 0.7+ 2, 0.5-0.7 0, under 0.5 0" in printed
    assert "| **all** | 3 | 2 |" in printed
    assert json.loads(out.read_text())["added_dates"] == {}


def ink_line(*runs: tuple[int, int], height: int = 40, width: int = 400):
    """A made-up line of print: dark columns at each (start, end), rows 10 to 30, on light grey."""
    import numpy as np
    gray = np.full((height, width), 220.0, np.float32)
    for a, b in runs:
        gray[10:30, a:b] = 30.0
    return gray


def test_a_padded_box_snaps_to_its_ink(load_script):
    script = load_script("draft_keywords")
    gray = ink_line((50, 60), (64, 74), (78, 88))                  # three letters, x 50 to 88
    assert script.tighten(gray, (40, 2, 98, 38), along_x=True) == (50, 10, 88, 30)


def test_a_keyword_stops_before_the_date_beside_it(load_script):
    script = load_script("draft_keywords")
    # 'EXP.' at 50 to 98, its date from 106: OCR's cut ran 8 px into the date
    gray = ink_line((50, 60), (64, 74), (78, 88), (94, 98), (106, 116), (120, 130))
    assert script.tighten(gray, (45, 5, 114, 35), along_x=True, stops=[(106, 130)]) == (50, 10, 98, 30)


def test_a_sideways_box_snaps_along_its_column(load_script):
    script = load_script("draft_keywords")
    gray = ink_line((50, 60), (64, 74), (78, 88)).T.copy()         # the same line, reading down
    assert script.tighten(gray, (2, 40, 38, 98), along_x=False) == (10, 50, 30, 88)


# --- Store set v3 (S5b) ---------------------------------------------------------

V3_TAGS = {"pack01": ["clean"], "pack08": ["sideways", "sideways"],          # pack01 is in made-up v1-train or -val
           "new1": ["clean", "clean"], "new2": ["clean"], "new3": ["sideways"], "new4": ["sideways"]}
DESIGNS_V3 = {f"trainset3/{d}-{k}.jpg": d for d, tags in V3_TAGS.items() for k in range(len(tags))}
LABELS_V3 = {p: label(V3_TAGS[d][int(p.split("-")[-1][0])]) for p, d in DESIGNS_V3.items()}


def test_a_v3_id_typed_another_way_than_v1s_is_refused():
    rows_v3 = rows({**DESIGNS_V3, "trainset3/pack01-0.jpg": "Pack01"})
    problems = store_train.check_designs(rows_v3, sorted(DESIGNS_V3), set(DESIGNS.values()), "v3")
    assert problems == ["Pack01 is the known design pack01 typed another way"]
    assert store_train.check_designs(rows(DESIGNS_V3), sorted(DESIGNS_V3), set(DESIGNS.values()), "v3") == []
    stray = rows(DESIGNS_V3) + [["samples1/pack01-0.jpg", "pack01"]]
    assert store_train.check_designs(stray, sorted(DESIGNS_V3), set(DESIGNS.values()), "v3")[0].endswith(
        "not a photo of v3: samples1/pack01-0.jpg")


def test_v3_keeps_a_known_designs_side_and_splits_the_rest_by_the_rule():
    split1 = store_train.split_v1(DESIGNS, LABELS)
    split = store_train.split_v3(DESIGNS_V3, LABELS_V3, DESIGNS, split1)
    for known in ("pack01", "pack08"):
        v1_side = {part for p, part in split1.items() if DESIGNS[p] == known}
        assert {split[p] for p, d in DESIGNS_V3.items() if d == known} == v1_side
    new = {d for d in DESIGNS_V3.values() if d.startswith("new")}
    val = {DESIGNS_V3[p] for p, part in split.items() if part == "val" and DESIGNS_V3[p] in new}
    assert len(val) == 2                         # one of 2 clean designs, one of 2 sideways designs
    assert store_train.split_v3(DESIGNS_V3, LABELS_V3, DESIGNS, split1) == split
    assert store_train.split_v1(DESIGNS, LABELS) == split1        # v1's split doesn't move


def test_only_v3s_photos_enter_v3s_split():
    split1 = store_train.split_v1(DESIGNS, LABELS)
    for stray in ("samples1/pack01-0.jpg", "testset2/x-0.jpeg"):
        with pytest.raises(ValueError, match="only store set v3"):
            store_train.split_v3({**DESIGNS_V3, stray: "pack01"}, {**LABELS_V3, stray: label("clean")},
                                 DESIGNS, split1)


def test_v3s_report_leaves_v1s_section_as_it_was(load_script, tmp_path, monkeypatch):
    script = load_script("store_designs")
    v1_text = "# Store pack designs and the v1 split\n\nv1's counts.\n"
    monkeypatch.setattr(script, "SUMMARY", tmp_path / "store_designs.md")
    (tmp_path / "store_designs.md").write_text(v1_text)
    monkeypatch.setattr(store, "DESIGNS_V3_SHA256", "d" * 64)
    monkeypatch.setattr(store, "LABELS_V3_SHA256", "e" * 64)
    split1 = store_train.split_v1(DESIGNS, LABELS)
    split = store_train.split_v3(DESIGNS_V3, LABELS_V3, DESIGNS, split1)
    section = script.summary_v3(DESIGNS_V3, LABELS_V3, DESIGNS, split1, split, "2026-11-04", "f" * 64)
    (tmp_path / "store_designs.md").write_text(v1_text + section)
    again = (tmp_path / "store_designs.md").read_text().split(script.V3_SECTION, 1)[0]
    assert again == v1_text
    assert "| design new to v3 | 4 | 5 | 5 |" in section
    names = {*DESIGNS_V3, *DESIGNS_V3.values(), *(p.split("/")[1] for p in DESIGNS_V3)}
    assert not [n for n in names if n in section]


def test_keyword_tools_take_v1_or_v3_and_never_v2():
    assert store_train.keyword_set("v1") is store.V1 and store_train.keyword_set("v3") is store.V3
    with pytest.raises(SystemExit, match="not v2"):
        store_train.keyword_set("v2")
    assert store_train.keywords_dir(store.V1) == store_train.KEYWORDS_DIR
    assert store_train.keywords_path(store.V1) == store_train.KEYWORDS_PATH
    assert store_train.keywords_dir(store.V3) == store.DATA_DIR / "v3" / "keywords"
    assert store_train.keywords_path(store.V3) == store.DATA_DIR / "v3" / "keywords.json"


def test_the_keyword_file_is_dated_by_its_review():
    tasks = export()
    assert store_train.reviewed_on(tasks) is None
    tasks[0]["annotations"][0]["updated_at"] = "2026-11-03T17:42:01.919561Z"
    tasks[1]["annotations"][0]["updated_at"] = "2026-11-02T09:00:00Z"
    assert store_train.reviewed_on(tasks) == "2026-11-03"
    assert store_train.keywords_file({}, {}, "2026-11-03", version="v3")["version"] == "v3"


def test_v3s_keyword_review_is_checked_against_v3s_photos():
    photos = ["trainset3/a.jpg"]
    task = {**reviewed("trainset3/a.jpg", []), "annotations": [{"result": [], "was_cancelled": False}]}
    assert store_train.check_reviewed([task], photos, "store set v3") == []
    stray = reviewed("samples1/a.jpg", [])
    assert store_train.check_reviewed([task, stray], photos, "store set v3") == [
        "samples1/a.jpg: labelled, but not in store set v3"]


# --- The export for training (S6) ---------------------------------------------
#
# Made-up phone photos: white, with a red "date" and a blue "keyword"
# off-centre in the upright frame, stored turned with an EXIF flag as phones
# store them, and bigger than MAX_SIDE, so the export applies the flag,
# shrinks and turns them. A box turned, mirrored or scaled the wrong way
# lands on white.

RED, BLUE = (230, 20, 20), (20, 20, 230)
# PIL's exif_transpose turns a stored photo upright; these turn an upright
# photo into the pixels a phone stores with each flag.
STORED = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
          4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
          6: Image.Transpose.ROTATE_90, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_270}
TALL, WIDE = (1800, 2400), (2400, 1800)          # upright sizes; both shrink by 2/3 to 1,600 px


def stored_photo(path, upright_size, date_rect, keyword_rect, flag):
    """Save a made-up photo, given in its upright frame, as a phone stores it with `flag`."""
    up = Image.new("RGB", upright_size, "white")
    up.paste(RED, date_rect)
    up.paste(BLUE, keyword_rect)
    exif = Image.Exif()
    exif[0x0112] = flag
    path.parent.mkdir(parents=True, exist_ok=True)
    (up.transpose(STORED[flag]) if flag in STORED else up).save(path, quality=95, exif=exif.tobytes())
    return store.sha256(path)


def xyxy(rect, size):
    return [rect[0] / size[0], rect[1] / size[1], rect[2] / size[0], rect[3] / size[1]]


# photo: (set, upright size, date rectangle, keyword rectangle, EXIF flag, design, side)
MADE_UP = {"samples1/a.jpg": ("v1", TALL, (300, 1500, 700, 1590), (90, 1500, 260, 1590), 6, "PouchCo", "train"),
           "samples2/b.jpg": ("v1", WIDE, (1500, 300, 1590, 900), (1500, 60, 1590, 240), 8, "JarCo", "val"),
           "samples3/c.jpg": ("v1", TALL, (1200, 2100, 1700, 2200), (1000, 2100, 1150, 2200), 3, "PouchCo", "train"),
           "samples3/d.jpg": ("v1", WIDE, (60, 1700, 500, 1780), (60, 1600, 200, 1680), 2, "BoxCo", "train"),
           "trainset3/e.jpg": ("v3", TALL, (200, 200, 800, 300), (40, 200, 180, 300), 1, "PouchCo", "train"),
           "trainset3/f.jpg": ("v3", WIDE, (1800, 1300, 2300, 1400), (1600, 1300, 1760, 1400), 5, "CanCo", "val")}


@pytest.fixture(scope="module")
def made_up(tmp_path_factory):
    """The made-up photos on disk, their frozen files as the export reads them, and the export of them."""
    root = tmp_path_factory.mktemp("photos")
    sets = {"v1": {"labels": {}, "keywords": {"photos": {}, "added_dates": {}}},
            "v3": {"labels": {}, "keywords": {"photos": {}, "added_dates": {}}}}
    for photo, (version, size, date_rect, keyword_rect, flag, _, _) in MADE_UP.items():
        sets[version]["labels"][photo] = {
            "sha256": stored_photo(root / photo, size, date_rect, keyword_rect, flag), "tag": "clean",
            "boxes": [{"label": "expiry", "xyxy": xyxy(date_rect, size)}]}
        sets[version]["keywords"]["photos"][photo] = [{"label": "due", "xyxy": xyxy(keyword_rect, size)}]
    designs = {p: v[5] for p, v in MADE_UP.items()}
    split = {p: v[6] for p, v in MADE_UP.items()}
    out, names = tmp_path_factory.mktemp("export") / "store", root / "train_names.json"
    plan = store_train.export_plan(sets, designs, split)
    manifest = store_train.write_export(plan, out, {"v1": {"labels": "a" * 64}}, root, names)
    return SimpleNamespace(root=root, sets=sets, designs=designs, split=split, plan=plan,
                           out=out, names=names, manifest=manifest)


def ink(im: Image.Image, colour) -> tuple[int, int, int, int]:
    """The pixel box (ends exclusive) round every pixel clearly of `colour`
    (JPEG and the shrink soften its edges by a pixel or so)."""
    a = np.asarray(im, np.int16)
    near = np.abs(a - np.array(colour)).max(axis=-1) < 90
    ys, xs = np.nonzero(near)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def test_every_box_sits_on_its_print_after_the_exif_turn_the_shrink_and_every_turn(made_up):
    """The plan's test: an EXIF-turned photo's exported boxes sit on its date,
    in each of the 4 turns, with 7 of EXIF's 8 flags (mirrors included)."""
    seen = 0
    for p in made_up.plan:
        version, size, *_ = MADE_UP[p.photo]
        for folder in store_train.FOLDERS[p.part]:
            path = made_up.out / "images" / folder / f"{p.name}.jpg"
            im = Image.open(path).convert("RGB")
            quarter = store_train.FOLDERS[p.part][folder]
            long, short = (1600, 1200)
            upright = (short, long) if size == TALL else (long, short)
            assert im.size == (upright if quarter % 2 == 0 else upright[::-1])
            date, keyword = derived.read_boxes(path)
            assert (date.label, keyword.label) == ("date", "due")
            for box, colour in ((date, RED), (keyword, BLUE)):
                drawn = [v * s for v, s in zip(box.xyxy, im.size * 2)]
                assert np.allclose(drawn, ink(im, colour), atol=2), (p.photo, folder, box.label)
            seen += 1
    assert seen == 4 * 4 + 2                     # 4 training photos in 4 turns, 2 validation photos once


def test_only_the_training_part_is_turned(made_up):
    for p in made_up.plan:
        on_disk = {f.parent.name for f in made_up.out.glob(f"images/*/{p.name}.jpg")}
        assert on_disk == set(store_train.FOLDERS[p.part])
    assert (made_up.out / "train.txt").read_text().splitlines()[:2] == ["./images/train_0/v1-0001.jpg",
                                                                        "./images/train_0/v1-0003.jpg"]
    assert (made_up.out / "val.txt").read_text() == "./images/val/v1-0002.jpg\n./images/val/v3-0002.jpg\n"
    assert made_up.manifest["counts"]["images"] == {"train_0": 4, "train_90": 4, "train_180": 4,
                                                    "train_270": 4, "val": 2}


def test_lists_name_their_photos_relative_to_themselves(made_up, tmp_path):
    moved = tmp_path / "elsewhere"
    shutil.copytree(made_up.out, moved)
    images = derived.read_list(moved / "train.txt")
    assert len(images) == 16 and all(i.is_file() and moved in i.parents for i in images)


def test_names_are_numbers_and_the_names_map_stays_out_of_the_export(made_up):
    files = [f for f in made_up.out.rglob("*") if f.is_file()]
    assert {f.name for f in files if f.parent == made_up.out} == {"manifest.json", "train.txt", "val.txt"}
    assert all(re.fullmatch(r"v[13]-\d{4}\.(jpg|txt)", f.name) for f in files if f.parent != made_up.out)
    text = "".join(f.read_text() for f in files if f.suffix in (".json", ".txt"))
    secret = {*MADE_UP, *(p.split("/")[1] for p in MADE_UP), *(Path(p).stem for p in MADE_UP if len(Path(p).stem) > 1),
              *made_up.designs.values(), *(lab["sha256"] for s in made_up.sets.values() for lab in s["labels"].values())}
    assert not [s for s in secret if s in text]
    names = json.loads(made_up.names.read_text())
    assert {n: v["photo"] for n, v in names["photos"].items()} == {
        "v1-0001": "samples1/a.jpg", "v1-0002": "samples2/b.jpg", "v1-0003": "samples3/c.jpg",
        "v1-0004": "samples3/d.jpg", "v3-0001": "trainset3/e.jpg", "v3-0002": "trainset3/f.jpg"}
    assert names["designs"] == {"d01": "BoxCo", "d02": "CanCo", "d03": "JarCo", "d04": "PouchCo"}
    assert names["manifest_sha256"] == store.sha256(made_up.out / "manifest.json")
    assert made_up.manifest["photos"]["v3-0001"] == {"part": "train", "design": "d04", "tag": "clean",
                                                     "labels": ["expiry", "due"], "added_dates": 0, "dropped": []}


def test_the_names_map_may_not_go_inside_the_export(made_up, tmp_path):
    out = tmp_path / "store"
    with pytest.raises(SystemExit, match="names map stays out"):
        store_train.write_export(made_up.plan, out, {}, made_up.root, out / "train_names.json")


def test_exported_photos_carry_no_metadata_and_read_the_same_in_opencv(made_up):
    cv2 = pytest.importorskip("cv2")
    for path in made_up.out.rglob("*.jpg"):
        with Image.open(path) as im:
            assert not im.getexif() and "icc_profile" not in im.info
            pixels = np.asarray(im.convert("RGB"))
        assert np.array_equal(cv2.imread(str(path))[..., ::-1], pixels)    # Ultralytics reads with OpenCV


def test_a_v2_photo_is_refused(made_up):
    v1 = {**made_up.sets["v1"], "labels": {**made_up.sets["v1"]["labels"], "testset2/x.jpg": label("clean")}}
    with pytest.raises(ValueError, match="testset2/x.jpg: a store set v2 photo"):
        store_train.export_plan({**made_up.sets, "v1": v1}, {**made_up.designs, "testset2/x.jpg": "JarCo"},
                                {**made_up.split, "testset2/x.jpg": "train"})
    with pytest.raises(ValueError, match="not v2"):
        store_train.export_plan({"v2": made_up.sets["v1"]}, made_up.designs, made_up.split)


def test_the_export_refuses_to_start_while_v2_isnt_frozen(load_script, monkeypatch):
    script = load_script("export_store_yolo")
    for constant in ("TEST_SET_V2_SHA256", "PHOTOS_V2_SHA256"):
        with monkeypatch.context() as m:
            m.setattr(store, constant, None)
            m.setattr(store_train, "read_designs", lambda *a: pytest.fail("read a file before checking v2"))
            with pytest.raises(SystemExit, match="v2 isn't frozen"):
                script.export()


def test_a_design_on_both_sides_a_photo_without_its_review_and_a_box_off_the_photo_are_refused(made_up):
    split = {**made_up.split, "samples1/a.jpg": "val"}       # PouchCo on both sides
    with pytest.raises(ValueError, match="design PouchCo on both sides"):
        store_train.export_plan(made_up.sets, made_up.designs, split)
    v3 = json.loads(json.dumps(made_up.sets["v3"]))
    del v3["keywords"]["photos"]["trainset3/e.jpg"]
    v3["labels"]["trainset3/f.jpg"]["boxes"][0]["xyxy"] = [0.9, 0.5, 1.1, 0.6]
    with pytest.raises(ValueError, match="2 problems") as refused:
        store_train.export_plan({**made_up.sets, "v3": v3}, made_up.designs, made_up.split)
    assert "trainset3/e.jpg: no keyword review" in str(refused.value)
    assert "trainset3/f.jpg: a expiry box outside the photo" in str(refused.value)


def test_the_dates_a_review_added_are_dates(made_up, tmp_path):
    v1 = json.loads(json.dumps(made_up.sets["v1"]))
    v1["keywords"]["added_dates"]["samples2/b.jpg"] = [{"label": "other_date", "xyxy": [0.1, 0.1, 0.2, 0.15]}]
    plan = store_train.export_plan({**made_up.sets, "v1": v1}, made_up.designs, made_up.split)
    manifest = store_train.write_export(plan, tmp_path / "store", {}, made_up.root, tmp_path / "names.json")
    assert [b.label for b in derived.read_boxes(tmp_path / "store/images/val/v1-0002.jpg")] == ["date", "date", "due"]
    assert manifest["photos"]["v1-0002"]["labels"] == ["expiry", "other_date", "due"]
    assert manifest["counts"]["dates"]["val"] == {"expiry": 2, "mfg": 0, "other_date": 1, "added": 1}


def test_a_box_under_2_px_across_is_left_out_and_counted(made_up, tmp_path):
    """As v1's stray mfg box, 1.1 by 3.8 px on its exported photo (S6): out
    of the label files, in the manifest's counts, and the frozen labels it
    came from as they were."""
    v1 = json.loads(json.dumps(made_up.sets["v1"]))
    w, h = 1600, 1200                                    # samples2/b.jpg as exported
    v1["labels"]["samples2/b.jpg"]["boxes"].append({"label": "mfg", "xyxy": [0.5, 0.5, 0.5 + 1.1 / w, 0.5 + 3.8 / h]})
    v1["keywords"]["added_dates"]["samples2/b.jpg"] = [
        {"label": "other_date", "xyxy": [0.1, 0.1, 0.1 + 3 / w, 0.1 + 1.5 / h]},      # a slip too
        {"label": "other_date", "xyxy": [0.2, 0.2, 0.2 + 2.5 / w, 0.2 + 2.5 / h]}]    # small, but kept
    frozen = json.dumps(v1["labels"])
    plan = store_train.export_plan({**made_up.sets, "v1": v1}, made_up.designs, made_up.split)
    manifest = store_train.write_export(plan, tmp_path / "store", {}, made_up.root, tmp_path / "names.json")
    assert [b.label for b in derived.read_boxes(tmp_path / "store/images/val/v1-0002.jpg")] == ["date", "date", "due"]
    entry = manifest["photos"]["v1-0002"]
    assert (entry["labels"], entry["added_dates"], entry["dropped"]) == (["expiry", "other_date", "due"], 1,
                                                                         ["mfg", "other_date"])
    assert manifest["counts"]["dropped"] == {"train": {}, "val": {"mfg": 1, "other_date": 1}}
    assert manifest["counts"]["dates"]["val"] == {"expiry": 2, "mfg": 0, "other_date": 1, "added": 1}
    assert made_up.manifest["counts"]["dropped"] == {"train": {}, "val": {}}
    assert json.dumps(v1["labels"]) == frozen


def test_the_export_writes_only_into_an_empty_folder_and_the_same_bytes_every_time(made_up, tmp_path):
    with pytest.raises(SystemExit, match="isn't empty"):
        store_train.write_export(made_up.plan, made_up.out, {}, made_up.root, tmp_path / "names.json")
    again = store_train.write_export(made_up.plan, tmp_path / "store", {"v1": {"labels": "a" * 64}},
                                     made_up.root, tmp_path / "names.json", made=made_up.manifest["made"])
    assert again == made_up.manifest
    assert (tmp_path / "store/manifest.json").read_bytes() == (made_up.out / "manifest.json").read_bytes()


def test_a_photo_changed_since_its_freeze_is_refused(made_up, tmp_path):
    root = tmp_path / "photos"
    shutil.copytree(made_up.root, root)
    with Image.open(root / "samples3/c.jpg") as im:
        im.convert("RGB").save(root / "samples3/c.jpg", quality=90)
    with pytest.raises(RuntimeError, match="samples3/c.jpg has changed since it was frozen"):
        store_train.write_export(made_up.plan, tmp_path / "store", {}, root, tmp_path / "names.json")


def test_the_copy_check_passes_a_copy_and_catches_every_difference(made_up, tmp_path):
    assert store_train.check_export(made_up.out) == []
    copy = tmp_path / "copy"
    shutil.copytree(made_up.out, copy)
    (copy / "labels/train_90/v1-0001.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (copy / "images/val/v3-0002.jpg").unlink()
    (copy / "images/train_0/v9-0001.jpg").write_bytes(b"stray")
    (copy / "names.json").write_text("{}")
    (copy / "val.txt").write_text("./images/val/v1-0002.jpg\n/elsewhere/x.jpg\n")
    assert store_train.check_export(copy) == [
        "missing: images/val/v3-0002.jpg", "changed: labels/train_90/v1-0001.txt", "changed: val.txt",
        "not in the manifest: images/train_0/v9-0001.jpg",
        "images/train_0: 5 .jpg files, the manifest counts 4",
        "images/val: 1 .jpg files, the manifest counts 2",
        "val.txt names /elsewhere/x.jpg, not one of the export's photos",
        "a names map in the export: names.json"]
    assert store_train.check_export(tmp_path) == [f"no manifest.json in {tmp_path}"]


def test_the_copy_check_runs_from_its_source_alone(made_up):
    """On the cluster it runs without this package (S6): its source and three imports."""
    source = "import hashlib, json\nfrom pathlib import Path\n" + inspect.getsource(store_train.check_export)
    space = {}
    exec(compile(source, "check_export", "exec"), space)
    assert space["check_export"](made_up.out) == []
