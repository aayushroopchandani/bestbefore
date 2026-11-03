import json

import pytest

from bestbefore import store, store_train


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
