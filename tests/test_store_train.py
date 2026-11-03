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
