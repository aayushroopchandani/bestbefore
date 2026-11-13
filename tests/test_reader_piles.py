"""The crop piles (reader/piles.py) and scripts/make_crops.py on made-up
sources: a generated set with its sidecar, an ExpDate split with its
annotations, and a turned copy of a validation photo."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bestbefore import derived, m2, m2b, m3, runs
from bestbefore.reader import piles, shards
from bestbefore.samples import Box

_ROT = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}


def _token(text, cls, box, **date):
    t = {"text": text, "cls": cls, "box": box}
    if cls == "date":
        t |= {"role": date.get("role", "expiry"), "format": date.get("format", "DD.MM.YY"),
              "parts": date.get("parts", [2027, 6, 5])}
    return t


def _block(quarter, tokens, orientation=None):
    orientation = orientation or {0: "upright", 1: "sideways", 2: "upside-down", 3: "sideways"}[quarter]
    return {"kind": "pair", "style": "dot-matrix", "band": "normal", "tokens": tokens,
            "geometry": {"orientation": orientation, "quarter": quarter, "angle": 1.0,
                         "surface": "flat", "perspective": None}}


def test_gen_takes_dates_codes_and_classless_tokens_of_three_characters():
    wanted = piles.gen_wanted
    assert wanted(_token("05.06.27", "date", [0, 0, 1, 1])) and wanted(_token("LOT A1", "code", None))
    assert wanted(_token("Rs.", None, None)) and wanted(_token(" 9 months from Mfg. ", None, None))
    assert not wanted(_token(" | ", None, None)) and not wanted(_token("Rs", None, None))
    assert not wanted(_token("EXP", "due", None)) and not wanted(_token("MFG:", "prod", None))


def test_generated_items_turn_upright_and_keep_the_sidecars_parts():
    line = {"image": "000007.jpg", "size": [400, 200], "blocks": [
        _block(1, [_token("Best Before", "due", [10, 10, 30, 90]),
                   _token("23.11.26", "date", [40, 20, 60, 100], parts=[2026, 11, 23]),
                   _token("Rs.", None, [70, 20, 80, 50]),
                   _token(" / ", None, [90, 20, 95, 30])]),
        _block(2, [_token("LOT 5X", "code", [200, 100, 300, 120]),
                   _token("MRP 99", None, None)]),
        _block(0, [_token("2026-01", "date", [300, 150, 380, 170], role="mfg", format="YYYY-MM", parts=[2026, 1, None])],
               orientation="tilted")]}
    items, no_box = piles.generated_items(line, "gen_v2", piles.gen_wanted, as_cut=False)
    assert no_box == 1                                       # MRP 99 lost its ink
    assert [i.text for i in items] == ["23.11.26", "Rs.", "LOT 5X", "2026-01"]
    assert [i.turn for i in items] == [3, 3, 2, 0]           # (4 - quarter) % 4
    assert all(i.tags["upright"] == 0 for i in items)        # stored upright
    date = items[0]
    assert date.parts == (2026, 11, 23) and date.role == "expiry" and date.tags["parts_from"] == "sidecar"
    assert piles.expdate_parts("23.11.26") == (2023, 11, 26)    # the frozen parser reads it year first
    assert date.xyxy == (0.1, 0.1, 0.15, 0.5) and date.source == "gen_v2/000007.jpg#b0t1"
    assert items[1].role is None and items[1].parts is None and items[2].cls == "code"
    assert items[3].tags["orientation"] == "tilted"


def test_a_suite_keeps_its_targets_as_cut_with_the_turn_that_stands_them_up():
    line = {"image": "000001.jpg", "size": [400, 200], "stratum": "sideways", "blocks": [
        _block(3, [_token("2027/01/02", "date", [10, 10, 30, 150], role="target", parts=[2027, 1, 2]),
                   _token("2025/01/02", "date", [40, 10, 60, 150], role="mfg", parts=[2025, 1, 2])])]}
    items, _ = piles.generated_items(line, "synth_val_v2", piles.target, as_cut=True)
    (item,) = items
    assert item.turn == 0 and item.tags["upright"] == 1 and item.tags["stratum"] == "sideways"


def test_the_turn_is_found_from_every_box():
    boxes = [(0.1, 0.2, 0.3, 0.25), (0.5, 0.6, 0.9, 0.7)]
    for q in (1, 2, 3):
        assert piles._the_turn(boxes, [derived.turn_box(b, q) for b in boxes], Path("x")) == q
    with pytest.raises(ValueError):
        piles._the_turn(boxes, [derived.turn_box(boxes[0], 1), derived.turn_box(boxes[1], 3)], Path("x"))


def test_expdate_test_is_never_cut():
    with pytest.raises(ValueError):
        next(piles._expdate("real-test", lambda n: True, {}))


# --- On made-up sources ------------------------------------------------------------------

def _bar(a: np.ndarray, box) -> None:
    """A dark bar over `box` (pixels), with a white notch at its top-left."""
    x1, y1, x2, y2 = box
    a[y1:y2, x1:x2] = 20
    a[y1:y1 + 4, x1:x1 + 4] = 255


def _fake_expdate(root: Path, monkeypatch) -> Path:
    """ExpDate's real-train with annotations: a.jpg for training, v.jpg
    for validation (in yolo/val.txt), and v.jpg turned 90 degrees as M2's
    real_val_rot copy."""
    expdate, generated = root / "expdate", root / "generated"
    monkeypatch.setenv("EXPDATE_DIR", str(expdate))
    monkeypatch.setenv("GENERATED_DIR", str(generated))
    train = expdate / "products-real" / "Products-Real" / "train"
    (train / "images").mkdir(parents=True)
    ann = {}
    for name, boxes in {"a.jpg": [("date", [20, 30, 120, 50], "2021.10.19"), ("due", [125, 30, 150, 50], None)],
                        "v.jpg": [("code", [10, 10, 60, 20], None), ("date", [40, 60, 160, 84], "21.05.26"),
                                  ("date", [40, 100, 140, 120], "2022/06/30")]}.items():
        a = np.full((150, 200, 3), 190, np.uint8)
        for _, b, _ in boxes:
            _bar(a, b)
        Image.fromarray(a).save(train / "images" / name, quality=95)
        ann[name] = {"width": 200, "height": 150,
                     "ann": [{"cls": c, "bbox": b, **({"transcription": t} if t else {})} for c, b, t in boxes]}
    (train / "annotations.json").write_text(json.dumps(ann))
    (expdate / "yolo").mkdir(parents=True)
    (expdate / "yolo" / "val.txt").write_text(f"{train / 'images' / 'v.jpg'}\n")
    with Image.open(train / "images" / "v.jpg") as im:
        image = im.convert("RGB")
    boxes = [Box(c, (b[0] / 200, b[1] / 150, b[2] / 200, b[3] / 150)) for c, b, _ in
             [(e["cls"], e["bbox"], None) for e in ann["v.jpg"]["ann"]]]
    turned, tboxes = derived.turn(image, boxes, 1)
    sample = derived.write_yolo(m2.derived_dir(), "real_val_rot", "v.jpg", turned, tboxes)
    derived.write_list(m2.list_path("real_val_rot"), [sample.image])
    return train


def test_real_val_and_its_turned_copy(tmp_path, monkeypatch):
    _fake_expdate(tmp_path, monkeypatch)
    (val,) = piles.photos("real-val")
    assert [i.text for i in val.items] == ["21.05.26", "2022/06/30"]
    assert val.items[1].parts == (2022, 6, 30) and val.items[0].tags["upright"] == 0
    (real,) = piles.photos("expdate-real")
    assert [i.text for i in real.items] == ["2021.10.19"] and real.items[0].source == "real-train/a.jpg#0"
    (rot,) = piles.photos("real-val-rot")
    assert [i.text for i in rot.items] == ["21.05.26", "2022/06/30"]
    assert [i.source for i in rot.items] == ["real_val_rot/v.jpg#1", "real_val_rot/v.jpg#2"]
    assert all(i.turn == 0 and i.tags["upright"] == 3 and i.tags["photo_turn"] == 1 for i in rot.items)


class _InProcess:
    """Stands in for multiprocessing.Pool: workers can't import a script
    loaded by path, so here every photo is cut in this process."""

    def __init__(self, workers: int):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def imap(self, fn, tasks, chunksize=1):
        return map(fn, tasks)


def _fake_gen_sample(root: Path, monkeypatch) -> None:
    """Three generated photos with their sidecar: blocks at quarters 0, 1
    and 2, every token a notched bar drawn as it lies on the photo."""
    generated = root / "generated"
    monkeypatch.setenv("GENERATED_DIR", str(generated))
    images = m2b.gen_dir() / "images" / "gen_v2_sample"
    images.mkdir(parents=True)
    # Tokens sit further apart than their margins, so each crop holds one bar.
    layouts = [(0, (400, 300), [("2027-03", "date", [50, 50, 200, 80]), ("LOT 7", "code", [50, 150, 120, 170])]),
               (1, (300, 400), [("MAR 2027", "date", [100, 60, 130, 260]), ("Rs.", None, [200, 60, 220, 100])]),
               (2, (500, 300), [("03/27", "date", [100, 200, 220, 230]), ("|", None, [400, 200, 406, 230])])]
    lines, paths = [], []
    for i, (q, (w, h), tokens) in enumerate(layouts):
        a = np.full((h, w, 3), 200, np.uint8)
        for _, _, box in tokens:
            # The bar as printed upright, then turned as the block was: its notch
            # sits at the print's top-left wherever the block turned.
            x1, y1, x2, y2 = box
            bw, bh = (x2 - x1, y2 - y1) if q % 2 == 0 else (y2 - y1, x2 - x1)
            upright = np.full((bh, bw, 3), 20, np.uint8)
            upright[:4, :4] = 255
            a[y1:y2, x1:x2] = np.rot90(upright, q)
        name = f"{i:06d}.jpg"
        Image.fromarray(a).save(images / name, quality=95)
        paths.append(images / name)
        lines.append({"image": name, "i": i, "size": [w, h], "blocks": [_block(q, [
            _token(t, c, b, role="expiry", format="MM/YY", parts=[2027, 3, None]) if c == "date" else _token(t, c, b)
            for t, c, b in tokens])], "background": {"boxes": []}})
    derived.write_list(m2b.list_path("gen_v2_sample"), paths)
    sidecar = m2b.sidecar_path("gen_v2_sample")
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("".join(json.dumps(line) + "\n" for line in lines))
    m2b.manifest_path("gen-v2-sample").parent.mkdir(parents=True)
    m2b.manifest_path("gen-v2-sample").write_text(json.dumps({"photos": 3, "generated_dates": 3}))


@pytest.fixture
def make_crops(load_script, tmp_path, monkeypatch):
    script = load_script("make_crops")
    monkeypatch.setattr(script, "Pool", _InProcess)
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    decisions = tmp_path / "runs" / "m3-decisions.json"
    monkeypatch.setattr(m3, "DECISIONS_PATH", decisions)
    decisions.parent.mkdir()
    decisions.write_text(json.dumps({"crop_recipe": {"pad": 0.0, "turn": False, "run": next(iter(m3.CROP_DEPARTURES))}}))
    return script


def test_make_crops_cuts_checks_and_stands_every_crop_upright(make_crops, tmp_path, monkeypatch):
    _fake_gen_sample(tmp_path, monkeypatch)
    root = tmp_path / "crops"
    recipe = make_crops.recipe_in_use()
    assert recipe["pad"] == 0.0 and recipe["turn"] is True       # the owner's departure, S2
    record = runs.start_run("m3-crops-test", {})
    made = make_crops.make("gen-sample", root, 2, None, False, recipe, record)
    assert made["count"] == 5 and made["skipped"] == {}           # "|" isn't wanted: 1 character
    manifest = json.loads((root / "gen-sample" / "manifest.json").read_text())
    assert manifest["crop_recipe"] == recipe and manifest["planned"] == {"date": 3, "code": 1, "None": 1}
    assert manifest["counts"]["turn"] == {"0": 2, "3": 2, "2": 1}
    assert manifest["sources"]["m2b/sidecars/gen_v2_sample.jsonl"] == hashlib.sha256(
        m2b.sidecar_path("gen_v2_sample").read_bytes()).hexdigest()
    env = shards.open_pile(root / "gen-sample")
    for crop in shards.records(env, "gen-sample"):
        dark = np.asarray(crop.image().convert("L")) < 100
        ys, xs = np.nonzero(dark)
        assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == pytest.approx(crop.box, abs=1)
        x1, y1 = round(crop.box[0]), round(crop.box[1])
        assert np.asarray(crop.image().convert("L"))[y1 + 1, x1 + 1] > 200, crop.text   # the notch: upright
    env.close()
    assert make_crops.check("gen-sample", root, 2)["ok"]


def test_check_finds_a_pile_that_lost_crops(make_crops, tmp_path, monkeypatch):
    _fake_gen_sample(tmp_path, monkeypatch)
    root = tmp_path / "crops"
    record = runs.start_run("m3-crops-test", {})
    make_crops.make("gen-sample", root, 1, None, False, make_crops.recipe_in_use(), record)
    lines = m2b.sidecar_path("gen_v2_sample").read_text().splitlines()
    first = json.loads(lines[0])
    first["blocks"][0]["tokens"].append(_token("LOT 8", "code", [50, 120, 120, 140]))
    m2b.sidecar_path("gen_v2_sample").write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    result = make_crops.check("gen-sample", root, 1)
    assert not result["ok"]
    assert any("the sources call for" in i for i in result["issues"])
    assert any("has changed since" in i for i in result["issues"])


def test_make_crops_refuses_a_margin_too_small_for_the_recipe(make_crops, monkeypatch):
    monkeypatch.setattr(m3, "crop_recipe_used", lambda d: {"pad": 0.45, "turn": True, "run": "r", "departures": []})
    with pytest.raises(SystemExit):
        make_crops.recipe_in_use()


def test_make_crops_never_overwrites_a_pile(make_crops, tmp_path, monkeypatch):
    _fake_gen_sample(tmp_path, monkeypatch)
    (tmp_path / "crops" / "gen-sample").mkdir(parents=True)
    with pytest.raises(SystemExit):
        make_crops.make("gen-sample", tmp_path / "crops", 1, None, False, {}, tmp_path)
