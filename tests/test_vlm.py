"""The VLM's answers, images and examples (reader/vlm.py, scripts/make_vlm_set.py)."""
import json

import numpy as np
import pytest
from PIL import Image

from bestbefore.reader import vlm
from bestbefore.samples import Expiry


# --- Answers as JSON ------------------------------------------------------------------------

def test_an_answer_writes_the_prompts_json():
    assert vlm.Answer(Expiry(2027, 3, 1), Expiry(2025, 3)).json() == '{"expiry": "2027-03-01", "mfg": "2025-03"}'
    assert vlm.Answer(None, None).json() == '{"expiry": null, "mfg": null}'


@pytest.mark.parametrize("text, expiry, mfg", [
    ('{"expiry": "2027-03-01", "mfg": "2025-03"}', Expiry(2027, 3, 1), Expiry(2025, 3)),
    ('```json\n{"expiry": "2027-03", "mfg": null}\n```', Expiry(2027, 3), None),
    ("The dates: {'expiry': '2027/03/01', 'mfg': None}. Done.", Expiry(2027, 3, 1), None),
    ('{"Expiry": "2027.3.1", "MFG": "2025-03-02",}', Expiry(2027, 3, 1), Expiry(2025, 3, 2)),
    ('{"expiry": "2027-03"}', Expiry(2027, 3), None),
])
def test_answers_are_parsed_leniently(text, expiry, mfg):
    parsed = vlm.parse_answer(text)
    assert parsed.answer == vlm.Answer(expiry, mfg)
    assert parsed.problems == ()


@pytest.mark.parametrize("value", ["2027-13", "2027-02-30", "03/2027", "soon", "2027", 2027])
def test_a_bad_date_is_none_and_said_so(value):
    parsed = vlm.parse_answer(json.dumps({"expiry": value, "mfg": "2025-01"}))
    assert parsed.answer == vlm.Answer(None, Expiry(2025, 1))
    assert parsed.problems == ("bad expiry",)


def test_null_in_any_spelling_is_no_problem():
    for null in (None, "null", "", "None", "N/A"):
        assert vlm.parse_answer(json.dumps({"expiry": null, "mfg": null})).problems == ()


def test_no_json_or_broken_json_answers_none():
    assert vlm.parse_answer("I can't see a date.") == vlm.Parsed(vlm.Answer(None, None), ("no json",))
    assert vlm.parse_answer('{"expiry": 2027-03}').problems == ("bad json",)
    assert vlm.parse_answer("{") .problems == ("no json",)


def test_every_answer_round_trips_through_its_json():
    for a in (vlm.Answer(Expiry(2028, 2, 29), None), vlm.Answer(None, Expiry(2024, 12)), vlm.Answer(None, None)):
        assert vlm.parse_answer(a.json()) == vlm.Parsed(a)


def test_the_confidence_is_the_least_sure_token_of_the_expirys_value():
    # Qwen writes each digit as its own token
    tokens = [('{"', 0.99), ("expiry", 0.99), ('":', 0.99), (' "', 0.98), ("2", 0.97), ("0", 0.96), ("2", 0.95),
              ("7", 0.40), ("-", 0.99), ("0", 0.90), ("3", 0.85), ('",', 0.99), (' "', 0.99), ("m", 0.2),
              ("fg", 0.3), ('":', 0.99), (" null", 0.1), ("}", 0.99)]
    assert vlm.value_confidence(tokens) == 0.40
    assert vlm.value_confidence(tokens, "mfg") is None
    assert vlm.value_confidence([('{"expiry": null}', 0.5)]) is None


# --- Image size -----------------------------------------------------------------------------

def test_the_model_size_is_qwens_smart_resize():
    smart = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl").smart_resize
    rng = np.random.default_rng(0)
    sizes = [(1600, 1200), (1200, 1600), (1008, 756), (1600, 1184), (40, 900), (300, 200), (4000, 3000)]
    sizes += [tuple(int(v) for v in rng.integers(40, 2400, 2)) for _ in range(300)]
    for w, h in sizes:
        th, tw = smart(h, w, factor=32, min_pixels=vlm.MIN_PIXELS, max_pixels=vlm.MAX_PIXELS)
        assert vlm.model_size(w, h) == (tw, th), (w, h)


def test_an_image_at_model_size_is_left_as_it_is():
    im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (768, 1024, 3), dtype=np.uint8))
    same = vlm.to_model_size(im)
    assert same.size == im.size and same.tobytes() == im.tobytes()
    big = vlm.to_model_size(Image.new("RGB", (1600, 1200)))
    assert big.size == (1600, 1184) and big.width * big.height <= vlm.MAX_PIXELS
    assert vlm.visual_tokens(*big.size) == 50 * 37


def test_the_processor_reads_a_model_size_image_as_it_would_resize_it():
    """Told not to resize, Qwen's processor reads our image exactly as it
    would after its own resize: the set's images reach the model unchanged."""
    transformers = pytest.importorskip("transformers")
    from bestbefore.config import checkpoint_dir
    folder = checkpoint_dir() / "pretrained" / "Qwen3-VL-2B-Instruct"
    if not (folder / "preprocessor_config.json").exists():
        pytest.skip("Qwen3-VL's processor files aren't here")
    proc = transformers.AutoImageProcessor.from_pretrained(folder)
    rng = np.random.default_rng(1)
    im = vlm.to_model_size(Image.fromarray(rng.integers(0, 255, (700, 1000, 3), dtype=np.uint8)))
    own = proc(images=[im], return_tensors="np", size={"shortest_edge": vlm.MIN_PIXELS,
                                                       "longest_edge": vlm.MAX_PIXELS})
    unresized = proc(images=[im], return_tensors="np", do_resize=False)
    assert (own["image_grid_thw"] == unresized["image_grid_thw"]).all()
    assert np.array_equal(own["pixel_values"], unresized["pixel_values"])


# --- Answers from labels, rules and sidecars ---------------------------------------------------

def _store_entry(boxes, life=()):
    return {"part": "train", "month": "2026-08",
            "boxes": [{"index": k, "label": label, "value": value, "text": "", "turn": 0}
                      for k, (label, value) in enumerate(boxes)],
            "life": [{"xyxy": [0.1, 0.1, 0.5, 0.2], "text": t, "turn": 0} for t in life]}


def test_a_store_answer_takes_the_frozen_values():
    entry = _store_entry([("expiry", "2027-03-01"), ("mfg", "2025-03-02"), ("other_date", None)])
    assert vlm.store_answer(entry) == (vlm.Answer(Expiry(2027, 3, 1), Expiry(2025, 3, 2)), "printed")


def test_a_derived_store_expiry_is_the_mfg_plus_its_shelf_life():
    entry = _store_entry([("mfg", "2025-01-10")], life=["Best Before 24 months from Mfg."])
    assert vlm.store_answer(entry) == (vlm.Answer(Expiry(2027, 1, 10), Expiry(2025, 1, 10)), "derived")
    entry = _store_entry([("mfg", "2024-12")], life=["BEST BEFORE 18 MONTHS FROM MANUFACTURE"])
    assert vlm.store_answer(entry)[0].expiry == Expiry(2026, 6)


def test_a_store_photo_without_an_expiry_answers_null():
    assert vlm.store_answer(_store_entry([("mfg", "2025-01")])) == (vlm.Answer(None, Expiry(2025, 1)), "none")
    # a shelf life with no mfg date in frame (the other-face jar) gives nothing to work out
    assert vlm.store_answer(_store_entry([], life=["Best Before 1 Year from Mfg. Dt."])) == (vlm.Answer(None, None), "none")


def test_two_expiry_values_on_one_photo_are_refused():
    with pytest.raises(ValueError):
        vlm.store_answer(_store_entry([("expiry", "2027-03"), ("expiry", "2027-04")]))


def _token(text, cls=None, role=None, parts=None):
    return {"text": text, "cls": cls, "role": role, "parts": parts, "box": [0, 0, 1, 1]}


def test_a_generated_blocks_answer_comes_from_its_sidecar():
    block = {"kind": "pair", "tokens": [_token("MFG", "prod"), _token("06.2025", "date", "mfg", [2025, 6, None]),
                                        _token("EXP.", "due"), _token("06.2028", "date", "expiry", [2028, 6, None])]}
    assert vlm.gen_answer(block) == vlm.Answer(Expiry(2028, 6), Expiry(2025, 6))
    block = {"kind": "single", "tokens": [_token("IMPORT", "prod"), _token("2026-01-02", "date", "import", [2026, 1, 2])]}
    assert vlm.gen_answer(block) == vlm.Answer(None, None)       # an import date is neither


def test_a_relative_blocks_expiry_is_worked_out():
    block = {"kind": "relative", "tokens": [_token("Best Before", "due"), _token("12 months from Mfg."),
                                            _token("MFG:", "prod"), _token("2030-03-28", "date", "mfg", [2030, 3, 28])]}
    assert vlm.gen_answer(block) == vlm.Answer(Expiry(2031, 3, 28), Expiry(2030, 3, 28))
    block["tokens"] = block["tokens"][2:]
    with pytest.raises(ValueError):
        vlm.gen_answer(block)


def _ann(*boxes):
    """ExpDate annotation entries: (cls, x1, y1, x2, y2[, transcription])."""
    out = []
    for cls, *rest in boxes:
        a = {"cls": cls, "bbox": list(rest[:4])}
        if len(rest) > 4:
            a["transcription"] = rest[4]
        out.append(a)
    return {"width": 1000, "height": 800, "ann": out}


def test_the_keyword_rule_takes_the_date_beside_a_due_mark():
    entry = _ann(("prod", 100, 100, 160, 120), ("date", 170, 100, 300, 120, "2024.01.05"),
                 ("due", 100, 140, 160, 160), ("date", 170, 140, 300, 160, "2026.01.04"))
    roles = vlm.expdate_roles(entry)
    assert (roles.answer, roles.why) == (vlm.Answer(Expiry(2026, 1, 4), Expiry(2024, 1, 5)), "due")


def test_a_due_mark_after_its_date_counts_too():
    # "2026.01.04 까지": the mark to the right of its date
    roles = vlm.expdate_roles(_ann(("date", 100, 100, 230, 120, "2026.01.04"), ("due", 236, 100, 280, 120)))
    assert (roles.answer, roles.why) == (vlm.Answer(Expiry(2026, 1, 4), None), "due")


def test_without_a_due_mark_the_only_date_or_else_the_latest():
    roles = vlm.expdate_roles(_ann(("date", 100, 100, 230, 120, "2026.01.04")))
    assert (roles.answer, roles.why) == (vlm.Answer(Expiry(2026, 1, 4), None), "only")
    roles = vlm.expdate_roles(_ann(("date", 100, 100, 230, 120, "2026.01.04"), ("date", 100, 400, 230, 420, "2027.05.06")))
    assert (roles.answer, roles.why) == (vlm.Answer(Expiry(2027, 5, 6), None), "latest")


def test_a_date_beside_a_prod_mark_is_never_the_expiry():
    roles = vlm.expdate_roles(_ann(("prod", 100, 60, 160, 80), ("date", 100, 100, 230, 120, "2026.01.04")))
    assert (roles.answer, roles.why) == (vlm.Answer(None, Expiry(2026, 1, 4)), "no expiry")


def test_photos_the_rule_cant_settle_are_dropped():
    two_dues = _ann(("due", 40, 100, 90, 120), ("date", 100, 100, 230, 120, "2026.01.04"),
                    ("date", 100, 130, 230, 150, "2027.01.04"), ("due", 236, 130, 280, 150))
    assert vlm.expdate_roles(two_dues).why == "dropped: two dates take one role"
    late_mfg = _ann(("prod", 40, 100, 90, 120), ("date", 100, 100, 230, 120, "2028.01.04"),
                    ("due", 40, 300, 90, 320), ("date", 100, 300, 230, 320, "2027.01.04"))
    assert vlm.expdate_roles(late_mfg).why == "dropped: the mfg comes after the expiry"
    bare = _ann(("date", 100, 100, 230, 120, "01.31"))       # 31 January, no year: the parser says 2031-01
    assert vlm.expdate_roles(bare).why == "dropped: a date of two numbers alone"
    unread = _ann(("date", 100, 100, 230, 120, "2027 SOON"))
    assert vlm.expdate_roles(unread).why == "dropped: a date doesn't parse"


def test_a_mark_far_off_its_line_or_column_is_not_beside_it():
    date = (100, 100, 200, 120)
    assert vlm.mark_gap(date, (40, 100, 90, 120)) == pytest.approx(0.5)
    assert vlm.mark_gap(date, (100, 60, 160, 80)) == pytest.approx(1.0)        # above
    assert vlm.mark_gap(date, (100, 200, 160, 220)) is None                     # 4 heights below
    assert vlm.mark_gap(date, (100 - 20 * 14, 100, 100 - 20 * 13, 120)) is None  # 13 heights to the left
    assert vlm.mark_gap(date, (300, 300, 360, 320)) is None


# --- Panels and regions ------------------------------------------------------------------------

def test_a_panel_holds_its_block_and_none_of_the_backgrounds_dates():
    rng = np.random.default_rng(0)
    for _ in range(400):
        w, h = (int(v) for v in rng.integers(400, 1600, 2))
        bx, by = rng.uniform(0, w - 200), rng.uniform(0, h - 100)
        place = (bx, by, bx + rng.uniform(30, 200), by + rng.uniform(15, 100))
        foreign = []
        for _ in range(int(rng.integers(0, 6))):
            fx, fy = rng.uniform(0, w - 120), rng.uniform(0, h - 30)
            f = (fx, fy, fx + rng.uniform(20, 120), fy + rng.uniform(8, 30))
            if not vlm.overlaps(f, place):
                foreign.append(f)
        rect = vlm.panel(place, (w, h), foreign, rng)
        assert rect is not None             # nothing overlaps the block, so a panel can always be had
        assert vlm.inside(place, rect)
        assert 0 <= rect[0] and 0 <= rect[1] and rect[2] <= w and rect[3] <= h
        assert not any(vlm.overlaps(rect, f) for f in foreign)


def test_no_panel_when_a_background_date_overlaps_the_block():
    rng = np.random.default_rng(0)
    assert vlm.panel((100, 100, 200, 140), (800, 600), [(150, 120, 260, 150)], rng) is None


def test_cut_back_gives_up_when_a_foreign_box_sits_between_kept_ones():
    assert vlm.cut_back((0, 0, 100, 100), [(10, 10, 20, 20), (70, 70, 80, 80)], [(40, 40, 50, 50)]) is None


def test_an_empty_region_keeps_clear_of_every_date():
    rng = np.random.default_rng(3)
    dates = [(100, 100, 300, 130), (500, 600, 700, 640)]
    for _ in range(100):
        rect = vlm.empty_region((1000, 800), dates, rng)
        if rect is None:
            continue
        assert not any(vlm.overlaps(rect, vlm._grown(d, 0.25 * (d[3] - d[1]))) for d in dates)


def test_an_mfg_region_holds_the_mfg_date_and_its_mark_only():
    rng = np.random.default_rng(4)
    mfg, mark, expiry = (200, 300, 330, 320), (140, 300, 190, 320), (200, 340, 330, 360)
    for _ in range(100):
        rect = vlm.mfg_region((1000, 800), mfg, mark, [expiry], rng)
        assert rect is not None and vlm.inside(mfg, rect) and vlm.inside(mark, rect)
        assert not vlm.overlaps(rect, expiry)


# --- Examples -----------------------------------------------------------------------------------

def test_an_example_round_trips(tmp_path):
    jpeg = vlm.encode(Image.new("RGB", (320, 256), (200, 10, 10)))
    with vlm.Writer(tmp_path / "set", batch=2) as w:
        for i in range(5):
            w.add(jpeg, vlm.Answer(Expiry(2027, i + 1), None), "gen", f"gen_v2/{i:06d}.jpg#b0", {"block": "pair"})
    from bestbefore.reader import shards
    env = shards.open_pile(tmp_path / "set")
    got = list(vlm.examples(env))
    assert shards.count(env) == 5 and [e.key for e in got] == [vlm.key_of(i) for i in range(5)]
    assert got[3].answer == vlm.Answer(Expiry(2027, 4), None) and got[3].image().size == (320, 256)
    assert vlm.get(env, 3) == got[3]
    assert all(vlm.example_problems(e) == [] for e in got)


def test_an_image_off_the_model_size_is_refused():
    with pytest.raises(ValueError):
        vlm.encode(Image.new("RGB", (300, 200)))
    ex = vlm.Example("vlm/000000000", vlm.shards.encode_jpeg(Image.new("RGB", (300, 200))), vlm.Answer(None, None),
                     "gen", "x")
    assert vlm.example_problems(ex) == ["vlm/000000000: (300, 200) isn't a model size"]


def test_an_epoch_lists_the_store_three_times():
    order = vlm.epoch(["store", "gen", "expdate-real", "store"], np.random.default_rng(0))
    assert sorted(order) == [0, 0, 0, 1, 2, 3, 3, 3]


# --- The store's answers against the frozen labels (the laptop's private data) -------------------

def test_store_answers_match_the_frozen_labels():
    """Every exported photo's answer is its frozen expiry (printed or
    derived) and mfg; null exactly on the other-face photos."""
    from bestbefore import store
    from bestbefore.reader import piles
    names_path = store.DATA_DIR / "train_names.json"
    if not names_path.exists() or piles.store_missing():
        pytest.skip("the store's labels and export are the laptop's")
    names = json.loads(names_path.read_text())["photos"]
    labels = {n: lab for v in (store.V1, store.V3) for n, lab in store.labels(v).items()}
    text = json.loads(piles.store_text_path().read_text())
    for number, entry in text["photos"].items():
        lab = labels[names[number]["photo"]]
        answer, how = vlm.store_answer(entry)
        if lab["tag"] == "other-face":
            assert answer.expiry is None and how == "none", number
        else:
            assert str(answer.expiry) == lab["expiry"], number
            assert how == ("derived" if lab["derived"] else "printed"), number
        assert answer.mfg is None or str(answer.mfg) == lab["mfg"], number
