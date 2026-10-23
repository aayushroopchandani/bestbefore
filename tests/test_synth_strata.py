"""synth-val's strata (docs/m2-synthetic-generator.md, "synth-val", S6):
each changes one thing from `clean`, draws only held-out fonts on
validation photos, and gives every photo one target date, which loads as
`exp` with the stratum as the photo's tag.

The backgrounds are a made-up ExpDate (conftest.build_fake_expdate); the
tests that draw need the fonts (python scripts/fetch_fonts.py).
"""
import json
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

from bestbefore import derived
from bestbefore.synth import compose, fonts, make
from bestbefore.synth.spec import CLEAN, GEN, SYNTH_VAL, strata_hash

needs_fonts = pytest.mark.skipif(not fonts.fetched(), reason="fonts not fetched: python scripts/fetch_fonts.py")


def _leaves(spec, prefix="") -> dict:
    out = {}
    for f in fields(spec):
        value = getattr(spec, f.name)
        out |= _leaves(value, f"{prefix}{f.name}.") if is_dataclass(value) else {f"{prefix}{f.name}": value}
    return out


def _pools():
    return (compose.load_backgrounds(GEN.backgrounds, GEN.compose.min_side),
            compose.load_backgrounds(CLEAN.backgrounds, CLEAN.compose.min_side))


def test_each_stratum_changes_one_thing_from_clean():
    clean = _leaves(CLEAN)
    changed = {name: sorted(k for k, v in _leaves(spec).items() if v != clean[k]) for name, spec in SYNTH_VAL}
    assert changed["clean"] == []
    assert changed["sideways"] == ["geometry.orientation"]
    assert changed["month-word"] == ["text.month_word"]
    assert changed["small"] == ["size.bands"]
    assert changed["dot-matrix"] == ["style.styles"]
    assert changed["curved"] == ["geometry.surface"]
    assert len(changed["mixed"]) > 1                         # the training spec, with one target
    assert [name for name, _ in SYNTH_VAL] == ["clean", "sideways", "month-word", "small", "dot-matrix",
                                               "curved", "mixed"]
    assert len(strata_hash()) == 64 and strata_hash() != strata_hash(SYNTH_VAL[::-1])


def test_clean_is_what_the_plan_says_and_keeps_the_training_proportions():
    assert dict(CLEAN.geometry.orientation) == {"upright": 1.0} and dict(CLEAN.geometry.surface) == {"flat": 1.0}
    assert CLEAN.text.month_word == 0 and dict(CLEAN.size.bands) == {"normal": 1.0}
    styles = dict(CLEAN.style.styles)
    assert styles.keys() == {"solid", "label"} and styles["solid"] / styles["label"] == pytest.approx(40 / 15)
    surfaces = dict(dict(SYNTH_VAL)["curved"].geometry.surface)
    assert surfaces["cylinder"] / surfaces["bend"] == pytest.approx(15 / 10) and sum(surfaces.values()) == 1
    for name, spec in SYNTH_VAL:     # what the plan doesn't name follows the training spec
        assert (spec.camera, spec.compose.opacity, spec.geometry.perspective, spec.text.single_word) == \
               (GEN.camera, GEN.compose.opacity, GEN.geometry.perspective, GEN.text.single_word), name


def test_synth_val_alone_uses_the_held_out_fonts_and_the_validation_photos():
    assert GEN.fonts == "train" and dict(GEN.backgrounds).keys() == {"train_real", "train_synth_clean"}
    for name, spec in SYNTH_VAL:
        assert spec.fonts == "held-out" and spec.backgrounds == (("val", 1.0),), name
        assert spec.text.kinds == (("single", 1.0),) and spec.text.single_role == "target", name
        assert spec.compose.two_blocks == 0 and spec.compose.block_tries > 1, name
    train, held_out = ({f.name for f in fonts.families(s)} for s in ("train", "held-out"))
    assert train and held_out and not train & held_out
    assert GEN.compose.block_tries == 1 and GEN.text.single_role == "expiry"


def test_a_training_list_never_takes_a_validation_photo(fake_expdate):
    gen, val = _pools()
    assert [b.path.name for b in val.by_list["val"]] == ["v1.jpg", "v2.jpg"]        # v3: under 640 px
    assert [b.path.name for bs in gen.by_list.values() for b in bs] == ["a.jpg", "b.jpg", "img_00001.jpg",
                                                                         "img_00002.jpg"]   # in list order
    assert [(b.width, b.height, len(b.boxes)) for b in val.by_list["val"]] == [(1100, 800, 1), (1050, 1000, 0)]


@needs_fonts
def test_the_generator_draws_fonts_from_its_specs_split(fake_expdate, monkeypatch):
    asked = []
    real = fonts.random_family
    monkeypatch.setattr(fonts, "random_family", lambda rng, split: asked.append(split) or real(rng, split))
    gen_pool, val_pool = _pools()
    gen = [make.make_one(i, GEN, gen_pool, 1) for i in range(3)]
    assert set(asked) == {"train"}
    asked.clear()
    val = [make.make_one(i, spec, val_pool, 2, stratum=name) for i, (name, spec) in enumerate(SYNTH_VAL)]
    assert set(asked) == {"held-out"}
    train, held_out = ({f.name for f in fonts.families(s)} for s in ("train", "held-out"))
    assert {b["family"] for m in gen for b in m.sidecar["blocks"]} <= train
    assert {b["family"] for m in val for b in m.sidecar["blocks"]} <= held_out
    assert {m.sidecar["background"]["list"] for m in val} == {"val"}
    assert [m.sidecar["stratum"] for m in val] == [name for name, _ in SYNTH_VAL]
    assert all("stratum" not in m.sidecar for m in gen)


@needs_fonts
def test_a_skipped_block_is_drawn_afresh_only_where_the_spec_says(fake_expdate, monkeypatch):
    gen_pool, val_pool = _pools()
    real, calls = make._place, []

    def first_fails(*args, **kwargs):
        calls.append(1)
        drawn, spot = real(*args, **kwargs)
        return (drawn, None) if len(calls) == 1 else (drawn, spot)

    monkeypatch.setattr(make, "_place", first_fails)
    one = make.make_one(0, CLEAN, val_pool, 2, stratum="clean").sidecar
    assert (one["wanted"], len(one["blocks"]), len(one["skipped"])) == (1, 1, 1)
    assert [t["role"] for t in one["blocks"][0]["tokens"] if t["cls"] == "date"] == ["target"]
    calls.clear()
    spec = replace(GEN, compose=replace(GEN.compose, two_blocks=0.0))
    two = make.make_one(0, spec, gen_pool, 1).sidecar
    assert (two["wanted"], len(two["blocks"]), len(two["skipped"])) == (1, 0, 1)    # training: no second try


@needs_fonts
def test_every_stratum_gives_one_target_which_loads_as_exp_with_its_stratum(fake_expdate, tmp_path):
    _, val_pool = _pools()
    images = tmp_path / "gen" / "images" / "synth_val"
    images.mkdir(parents=True)
    derived.label_path(images / "x.jpg").parent.mkdir(parents=True)
    made = [make.make_one(i, spec, val_pool, 2, stratum=name) for i, (name, spec) in enumerate(SYNTH_VAL)]
    paths = [make.write_one(m, images) for m in made]
    derived.write_list(tmp_path / "synth_val.txt", paths)
    sidecar = tmp_path / "synth_val.jsonl"
    sidecar.write_text("".join(json.dumps(m.sidecar) + "\n" for m in made))
    samples = derived.load_yolo_set(tmp_path / "synth_val.txt", sidecar)
    assert [s.tag for s in samples] == [name for name, _ in SYNTH_VAL]
    for s, m in zip(samples, made):
        dates = [t for b in m.sidecar["blocks"] for t in b["tokens"] if t["cls"] == "date"]
        assert [t["role"] for t in dates] == ["target"]
        exp = [b for b in s.boxes if b.label == "exp"]
        assert len(exp) == 1
        w, h = m.sidecar["size"]
        x1, y1, x2, y2 = dates[0]["box"]
        assert np.allclose(exp[0].xyxy, (x1 / w, y1 / h, x2 / w, y2 / h), atol=2e-6)
    plain = derived.load_yolo_set(tmp_path / "synth_val.txt")         # without the sidecar: ordinary dates
    assert all(b.label != "exp" for s in plain for b in s.boxes) and all(s.tag is None for s in plain)
