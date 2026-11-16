"""M3's training stream and the visibility measure (reader/data.py,
docs/m3-date-reader.md "The mix", "Augmentation", S5)."""
import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from bestbefore import m3
from bestbefore.reader import charset, crops, data, onnx_run, shards
from conftest import FAKE_DESIGNS, render_crop, write_measure


# --- The visibility measure -------------------------------------------------------------

GREY = (128, 128, 128)


def _measured(text="12/2027", **kw) -> float:
    crop, box = render_crop(text, np.random.default_rng(7), **kw)
    return data.visibility(np.asarray(crop), box)


def test_visibility_falls_with_the_prints_contrast_under_fixed_noise():
    values = [_measured(back=GREY, ink=tuple(128 - c for c in (k, k, k)), noise=6.0) for k in (96, 48, 24, 12, 6, 0)]
    assert values == sorted(values, reverse=True) and len(set(values)) == len(values)
    assert values[0] > 10                       # clear print: far over noise
    assert 0.5 < values[-1] < 1.6               # no print, white noise alone: about 1


def test_print_visible_in_colour_only_still_scores():
    back, ink = (90, 140, 120), (200, 80, 150)  # the same luma (Rec. 601), different colours
    luma = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    assert abs(luma(back) - luma(ink)) < 1.5
    crop, box = render_crop("12/2027", np.random.default_rng(7), back=back, ink=ink, noise=6.0)
    colour = data.visibility(np.asarray(crop), box)
    grey = np.asarray(crop.convert("L").convert("RGB"))         # the same crop, colour taken away
    assert colour > 5 and colour > 4 * data.visibility(grey, box)


def test_visibility_ignores_slow_shading_and_a_crops_size():
    crop, box = render_crop("12/2027", np.random.default_rng(7), back=GREY, ink=(78, 78, 78), noise=3.0)
    a = np.asarray(crop).astype(np.float32)
    shaded = np.clip(a + np.linspace(-60, 60, a.shape[1])[None, :, None], 0, 255).astype(np.uint8)
    flat = data.visibility(np.asarray(crop), box)
    assert data.visibility(shaded, box) == pytest.approx(flat, rel=0.25)
    big = np.asarray(crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC))
    assert data.visibility(big, tuple(2 * v for v in box)) == pytest.approx(flat, rel=0.25)


def test_a_flat_crop_scores_zero():
    assert data.visibility(np.full((40, 120, 3), 3, np.uint8), (20, 10, 100, 30)) == 0.0


def test_a_measure_is_read_only_if_it_fits_its_pile(fake_piles, tmp_path):
    folder = fake_piles / "gen"
    n = shards.count(shards.open_pile(folder))
    assert len(data.read_visibility(folder, n)) == n
    with pytest.raises(RuntimeError, match="records"):
        data.read_visibility(folder, n + 1)
    other = tmp_path / "gen"
    other.mkdir()
    for f in ("manifest.json", data.VISIBILITY_FILE, data.VISIBILITY_RECORD):
        (other / f).write_bytes((folder / f).read_bytes())
    (other / "manifest.json").write_text('{"remade": true}')
    with pytest.raises(RuntimeError, match="changed"):
        data.read_visibility(other, n)
    write_measure(other, np.ones(n))
    record = json.loads((other / data.VISIBILITY_RECORD).read_text())
    (other / data.VISIBILITY_RECORD).write_text(json.dumps(record | {"code": "an older measure"}))
    with pytest.raises(RuntimeError, match="another version"):
        data.read_visibility(other, n)


# --- What may be drawn --------------------------------------------------------------------

def test_training_refuses_without_a_cut_off(fake_piles):
    with pytest.raises(RuntimeError, match="VISIBLE_MIN"):
        data.pile_index(fake_piles, "gen", None)
    assert data.pile_index(fake_piles, "store", None).count == 12     # the store is never filtered


def test_a_crop_under_the_cut_off_is_never_drawn(fake_piles, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "gen").symlink_to(fake_piles / "gen")
    values = data.read_visibility(fake_piles / "gen", 20)
    cut = float(np.sort(values)[5])                      # the 5 lowest fall under it
    ix = data.pile_index(fake_piles, "gen", cut)
    assert sorted(ix.kept.tolist()) == sorted(np.flatnonzero(values >= cut).tolist()) and len(ix.kept) == 15
    mix = data.Mix({"gen": ix})
    rng = np.random.default_rng(0)
    drawn = Counter(mix.draw(rng)[1] for _ in range(3000))
    assert set(drawn) == set(ix.kept.tolist())           # every kept crop, never a dropped one


def test_the_mix_draws_piles_by_weight_normalised_over_the_arm(fake_piles):
    for arm, expected in [("all", {"expdate-real": 0.10, "expdate-synth": 0.20, "gen": 0.695, "store": 0.005}),
                          ("exp", {"expdate-real": 1 / 3, "expdate-synth": 2 / 3})]:
        mix = data.Mix.for_arm(fake_piles, arm, visible_min=0.0)
        assert dict(zip(mix.names, mix.shares)) == pytest.approx(expected)
        rng = np.random.default_rng(1)
        counts = Counter(mix.names[mix.draw(rng)[0]] for _ in range(40_000))
        for name, share in expected.items():
            assert counts[name] / 40_000 == pytest.approx(share, abs=0.01 if share > 0.05 else 0.002)


def test_the_store_picks_a_design_first(fake_piles):
    ix = data.pile_index(fake_piles, "store", None)
    assert sorted(len(d) for d in ix.designs) == [1, 3, 8]
    mix = data.Mix({"store": ix})
    rng = np.random.default_rng(2)
    design = {i: FAKE_DESIGNS[i] for i in range(12)}
    counts = Counter(design[mix.draw(rng)[1]] for _ in range(9000))
    for d in ("d01", "d02", "d03"):                     # 8, 3 and 1 crops, each design a third
        assert counts[d] / 9000 == pytest.approx(1 / 3, abs=0.03)


def test_draws_are_seeded_per_run_and_per_worker(fake_piles):
    mix = data.Mix.for_arm(fake_piles, "all", visible_min=0.0)

    def draws(seed):            # the stream seeds each worker's generator with [run seed, worker]
        rng = np.random.default_rng(seed)
        return [mix.draw(rng) for _ in range(200)]
    assert draws([0, 0]) == draws([0, 0])
    assert draws([0, 0]) != draws([0, 1])               # another worker
    assert draws([0, 0]) != draws([1, 0])               # another run's seed


def test_the_stream_is_the_same_for_a_seed(fake_piles):
    mix = data.Mix.for_arm(fake_piles, "all", visible_min=0.0)
    first = lambda seed: next(iter(data.Stream(fake_piles, mix, 8, 0.0, seed, pool=2)))
    a, b, c = first(3), first(3), first(4)
    assert all(torch_equal(a[k], b[k]) for k in a)
    assert not torch_equal(a["x"], c["x"]) or not torch_equal(a["targets"], c["targets"])


def torch_equal(a, b) -> bool:
    return a.shape == b.shape and bool((a == b).all())


def test_a_batch_holds_the_crops_and_their_texts_as_classes(fake_piles):
    mix = data.Mix.for_arm(fake_piles, "all", visible_min=0.0)
    batch = next(iter(data.Stream(fake_piles, mix, 8, 0.0, 0, augment=None, pool=1)))
    assert batch["x"].shape[:3] == (8, 3, 48) and batch["x"].shape[3] % 8 == 0
    assert int(batch["lengths"].sum()) == len(batch["targets"])
    assert int(batch["targets"].min()) > charset.BLANK
    assert set(batch["piles"].tolist()) <= set(range(len(mix.names)))
    assert max(batch["widths"].tolist()) <= batch["x"].shape[3]


# --- Views, augmentation, buckets ----------------------------------------------------------

def test_padding_and_jitter_stay_inside_the_margin():
    rng = np.random.default_rng(5)
    photo = Image.new("RGB", (3000, 3000), (120, 120, 120))
    worst = m3.EDGE_JITTER
    for _ in range(300):
        h = rng.uniform(4, 150)
        w = h * rng.uniform(0.3, 15)
        x, y = rng.uniform(500, 2400 - w if w < 1900 else 500), rng.uniform(500, 2300)
        crop, box = crops.cut(photo, (x, y, x + w, y + h))
        for pad in (0.0, 0.15, 0.4):                    # S2's paddings; m3.crop_recipe_used's is 0
            if pad + worst > crops.MARGIN:
                continue
            for j in (worst, -worst):
                left, top, right, bottom = crops.view_rect(box, (10**6, 10**6), pad, (j, j, j, j))
                assert left >= -1 and top >= -1 and right <= crop.width + 1 and bottom <= crop.height + 1
                clipped = crops.view_rect(box, crop.size, pad, (j, j, j, j))
                assert clipped[2] > clipped[0] and clipped[3] > clipped[1]


def test_a_view_without_augmentation_is_the_plain_view():
    crop, box = render_crop("08/2027", np.random.default_rng(0))
    rgb = np.asarray(crop)
    none = data.Augment(jitter=0.0, p_turn=0, p_shrink=0, p_blur=0, p_jpeg=0, p_colour=0, p_invert=0)
    assert np.array_equal(data.augmented_view(rgb, box, 0.0, np.random.default_rng(0), none),
                          data.plain_view(rgb, box, 0.0))


def test_augmented_views_are_crops_of_the_print():
    crop, box = render_crop("08/2027", np.random.default_rng(0))
    rgb = np.asarray(crop)
    always = data.Augment(p_turn=1, p_shrink=1, p_blur=1, p_jpeg=1, p_colour=1, p_invert=1)
    rng = np.random.default_rng(1)
    plain = data.plain_view(rgb, box, 0.0)
    for _ in range(20):
        v = data.augmented_view(rgb, box, 0.0, rng, always)
        assert v.dtype == np.uint8 and v.ndim == 3 and v.shape[2] == 3
        # jitter moves each edge by 10% of the height at most; a turn of 4 degrees boxes the
        # tilted print, taller by up to its width x sin 4 degrees
        assert 0.75 * plain.shape[0] <= v.shape[0] <= 1.25 * plain.shape[0] + 0.15 * plain.shape[1] + 2


def test_width_buckets_pad_less_than_random_batches():
    rng = np.random.default_rng(0)
    views = [np.zeros((int(rng.integers(16, 64)), int(rng.integers(20, 700)), 3), np.uint8) for _ in range(512)]
    bucketed = data.width_buckets(views, 32, rng)
    order = rng.permutation(len(views)).tolist()
    random_batches = [order[i:i + 32] for i in range(0, len(order), 32)]
    assert sorted(i for b in bucketed for i in b) == list(range(512))
    assert data.padded_share(views, bucketed) < 0.5 * data.padded_share(views, random_batches)


def test_suites_are_read_as_plain_views(fake_piles):
    views, texts = data.suite(fake_piles, "real-val", 0.0)
    assert len(views) == len(texts) == 8
    env = shards.open_pile(fake_piles / "real-val")
    first = shards.get(env, "real-val", 0)
    env.close()
    assert np.array_equal(views[0], data.plain_view(np.asarray(first.image()), first.box, 0.0))
    assert onnx_run.resized_width(views[0].shape[1], views[0].shape[0]) > 48


# --- scripts/visibility.py ---------------------------------------------------------------

class _InProcess:
    """Stands in for multiprocessing.Pool: workers can't import a script
    loaded by path (as in test_reader_piles.py)."""

    def __init__(self, workers: int):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def map(self, fn, tasks):
        return list(map(fn, tasks))


@pytest.fixture
def copied_piles(fake_piles, tmp_path):
    import shutil
    root = tmp_path / "crops"
    shutil.copytree(fake_piles, root)
    return root


def test_measure_writes_each_records_value_in_key_order(load_script, monkeypatch, copied_piles):
    script = load_script("visibility")
    monkeypatch.setattr(script, "Pool", _InProcess)
    (copied_piles / "gen" / data.VISIBILITY_FILE).unlink()
    record = script.measure("gen", copied_piles, workers=3)
    values = data.read_visibility(copied_piles / "gen", 20)
    env = shards.open_pile(copied_piles / "gen")
    expected = [data.visibility(np.asarray(c.image()), c.box) for c in shards.records(env, "gen")]
    env.close()
    assert values.dtype == np.float32 and values.tolist() == pytest.approx(expected, rel=1e-6)
    assert record["count"] == 20 and record["code"] == data.visibility_code()


def test_sheets_and_the_check(load_script, copied_piles):
    script = load_script("visibility")
    out = copied_piles / "sheets" / "visibility"
    paths = script.sheets("gen", copied_piles, out)
    assert [p.name for p in paths] == ["gen-bands.png", "gen-lowest.png"] and all(p.exists() for p in paths)
    with pytest.raises(SystemExit):
        script.sheets("store", copied_piles, out)              # the retailer's photos: never on these sheets
    values = data.read_visibility(copied_piles / "gen", 20)
    cut = float(np.sort(values)[3])                            # drops 3 of 20: over 10%, so STOP
    result = script.check(cut, ["gen"], copied_piles, out)
    assert result["piles"]["gen"]["dropped"] == 3 and result["stop"] == ["gen"]
    assert (out / f"gen-dropped-{cut:g}.png").exists() and (out / f"gen-kept-above-{cut:g}.png").exists()
    assert script.check(float(values.min()), ["gen"], copied_piles, out)["stop"] == []
