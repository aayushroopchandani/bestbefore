"""Camera effects on the whole photo (docs/m2-synthetic-generator.md, "Camera
effects"): each by its share, each as strong as the spec says and no more.
"""
import math

import numpy as np
import pytest

from bestbefore.synth import camera
from bestbefore.synth.spec import CAMERA


def _flat(value: float = 128.0, w: int = 400, h: int = 300) -> np.ndarray:
    return np.full((h, w, 3), value, np.float32)


def test_draw_follows_the_spec():
    rng = np.random.default_rng(0)
    drawn = [camera.draw(rng, 1600, 1200, [(100, 100, 300, 160)]) for _ in range(10_000)]
    for name in ("uneven", "dim", "glare", "defocus", "motion", "cast"):
        assert abs(np.mean([name in e for e in drawn]) - getattr(CAMERA, name)) < 0.03, name
    noisy = CAMERA.noise + (1 - CAMERA.noise) * CAMERA.dim           # dim light brings noise too
    assert abs(np.mean(["noise" in e for e in drawn]) - noisy) < 0.03
    assert all("noise" in e for e in drawn if "dim" in e)
    for e in drawn:
        assert CAMERA.defocus_sigma[0] <= e.get("defocus", 0.3) <= CAMERA.defocus_sigma[1]
        assert CAMERA.noise_sigma[0] <= e.get("noise", 2.0) <= CAMERA.noise_sigma[1]
        assert CAMERA.dim_level[0] <= e.get("dim", 0.5) <= CAMERA.dim_level[1]
        assert 0.10 <= abs(e["uneven"]["ramp"]) <= 0.25 if "uneven" in e else True
        assert all(abs(c) <= CAMERA.cast_shift for c in e.get("cast", ()))
        assert 3 <= e.get("motion", {"length": 3})["length"] <= 7
        for g in e.get("glare", ()):
            assert 0.05 * 1200 <= g["radius"] <= 0.25 * 1200
    glares = [g for e in drawn for g in e.get("glare", ())]
    near = np.mean([100 <= g["centre"][0] <= 300 and 100 <= g["centre"][1] <= 160 for g in glares])
    assert 0.45 <= near <= 0.6                     # half near the block, the rest anywhere (a few land on it)


def test_no_effect_leaves_the_photo_as_it_is():
    photo = np.random.default_rng(0).uniform(0, 255, (50, 60, 3)).astype(np.float32)
    assert np.array_equal(camera.apply(photo.copy(), {}, np.random.default_rng(0)), photo)


def test_the_same_seed_gives_the_same_photo():
    photo = np.random.default_rng(1).uniform(0, 255, (120, 160, 3)).astype(np.float32)
    effects = camera.draw(np.random.default_rng(3), 160, 120, [])
    effects |= {"noise": 4.0, "defocus": 0.8}
    a = camera.apply(photo.copy(), effects, np.random.default_rng(9))
    b = camera.apply(photo.copy(), effects, np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_noise_has_the_sigma_asked_for():
    out = camera.apply(_flat(), {"noise": 4.0}, np.random.default_rng(0))
    assert out.std() == pytest.approx(4.0, rel=0.03) and abs(out.mean() - 128) < 0.1


@pytest.mark.parametrize("kind", ["linear", "radial"])
def test_uneven_light_stays_within_its_ramp(kind):
    u = {"kind": kind, "ramp": 0.25, "angle": 30.0, "centre": [100.0, 50.0]}
    f = camera.ramp(u, 400, 300)
    assert f.min() >= 0.75 - 1e-5 and f.max() <= 1.25 + 1e-5
    assert f.max() - f.min() > 0.45                                     # it does span the ramp
    if kind == "radial":
        assert f[50, 100] == pytest.approx(1.25, abs=1e-3)               # brightest at its centre


def test_dim_light_scales_the_brightness():
    out = camera.apply(_flat(200.0), {"dim": 0.6}, np.random.default_rng(0))
    assert np.allclose(out, 120.0)


def test_glare_brightens_its_centre_and_fades():
    g = {"centre": [200.0, 150.0], "radius": 60.0, "aspect": 0.5, "angle": 0.0, "strength": 0.8}
    out = camera.apply(_flat(100.0), {"glare": [g]}, np.random.default_rng(0))
    assert out[150, 200, 0] == pytest.approx(100 + 0.8 * 155, abs=0.5)    # towards white by its strength
    assert out[150, 200 + 90, 0] < 105 and out[150 + 45, 200, 0] < 105    # 3 sigmas out: nearly nothing
    assert out[10, 10, 0] == 100.0


@pytest.mark.parametrize("length,angle", [(3, 0.0), (7, 45.0), (6, 100.0)])
def test_a_motion_streak_sums_to_one_and_has_its_length(length, angle):
    k = camera.motion_kernel(length, angle)
    assert k.sum() == pytest.approx(1.0) and k.shape[0] % 2 == 1
    ys, xs = np.nonzero(k > 1e-3)
    span = math.hypot(xs.max() - xs.min(), ys.max() - ys.min())
    assert length - 1.5 <= span <= length + 1.5


def test_defocus_softens_and_the_cast_tints():
    checks = np.indices((60, 60)).sum(axis=0) % 2 * 200.0
    photo = np.repeat(checks[..., None], 3, axis=2).astype(np.float32)
    assert camera.apply(photo.copy(), {"defocus": 1.0}, np.random.default_rng(0)).std() < 0.5 * photo.std()
    tinted = camera.apply(_flat(100.0), {"cast": [0.08, 0.0, -0.08]}, np.random.default_rng(0))
    assert np.allclose(tinted[0, 0], (108.0, 100.0, 92.0))
