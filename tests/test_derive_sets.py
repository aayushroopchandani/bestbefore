"""scripts/derive_sets.py --only rot180 (docs/m2b-store-and-size.md, S9):
M2b's three 180° sets, written under $GENERATED_DIR/m2b/ with M2's
derived sets, lists and manifest left exactly as they were.

The photos are a made-up ExpDate (conftest.build_fake_expdate).
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bestbefore import derived, m2, m2b, sets
from bestbefore.images import load_upright


def _fingerprint(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


class _InProcess:
    """Stands in for multiprocessing.Pool: the script's workers can't import
    a script loaded by path, so here every photo is made in this process."""

    def __init__(self, workers: int):
        self.workers = workers

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def starmap(self, fn, tasks):
        return list(itertools.starmap(fn, tasks))


def _run(script, monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["derive_sets.py", *args])
    monkeypatch.setattr(script, "Pool", _InProcess)
    script.main()


@pytest.fixture
def expdate(fake_expdate, monkeypatch):
    """The made-up ExpDate with a train_synth.txt: its two clean photos and
    the 14 copies, which derive_sets.py leaves out (only names: they are
    dropped before any is opened)."""
    yolo = fake_expdate.parent / "expdate" / "yolo"
    names = ["img_00001.jpg", "img_00002.jpg", *sorted(m2.REUSED_VAL_COPIES)]
    (yolo / "train_synth.txt").write_text("".join(f"./images/synth/{n}\n" for n in names))
    return yolo


def test_rot180_writes_m2bs_three_sets_and_leaves_m2s_alone(expdate, load_script, monkeypatch, tmp_path):
    from bestbefore import runs
    script = load_script("derive_sets")
    count = itertools.count()               # each run its own folder: two can start in the same second
    monkeypatch.setattr(script, "start_run", lambda name, config, seed=None: runs.start_run(
        name, config, seed=seed, root=tmp_path / "runs" / str(next(count))))
    _run(script, monkeypatch, "--only", "lists", "--workers", "1")       # M2's own first, as on both machines,
    for kind, text in (("images", "a photo"), ("labels", "0 0.5 0.5 0.1 0.1\n")):  # and stand-ins for its turns
        stand_in = m2.derived_dir() / kind / "real_rot" / f"a.{'jpg' if kind == 'images' else 'txt'}"
        stand_in.parent.mkdir(parents=True)
        stand_in.write_text(text)
    derived.write_list(m2.list_path("real_rot"), [m2.derived_dir() / "images" / "real_rot" / "a.jpg"])
    before = _fingerprint(m2.root())
    assert len(before) == 5 and "manifests/derived.json" in before
    _run(script, monkeypatch, "--only", "rot180", "--workers", "2")
    assert _fingerprint(m2.root()) == before                              # nothing of M2's written

    manifest = json.loads(m2b.manifest_path("derived").read_text())
    sizes = {"real_val_rot180": 3, "real_rot180": 2, "synth_rot180": 2}
    assert manifest["sets"].keys() == sizes.keys()
    for name, n in sizes.items():
        entry = manifest["sets"][name]
        assert (entry["photos"], entry["turned"], entry["worst_box_error_px"]) == (n, {"180": n}, 0.0), name
        assert m2b.list_missing(name) is None, name
        listed = derived.read_list(m2b.list_path(name))
        assert len(listed) == n and all(p.parent == m2b.derived_dir() / "images" / name for p in listed)
    assert not m2.list_path("real_rot180").exists() and not (m2.derived_dir() / "images" / "real_rot180").exists()

    # each photo is its source turned a half turn, and so are its boxes
    for name, source in (("real_val_rot180", "val"), ("real_rot180", "train_real")):
        for src, out in zip(derived.read_list(expdate / f"{source}.txt"), derived.read_list(m2b.list_path(name))):
            assert out.name == src.name
            a = np.asarray(load_upright(src).transpose(Image.Transpose.ROTATE_180), np.float32)
            b = np.asarray(load_upright(out), np.float32)
            assert a.shape == b.shape and np.abs(a - b).mean() < 3        # JPEG at quality 95, as M2's turns
            turned = [(bx.label, derived.turn_box(bx.xyxy, 2)) for bx in derived.read_boxes(src)]
            back = [(bx.label, bx.xyxy) for bx in derived.read_boxes(out)]
            assert [l for l, _ in turned] == [l for l, _ in back]
            assert all(np.allclose(p, q, atol=1e-6) for (_, p), (_, q) in zip(turned, back))
    synth = [p.name for p in derived.read_list(m2b.list_path("synth_rot180"))]
    assert synth == ["img_00001.jpg", "img_00002.jpg"]                    # without the 14 copies

    samples = sets.load_set("real-val-rot180")
    assert [s.id for s in samples] == ["v1.jpg", "v2.jpg", "v3.jpg"]
    assert sum(b.label == "date" for s in samples for b in s.boxes) == 1   # v1's one date, turned
    run = next((tmp_path / "runs").rglob("*-m2b-derive"))
    assert json.loads((run / "manifest.json").read_text()) == manifest

    _run(script, monkeypatch, "--only", "rot180", "--workers", "1")         # again: the same photos and labels
    assert _fingerprint(m2.root()) == before
