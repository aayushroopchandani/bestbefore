import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def load_script():
    """Import a script from scripts/ as a module, to test its functions."""
    def load(name: str):
        spec = importlib.util.spec_from_file_location(f"scripts_{name}", SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module          # dataclasses look their module up here
        spec.loader.exec_module(module)
        return module
    return load


def _texture(w: int, h: int, seed: int):
    """A photo-like surface: smooth light and dark patches, some grain."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(40, 230, (6, 8, 3)).astype(np.uint8)
    im = np.asarray(Image.fromarray(coarse).resize((w, h), Image.Resampling.BICUBIC), np.float32)
    return Image.fromarray(np.clip(im + rng.normal(0, 6, im.shape), 0, 255).astype(np.uint8))


def build_fake_expdate(root: Path, mp: pytest.MonkeyPatch) -> Path:
    """A made-up ExpDate (training, synthetic, validation and test photos
    with label files, and their lists) and $GENERATED_DIR under root, the
    real fonts linked in, the environment pointing at them; returns
    $GENERATED_DIR. Lists name their photos relative to themselves, so a
    copied tree still works. v3.jpg (590 px) is too small to be a
    background; img_00948.jpg is one of the 14 copies of a validation photo."""
    from bestbefore import derived
    from bestbefore.samples import Box
    from bestbefore.synth import fonts
    real_fonts = fonts.font_dir()
    photos = {  # name: (folder, size, boxes); validation photos sit in real-train, as ExpDate's do
        "a.jpg": ("real-train", (1200, 900), (Box("date", (0.1, 0.1, 0.3, 0.14)),)),
        "b.jpg": ("real-train", (800, 600), ()),
        "v1.jpg": ("real-train", (1100, 800), (Box("date", (0.5, 0.5, 0.7, 0.55)),)),
        "v2.jpg": ("real-train", (1050, 1000), ()),
        "v3.jpg": ("real-train", (590, 400), ()),
        "img_00001.jpg": ("synth", (1100, 1300), (Box("code", (0.4, 0.4, 0.6, 0.43)),)),
        "img_00002.jpg": ("synth", (700, 500), ()),
        "img_00948.jpg": ("synth", (800, 800), ()),
        "test_00001.jpg": ("real-test", (900, 700), ()),
    }
    lists = {"train_real": ("a.jpg", "b.jpg"), "val": ("v1.jpg", "v2.jpg", "v3.jpg"),
             "train_synth_clean": ("img_00001.jpg", "img_00002.jpg")}
    expdate, generated = root / "expdate", root / "generated"
    for k, (name, (folder, (w, h), boxes)) in enumerate(photos.items()):
        derived.write_yolo(expdate / "yolo", folder, name, _texture(w, h, k), boxes)
    for name, names in lists.items():
        path = (generated / "m2" / "lists" if name == "train_synth_clean" else expdate / "yolo") / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        images = [expdate / "yolo" / "images" / photos[n][0] / n for n in names]
        path.write_text("".join(f"./{os.path.relpath(p, path.parent)}\n" for p in images))
    (generated / "fonts").symlink_to(real_fonts)
    mp.setenv("EXPDATE_DIR", str(expdate))
    mp.setenv("GENERATED_DIR", str(generated))
    return generated


@pytest.fixture
def fake_expdate(tmp_path, monkeypatch) -> Path:
    """build_fake_expdate in this test's temporary folder."""
    return build_fake_expdate(tmp_path, monkeypatch)


@pytest.fixture(scope="session")
def fake_expdate_builder():
    """build_fake_expdate itself, for fixtures wider than one test."""
    return build_fake_expdate


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """A fake squeue and sbatch first on PATH. squeue lists our jobs 72914
    (running) and 72915 (waiting); sbatch writes its arguments to
    <bin>/sbatch.args and answers job 72999, or refuses as the 5-job cap
    does when SBATCH_REFUSE=1."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    commands = {
        "squeue": 'case "$*" in\n'
                  '  *%T*) echo "72914 RUNNING"; echo "72915 PENDING";;\n'
                  '  *) echo "72914"; echo "72915";;\n'
                  'esac\n',
        "sbatch": 'echo "$*" > "$(dirname "$0")/sbatch.args"\n'
                  'if [ "$SBATCH_REFUSE" = 1 ]; then\n'
                  '  echo "sbatch: error: QOSMaxSubmitJobPerUserLimit" >&2; exit 1\n'
                  'fi\n'
                  'echo 72999\n',
    }
    for name, body in commands.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir
