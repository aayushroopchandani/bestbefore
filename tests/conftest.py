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


# --- M3 S5: made-up crop piles --------------------------------------------------------

def render_crop(text: str, rng, size: int = 28, back=(200, 200, 200), ink=None, noise: float = 4.0):
    """A made-up stored crop: `text` printed on a plain surface with
    Gaussian noise, cut as scripts/make_crops.py cuts (crops.cut: half the
    box's short side of margin). Returns the crop (PIL) and the box in it."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from bestbefore.reader import crops
    font = ImageFont.load_default(size=size)
    ink = ink or tuple(max(0, c - 150) for c in back)
    l, t, r, b = font.getbbox(text)
    w, h = r - l + 2 * size, b - t + 2 * size
    image = Image.new("RGB", (w, h), back)
    ImageDraw.Draw(image).text((size - l, size - t), text, font=font, fill=ink)
    noisy = np.asarray(image).astype(np.float32) + rng.normal(0, noise, (h, w, 3))
    image = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    return crops.cut(image, (size, size, size + r - l, size + b - t))


def fake_text(rng) -> str:
    d, m, y = int(rng.integers(1, 29)), int(rng.integers(1, 13)), int(rng.integers(24, 32))
    return [f"{d:02d}/{m:02d}/20{y}", f"{m:02d}/20{y}", f"EXP {d:02d}.{m:02d}.{y}", f"L{d}{m}{y}"][int(rng.integers(4))]


# pile: (crops, designs or None). The store's designs hold 8, 3 and 1 crops.
FAKE_PILES = {"expdate-real": 10, "expdate-synth": 10, "gen": 20, "store": 12, "real-val": 8, "synth-val-v2": 8}
FAKE_DESIGNS = ("d01",) * 8 + ("d02",) * 3 + ("d03",)


def write_measure(folder: Path, values) -> None:
    """A pile's visibility as scripts/visibility.py writes it, with given values."""
    import json
    import numpy as np
    from bestbefore.reader import data, piles
    np.save(folder / data.VISIBILITY_FILE, np.asarray(values, np.float32))
    (folder / data.VISIBILITY_RECORD).write_text(json.dumps({
        "count": len(values), "code": data.visibility_code(),
        "manifest_sha256": piles.sha256_file(folder / "manifest.json")}))


def build_fake_piles(root: Path, seed: int = 0, sizes: dict | None = None) -> Path:
    """Made-up piles under root (each an LMDB with a manifest), training
    piles measured by data.visibility; returns root."""
    import json
    import numpy as np
    from bestbefore.reader import data, piles, shards
    rng = np.random.default_rng(seed)
    for pile, n in (sizes or FAKE_PILES).items():
        folder = root / pile
        values = []
        with shards.Writer(folder, pile) as w:
            for i in range(n):
                text = fake_text(rng)
                crop, box = render_crop(text, rng)
                tags = {"upright": 0}
                if pile == "store":
                    tags |= {"design": FAKE_DESIGNS[i % len(FAKE_DESIGNS)], "part": "train", "tag": "clean"}
                w.add(jpeg=shards.encode_jpeg(crop), box=box, text=text, cls="date", source=f"{pile}#{i}",
                      tags=tags)
                values.append(data.visibility(np.asarray(crop), box))
        (folder / "manifest.json").write_text(json.dumps({"pile": pile, "count": n}))
        if pile in piles.VISIBILITY_FILTERED:
            write_measure(folder, values)
    return root


@pytest.fixture(scope="session")
def fake_piles(tmp_path_factory) -> Path:
    """Made-up crop piles for M3's training stream, shared by a session's tests (read only)."""
    return build_fake_piles(tmp_path_factory.mktemp("piles"))
