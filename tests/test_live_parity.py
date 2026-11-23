"""M2c S1's parity run (docs/m2c-live-boxes.md, S1): rect reaching
Ultralytics, the padding each photo is checked for, the rules read off a
run's records, and the CPU job's steps."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from bestbefore import detector, live

T = 0.1641


def record(confs, ap, expiry, set_name, variant):
    """A parity record holding only what the rules read."""
    return {"set": set_name, "variant": variant, "threshold": T, "ap50_date": ap,
            "dates": {"conf": confs, "expiry": expiry}}


def test_rect_reaches_ultralytics_and_square_stays_the_default():
    class Model:
        rect = None

        def predict(self, source, imgsz, conf, device, rect, verbose):
            self.rect = rect
            boxes = SimpleNamespace(xyxyn=np.zeros((0, 4)), cls=np.zeros(0), conf=np.zeros(0))
            return [SimpleNamespace(boxes=boxes, names={0: "date"}) for _ in source]

    model, image = Model(), Image.new("RGB", (40, 20))
    detector.predict(model, [image], imgsz=64, rect=True)
    assert model.rect is True
    detector.predict(model, [image], imgsz=64)
    assert model.rect is False


def test_the_parity_run_refuses_padding_it_wasnt_asked_for(load_script):
    script = load_script("live_parity")
    assert script.padding_problems([[1, 3, 1280, 1280]], False, 1280, 32) == []
    assert script.padding_problems([[1, 3, 1280, 736]], False, 1280, 32)
    assert script.padding_problems([[1, 3, 1280, 736], [1, 3, 960, 1280]], True, 1280, 32) == []
    assert script.padding_problems([[1, 3, 1280, 1280]], True, 1280, 32) == []      # a square photo is square tight
    assert script.padding_problems([[1, 3, 1280, 730]], True, 1280, 32)
    assert script.padding_problems([[1, 3, 640, 480]], True, 1280, 32)


def test_the_job_checks_the_files_first_and_keeps_everything_owner_only():
    text = (Path(__file__).resolve().parents[1] / "slurm" / "m2c_cpu.sbatch").read_text()
    task = text.split("parity)\n", 1)[1].split(";;", 1)[0]
    steps = [line.strip() for line in task.splitlines() if "scripts/" in line]
    assert steps[0] == "python scripts/export_live.py check"
    assert steps[1] == 'python scripts/export_live.py int8 --threads "$SLURM_CPUS_PER_TASK"'
    assert "live_parity.py run --threads" in steps[2] and "live_parity.py rules" in steps[3]
    assert task.strip().startswith("umask 077") and "export YOLO_AUTOINSTALL=false" in text
    settings = dict(line[len("#SBATCH --"):].split("=", 1) for line in text.splitlines() if line.startswith("#SBATCH --"))
    assert (settings["partition"], settings["cpus-per-task"], settings["mem"], settings["time"]) == \
        ("short", "24", "32G", "02:00:00")
    assert "set -euo pipefail" in text and "squeue --me" in text


def test_the_parity_scripts_rules_read_each_set_and_wait_for_all_five(load_script, tmp_path):
    script = load_script("live_parity")
    good = {"pt-square": 0.96, "fp32-square": 0.96, "fp32-tight": 0.955,
            "int8-square": 0.90, "int8-tight": 0.90, "int8-pct-square": 0.955, "int8-pct-tight": 0.952}
    for name in live.SETS:
        for variant, ap in good.items():
            (tmp_path / f"parity-{name}-{variant}.json").write_text(
                json.dumps(record([0.9, 0.5], ap, [True, False], name, variant)))
    (tmp_path / "run.json").write_text("{}")
    out = script.rules(tmp_path)
    assert out["R0"] is True and out["R1"] is True
    assert out["R2 int8 tight"] is False and out["R2 int8-pct tight"] is True    # read at R1's padding
    assert out["R2 webassembly precision (accuracy half)"] == "int8-pct"          # the fallback, once
    assert json.loads((tmp_path / "rules.json").read_text())["R0"] is True
    (tmp_path / "parity-real-val-small-fp32-tight.json").unlink()
    out = script.rules(tmp_path)
    assert out["R0"] is True and out["R1"].startswith("incomplete: no real-val-small")
    assert "R2 webassembly precision (accuracy half)" not in out
