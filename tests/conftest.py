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
