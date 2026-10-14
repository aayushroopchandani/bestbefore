import json

import pytest

from bestbefore import config, runs


def test_missing_variable_names_itself(monkeypatch):
    monkeypatch.delenv("SCANIFY_DIR", raising=False)
    with pytest.raises(RuntimeError, match="SCANIFY_DIR"):
        config.scanify_dir()


def test_relative_paths_are_from_the_repo_root(monkeypatch):
    monkeypatch.setenv("EXPDATE_DIR", "data/expdate")
    assert config.expdate_dir() == config.REPO_ROOT / "data" / "expdate"


def test_checkpoint_dir(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", "data/checkpoints")
    assert config.checkpoint_dir() == config.REPO_ROOT / "data" / "checkpoints"


def test_run_record(monkeypatch, tmp_path):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    run_dir = runs.start_run("demo", {"split": "real-test"}, seed=0)
    record = json.loads((run_dir / "run.json").read_text())
    assert run_dir.name.endswith("-demo")
    assert record["seed"] == 0
    assert record["config"] == {"split": "real-test"}
    assert len(record["git_commit"]) == 40
    assert "numpy" in record["packages"]
