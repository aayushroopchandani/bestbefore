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


def test_generated_dir_expands_home(monkeypatch):
    monkeypatch.setenv("GENERATED_DIR", "~/bestbefore-data/generated")
    assert config.generated_dir() == config.Path.home() / "bestbefore-data" / "generated"


def test_run_record(monkeypatch, tmp_path):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    run_dir = runs.start_run("demo", {"split": "real-test"}, seed=0)
    record = json.loads((run_dir / "run.json").read_text())
    assert run_dir.name.endswith("-demo")
    assert record["seed"] == 0
    assert record["config"] == {"split": "real-test"}
    assert len(record["git_commit"]) == 40
    assert "numpy" in record["packages"]


def test_run_records_are_replaced_whole(monkeypatch, tmp_path):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    run_dir = runs.start_run("demo", {"a": 1})
    runs.update_run(run_dir, status="finished", path=tmp_path)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "finished" and record["path"] == str(tmp_path)
    assert record["config"] == {"a": 1} and isinstance(record["git_dirty"], bool)
    assert [p.name for p in run_dir.iterdir()] == ["run.json"]       # no temporary file left
