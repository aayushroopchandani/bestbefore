"""Generating at scale and the check (docs/m2-synthetic-generator.md, S6).

A made-up ExpDate (conftest.build_fake_expdate) stands in for the real
one, so the pools are loaded through the real guards. The tests that draw
need the fonts (python scripts/fetch_fonts.py).
"""
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from bestbefore import derived, m2, sets
from bestbefore.synth import batch, fonts
from bestbefore.synth.batch import Layout
from bestbefore.synth.spec import GEN, SYNTH_VAL

needs_fonts = pytest.mark.skipif(not fonts.fetched(), reason="fonts not fetched: python scripts/fetch_fonts.py")

SMALL = Layout(gen=6, gen10k=4, task_size=3, per_stratum=1)      # 2 gen tasks, then 7 synth-val photos


@pytest.fixture(scope="module")
def made(tmp_path_factory, fake_expdate_builder) -> Path:
    """Every task of SMALL made once (in this process), unchecked: copied by
    the tests that break it."""
    if not fonts.fetched():
        pytest.skip("fonts not fetched")
    root = tmp_path_factory.mktemp("made")
    with pytest.MonkeyPatch.context() as mp:
        fake_expdate_builder(root, mp)
        batch.make_tasks(SMALL.tasks(), workers=1, layout=SMALL)
    return root


@pytest.fixture
def good(made, tmp_path, monkeypatch) -> Path:
    """A fresh copy of the made tasks, with the environment pointing at it."""
    root = tmp_path / "copy"
    shutil.copytree(made, root, symlinks=True)
    for name, path in (("EXPDATE_DIR", root / "expdate"), ("GENERATED_DIR", root / "generated")):
        monkeypatch.setenv(name, str(path))
    return root


def _files(generated: Path) -> dict[str, bytes]:
    """Every image, label file and joined sidecar, by path under gen/; the
    lists with their root taken out (they hold absolute paths)."""
    out = {}
    gen = generated / "m2" / "gen"
    for folder in ("images", "labels"):
        out |= {str(p.relative_to(gen)): p.read_bytes() for p in sorted((gen / folder).rglob("*")) if p.is_file()}
    out |= {p.name: p.read_bytes() for p in (gen / "sidecars").glob("*.jsonl")}
    for p in sorted((generated / "m2" / "lists").glob("gen*.txt")):
        out[p.name] = p.read_text().replace(str(generated), "<root>").encode()
    return out


# --- Tasks and shares -----------------------------------------------------------------------------

def test_layout_and_tasks():
    tasks = batch.LAYOUT.tasks()
    assert len(tasks) == 51 and [t.number for t in tasks] == list(range(51))
    assert tasks[0].photos == range(0, 1000) and tasks[49].photos == range(49_000, 50_000)
    assert (tasks[50].set, tasks[50].photos) == ("synth-val", range(0, 2100))
    assert [t.number for t in batch.parse_tasks("0-4,50")] == [0, 1, 2, 3, 4, 50]
    assert [t.number for t in batch.parse_tasks("all")] == list(range(51))
    with pytest.raises(ValueError):
        batch.parse_tasks("51")
    assert [batch.LAYOUT.spec("synth-val", i)[1] for i in (0, 299, 300, 2099)] == \
           ["clean", "clean", "sideways", "mixed"]
    assert batch.LAYOUT.spec("gen", 7) == (GEN, None)


def test_a_share_is_exact_where_fixed_within_3_points_or_3_standard_errors():
    row = lambda spec, measured, n: {"spec": spec, "measured": measured, "n": n}
    assert batch.share_ok(row(1.0, 1.0, 300)) and not batch.share_ok(row(1.0, 0.9967, 300))
    assert not batch.share_ok(row(0.0, 0.0033, 300))
    assert batch.share_ok(row(0.5, 0.529, 50_000)) and not batch.share_ok(row(0.5, 0.531, 50_000))
    assert batch.share_ok(row(0.5, 0.58, 300)) and not batch.share_ok(row(0.5, 0.59, 300))   # 3 SE = 8.7 points
    assert batch.share_ok(row(0.5, None, 0))


# --- The guards are the ones used --------------------------------------------------------------------

@pytest.mark.parametrize("intruder, folder, into", [("v1.jpg", "real-train", "train_real"),
                                                   ("test_00001.jpg", "real-test", "train_real"),
                                                   ("img_00948.jpg", "synth", "train_synth_clean")])
def test_training_pools_refuse_validation_and_test_photos_and_the_copies(fake_expdate, intruder, folder, into):
    path = m2.list_file(into)
    derived.write_list(path, derived.read_list(path) + [fake_expdate.parent / "expdate" / "yolo" / "images" /
                                                        folder / intruder])
    with pytest.raises(ValueError, match="never a background"):
        batch.load_pools(["gen"])


def test_each_set_gets_its_own_pool(fake_expdate):
    pools = batch.load_pools(["gen", "synth-val"])
    assert pools["synth-val"].by_list.keys() == {"val"}
    assert pools["gen"].by_list.keys() == {"train_real", "train_synth_clean"}


# --- Same files however the work is split ------------------------------------------------------------

@needs_fonts
def test_two_different_splits_write_identical_files(good, tmp_path, monkeypatch, fake_expdate_builder):
    """SMALL made in one process, all at once, against a layout of other
    task sizes made on 2 worker processes, task by task in reverse order:
    every image, label file, joined sidecar and list is the same."""
    assert batch.passed(batch.check(SMALL, workers=1))
    one = _files(good / "generated")
    other = replace(SMALL, task_size=2)                      # gen in 3 tasks of 2, not 2 of 3
    generated = fake_expdate_builder(tmp_path / "other", monkeypatch)
    pools = batch.load_pools(["gen", "synth-val"])
    for task in reversed(other.tasks()):
        batch.make_tasks([task], workers=2, pools=pools, layout=other)
    assert batch.passed(batch.check(other, workers=2))
    two = _files(generated)
    assert one.keys() == two.keys() and len(one) == 2 * (6 + 7) + 2 + 2
    assert [k for k in one if one[k] != two[k]] == []


# --- The check ---------------------------------------------------------------------------------------

@needs_fonts
def test_the_check_vouches_for_a_good_set(good):
    results = batch.check(SMALL, workers=1)
    assert batch.passed(results), {n: r["manifest"]["problems"] for n, r in results.items()}
    gen, val = results["gen"]["manifest"], results["synth-val"]["manifest"]
    assert (gen["complete"], gen["photos"], val["complete"], val["photos"]) == (True, 6, True, 7)
    assert gen["fonts"] == {f: fonts.SHA256[f] for f in gen["fonts"]} and gen["made_by"][0]["git_commit"]
    for name in ("gen50k", "gen10k", "synth_val"):
        assert m2.list_missing(name) is None, name
    assert m2.count_listed("gen10k") == 4 and \
        derived.read_list(m2.list_path("gen10k")) == derived.read_list(m2.list_path("gen50k"))[:4]
    samples = sets.load_set("synth-val")                      # through the list, as score_suite reads it
    assert [s.tag for s in samples] == [name for name, _ in SYNTH_VAL]
    assert all([b.label for b in s.boxes].count("exp") == 1 for s in samples)
    assert results["synth-val"]["hashes"]["images"]["000006.jpg"] == \
        hashlib.sha256((batch.images_dir("synth-val") / "000006.jpg").read_bytes()).hexdigest()


def _break(how: str, generated: Path) -> str:
    """Break the made set one way; returns what the check must say."""
    gen = generated / "m2" / "gen"
    image = gen / "images" / "gen50k" / "000004.jpg"
    if how == "missing image":
        image.unlink()
        return "000004.jpg: FileNotFoundError"
    if how == "cut-off image":
        image.write_bytes(image.read_bytes()[:3000])
        return "000004.jpg: OSError"
    if how == "label byte":
        labels = gen / "labels" / "gen50k" / "000004.txt"
        labels.write_text(labels.read_text().replace("0.", "1.", 1))
        return "000004.jpg: its label file is not the one"
    if how == "extra image":
        shutil.copy(image, image.with_name("000099.jpg"))
        return "1 images beyond the set's 6"
    if how == "task not made":
        (gen / "sidecars" / "tasks" / "01.json").unlink()
        return "gen: tasks 1 not made"
    if how == "lines swapped":
        part = gen / "sidecars" / "tasks" / "00.jsonl"
        a, b, c = part.read_text().splitlines()
        part.write_text("\n".join((b, a, c)) + "\n")
        return "gen line 0: is 000001.jpg"
    if how == "another spec":
        record = gen / "sidecars" / "tasks" / "00.json"
        record.write_text(json.dumps(json.loads(record.read_text()) | {"spec_sha256": "0" * 64}))
        return "task 0: made with spec 000000000000"
    if how == "a validation background":
        part = gen / "sidecars" / "tasks" / "00.jsonl"
        lines = [json.loads(line) for line in part.read_text().splitlines()]
        lines[0]["background"] |= {"list": "train_real", "image": "v1.jpg"}
        part.write_text("".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines))
        return "a background no train_real photo may be"
    if how == "a held-out font":
        part = gen / "sidecars" / "tasks" / "00.jsonl"
        lines = [json.loads(line) for line in part.read_text().splitlines()]
        next(line for line in lines if line["blocks"])["blocks"][0]["family"] = "IBM Plex Mono"
        part.write_text("".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines))
        return "font IBM Plex Mono, outside train"
    raise ValueError(how)


@needs_fonts
@pytest.mark.parametrize("how", ["missing image", "cut-off image", "label byte", "extra image", "task not made",
                                 "lines swapped", "another spec", "a validation background", "a held-out font"])
def test_the_check_fails_a_broken_set_and_says_why(good, how):
    said = _break(how, good / "generated")
    results = batch.check(SMALL, workers=1)
    gen = results["gen"]["manifest"]
    assert not batch.passed(results) and gen["complete"] is False
    assert any(said in p for p in gen["problems"]), gen["problems"]
    assert m2.list_missing("gen50k") is not None and m2.list_missing("gen10k") is not None
    assert results["synth-val"]["manifest"]["complete"] is True      # the other set is still vouched for


# --- Releasing the GPU jobs --------------------------------------------------------------------------

@pytest.fixture
def held_jobs(tmp_path, monkeypatch) -> Path:
    """A fake squeue listing job 72914 waiting and 72915 and 73001 held,
    and a fake scontrol whose releases squeue then shows; both log their
    calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    commands = {
        "squeue": 'echo "$*" >> "$(dirname "$0")/calls"\n'
                  'for j in 72914:Priority 72915:JobHeldUser 73001:JobHeldUser; do\n'
                  '  id=${j%%:*}; r=${j#*:}\n'
                  '  grep -qx "$id" "$(dirname "$0")/released" 2>/dev/null && r=None\n'
                  '  echo "$id $r"\n'
                  'done\n',
        "scontrol": 'echo "$*" >> "$(dirname "$0")/calls"\n'
                    'echo "$2" >> "$(dirname "$0")/released"\n',
    }
    for name, body in commands.items():
        (bin_dir / name).write_text("#!/bin/sh\n" + body)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def test_release_frees_every_held_job_of_the_name(held_jobs):
    assert batch.release_held("m2") == ["72915", "73001"]
    calls = (held_jobs / "calls").read_text().splitlines()
    assert "release 72915" in calls and "release 73001" in calls and "release 72914" not in calls
    assert any("--name=m2" in c and "--states=PENDING" in c for c in calls)


def _fake_check(passes: bool):
    manifest = {"complete": passes, "photos": 1, "expected": 1, "problem_count": 0 if passes else 1,
                "problems": [] if passes else ["gen: tasks 5-49 not made (no record)"], "shares": []}
    return lambda layout, workers: {"gen": {"manifest": manifest, "hashes": {}},
                                    "synth-val": {"manifest": manifest | {"shares": {}}, "hashes": {}}}


@pytest.mark.parametrize("passes", [True, False])
def test_the_job_releases_only_after_a_check_that_passed(load_script, held_jobs, tmp_path, monkeypatch, passes):
    generate = load_script("generate")
    monkeypatch.setattr(generate.batch, "check", _fake_check(passes))
    monkeypatch.setattr(generate, "start_run", lambda name, config: tmp_path / "run")
    (tmp_path / "run").mkdir()
    if passes:
        generate.check(workers=1, release="m2")
    else:
        with pytest.raises(SystemExit, match="releasing nothing"):
            generate.check(workers=1, release="m2")
    released = held_jobs / "released"
    assert (released.read_text().split() if released.exists() else []) == (["72915", "73001"] if passes else [])
    outcome = json.loads((tmp_path / "run" / "outcome.json").read_text())
    assert outcome["passed"] is passes and outcome["released"] == (["72915", "73001"] if passes else None)
