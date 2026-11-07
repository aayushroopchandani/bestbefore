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

import numpy as np
import pytest

from bestbefore import derived, m2, m2b, sets
from bestbefore.synth import batch, fonts, make
from bestbefore.synth.batch import Layout, LayoutV2
from bestbefore.synth.spec import GEN, GEN_V2, SYNTH_VAL, SYNTH_VAL_V2, spec_dict, spec_hash, strata_hash

needs_fonts = pytest.mark.skipif(not fonts.fetched(), reason="fonts not fetched: python scripts/fetch_fonts.py")

SMALL = Layout(gen=6, gen10k=4, task_size=3, per_stratum=1)      # 2 gen tasks, then 7 synth-val photos
SMALL_V2 = LayoutV2(gen=6, task_size=3, per_stratum=1)            # M2b's: 2 gen-v2 tasks, then 8 synth-val-v2 photos


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
    assert gen["spec"] == spec_dict(GEN) and "v2" not in gen["spec"]          # M2's specs, as before gen-v2
    assert all("v2" not in spec for spec in val["spec"].values())
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


# === M2b S9: gen-v2 and synth-val-v2, apart from M2's ==================================================

def _fingerprint(folder: Path) -> dict[str, str]:
    """Every file under a folder, by path, with its SHA-256."""
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_m2bs_layout_and_tasks():
    tasks = batch.LAYOUT_V2.tasks()
    assert len(tasks) == 201 and [t.number for t in tasks] == list(range(201))
    assert {t.set for t in tasks[:200]} == {"gen-v2"} and tasks[0].photos == range(0, 1000)
    assert tasks[199].photos == range(199_000, 200_000)
    assert (tasks[200].set, tasks[200].photos) == ("synth-val-v2", range(0, 2400))
    assert [t.number for t in batch.parse_tasks("0-4,200", batch.LAYOUT_V2)] == [0, 1, 2, 3, 4, 200]
    with pytest.raises(ValueError):
        batch.parse_tasks("201", batch.LAYOUT_V2)
    assert [batch.LAYOUT_V2.spec("synth-val-v2", i)[1] for i in (0, 299, 300, 2399)] == \
           ["upside-down", "upside-down", "glued", "nozzle-rows"]
    assert batch.LAYOUT_V2.spec("synth-val-v2", 1500) == (dict(SYNTH_VAL_V2)["embossed"], "embossed")
    assert batch.LAYOUT_V2.spec("gen-v2", 7) == (GEN_V2, None)
    assert batch.LAYOUT_V2.lists("gen-v2") == {"gen_v2": 200_000}
    assert batch.LAYOUT_V2.lists("synth-val-v2") == {"synth_val_v2": 2400}
    assert batch.spec_sha256("gen-v2") == spec_hash(GEN_V2) and batch.spec_sha256("synth-val-v2") == \
        strata_hash(SYNTH_VAL_V2)
    # M2's layout as it was: its fields (in its manifests), sets, tasks and lists
    assert batch.LAYOUT.sets == ("gen", "synth-val") and batch.LAYOUT_V2.sets == ("gen-v2", "synth-val-v2")
    assert batch.SETS == ("gen", "synth-val", "gen-v2", "synth-val-v2")
    from dataclasses import asdict
    assert asdict(batch.LAYOUT) == {"gen": 50_000, "gen10k": 10_000, "task_size": 1_000, "per_stratum": 300}
    assert batch.LAYOUT.lists("gen") == {"gen50k": 50_000, "gen10k": 10_000}
    assert batch.LAYOUT.lists("synth-val") == {"synth_val": 2100}
    assert batch.spec_sha256("synth-val") == strata_hash() and batch.spec_sha256("gen") == spec_hash(GEN)


def test_m2bs_tasks_and_files_live_under_m2b(fake_expdate):
    m2_tasks, v2_tasks = batch.LAYOUT.tasks(), batch.LAYOUT_V2.tasks()
    assert batch.task_files(m2_tasks[50])[1] == m2.gen_dir() / "sidecars" / "tasks" / "50.json"     # as M2 S6's
    assert batch.task_files(m2_tasks[7])[0] == m2.gen_dir() / "sidecars" / "tasks" / "07.jsonl"
    assert batch.task_files(v2_tasks[200])[1] == m2b.gen_dir() / "sidecars" / "tasks" / "200.json"
    assert batch.task_files(v2_tasks[7])[0] == m2b.gen_dir() / "sidecars" / "tasks" / "007.jsonl"
    assert batch.images_dir("gen-v2") == m2b.gen_dir() / "images" / "gen_v2"
    assert batch.images_dir("synth-val-v2") == m2b.gen_dir() / "images" / "synth_val_v2"
    assert batch.images_dir("gen") == m2.gen_dir() / "images" / "gen50k"
    pools = batch.load_pools(["gen-v2", "synth-val-v2"])
    assert pools["synth-val-v2"].by_list.keys() == {"val"}
    assert pools["gen-v2"].by_list.keys() == {"train_real", "train_synth_clean"}


@pytest.fixture(scope="module")
def made_v2(made, tmp_path_factory) -> Path:
    """M2's SMALL made and checked, then M2b's SMALL_V2 made beside it,
    unchecked: copied by the tests."""
    root = tmp_path_factory.mktemp("made_v2")
    shutil.copytree(made, root, symlinks=True, dirs_exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        for name, path in (("EXPDATE_DIR", root / "expdate"), ("GENERATED_DIR", root / "generated")):
            mp.setenv(name, str(path))
        assert batch.passed(batch.check(SMALL, workers=1))
        batch.make_tasks(SMALL_V2.tasks(), workers=1, layout=SMALL_V2)
    return root


@pytest.fixture
def good_v2(made_v2, tmp_path, monkeypatch) -> Path:
    root = tmp_path / "copy"
    shutil.copytree(made_v2, root, symlinks=True)
    for name, path in (("EXPDATE_DIR", root / "expdate"), ("GENERATED_DIR", root / "generated")):
        monkeypatch.setenv(name, str(path))
    return root


@needs_fonts
def test_m2bs_check_vouches_for_its_sets_and_leaves_m2s_alone(good_v2):
    generated = good_v2 / "generated"
    m2_before = _fingerprint(generated / "m2")
    results = batch.check(SMALL_V2, workers=1)
    assert batch.passed(results), {n: r["manifest"]["problems"] for n, r in results.items()}
    assert _fingerprint(generated / "m2") == m2_before                    # M2's files, untouched
    gen, val = results["gen-v2"]["manifest"], results["synth-val-v2"]["manifest"]
    assert (gen["complete"], gen["photos"], val["complete"], val["photos"]) == (True, 6, True, 8)
    assert gen["spec"] == spec_dict(GEN_V2) and gen["spec_sha256"] == spec_hash(GEN_V2) and gen["seed"] == 4
    assert val["spec_sha256"] == strata_hash(SYNTH_VAL_V2) and val["seed"] == 5
    assert list(val["strata"]) == [name for name, _ in SYNTH_VAL_V2] and val["layout"] == \
        {"gen": 6, "task_size": 3, "per_stratum": 1}
    # the sizes: gen-v2's untilted dates against the store's, synth-val-v2's digits in clean's band
    assert set(gen["sizes"]) == {"untilted", "all", "store"} and gen["sizes"]["store"] == make.STORE_SIZES
    assert gen["sizes"]["untilted"]["dates"] == len(make.date_sizes(
        [json.loads(line) for line in m2b.sidecar_path("gen_v2").read_text().splitlines()], tilted=False))
    assert set(val["sizes"]) == {name for name, _ in SYNTH_VAL_V2}
    assert all(v["outside_band"] == 0 and v["blocks"] == 1 for v in val["sizes"].values())
    for stratum, rows in val["shares"].items():                     # every share a stratum fixes, exactly
        assert all(r["ok"] for r in rows), stratum
    # written under m2b/, vouched for as the chain reads it
    for name in ("gen_v2", "synth_val_v2"):
        assert m2b.list_missing(name) is None, name
    assert m2b.manifest_path("gen-v2").exists() and not (generated / "m2" / "manifests" / "gen-v2.json").exists()
    assert (m2b.gen_dir() / "sidecars" / "tasks" / "002.json").exists()
    samples = sets.load_set("synth-val-v2")
    assert [s.tag for s in samples] == [name for name, _ in SYNTH_VAL_V2]
    assert all([b.label for b in s.boxes].count("exp") == 1 for s in samples)
    assert len(sets.load_set("gen-v2")) == 6
    # and M2's own check still says what it said
    again = batch.check(SMALL, workers=1)
    assert batch.passed(again) and "sizes" not in again["gen"]["manifest"]
    assert again["synth-val"]["manifest"]["strata"].keys() == {name for name, _ in SYNTH_VAL}


def _break_v2(how: str, generated: Path) -> tuple[str, str]:
    """Break M2b's made sets one way; returns the set and what the check must say."""
    tasks = generated / "m2b" / "gen" / "sidecars" / "tasks"

    def edit(part: str, change) -> None:
        lines = [json.loads(line) for line in (tasks / part).read_text().splitlines()]
        change(lines)
        (tasks / part).write_text("".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines))

    if how == "digits off the band":
        edit("002.jsonl", lambda lines: lines[1]["blocks"][0].update(print_height=0.05))     # the glued stratum's
        return "synth-val-v2", "glued: 1 of 1 blocks' digits outside their band"
    if how == "two targets":
        def two(lines):
            dates = [t for t in lines[2]["blocks"][0]["tokens"] if t["cls"] == "date"]      # the line-pair's
            for t in dates:
                t["role"] = "target"
        edit("002.jsonl", two)
        return "synth-val-v2", "000002.jpg: 2 targets among 2 new dates, not 1"
    if how == "a target in gen-v2":
        def target(lines):
            line = next(line for line in lines if line["blocks"])
            next(t for t in line["blocks"][0]["tokens"] if t["cls"] == "date")["role"] = "target"
        edit("000.jsonl", target)
        return "gen-v2", "a target date outside synth-val"
    if how == "a task of M2's spec":
        record = tasks / "001.json"
        record.write_text(json.dumps(json.loads(record.read_text()) | {"spec_sha256": spec_hash(GEN)}))
        return "gen-v2", "task 1: made with spec 3fc141781fcf"
    raise ValueError(how)


@needs_fonts
@pytest.mark.parametrize("how", ["digits off the band", "two targets", "a target in gen-v2", "a task of M2's spec"])
def test_m2bs_check_fails_a_broken_set_and_says_why(good_v2, how):
    name, said = _break_v2(how, good_v2 / "generated")
    results = batch.check(SMALL_V2, workers=1)
    manifest = results[name]["manifest"]
    assert not batch.passed(results) and manifest["complete"] is False
    assert any(said in p for p in manifest["problems"]), manifest["problems"]
    assert m2b.list_missing(m2b.GENERATED_SETS[name]) is not None


def _sidecar(sizes: list[tuple[str, float]], long_side: int = 1000) -> dict:
    """A made-up gen-v2 sidecar line: one block per (orientation, short side
    over the long side), each with one date."""
    return {"size": [long_side, 800], "blocks": [
        {"geometry": {"orientation": o}, "tokens": [{"cls": "date", "box": [0, 0, 300, round(f * long_side)]}]}
        for o, f in sizes]}


def test_gen_v2s_sizes_are_checked_on_untilted_dates_against_the_stores():
    store = make.STORE_SIZES
    assert store["median"] == 0.01736 and abs(sum(v for k, v in store.items() if k != "median") - 1) < 1e-9
    from bestbefore.synth.spec import V2Size
    assert V2Size().deciles[5] == store["median"]                       # the median is the middle decile
    # 1,000 dates spread as the store's (35.5 / 42.0 / 22.5%, median 1.736%), and 400 tilted ones all big
    rng = np.random.default_rng(0)
    like = ([("upright", f) for f in rng.uniform(0.010, 0.0149, 355)] + [("sideways", f) for f in
            rng.uniform(0.0151, 0.0173, 160)] + [("upright", f) for f in rng.uniform(0.0175, 0.0249, 260)] +
            [("upside-down", f) for f in rng.uniform(0.0251, 0.04, 225)])
    tilted = [("tilted", f) for f in rng.uniform(0.05, 0.08, 400)]
    sizes, problems = batch.size_check("gen-v2", [_sidecar(like + tilted)])
    assert problems == [] and sizes["untilted"]["dates"] == 1000 and sizes["all"]["dates"] == 1400
    assert sizes["all"]["2.5% and up"] > 0.4                            # tilted boxes run big, and don't count
    big = [(o, f * 1.6) for o, f in like]
    sizes, problems = batch.size_check("gen-v2", [_sidecar(big)])
    assert any("2.5% and up" in p for p in problems) and any("median" in p for p in problems)
    assert batch.size_check("gen", [_sidecar(big)]) == (None, [])        # M2's sets: no size check
    assert batch.size_check("synth-val", []) == (None, [])


def test_generates_m2b_options_keep_m2s_commands_as_they_were(load_script, monkeypatch):
    generate = load_script("generate")
    for args in (["--milestone", "m2b", "--check", "--release", "m2"],          # M2b's releases nothing
                 ["--set", "gen-v2-sample", "--count", "5", "--milestone", "m2b"]):
        monkeypatch.setattr("sys.argv", ["generate.py", *args])
        with pytest.raises(SystemExit) as stop:
            generate.main()
        assert stop.value.code == 2, args
    called = []
    monkeypatch.setattr(generate.fonts, "fetched", lambda: True)
    for args, want in ((["--tasks", "0-4,50"], ("tasks", "m2")), (["--check", "--release", "m2"], ("check", "m2")),
                       (["--milestone", "m2b", "--tasks", "0-4,200"], ("tasks", "m2b")),
                       (["--milestone", "m2b", "--check"], ("check", "m2b"))):
        monkeypatch.setattr(generate, "tasks", lambda spec, workers, milestone: called.append(("tasks", milestone)))
        monkeypatch.setattr(generate, "check", lambda workers, release, milestone: called.append(("check", milestone)))
        monkeypatch.setattr("sys.argv", ["generate.py", *args])
        generate.main()
        assert called[-1] == want, args


@needs_fonts
def test_generate_makes_and_checks_m2bs_sets_in_runs_of_their_own(load_script, good, tmp_path, monkeypatch):
    from bestbefore import runs
    generate = load_script("generate")
    monkeypatch.setitem(generate.LAYOUTS, "m2b", SMALL_V2)
    count = iter(range(100))
    monkeypatch.setattr(generate, "start_run", lambda name, config, seed=None: runs.start_run(
        name, config, seed=seed, root=tmp_path / "runs" / str(next(count))))
    m2_before = _fingerprint(good / "generated" / "m2")
    generate.tasks("0-2", 1, "m2b")
    generate.check(1, None, "m2b")
    made = {p.name[16:]: p for p in (tmp_path / "runs").rglob("*-m2b-generate*")}
    assert made.keys() == {"m2b-generate", "m2b-generate-check"}
    summary = json.loads((made["m2b-generate"] / "summary.json").read_text())
    assert summary["photos"] == 14 and summary["tasks"] == [0, 1, 2]
    check = made["m2b-generate-check"]
    assert json.loads((check / "outcome.json").read_text())["passed"] is True
    assert json.loads((check / "gen-v2.json").read_text())["complete"] is True
    assert (check / "synth-val-v2-shares.md").read_text().count("### ") == 8          # a table per stratum
    assert set(json.loads((check / "hashes.json").read_text())) == {"gen-v2", "synth-val-v2"}
    assert _fingerprint(good / "generated" / "m2") == m2_before


def test_m2bs_generation_job_runs_its_steps_in_order_and_releases_nothing():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "slurm" / "m2b_generate.sbatch").read_text()
    steps = [line for line in text.splitlines() if line.startswith("python scripts/")]
    assert steps == ["python scripts/render_sheet.py --v2",
                     'python scripts/generate.py --set gen-v2-sample --count 500 --workers "$SLURM_CPUS_PER_TASK"',
                     'python scripts/generate.py --milestone m2b --tasks 0-200 --workers "$SLURM_CPUS_PER_TASK"',
                     'python scripts/generate.py --milestone m2b --check --workers "$SLURM_CPUS_PER_TASK"',
                     'python scripts/derive_sets.py --only rot180 --workers "$SLURM_CPUS_PER_TASK"']
    assert "--release" not in text and "scontrol" not in text and "set -euo pipefail" in text   # a failed check stops it
    settings = dict(line[len("#SBATCH --"):].split("=", 1) for line in text.splitlines() if line.startswith("#SBATCH --"))
    assert (settings["partition"], settings["cpus-per-task"]) == ("short", "24")
    hours, minutes, _ = map(int, settings["time"].split(":"))
    assert hours * 60 + minutes <= 6 * 60                                   # short's limit
    assert [t.number for t in batch.parse_tasks("0-200", batch.LAYOUT_V2)] == list(range(201))
