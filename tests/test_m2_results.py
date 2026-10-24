"""M2's results tables (docs/m2-synthetic-generator.md, S7), on made-up run
records: cells, every table, the rules as summarise.py re-applies them, the
store split by cause, and partial results shown as "not yet"."""
import json
import random

import pytest

from bestbefore import m2, runs, store
from bestbefore import m2_results as r
from bestbefore.detection_eval import Prediction, match_photo


def summary(found, total=187, expiry=None, precision=0.9, false=0.1):
    out = {"dates_found": found, "dates": total, "precision": precision,
           "false_boxes_per_photo": false}
    if expiry is not None:
        out |= {"expiry_found": expiry[0], "expiry_dates": expiry[1]}
    return out


def entry(found, map50=0.99, boxed=None, expiry=None, total=187):
    return {"iou0.5": summary(found, total, expiry), "iou0.3": summary(found, total, expiry),
            "any_conf": summary(boxed or found, total, expiry), "map50": {"date": map50}}


def suite(real_val=0.99, hard=0.5, threshold=0.2, found=180, strata=None):
    """One run's suite.json: the same mAP on every hard set."""
    out = {"threshold": threshold, "real-val": entry(found, real_val),
           **{name: entry(found, hard) for name in m2.HARD}}
    if strata:
        out["synth-val"]["by_tag"] = {
            s: {"iou0.5": summary(0, 300, (k, 300)), "any_conf": summary(0, 300, (k + 10, 300))}
            for s, k in strata.items()}
    return out


def cells(text: str, row: str) -> dict[str, str]:
    """The first markdown table row in `text` whose first cell is `row`, by column."""
    head = None
    for line in text.splitlines():
        if not line.startswith("|"):
            head = None
            continue
        values = [c.strip() for c in line.strip("|").split("|")]
        if head is None:
            head = values
        elif values[0] == row:
            return dict(zip(head, values))
    raise KeyError(row)


@pytest.fixture
def records(monkeypatch, tmp_path):
    """Empty run folders: results/runs/ and data/store/runs/."""
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path / "store-runs")
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


def finish(run: str, suite_json=None, **files):
    """A finished M2 run's record, maybe scored (suite.json and other files)."""
    run_dir = runs.start_run(f"m2-{run}", {"imgsz": 640})
    runs.update_run(run_dir, status="finished")
    for name, data in ({"suite.json": suite_json} | files).items():
        if data is not None:
            (run_dir / name).write_text(json.dumps(data))
    return run_dir


# --- Cells --------------------------------------------------------------------------

def test_a_count_cell_is_the_mean_over_the_seeds_in_with_the_range():
    assert r.count_cell([(176, 187), (178, 187), (181, 187)]) == "178.3/187 (176–181)"
    assert r.count_cell([(5, 10), None, None]) == "5/10"
    assert r.count_cell([(0, 15)] * 3) == "0/15"
    assert r.count_cell([None, None, None]) == "not yet"
    assert r.count_cell([r._NA, None, None]) == "–"      # scored, but not this number
    with pytest.raises(ValueError, match="different sets"):
        r.count_cell([(1, 10), (2, 11)])


def test_a_value_cell_is_the_mean_with_the_range():
    assert r.value_cell([0.989, 0.993, 0.991]) == "0.991 (0.989–0.993)"
    assert r.value_cell([0.5, None, 0.5]) == "0.500"
    assert r.value_cell([0.0312, 0.0298], 2) == "0.03"
    assert r.value_cell([None] * 3) == "not yet"
    assert r.seeds_cell([{}, None, {}]) == "2/3"


def test_each_tells_a_seed_not_in_from_a_number_not_recorded():
    seeds = [{"a": 1}, None, {"b": 2}]
    assert r.each(seeds, lambda s: s["a"]) == [1, None, r._NA]


# --- The validation suite -----------------------------------------------------------

def test_the_suite_table_shows_partial_arms_and_missing_sets_as_not_yet():
    skipped = suite(found=170)
    skipped["real-val-rot"] = {"missing": "real_val_rot not made yet"}
    suites = {"base": [suite(found=180), skipped, None], "rot": [None] * 3}
    base = cells(r.suite_table(suites, "real-val"), "base")
    assert base["seeds"] == "2/3" and base["dates found"] == "175/187 (170–180)"
    assert base["mAP@0.5 date"] == "0.990"
    assert cells(r.suite_table(suites, "real-val-rot"), "base")["dates found"] == "180/187"
    rot = cells(r.suite_table(suites, "real-val"), "rot")
    assert rot["seeds"] == "0/3" and rot["dates found"] == "not yet"
    # synth-val's headline is its targets, loaded as expiry dates.
    synth = suite()
    synth["synth-val"] = entry(900, expiry=(1500, 2100), total=4494)
    row = cells(r.suite_table({"gen": [synth, None, None]}, "synth-val"), "gen")
    assert row["targets found"] == "1500/2100" and row["dates found"] == "900/4494"


def test_the_rule_table_marks_eligibility_only_for_complete_arms():
    suites = {"base": [suite(0.990, 0.50)] * 3, "rot": [suite(0.975, 0.80)] * 3,
              "gen": [suite(0.990, 0.90), None, None]}
    table = r.map_table(suites, rule=True)
    assert cells(table, "base")["eligible"] == "yes"
    assert cells(table, "rot")["eligible"] == "no"           # 1.5 points under base on real-val
    assert cells(table, "gen")["eligible"] == "not yet"
    assert cells(table, "rot")["hard sets, mean"] == "0.800"
    assert cells(table, "gen")["threshold"] == "0.200"
    assert "eligible" not in r.map_table(suites)


def test_the_strata_table_counts_targets_per_stratum():
    strata = {s: 100 + i for i, s in enumerate(r.STRATA)}
    suites = {"gen": [suite(strata=strata), suite(strata={**strata, "sideways": 110}), None]}
    row = cells(r.strata_table(suites), "gen")
    assert row["clean"] == "100/300" and row["sideways"] == "105.5/300 (101–110)"
    assert cells(r.strata_table(suites, "any_conf"), "gen")["clean"] == "110/300"


def candidates(hard: dict[str, float]):
    """Every candidate's three seeds scored: real-val 0.99, one hard-set mAP each."""
    for arm in m2.CANDIDATES:
        for run in r.seed_runs(arm):
            finish(run, suite(0.99, hard[arm], threshold=0.25))


def test_the_m2_detector_is_decided_and_checked_against_the_chain(records):
    assert "not decided yet, 0 of 15" in r.detector_decision()
    candidates({"base": 0.5, "rot": 0.8, "gen": 0.85, "gen-rot": 0.855, "gen-rot-1280": 0.86})
    text = r.detector_decision()
    assert "by the rule: `gen`**" in text and "`m2-gen-s0`" in text      # within 1 point, cheapest
    assert "hasn't recorded" in text
    m2.decisions_path().write_text(json.dumps(m2.decisions()))          # as the chain writes it
    assert "agrees" in r.detector_decision()
    chain = m2.decisions()
    chain["m2_detector"]["arm"] = "gen-rot"
    m2.decisions_path().write_text(json.dumps(chain))
    assert "**Warning:** the chain's record differs: arm: chain 'gen-rot', here 'gen'" in (
        r.detector_decision())


def test_a_chain_decision_the_runs_here_dont_support_is_a_warning(records):
    m2.decisions_path().write_text(json.dumps({"pilot": {"passes": True}}))
    assert "**Warning:** the chain has decided this" in r.pilot_line()
    assert r.chain_check("loader") == ""                  # nobody has decided it yet


def test_the_pilot_line(records):
    pilot = suite()
    pilot["synth-val"]["any_conf"] |= {"expiry_found": 1300, "expiry_dates": 2100,
                                       "expiry_recall": 0.619}
    finish("pilot", pilot)
    assert "**passed**" in r.pilot_line() and "0.619" in r.pilot_line()


def speed_run(run: str, epoch_times):
    """A finished speed run as pulled to the laptop: results.csv in its run
    folder, its save_dir a cluster path that doesn't exist here."""
    run_dir = finish(run)
    runs.update_run(run_dir, train_minutes=3.0, config={"ram_check": None},
                    applied={"save_dir": "/cluster/ckpt/m2/x", "train_images": 12_000,
                             "ram_cache": False})
    (run_dir / "results.csv").write_text(
        "epoch,time\n" + "".join(f"{i + 1},{t}\n" for i, t in enumerate(epoch_times)))
    return run_dir


def test_the_loader_block_reads_the_pulled_results_csv(records):
    assert "Not decided yet: 0 of 3" in r.loader_block()
    # Epoch 2 takes 30, 25 and 24 s: 400, 480 and 500 images/s; b is within 10% of c.
    for k, times in (("a", (40, 70, 100)), ("b", (40, 65, 90)), ("c", (90, 114, 138))):
        speed_run(f"speed-{k}", times)
    text = r.loader_block()
    assert cells(text, "m2-speed-b")["images/s, epoch 2"] == "480.0"
    assert cells(text, "m2-speed-b")["epoch 2 s"] == "25.0"
    assert "by the rule: `b` (24 workers, no cache)**" in text and "96.0% of the fastest" in text
    assert m2.results_csv(m2.finished_run("speed-a")).parent == m2.finished_run("speed-a")


def test_more_photos_help_needs_more_than_one_point():
    def seeds(hard):
        return [suite(0.99, hard)] * 3

    assert not m2.more_photos_help(seeds(0.80), seeds(0.79))      # exactly 1 point: no
    assert m2.more_photos_help(seeds(0.80), seeds(0.789))
    assert not m2.more_photos_help(seeds(0.70), seeds(0.75))


def test_eligible_takes_any_subset_with_base():
    suites = {"base": [suite(0.99, 0.5)] * 3, "gen": [suite(0.975, 0.9)] * 3}
    assert m2.eligible(suites) == ["base"]


# --- The store photos -----------------------------------------------------------------

def store_json(found: dict[str, tuple[int, int]], boxed: dict[str, tuple[int, int]]) -> dict:
    """A store.json: expiry dates found and boxed per tag; 'all' is their sum."""
    def level(counts):
        return {t: summary(0, 0, c) for t, c in counts.items()}

    at5, anyc = level(found), level(boxed)
    by_tag = {t: {"iou0.5": at5[t], "iou0.3": at5[t], "any_conf": anyc[t]} for t in found}
    by_tag["other-face"] = {lv: summary(1, 7) for lv in ("iou0.5", "iou0.3", "any_conf")}

    def total(counts):
        return summary(40, 199, tuple(map(sum, zip(*counts.values()))), precision=0.95)

    return {"iou0.5": total(found), "iou0.3": total(found), "any_conf": total(boxed),
            "by_tag": by_tag}


def test_store_tables_put_m1_first_and_only_tags_with_an_expiry():
    m1 = store_json({"sideways": (0, 15), "clean": (6, 31)}, {"sideways": (0, 15), "clean": (20, 31)})
    gen = store_json({"sideways": (9, 15), "clean": (20, 31)}, {"sideways": (12, 15), "clean": (25, 31)})
    rows = {r.M1_ROW: [m1, m1, m1], "gen": [gen, None, None], "rot": [None] * 3}
    groups = r.tag_groups(rows)
    assert groups == ["sideways", "clean", "all"]           # store.TAGS order; no other-face
    tags = {name: r.each(seeds, r.by_tag) for name, seeds in rows.items()}
    table = r.group_table(tags, groups, "iou0.5")
    assert table.splitlines()[2].startswith(f"| {r.M1_ROW} | 3/3 | 0/15 | 6/31 | 6/46 |")
    assert cells(table, "gen")["sideways"] == "9/15" and cells(table, "rot")["all"] == "not yet"
    assert cells(r.group_table(tags, groups, "any_conf"), "gen")["sideways"] == "12/15"
    overall = r.scores_table(rows)
    assert cells(overall, "gen")["expiry found"] == "29/46" and "mAP@0.5 date" not in overall


def test_m1s_test_scores_lack_any_confidence_and_show_a_dash():
    m1_test = {level: summary(697, 733, (637, 665)) for level in ("iou0.5", "iou0.3")}
    m1_test["map50"] = {"date": 0.993}
    table = r.scores_table({r.M1_ROW: [m1_test, None, None]})
    row = cells(table, r.M1_ROW)
    assert row["expiry boxed at any confidence"] == "–" and row["mAP@0.5 date"] == "0.993"


def label(tag, fmt, height=0.03, derived=False):
    """A store label with one expiry box of the given height (a square photo)."""
    kind = "mfg" if derived else "expiry"
    return {"tag": tag, "format": fmt, "derived": derived,
            "boxes": [{"label": kind, "xyxy": [0.1, 0.0, 0.4, height]}]}


@pytest.mark.parametrize("tag, fmt, height, cause", [
    ("clean", "Month YYYY", 0.03, r.FULL_NAME),
    ("small-print", "DDth Month YYYY", 0.01, r.FULL_NAME),
    ("clean", "MON YYYY", 0.03, r.ABBREVIATION),
    ("dot-matrix", "YYYY.MON.DD", 0.03, r.ABBREVIATION),
    ("clean", "DD/MM/YYYY", 0.0149, "numbers: print under 1.5%"),
    ("clean", "DD/MM/YYYY", 0.015, "numbers: print 1.5 to 2.5%"),
    ("dot-matrix", "MM/YYYY", 0.0249, "numbers: print 1.5 to 2.5%"),
    ("clean", "DD-MM-YYYY", 0.025, "numbers: print 2.5% and up"),
    ("sideways", "Month YYYY", 0.01, r.SIDEWAYS),            # sideways first, whatever the format
    ("sideways", "DD/MM/YYYY", 0.05, r.SIDEWAYS),
    ("other-face", "DD/MM/YYYY", 0.03, None),
])
def test_cause_of(tag, fmt, height, cause):
    assert r.cause_of(label(tag, fmt, height), (1000, 1000)) == cause


def test_print_height_is_over_the_long_side():
    # 0.02 of a 1000 px tall photo whose long side is 2000 px: 1%.
    assert r.cause_of(label("clean", "DD/MM/YYYY", 0.02), (2000, 1000)) == "numbers: print under 1.5%"


def test_store_causes_count_found_and_boxed_and_skip_what_has_no_expiry_box():
    labels = {"a": label("clean", "Month YYYY"), "b": label("sideways", "MON YYYY"),
              "c": label("other-face", "DD/MM/YYYY"), "d": label("clean", "DD/MM/YYYY", derived=True)}
    sizes = dict.fromkeys(labels, (1000, 1000))
    box = [0.1, 0.0, 0.4, 0.03]
    boxes = {"a": [[*box, 0.9]], "b": [[*box, 0.1]], "c": [[*box, 0.9]], "d": [[*box, 0.9]]}
    split = r.store_causes(labels, sizes, boxes, threshold=0.5)
    assert split[r.FULL_NAME]["iou0.5"] == {"expiry_dates": 1, "expiry_found": 1}
    assert split[r.SIDEWAYS]["iou0.5"] == {"expiry_dates": 1, "expiry_found": 0}
    assert split[r.SIDEWAYS]["any_conf"] == {"expiry_dates": 1, "expiry_found": 1}
    assert sum(c["iou0.5"]["expiry_dates"] for c in split.values()) == 2   # c and d: nothing to find


def test_one_matching_gives_what_matching_at_each_threshold_gives():
    """store_causes matches once, at every confidence; m1-store-misses
    matched again for each level. They must agree on any photo."""
    rng = random.Random(0)
    for _ in range(300):
        truths = [[x := rng.random() * 0.8, y := rng.random() * 0.8, x + 0.1, y + 0.05]
                  for _ in range(rng.randint(1, 4))]
        preds = [Prediction("date", (t[0] + rng.uniform(-0.04, 0.04), t[1] + rng.uniform(-0.02, 0.02),
                                     t[2] + rng.uniform(-0.04, 0.04), t[3] + rng.uniform(-0.02, 0.02)),
                            round(rng.random(), 2))
                 for t in rng.choices(truths, k=rng.randint(0, 6))]
        threshold = rng.choice((0.3, 0.5, 0.72))
        _, once = match_photo(preds, truths, 0.5)
        _, again = match_photo([p for p in preds if p.conf >= threshold], truths, 0.5)
        assert [c is not None and c >= threshold for c in once] == [c is not None for c in again]


def test_the_causes_must_add_up_to_store_json():
    split = {c: {lv: {"expiry_dates": 0, "expiry_found": 0} for lv in r.LEVELS} for c in r.CAUSES}
    split[r.SIDEWAYS] = {"iou0.5": {"expiry_dates": 15, "expiry_found": 2},
                         "any_conf": {"expiry_dates": 15, "expiry_found": 5}}
    scores = {"iou0.5": summary(0, 0, (2, 15)), "any_conf": summary(0, 0, (5, 15))}
    assert "causes add up" in r.cause_check({"gen": [scores]}, {"gen": [split]})
    scores["iou0.5"] = summary(0, 0, (3, 15))
    assert "**Warning:**" in r.cause_check({"gen": [scores]}, {"gen": [split]})
    assert r.cause_check({"gen": [None]}, {"gen": [None]}) == ""


def test_private_predictions_are_the_newest_kept_for_that_run(records):
    for stamp, threshold in (("20261101-100000", 0.5), ("20261101-110000", 0.25)):
        run = store.RUNS_DIR / f"{stamp}-detector-store-m2-gen-s0"
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"config": {"threshold": threshold}}))
        (run / "predictions.json").write_text(json.dumps({"p": [[0, 0, 1, 1, threshold]]}))
    assert r.private_predictions("m2-gen-s0") == (0.25, {"p": [[0, 0, 1, 1, 0.25]]})
    assert r.private_predictions("m2-gen-s1") is None
    assert r.private_predictions("gen-s0") is None                 # M2 names start m2-


# --- Partial results: nothing crashes -----------------------------------------------------

def test_every_block_and_the_report_run_on_records_not_in_yet(records, load_script):
    for block in (r.loader_block, r.pilot_line, r.detector_decision, r.ablation_block,
                  r.data_size_block, r.input_size_block, r.runs_table, r.generation_block):
        assert "yet" in block()
    report = load_script("m2_report").report()
    assert report.startswith("# M2:") and "_Not yet: the M2 detector is chosen" in report
    # One run in: its rows fill, the others stay "not yet".
    finish("base-s0", suite())
    text = r.ablation_block()
    assert cells(text, "base")["seeds"] == "1/3" and cells(text, "rot")["seeds"] == "0/3"
