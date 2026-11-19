"""M2b's results tables (docs/m2b-store-and-size.md, S11 to S13), on
made-up run records: the rules as summarise.py works them out again and
checks them against the chain's record, R3 read back, every table, ft2's
frozen layers, compute, the inference add-ons with R4 and the what-if
worked out again, and records not in yet shown as "not yet"."""
import json
from itertools import combinations

import pytest

from bestbefore import ensemble, final_test, m2, m2b, runs
from bestbefore import m2_results as m2r
from bestbefore import m2b_results as r


def summ(found, total, expiry=None, precision=0.9, false=0.1):
    out = {"dates_found": found, "dates": total, "precision": precision, "false_boxes_per_photo": false}
    if expiry is not None:
        out |= {"expiry_found": expiry[0], "expiry_dates": expiry[1]}
    return out


def entry(map50, found=50, total=63, expiry=(20, 29), m2_found=40, boxed=60, keywords=None):
    """One set of an M2b suite at both thresholds; `keywords` adds the
    keyword classes' mAP@0.5 beside the dates'."""
    return {"iou0.5": summ(found, total, expiry), "iou0.3": summ(found, total, expiry),
            "any_conf": summ(boxed, total, expiry, precision=0.2, false=9.0),
            "at_threshold_m2": {"iou0.5": summ(m2_found, total, (expiry[0] - 5, expiry[1]), precision=0.95),
                                "iou0.3": summ(m2_found, total, (expiry[0] - 5, expiry[1]))},
            "map50": {"date": map50, **(keywords or {})}}


def m2b_suite(v1=0.85, real=0.99, synth=0.99, threshold=0.2, threshold_m2=0.4, minutes=2.0, upside=250,
              keywords=None):
    """A whole suite-m2b.json: v1-val by tag and by size, both synthetic
    sets by stratum."""
    out = {"threshold": threshold, "threshold_m2": threshold_m2, "scored": {"minutes": minutes},
           **{name: entry(0.95) for name in m2b.SUITE}}
    out["real-val"], out["synth-val"] = entry(real), entry(synth)
    out["v1-val"] = entry(v1, keywords=keywords)
    tags = {"clean": (12, 13), "sideways": (4, 6), "dot-matrix": (3, 3), "small-print": (1, 7)}
    out["v1-val"]["by_tag"] = {t: {"iou0.5": summ(0, 0, c), "any_conf": summ(0, 0, (c[1], c[1])),
                                   "at_threshold_m2": {"iou0.5": summ(0, 0, (0, c[1]))}}
                               for t, c in tags.items()}
    out["v1-val"]["by_tag"]["other-face"] = {lv: summ(1, 2) for lv in ("iou0.5", "any_conf")}
    out["v1-val"]["by_tag"]["other-face"]["at_threshold_m2"] = {"iou0.5": summ(0, 2)}
    out["v1-val"]["by_size"] = {band: {"dates": 20, "found": 15 + k, "found_at_threshold_m2": 10,
                                       "boxed_at_any_confidence": 19}
                                for k, (band, _, _) in enumerate(m2b.SIZE_BANDS)}
    for name, strata in (("synth-val-v2", r.STRATA_V2), ("synth-val", m2r.STRATA)):
        out[name]["by_tag"] = {s: {"iou0.5": summ(0, 0, (290, 300)), "any_conf": summ(0, 0, (299, 300))}
                               for s in strata}
    out["synth-val-v2"]["by_tag"]["upside-down"]["iou0.5"] = summ(0, 0, (upside, 300))
    return out


def cells(text: str, row: str, column: int = 0) -> dict[str, str]:
    """The first markdown table row in `text` whose cell `column` is `row`."""
    head = None
    for line in text.splitlines():
        if not line.startswith("|"):
            head = None
            continue
        values = [c.strip() for c in line.strip("|").split("|")]
        if head is None:
            head = values
        elif values[column] == row:
            return dict(zip(head, values))
    raise KeyError(row)


@pytest.fixture
def records(monkeypatch, tmp_path):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


def finish(name: str, suite: dict | None = None, milestone: str = "m2b", **record):
    """A finished run, scored on the M2b suite if `suite` is given."""
    run_dir = runs.start_run(f"{milestone}-{name}", record.pop("config", {}))
    runs.update_run(run_dir, status="finished", **record)
    if suite is not None:
        (run_dir / ("suite-m2b.json" if milestone == "m2b" else "suite.json")).write_text(json.dumps(suite))
    return run_dir


def arm(name: str, v1s, **kw):
    for s, v1 in zip(m2b.SEEDS, v1s):
        finish(f"{name}-s{s}", m2b_suite(v1, **kw))


def rescore(name: str, suite: dict, milestone: str = "m2b"):
    """A finished run's suite, written again."""
    run_dir = (m2b if milestone == "m2b" else m2).finished_run(name)
    (run_dir / ("suite-m2b.json" if milestone == "m2b" else "suite.json")).write_text(json.dumps(suite))


def chain_writes():
    """The chain's record, as it writes it."""
    m2b.decisions_path().write_text(json.dumps(m2b.decisions()))


# --- The rules -------------------------------------------------------------------------

def test_nothing_in_yet_is_not_yet_everywhere(records):
    assert "not yet, 0 of 3 control seeds" in r.noise_line()
    assert "R1: not yet" in r.r1_line() and "R2: not yet" in r.r2_line() and "R3: not yet" in r.r3_line()
    assert r.record_check() == "" and r.early_read_check() == "" and r.control_compute_line() == ""
    assert cells(r.r2_table(), "n1280-ft")["guard"] == "not yet"
    assert cells(r.r3_table(), "m1280-ft")["R3"] == "not yet"
    for block in (r.fine_tune_block(), r.gen_v2_block(), r.model_size_block(),   # every table builds on nothing
                  r.inference_block()):
        assert "not yet" in block and "**Warning:**" not in block


def test_the_band_r1_and_r2_are_worked_out_again_and_checked_against_the_chain(records):
    arm("n1280", (0.80, 0.85, 0.82))
    assert "**The noise band: 0.0500.**" in r.noise_line()
    assert "0.8000, 0.8500, 0.8200" in r.noise_line()
    assert "hasn't recorded" in r.noise_line()
    assert "isn't here yet" in r.record_check()
    chain_writes()
    assert "agrees" in r.noise_line() and "is the same" in r.record_check()
    arm("s1280", (0.84, 0.86, 0.85))
    finish("m1280-s0", m2b_suite(0.87))
    arm("n1280-ft", (0.90, 0.91, 0.92))
    arm("n1280-ft2", (0.93, 0.94, 0.95))
    chain_writes()
    assert "**R1, the `m` probe: passes**" in r.r1_line() and "above every one" in r.r1_line()
    assert "**R2, by the rule: `ft`**" in r.r2_line()                 # 0.94 within 0.05 of 0.91
    assert "within the band (0.0500) of the best, `ft2`'s 0.9400" in r.r2_line()
    assert "agrees" in r.r2_line() and "noise_store, R1, R2" in r.record_check()
    chain = m2b.decisions()
    chain["R2"]["recipe"] = "ft2"
    m2b.decisions_path().write_text(json.dumps(chain))
    assert "**Warning:** the chain's record differs: recipe: chain 'ft2', here 'ft'" in r.r2_line()
    assert "**Warning:**" in r.record_check() and "in R2." in r.record_check()


def test_r2_says_why(records):
    arm("n1280", (0.80, 0.80, 0.80))
    arm("n1280-ft", (0.95, 0.95, 0.95))
    arm("n1280-ft2", (0.90, 0.90, 0.90))
    line = r.r2_line()
    assert "`ft`**, the highest mean on v1-val, 0.9500; `ft2` 0.9000 (-0.0500)" in line
    assert "Both recipes pass the guard (its floors, the control's mean less 0.01: real-val 0.9800" in line
    table = r.r2_table()
    assert cells(table, "n1280 (control)")["guard"] == "–"
    assert cells(table, "n1280-ft")["v1-val"] == "0.9500" and cells(table, "n1280-ft")["guard"] == "passes"


def test_r2_names_a_recipe_the_guard_left_out_and_says_when_none_passes(records):
    arm("n1280", (0.80, 0.80, 0.80))
    arm("n1280-ft", (0.85, 0.85, 0.85))
    arm("n1280-ft2", (0.95, 0.95, 0.95), synth=0.95)      # forgets synth-val
    assert "`ft`**" in r.r2_line() and "Out by the guard: ft2." in r.r2_line()
    assert cells(r.r2_table(), "n1280-ft2")["guard"] == "fails"
    for s in m2b.SEEDS:                                   # ft forgets real-val too
        rescore(f"n1280-ft-s{s}", m2b_suite(0.85, real=0.95))
    assert "**R2: no recipe passed the guard**" in r.r2_line()


def test_r3_names_the_cheapest_in_the_band_and_how_far_each_sits_from_its_edge(records):
    arm("n1280", (0.70, 0.80, 0.75))                      # the band: 0.10
    arm("s1280", (0.76, 0.78, 0.80))
    arm("m1280", (0.85, 0.83, 0.84))                      # s0 above every s1280 seed: R1 passes
    arm("n1280-ft", (0.90, 0.91, 0.92))
    arm("n1280-ft2", (0.89, 0.89, 0.89))                  # R2: ft, the highest
    arm("s1280-ft", (0.93, 0.94, 0.95))
    arm("m1280-ft", (0.97, 0.98, 0.99))                   # the best, 0.98: the edge is 0.88
    chain_writes()
    line, table = r.r3_line(), r.r3_table()
    assert ("**R3, by the rule: `n1280-ft`**, the cheapest candidate within the band (0.1000) of the best, "
            "`m1280-ft`'s 0.9800: the band's edge is 0.8800, and `n1280-ft`'s 0.9100 sits 0.0300 above it.") in line
    assert "Every candidate passes the guard." in line
    assert "Within the band too, and dearer: `s1280-ft` (+0.0600), `m1280-ft` (+0.1000)." in line
    assert "Below the edge: `n1280` (-0.1300), `s1280` (-0.1000), `m1280` (-0.0400)." in line
    assert "Nearest the edge: `m1280`, 0.0400 below it." in line
    assert "the M2b detector is `n1280-ft-s0` (v1-val 0.9000) at its threshold_store, 0.2000" in line
    assert "agrees" in line
    assert [cells(table, a)["R3"] for a in ("n1280 (control)", "n1280-ft", "s1280", "s1280-ft", "m1280",
                                            "m1280-ft")] == ["below the band", "**chosen**", "below the band",
                                                             "in the band", "below the band", "in the band"]
    chosen = cells(table, "n1280-ft")
    assert chosen["from the band's edge"] == "+0.0300" and chosen["v1-val"] == "0.9100 (0.9000–0.9200)"
    assert chosen["guard"] == "passes" and cells(table, "n1280 (control)")["guard"] == "passes"
    chain = m2b.decisions()
    chain["R3"]["arm"] = "s1280-ft"
    m2b.decisions_path().write_text(json.dumps(chain))
    assert "**Warning:** the chain's record differs: arm: chain 's1280-ft', here 'n1280-ft'" in r.r3_line()


def test_r3_without_m_once_r1_fails_and_a_candidate_out_by_the_guard(records):
    arm("n1280", (0.80, 0.81, 0.82))                      # a spread of 0.02: the band is its floor, 0.03
    arm("s1280", (0.84, 0.85, 0.86))
    finish("m1280-s0", m2b_suite(0.85))                   # not above s1280-s2: R1 fails
    arm("n1280-ft", (0.90, 0.90, 0.90))
    arm("n1280-ft2", (0.88, 0.88, 0.88))
    arm("s1280-ft", (0.95, 0.95, 0.95), synth=0.95)       # forgets synth-val
    chain_writes()
    line, table = r.r3_line(), r.r3_table()
    assert "m1280" not in table and "m1280" not in line
    assert cells(table, "s1280-ft")["guard"] == "fails" and cells(table, "s1280-ft")["R3"] == "fails the guard"
    assert "**R3, by the rule: `n1280-ft`**, the highest mean on v1-val, 0.9000." in line
    assert "Out by the guard: s1280-ft." in line and "Within the band too" not in line
    assert "Below the edge: `n1280` (-0.0600), `s1280` (-0.0200)." in line          # the edge: 0.87


# --- The tables ----------------------------------------------------------------------

def test_the_map_and_threshold_tables(records):
    arm("n1280", (0.70, 0.80, 0.75))
    finish("n1280-ft-s0", m2b_suite(0.9, threshold=0.25))
    suites = r.arm_suites(r.FINE_TUNE)
    row = cells(r.map_table(suites), "n1280 (control)")
    assert row["v1-val"] == "0.750 (0.700–0.800)" and row["real-val"] == "0.990"
    assert cells(r.map_table(suites), "n1280-ft")["seeds"] == "1/3"
    assert cells(r.map_table(suites), "n1280-ft2")["v1-val"] == "not yet"
    table = r.threshold_table(suites)
    store_row = cells(table, "store (v1-val)", column=1)              # the control's first
    assert store_row["value"] == "0.2000" and store_row["dates found"] == "50/63"
    assert store_row["expiry found"] == "20/29" and store_row["precision"] == "0.900"
    m2_row = cells(table, "M2's (real sets)", column=1)
    assert m2_row["value"] == "0.4000" and m2_row["dates found"] == "40/63"
    assert m2_row["expiry found"] == "15/29" and m2_row["precision"] == "0.950"


def test_the_threshold_on_m2s_weights_seed_by_seed_and_m2s_own(records):
    arm("n1280", (0.70, 0.80, 0.75), threshold=0.01, threshold_m2=0.34)
    table = r.control_threshold_table()
    assert cells(table, "m2b-n1280-s2")["value"] == "0.0100"
    assert "seeds" not in cells(table, "m2b-n1280-s0")
    assert r.m2_threshold_check() == ""                          # M2's records not here
    for s in m2b.SEEDS:
        finish(f"gen-rot-1280-s{s}", {"threshold": 0.34}, milestone="m2")
    assert "is each control seed's own in M2's records (0.3400, 0.3400, 0.3400; 3 of 3" in (
        r.m2_threshold_check())
    rescore("gen-rot-1280-s1", {"threshold": 0.35}, milestone="m2")
    assert "**Warning:**" in r.m2_threshold_check() and "seed 1" in r.m2_threshold_check()


def test_v1_val_by_tag_only_where_an_expiry_is_and_by_size(records):
    arm("n1280", (0.70, 0.80, 0.75))
    suites = r.arm_suites((m2b.CONTROL,))
    assert r.expiry_tags(suites) == ["sideways", "dot-matrix", "small-print", "clean"]   # store.TAGS order
    row = cells(r.tag_table(suites), "n1280 (control)")
    assert "other-face" not in row
    assert row["sideways"] == "4/6" and row["all expiry dates"] == "20/29" and row["all dates"] == "50/63"
    assert cells(r.tag_table(suites, "at_threshold_m2"), "n1280 (control)")["sideways"] == "0/6"
    assert cells(r.tag_table(suites, "any_conf"), "n1280 (control)")["small-print"] == "7/7"
    at_m2 = cells(r.tag_table(suites, "at_threshold_m2"), "n1280 (control)")
    assert at_m2["all expiry dates"] == "15/29" and at_m2["all dates"] == "40/63"
    sizes = cells(r.size_table(suites), "n1280 (control)")
    assert (sizes["under 1.5%"], sizes["1.5-2.5%"], sizes["2.5% and up"]) == ("15/20", "16/20", "17/20")
    assert cells(r.size_table(suites, "boxed_at_any_confidence"), "n1280 (control)")["under 1.5%"] == "19/20"


def test_v1_val_by_class_the_dates_and_their_keyword_boxes(records):
    finish("n1280-ft-s0", m2b_suite(0.90, keywords={"due": 0.80, "prod": 0.60, "code": 0.40}))
    finish("n1280-ft-s1", m2b_suite(0.92, keywords={"due": 0.82, "prod": 0.70}))     # no code box scored
    finish("s1280-ft-s0", m2b_suite(0.95))
    table = r.class_table(r.arm_suites(("n1280-ft", "s1280-ft")))
    row = cells(table, "n1280-ft")
    assert list(row)[2:] == ["date", "due", "prod", "code"]
    assert (row["seeds"], row["date"], row["prod"], row["code"]) == ("2/3", "0.910 (0.900–0.920)",
                                                                     "0.650 (0.600–0.700)", "0.400")
    assert cells(table, "s1280-ft")["due"] == "–"


def test_step_ones_early_read_is_only_a_check(records):
    suite = m2b_suite(0.80)
    early = {"threshold": suite["threshold_m2"], **{name: suite[name] for name in m2.SUITE}}
    finish("n1280-s0", suite)
    finish("gen-rot-1280-s0", early, milestone="m2")                  # n's early read is M2's own record
    finish("s1280-s0", suite).joinpath("suite.json").write_text(json.dumps(early))
    finish("m1280-s0", suite)                                          # no early read: left out
    line = r.early_read_check()
    assert line.startswith("Step 1's early read") and "for 2 of 9 step-1 runs" in line
    assert "differs by at most 0.0000, and its threshold from the M2b suite's `threshold_m2` by at most 0.0000" in line
    moved = json.loads(json.dumps(early))
    moved["real-val-small"]["map50"]["date"] -= 0.02
    m2b.finished_run("s1280-s0").joinpath("suite.json").write_text(json.dumps(moved))
    assert r.early_read_check().startswith("**Warning:**") and "at most 0.0200" in r.early_read_check()


def test_strata_of_both_synthetic_sets(records):
    finish("n1280-ft-s0", m2b_suite(upside=280))
    finish("n1280-ft-s1", m2b_suite(upside=290))
    suites = r.arm_suites(("n1280-ft",))
    row = cells(r.strata_table(suites), "n1280-ft")
    assert list(row)[2:] == list(r.STRATA_V2)
    assert row["upside-down"] == "285/300 (280–290)" and row["glued"] == "290/300"
    assert cells(r.strata_table(suites, "any_conf"), "n1280-ft")["upside-down"] == "299/300"
    m2_strata = cells(r.strata_table(suites, "iou0.5", "synth-val", m2r.STRATA), "n1280-ft")
    assert list(m2_strata)[2:] == list(m2r.STRATA)


def drift(changed_conv=0):
    def kind(values, changed=0):
        return {"tensors": 1, "values": values, "changed": changed, "max_abs": 0.0, "mean_abs": 0.0}
    return {"layers": 11, "conv": kind(1000, changed_conv), "bn_affine": kind(80), "bn_running_mean": kind(40),
            "bn_running_var": kind(40), "trained": kind(2000, 1990)}


def test_ft2s_frozen_layers(records):
    finish("n1280-ft2-s0", stages=[{"frozen": drift()}, {}])
    table = r.frozen_table()
    row = cells(table, "m2b-n1280-ft2-s0")
    assert row["frozen layers"] == "0 to 10" and row["convolution weights"] == "0 of 1,000"
    assert row["running variances"] == "0 of 40" and row["values trained"] == "1,990 of 2,000 (99.5%)"
    assert cells(table, "m2b-n1280-ft2-s1")["convolution weights"] == "not yet"


def results_csv(folder, times):
    folder.mkdir(exist_ok=True)
    (folder / "results.csv").write_text("epoch,time\n" + "".join(f"{i + 1},{t}\n" for i, t in enumerate(times)))


def test_compute_from_the_records_and_the_pulled_results_csv(records):
    finish("n1280-s0", m2b_suite(minutes=5.4), config={"trained": False})
    applied = {"train_images": 1000, "save_dir": "/cluster/not/here"}
    one = finish("n1280-ft-s0", m2b_suite(minutes=1.8), applied=applied, epochs_run=4, train_minutes=1.0,
                 stages=[{"applied": applied, "epochs_run": 4, "train_minutes": 1.0}])
    results_csv(one, (12, 22, 32, 42))                                 # 10 s an epoch: 100 images/s
    two = finish("n1280-ft2-s0", applied=applied, epochs_run=6, train_minutes=2.0,
                 stages=[{"applied": applied, "epochs_run": 3, "train_minutes": 0.5},
                         {"applied": applied, "epochs_run": 3, "train_minutes": 1.5}])
    results_csv(two.parent / f"{two.name}-stage1", (6, 11, 16))         # 5 s: 200 images/s
    results_csv(two, (20, 40, 60))                                      # 20 s: 50 images/s
    text = r.runs_table(["n1280-s0", "n1280-ft-s0", "n1280-ft2-s0", "n1280-ft2-s1"])
    control = cells(text, "m2b-n1280-s0")
    assert control["trained"] == "no: M2's weights" and control["images/s"] == "–"
    assert control["M2b suite minutes"] == "5.4" and control["M2 suite minutes"] == "–"
    ft = cells(text, "m2b-n1280-ft-s0")
    assert (ft["images/s"], ft["epoch 1 s"], ft["setup minutes"]) == ("100", "12", "0.3")
    ft2 = cells(text, "m2b-n1280-ft2-s0")
    assert ft2["epochs"] == "3 + 3" and ft2["train minutes"] == "0.5 + 1.5"
    assert ft2["images/s"] == "200 + 50" and ft2["M2b suite minutes"] == "–"
    assert cells(text, "m2b-n1280-ft2-s1")["trained"] == "not yet"
    assert "Over these runs: 0.05 h of training (setup included) and 0.12 h of scoring" in text


def test_ns_step_one_compute_is_m2s_own_record(records):
    for s, minutes in zip(m2b.SEEDS, (201.8, 203.4, 204.3)):
        run_dir = finish(f"gen-rot-1280-s{s}", milestone="m2", applied={"train_images": 1000},
                         train_minutes=minutes)
        results_csv(run_dir, (12, 22, 32, 42))                          # 10 s an epoch: 100 images/s
    assert ("`n`'s step 1 is M2's `gen-rot-1280` (3 of 3 seeds, from M2's own records): 203.2 (201.8–204.3) "
            "minutes a run at 100 images a second") in r.control_compute_line()


def test_every_run_and_the_size_arms_take_r2s_recipe_for_s_and_m(records):
    names = r.every_run()
    assert names[:3] == ["n1280-s0", "n1280-s1", "n1280-s2"] and "s1280-ft-s0" in names
    assert "m2b-smoke" not in names and len(names) == 3 * 8
    assert r.size_arms() == ("n1280", "s1280", "m1280", "n1280-ft", "s1280-ft", "m1280-ft")
    arm("n1280", (0.80, 0.80, 0.80))
    arm("n1280-ft", (0.85, 0.85, 0.85))
    arm("n1280-ft2", (0.95, 0.95, 0.95))
    assert {"s1280-ft2-s2", "m1280-ft2-s0"} <= set(r.every_run())
    assert r.size_arms()[3:] == ("n1280-ft2", "s1280-ft2", "m1280-ft2")


def test_m2s_chain_check_still_names_m2s_record(records):
    m2.decisions_path().write_text(json.dumps({}))
    assert m2r.chain_check("pilot") == ""
    m2b.decisions_path().write_text(json.dumps({"noise_store": {"band": 0.03}}))
    assert "**Warning:** the chain has decided this" in r.chain_check("noise_store")


# --- S13: the inference add-ons ------------------------------------------------------------

def r3_decided():
    """The chain's record with R3 at n1280-ft (band 0.10, threshold 0.2), as
    in test_r3_names_the_cheapest_candidate_within_the_band."""
    arm("n1280", (0.70, 0.80, 0.75))
    arm("s1280", (0.76, 0.78, 0.80))
    arm("m1280", (0.85, 0.83, 0.84))
    arm("n1280-ft", (0.90, 0.91, 0.92))
    arm("n1280-ft2", (0.89, 0.89, 0.89))
    arm("s1280-ft", (0.93, 0.94, 0.95))
    arm("m1280-ft", (0.97, 0.98, 0.99))
    chain_writes()


def inference(v1_aps: dict, real=0.99, worst=0.0005):
    """scripts/m2b_inference.py's inference.json: every configuration with
    its v1-val AP from `v1_aps`, R4 and the what-if walked as the script
    walks them."""
    configs = {}
    for k in range(len(m2b.ADDONS) + 1):
        for addons in combinations(m2b.ADDONS, k):
            name, used = m2b.configuration(addons), ensemble.passes(addons)
            v1 = m2b_suite(v1_aps[name])["v1-val"]
            v1["iou0.5"]["boxes_kept"] = 56
            configs[name] = {"addons": list(addons), "passes": [p.name for p in used], "threshold": 0.3,
                             "v1-val": v1, "real-val": entry(real), "cpu_ms_per_photo": 170.0 * len(used)}
    aps = {n: {"v1-val": c["v1-val"]["map50"]["date"], "real-val": real} for n, c in configs.items()}
    decided = m2b.decisions()
    record = {"arm": "n1280-ft", "run": "n1280-ft-s0", "threshold_record": decided["R3"]["threshold"],
              "configurations": configs,
              "sources": {"v1-val": {"processor": "Apple M2", "device": "cpu", "torch_threads": 1,
                                     "torch_threads_used": 7, "photos": 34, "git_commit": "e38ae84c"},
                          "real-val": {"processor": "Xeon", "device": "cpu", "torch_threads": 24,
                                       "torch_threads_used": None, "photos": 165, "git_commit": "e38ae84c"}},
              "ap_check": {"within": 0.01, "largest_real_val_difference": worst, "agrees": worst <= 0.01,
                           "seeds": {f"s{s}": {name: {"ours": 0.95, "ultralytics": 0.95 - worst, "difference": worst}
                                               for name in ("real-val", "v1-val")} for s in m2b.SEEDS}}}
    for key, band in (("R4", decided["noise_store"]["band"]), ("what_if", m2b.WHAT_IF_BAND)):
        walk = m2b.decide_addons(aps, band)
        record[key] = {**walk, "threshold": 0.3 if walk["adopted"] else record["threshold_record"]}
    run_dir = finish("inference")
    (run_dir / "inference.json").write_text(json.dumps(record))
    return run_dir


APS = {"none": 0.94, "ensemble": 0.98, "rotation": 0.97, "imgsz-1536": 0.95, "ensemble+rotation": 0.99,
       "ensemble+imgsz-1536": 0.985, "rotation+imgsz-1536": 0.96, "ensemble+rotation+imgsz-1536": 0.99}


def test_r4_and_the_what_if_are_worked_out_again_and_checked(records):
    r3_decided()
    run_dir = inference(APS)
    block = r.inference_block()
    assert "**Warning:**" not in block
    assert ("**R4, as written (band 0.1000, the noise band): adopts nothing**. Gains on v1-val over what it had "
            "kept, in R4's order: ensemble +0.0400, rotation +0.0300, imgsz-1536 +0.0100") in block
    assert "So the M2b detector is `n1280-ft-s0` alone at its threshold_store, 0.2000." in block
    # The what-if keeps the ensemble (+0.04 > 0.03), then measures rotation on it (+0.01): out.
    assert "(band 0.0300, the plan's floor; it decides nothing): adopts ensemble**" in block
    assert "for the plan's Later, not for S14" in block
    record = json.loads((run_dir / "inference.json").read_text())
    table = r.walk_table(record, "what_if")
    assert [cells(table, a)["kept"] for a in m2b.ADDONS] == ["**kept**", "no", "no"]
    assert cells(table, "rotation")["against"] == "ensemble" and cells(table, "rotation")["gain"] == "+0.0100"
    assert "R4 and the what-if are the record's" in r.r4_check(record)
    record["R4"]["adopted"] = ["ensemble"]                           # a record that disagrees with the rule
    (run_dir / "inference.json").write_text(json.dumps(record))
    assert "**Warning:** R4: worked out again, it adopts nothing, the record ['ensemble']" in r.inference_block()
    record["R4"]["adopted"], record["threshold_record"] = [], 0.5
    (run_dir / "inference.json").write_text(json.dumps(record))
    assert "threshold_record: the record's 0.5, R3's 0.2" in r.inference_block()


def test_every_configuration_with_counts_beside_rates_and_the_ap_check(records):
    r3_decided()
    first = inference(APS)
    record = json.loads((first / "inference.json").read_text())
    first.rename(first.with_name("20261117-000000-m2b-inference"))   # older, and its name free again
    table = r.configuration_table(record)
    none = cells(table, "none")
    assert (none["passes"], none["v1-val AP"], none["dates found"], none["expiry found"]) == ("1", "0.9400", "50/63",
                                                                                             "20/29")
    assert (none["false boxes"], none["CPU ms a photo"]) == ("6", "170")
    assert cells(table, "ensemble+rotation+imgsz-1536")["passes"] == "12"
    assert "v1-val on Apple M2 (cpu, 7 torch threads" in r.sources_line(record)
    assert "real-val on Xeon (cpu, torch's threads not recorded" in r.sources_line(record)
    suites = r.configuration_suites(record)
    assert cells(r.tag_table(suites), "rotation")["sideways"] == "4/6"
    assert cells(r.size_table(suites), "none")["under 1.5%"] == "15/20"
    assert "**The AP check passes**: on real-val ours is within 0.0005" in r.inference_block()
    inference(APS, worst=0.02)                                        # the newest record is read
    assert "**Warning:** our AP and Ultralytics' differ by 0.0200 on real-val" in r.inference_block()


def test_the_timing_and_the_device_check(records):
    r3_decided()
    assert "_The timing: not yet" in r.inference_block() and "_The device check: not yet._" in r.inference_block()
    timing = {"config": {"processor": "Apple M2", "torch_threads": 7, "photos": 34, "photo_long_side_median": 1600,
                         "warmup": 2, "repeats": 3},
              "timed": {"n1280-ft-s0@1280": {"median_ms": 175.0, "mean_ms": 178.0, "repeat_mean_ms": [170, 179, 185]},
                        "m1280-ft-s0@1280": {"median_ms": 819.0, "mean_ms": 905.0, "repeat_mean_ms": [890, 835, 989]}}}
    finish("inference-timing").joinpath("timing.json").write_text(json.dumps(timing))
    table = r.timing_table(r.timing_record())
    assert cells(table, "m1280-ft-s0@1280")["against the first"] == "4.68×"
    assert cells(table, "n1280-ft-s0@1280")["each repeat's mean"] == "170, 179, 185"
    run_dir = m2b.finished_run("n1280-ft-s0")
    here = m2b_suite(0.91)
    here["v1-val"]["iou0.5"]["dates_found"] = 49                     # one date moved
    (run_dir / "device-check-cpu").mkdir()
    (run_dir / "device-check-cpu" / "suite-m2b.json").write_text(json.dumps(here))
    line = r.device_check_line()
    assert line.startswith("**The device check passes**") and "dates found 50/63 on the GPU, 49/63 here" in line
    here["v1-val"]["iou0.5"]["expiry_found"] = 18                    # two more: three in all
    (run_dir / "device-check-cpu" / "suite-m2b.json").write_text(json.dumps(here))
    assert "**Warning:** more than two dates moved." in r.device_check_line()


# --- S14: the final test, store set v2 and ExpDate test ---------------------------------------

def level(expiry, dates, precision=0.9, false=0.2):
    return {"photos": 99, "expiry_found": expiry[0], "expiry_dates": expiry[1], "dates_found": dates[0],
            "dates": dates[1], "precision": precision, "false_boxes_per_photo": false, "boxes_kept": dates[0] + 20}


def v2_record(k=0, unseen_ap=0.7, threshold=0.2, sha="sha", device="cpu"):
    """A store-v2.json whose breakdowns add up to its counts: `k` more
    expiries found, all in unseen designs and month words."""
    design = {"iou0.5": {final_test.SEEN: ((3, 4), (15, 21)), final_test.VALIDATION: ((5, 7), (18, 23)),
                         final_test.UNSEEN: ((32 + k, 57), (117 + k, 168))},
              "any_conf": {final_test.SEEN: ((4, 4), (20, 21)), final_test.VALIDATION: ((7, 7), (22, 23)),
                           final_test.UNSEEN: ((49, 57), (158, 168))}}
    causes = {"iou0.5": (15 + k, 2, 10, 10, 3, 0), "any_conf": (22, 4, 14, 12, 7, 1)}
    of = (25, 6, 15, 12, 9, 1)
    tags = {"dot-matrix": (20, 28), "clean": (12 + k, 24), "sideways": (3, 10), "small-print": (4, 5), "curved": (1, 1)}
    out = {"iou0.5": level((40 + k, 68), (150 + k, 212)), "iou0.3": level((40 + k, 68), (150 + k, 212)),
           "any_conf": level((60, 68), (200, 212), precision=0.3, false=5.0),
           "by_tag": {t: {lv: level(c, (c[0] * 2, c[1] * 2)) for lv in ("iou0.5", "iou0.3", "any_conf")}
                      for t, c in tags.items()},
           "by_design": {g: {lv: level(*design[lv][g]) for lv in design} for g in final_test.DESIGN_GROUPS},
           "by_cause": {c: {lv: {"expiry_found": causes[lv][i], "expiry_dates": of[i]} for lv in causes}
                        for i, c in enumerate(final_test.CAUSES)},
           "ap50": {"date": 0.8, "by_tag": {t: 0.8 for t in tags},
                    "by_design": {final_test.SEEN: 0.9, final_test.VALIDATION: 0.85, final_test.UNSEEN: unseen_ap}},
           "scored": {"set": "store-v2", "threshold": threshold, "weights_sha256": sha, "device": device,
                      "batch": 1}}
    out["by_tag"]["other-face"] = {lv: {"photos": 24, "dates": 11, "dates_found": 5, "precision": 0.5,
                                        "false_boxes_per_photo": 0.4} for lv in ("iou0.5", "iou0.3", "any_conf")}
    return out


def final_folder(name: str, threshold: float = 0.2, v1: float = 0.85):
    """A finished run of any milestone by eval_detector.py's name, its own
    record holding `threshold`."""
    if name.startswith("m2b-"):
        return finish(name[4:], m2b_suite(v1, threshold=threshold))
    if name.startswith("m2-"):
        return finish(name[3:], {"threshold": threshold}, milestone="m2")
    run_dir = runs.start_run(f"m1-{name}", {})
    (run_dir / "val.json").write_text(json.dumps({"threshold": threshold}))
    return run_dir


def scored_v2(name: str, k: int = 0, unseen_ap: float = 0.7, v1: float = 0.85):
    run_dir = final_folder(name, v1=v1)
    threshold, sha = r.final_expected(name)
    (run_dir / "store-v2.json").write_text(json.dumps(v2_record(k, unseen_ap, threshold, sha)))
    return run_dir


def test_the_final_test_with_nothing_in_is_not_yet(records):
    block = r.final_test_block()
    assert "not yet" in block and "**Warning:**" not in block
    assert r.final_check() == "" and r.private_check() == ""


def test_store_v2_every_record_by_tag_cause_and_design_with_rates_and_ranges(records):
    for name in (*final_test.M1_RUNS, *final_test.M2_RUNS, "m2b-n1280-s0", "m2b-n1280-s1", "m2b-n1280-s2"):
        scored_v2(name)
    for s, k in zip(m2b.SEEDS, (8, 9, 10)):
        scored_v2(f"m2b-n1280-ft-s{s}", k)
    rows = r.final_rows()
    table = r.final_table(rows)
    ft = r.FINAL_LABELS["n1280-ft"]
    assert cells(table, ft)["expiry found"] == "49/68, 72% (48–50)"
    assert cells(table, r.FINAL_LABELS["n1280"])["expiry found"] == "40/68, 59%"
    assert cells(table, ft)["all dates found"] == "159/212, 75% (158–160)"
    assert cells(table, ft)["expiry boxed at any confidence"] == "60/68, 88%"
    assert cells(table, "s1280")["expiry found"] == "not yet" and cells(table, "s1280")["seeds"] == "0/3"
    tags = r.found_by(rows, "by_tag")
    assert "other-face" not in tags.splitlines()[0]                      # no expiry there
    assert cells(tags, ft)["clean"] == "21/24, 88% (20–22)"
    assert "other-face" in r.value_by(rows, "by_tag", r.precision_of("by_tag")).splitlines()[0]
    causes = r.found_by(rows, "by_cause")
    assert cells(causes, ft)[final_test.MONTH_WORD] == "24/25, 96% (23–25)"
    assert cells(causes, ft)[final_test.UPSIDE_DOWN] == "0/1, 0%"
    assert cells(r.found_by(rows, "by_cause", "any_conf"), ft)[final_test.UPSIDE_DOWN] == "1/1, 100%"
    designs = r.found_by(rows, "by_design")
    assert cells(designs, ft)[final_test.UNSEEN] == "41/57, 72% (40–42)"
    assert cells(r.found_by(rows, "by_design", what="dates"), ft)[final_test.SEEN] == "15/21, 71%"
    assert cells(r.value_by(rows, "by_design", r.ap_of("by_design")), ft)[final_test.UNSEEN] == "0.700"
    head = r.headline_table()
    assert cells(head, "M2b (`n1280-ft-s0`)")["v2: expiry found"] == "48/68, 71%"
    assert cells(head, "M2b (`n1280-ft-s0`)")["v2: its arm's 3 seeds"] == "49/68, 72% (48–50)"
    assert cells(head, "M2b (`n1280-ft-s0`)")["ExpDate test: expiry found"] == "not yet"
    assert "Every record in so far (12 of 30 on store set v2, 0 of 3 on ExpDate test)" in r.final_check()


def test_learned_packs_reads_unseen_designs_against_the_control_by_ap(records):
    for s, (ap, v1) in enumerate(((0.60, 0.70), (0.62, 0.75), (0.61, 0.72))):
        scored_v2(f"m2b-n1280-s{s}", unseen_ap=ap, v1=v1)
    for arm, aps, v1s in (("n1280-ft", (0.70, 0.72, 0.71), (0.94, 0.95, 0.93)),
                          ("n1280-ft2", (0.58, 0.66, 0.61), (0.93, 0.94, 0.93)),
                          ("n1280-ft-gen50k", (0.50, 0.52, 0.55), (0.92, 0.93, 0.74))):
        for s, (ap, v1) in enumerate(zip(aps, v1s)):
            scored_v2(f"m2b-{arm}-s{s}", unseen_ap=ap, v1=v1)
    table = r.packs_table()
    assert cells(table, "n1280-ft")["v2 unseen, against the control"] == "lifted: ranges apart"
    assert cells(table, "n1280-ft")["v1-val, against the control"] == "lifted: ranges apart"
    assert cells(table, "n1280-ft2")["v2 unseen, against the control"] == "not lifted: ranges overlap"
    assert cells(table, "n1280-ft-gen50k")["v2 unseen, against the control"] == "lowered: ranges apart"
    assert cells(table, "n1280-ft-gen50k")["v1-val, against the control"] == "not lifted: ranges overlap"
    assert cells(table, "s1280-ft")["v2 unseen, against the control"] == "not yet"
    assert cells(table, "n1280 (control)")["v2 unseen: AP (ours)"] == "0.610 (0.600–0.620)"


def test_expdate_test_sets_the_m2b_detector_beside_m1_and_m2_at_both_thresholds(records):
    def test_record(found, extra=None):
        return {"threshold": 0.3, "iou0.5": level((found, 665), (found + 68, 733)), "map50": {"date": 0.995},
                **(extra or {})}

    for name, found in zip(final_test.M1_RUNS, (637, 651, 650)):           # M1's records: no any_conf, no AP of ours
        final_folder(name).joinpath("real-test.json").write_text(json.dumps(test_record(found)))
    for name in final_test.M2_RUNS:
        final_folder(name).joinpath("real-test.json").write_text(json.dumps(test_record(662, {
            "any_conf": level((664, 665), (731, 733))})))
    for name in final_test.EXPDATE_TEST:
        extra = {"any_conf": level((665, 665), (733, 733)), "ap50_ours": {"date": 0.994},
                 "at_threshold_m2": {"threshold": 0.4, "iou0.5": level((660, 665), (726, 733), precision=0.99)}}
        final_folder(name).joinpath("real-test.json").write_text(json.dumps(test_record(663, extra)))
    table = r.expdate_table()
    assert cells(table, m2r.M1_ROW)["expiry found"] == "646/665, 97% (637–651)"
    assert cells(table, m2r.M1_ROW)["expiry boxed at any confidence"] == "–"
    assert cells(table, r.M2_ROW)["AP@0.5 date (ours)"] == "–"
    store_row, m2_row = (cells(table, f"M2b (`n1280-ft`) at {t}") for t in ("threshold_store", "M2's rule's threshold"))
    assert (store_row["expiry found"], m2_row["expiry found"]) == ("663/665, 100%", "660/665, 99%")
    assert (store_row["threshold"], m2_row["threshold"]) == ("0.3000", "0.4000")
    assert m2_row["precision"] == "0.990" and m2_row["expiry boxed at any confidence"] == "665/665, 100%"


def test_the_final_check_names_a_wrong_threshold_hash_device_or_breakdown(records):
    run_dir = scored_v2("m2b-n1280-ft-s0")
    assert r.final_check().startswith("Every record in so far (1 of 30")
    good = json.loads((run_dir / "store-v2.json").read_text())
    for change, words in (({"threshold": 0.2001}, "threshold 0.2001, not 0.2"),
                          ({"weights_sha256": "abc"}, "weights' SHA-256 'abc'"),
                          ({"device": "mps"}, "device 'mps', not 'cpu'"),
                          ({"batch": 16}, "photos a call 16, not 1")):
        (run_dir / "store-v2.json").write_text(json.dumps({**good, "scored": {**good["scored"], **change}}))
        assert "**Warning:**" in r.final_check() and words in r.final_check()
    bad = json.loads(json.dumps(good))
    bad["by_cause"][final_test.MONTH_WORD]["iou0.5"]["expiry_found"] += 1
    (run_dir / "store-v2.json").write_text(json.dumps(bad))
    assert "its causes give 41/68, its counts 40/68" in r.final_check()


def test_the_kept_boxes_give_each_records_counts_again(records, monkeypatch, tmp_path):
    from pathlib import Path

    from bestbefore.samples import Box, Sample
    monkeypatch.setattr(final_test, "RUNS_DIR", tmp_path / "v2-runs")
    photo = Sample("testset2/a.jpeg", Path("a.jpeg"), 1200, 1600, (Box("expiry", (0.1, 0.5, 0.4, 0.52)),), tag="clean")
    v2 = final_test.V2([photo], {photo.id: final_test.UNSEEN}, {photo.id: ["numbers: print 1.5 to 2.5%"]})
    run_dir = final_folder("m2b-n1280-ft-s0")
    threshold, sha = r.final_expected("m2b-n1280-ft-s0")
    record = {"iou0.5": {"dates_found": 1, "expiry_found": 1, "boxes_kept": 2},
              "scored": {"threshold": threshold, "weights_sha256": sha, "device": "cpu"}}
    (run_dir / "store-v2.json").write_text(json.dumps(record))
    assert "no per-photo boxes" in r.private_check(load=lambda: v2)
    kept = runs.start_run("detector-store-v2-m2b-n1280-ft-s0", {}, root=final_test.RUNS_DIR)
    boxes = {photo.id: [[0.1, 0.5, 0.4, 0.52, 0.9], [0.6, 0.6, 0.7, 0.7, 0.3], [0.0, 0.0, 0.1, 0.1, 0.1]]}
    (kept / "predictions.json").write_text(json.dumps(boxes))
    assert r.private_check(load=lambda: v2) == ("For all 1 v2 records in, the per-photo boxes kept in "
                                                 "`data/store/v2/runs/` give the record's counts again.")
    boxes[photo.id][1][4] = 0.1                                          # a box the record kept, now under the threshold
    (kept / "predictions.json").write_text(json.dumps(boxes))
    assert "boxes_kept 1 from its boxes, 2 in its record" in r.private_check(load=lambda: v2)

    def unreadable():
        raise RuntimeError("no store photos here")
    assert "weren't checked here" in r.private_check(load=unreadable)


def test_each_final_record_expects_its_own_gpu_threshold_and_its_milestones_hash(records):
    from bestbefore import m1
    for name, threshold in (("n640", 0.7279), ("m2-gen-rot-1280-s0", 0.3424), ("m2b-n1280-s0", 0.0055),
                            ("m2b-n1280-ft-s0", 0.1641)):
        final_folder(name, threshold=threshold)
    assert r.final_expected("n640") == (0.7279, m1.BEST_PT_SHA256["n640"])
    assert r.final_expected("m2-gen-rot-1280-s0") == (0.3424, m2.LAST_PT_SHA256["gen-rot-1280-s0"])
    assert r.final_expected("m2b-n1280-s0") == (0.0055, m2.LAST_PT_SHA256["gen-rot-1280-s0"])    # the control: M2's
    assert r.final_expected("m2b-n1280-ft-s0") == (0.1641, m2b.LAST_PT_SHA256["n1280-ft-s0"])
    assert r.final_expected("m2b-s1280-s0") == (None, None)                                     # no such run here



def test_the_device_checks_at_16_and_at_one_photo_a_call(records):
    for run in r.DEVICE_CHECK_RUNS:
        finish(run, m2b_suite(0.95))
    table = r.device_checks_table()
    assert cells(table, "m2b-n1280-ft-s0")["check"] == "not yet"
    run_dir = m2b.finished_run("s1280-ft-s0")
    here = m2b_suite(0.94)
    here["v1-val"]["iou0.5"]["dates_found"] = 49                       # one date moved
    for folder, batch in (("device-check-cpu", None), ("device-check-cpu-batch1", 1)):
        (run_dir / folder).mkdir()
        (run_dir / folder / "suite-m2b.json").write_text(json.dumps(
            here | {"scored": {"device": "cpu"} | ({"batch": batch} if batch else {})}))
    rows = [line for line in r.device_checks_table().splitlines() if "m2b-s1280-ft-s0" in line]
    assert len(rows) == 2 and all("| passes |" in line for line in rows)
    assert "| 1 | 50/63 → 49/63 | 20/29 → 20/29 | 60/63 → 60/63 | 0.9500 → 0.9400 | 1 | passes |" in rows[1]
    assert "**Warning:**" not in r.device_checks_table()
    here["v1-val"]["any_conf"]["dates_found"] = 58                     # two more: three in all
    (run_dir / "device-check-cpu-batch1" / "suite-m2b.json").write_text(json.dumps(here | {"scored": {"batch": 1}}))
    assert "**Warning:** a device check didn't pass: m2b-s1280-ft-s0 at 1 a call: fails." in r.device_checks_table()
    (run_dir / "device-check-cpu-batch1" / "suite-m2b.json").write_text(json.dumps(m2b_suite(0.95)))  # made at 16
    assert "made at 16 a call" in r.device_checks_table()
