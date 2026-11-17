"""M2b's results tables (docs/m2b-store-and-size.md, S11 and S12), on
made-up run records: the rules as summarise.py works them out again and
checks them against the chain's record, R3 read back, every table, ft2's
frozen layers, compute, and records not in yet shown as "not yet"."""
import json

import pytest

from bestbefore import m2, m2b, runs
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
    for block in (r.fine_tune_block(), r.gen_v2_block(), r.model_size_block()):   # every table builds on nothing
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
