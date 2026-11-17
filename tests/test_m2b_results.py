"""M2b's results tables (docs/m2b-store-and-size.md, S11), on made-up run
records: the rules as summarise.py works them out again and checks them
against the chain's record, every table, ft2's frozen layers, compute, and
records not in yet shown as "not yet"."""
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


def entry(map50, found=50, total=63, expiry=(20, 29), m2_found=40, boxed=60):
    """One set of an M2b suite at both thresholds."""
    return {"iou0.5": summ(found, total, expiry), "iou0.3": summ(found, total, expiry),
            "any_conf": summ(boxed, total, expiry, precision=0.2, false=9.0),
            "at_threshold_m2": {"iou0.5": summ(m2_found, total, (expiry[0] - 5, expiry[1]), precision=0.95),
                                "iou0.3": summ(m2_found, total, (expiry[0] - 5, expiry[1]))},
            "map50": {"date": map50}}


def m2b_suite(v1=0.85, real=0.99, synth=0.99, threshold=0.2, threshold_m2=0.4, minutes=2.0, upside=250):
    """A whole suite-m2b.json: v1-val by tag and by size, both synthetic
    sets by stratum."""
    out = {"threshold": threshold, "threshold_m2": threshold_m2, "scored": {"minutes": minutes},
           **{name: entry(0.95) for name in m2b.SUITE}}
    out["real-val"], out["synth-val"], out["v1-val"] = entry(real), entry(synth), entry(v1)
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
    assert "R1: not yet" in r.r1_line() and "R2: not yet" in r.r2_line()
    assert r.record_check() == ""
    assert cells(r.r2_table(), "n1280-ft")["guard"] == "not yet"
    for block in (r.fine_tune_block(), r.gen_v2_block()):        # every table builds on nothing
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


def test_every_run_takes_r2s_recipe_for_s_and_m(records):
    names = r.every_run()
    assert names[:3] == ["n1280-s0", "n1280-s1", "n1280-s2"] and "s1280-ft-s0" in names
    assert "m2b-smoke" not in names and len(names) == 3 * 8
    arm("n1280", (0.80, 0.80, 0.80))
    arm("n1280-ft", (0.85, 0.85, 0.85))
    arm("n1280-ft2", (0.95, 0.95, 0.95))
    assert {"s1280-ft2-s2", "m1280-ft2-s0"} <= set(r.every_run())


def test_m2s_chain_check_still_names_m2s_record(records):
    m2.decisions_path().write_text(json.dumps({}))
    assert m2r.chain_check("pilot") == ""
    m2b.decisions_path().write_text(json.dumps({"noise_store": {"band": 0.03}}))
    assert "**Warning:** the chain has decided this" in r.chain_check("noise_store")
