import json

from bestbefore.m1 import END, START, decide_c1, keep_synthetic, load_run, replace_results


def scores(**map50):
    return {name: {"map50": {"date": v}} for name, v in map50.items()}


def test_clear_winner():
    assert decide_c1(scores(n640=0.80, n1280=0.84, s640=0.82, s1280=0.90)) == "s1280"


def test_within_one_point_the_cheapest_wins():
    # s1280 is best, but n1280 is within 0.01 and comes first in the order.
    assert decide_c1(scores(n640=0.80, n1280=0.895, s640=0.89, s1280=0.90)) == "n1280"
    # Exactly 1 point behind still counts as a tie.
    assert decide_c1(scores(n640=0.89, n1280=0.85, s640=0.85, s1280=0.90)) == "n640"


def test_keep_synthetic_unless_real_only_clearly_wins():
    assert keep_synthetic({"map50": {"date": 0.90}}, {"map50": {"date": 0.905}})
    assert keep_synthetic({"map50": {"date": 0.90}}, {"map50": {"date": 0.85}})
    assert not keep_synthetic({"map50": {"date": 0.90}}, {"map50": {"date": 0.92}})


def test_replace_results_keeps_the_rest_of_the_readme():
    readme = f"# Title\n\n{START}\nold table\n{END}\n\n## What it means\n"
    out = replace_results(readme, "new table")
    assert out == f"# Title\n\n{START}\nnew table\n{END}\n\n## What it means\n"
    assert replace_results(out, "newer").count(START) == 1


def test_load_run_takes_the_newest_finished_run(tmp_path):
    for stamp, finished in (("20261015-100000", True), ("20261015-110000", True),
                            ("20261015-120000", False)):     # newest crashed
        d = tmp_path / f"{stamp}-m1-n640"
        d.mkdir()
        (d / "run.json").write_text(json.dumps({"stamp": stamp}))
        if finished:
            (d / "val.json").write_text("{}")
    record, _ = load_run("n640", tmp_path)
    assert record["stamp"] == "20261015-110000"
    assert load_run("s640", tmp_path) is None
    assert load_run("n640-real", tmp_path) is None    # n640 doesn't match n640-real
