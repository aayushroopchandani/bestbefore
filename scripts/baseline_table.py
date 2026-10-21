"""Write results/m0_baselines.md: baselines A and B side by side, from their
latest runs (M0's deliverable, docs/m0-data-and-test-set.md, C3).

    python scripts/baseline_table.py

Reads the run.json of the newest run of each baseline on each set (made by
scripts/run_baseline.py) and nothing else, so the table holds exactly what
the runs measured. Counts are k/n: with 16 to 33 store photos per tag, one
photo moves a rate by 3 to 6 points.
"""
from __future__ import annotations

import json
from pathlib import Path

from bestbefore import store
from bestbefore.config import REPO_ROOT
from bestbefore.runs import latest_run
from bestbefore.scoring import OUTCOMES
from bestbefore.sets import runs_root
from bestbefore.tables import markdown_table

OUT = REPO_ROOT / "results" / "m0_baselines.md"
READERS = ("a", "b")


def load(reader: str, set_name: str) -> tuple[Path, dict]:
    run = latest_run(f"baseline-{reader}-{set_name}", runs_root(set_name))
    if run is None:
        raise SystemExit(f"no baseline {reader} run on {set_name}: "
                         f"python scripts/run_baseline.py --reader {reader} --set {set_name}")
    return run, json.loads((run / "run.json").read_text())


def cells(t: dict | None) -> dict[str, str]:
    """Right / wrong / none for one reader, as k/n."""
    return {o: f"{t[o]}/{t['n']}" if t else "-" for o in OUTCOMES}


def row(name: str, n: int, per_reader: dict[str, dict | None]) -> dict[str, str]:
    out = {"": name, "n": f"{n}"}
    for reader, t in per_reader.items():
        out.update({f"{reader.upper()} {o}": v for o, v in cells(t).items()})
    return out


def expdate_table(runs: dict[str, dict]) -> str:
    s = {r: runs[r]["summary"] for r in READERS}
    table = markdown_table([row("photos with a marked expiry", s["a"]["all"]["n"],
                                {r: s[r]["all"] for r in READERS})], left=("",))
    skipped = {r: s[r]["not_scored"].get("no expiry marked", {"n": 0, "date_read": 0})
               for r in READERS}
    return (table + f"\n\nNot scored: {skipped['a']['n']} photos with no marked expiry "
            f"(A read a date from {skipped['a']['date_read']}, "
            f"B from {skipped['b']['date_read']}).")


def store_table(runs: dict[str, dict]) -> str:
    s = {r: runs[r]["summary"] for r in READERS}
    rows = []
    for tag in store.TAGS[1:]:          # other-face has no expiry to read: below
        per_reader = {r: s[r]["by_tag"].get(tag) for r in READERS}
        rows.append(row(tag, (per_reader["a"] or {"n": 0})["n"], per_reader))
    rows.append(row("**all readable**", s["a"]["all"]["n"], {r: s[r]["all"] for r in READERS}))
    other = {r: s[r]["not_scored"].get("other-face", {"n": 0, "date_read": 0}) for r in READERS}
    return (markdown_table(rows, left=("",))
            + f"\n\n`other-face` photos (expiry not in the photo, so any date read is "
            f"the wrong one): A returned a date for {other['a']['date_read']}/{other['a']['n']}, "
            f"B for {other['b']['date_read']}/{other['b']['n']}.")


def provenance(set_name: str, reader: str, run: Path, record: dict) -> str:
    commit = record["git_commit"][:7] + (" + uncommitted changes" if record["git_dirty"] else "")
    line = f"- {reader.upper()} on {set_name}: run `{run.name}`, this repo at `{commit}`"
    if "scanify_commit" in record["config"]:
        line += f", Scanify at `{record['config']['scanify_commit'][:7]}`"
    if "reproduction" in record:
        c = record["reproduction"]
        line += (f"; on Scanify's own {c['n']} photos: {c['right']} right, {c['wrong']} wrong, "
                 f"{c['none']} none ({'reproduces' if c['matches_scanify'] else 'does NOT reproduce'}"
                 " Scanify's 32 / 0 / 12)")
    return line + "."


def main() -> None:
    sets = {"real-test": {}, "store": {}}
    lines = []
    for set_name in sets:
        for reader in READERS:
            run, record = load(reader, set_name)
            sets[set_name][reader] = record
            lines.append(provenance(set_name, reader, run, record))

    md = (
        "# M0 baselines\n\n"
        "The two readers every later model has to beat, scored the same way\n"
        "(`bestbefore.scoring`): **right** is the exact day, month and year, or\n"
        "month and year when the pack prints no day; **wrong** is any other date;\n"
        "**none** is no date. Wrong is worse than none: a wrong date can mark a\n"
        "good batch as expired.\n\n"
        "- **A:** RapidOCR 1.2.3 on the whole upright photo, then a date regex,\n"
        "  then the latest date found.\n"
        "- **B:** Scanify's reader (the same RapidOCR plus resizing, sharpening\n"
        "  and hand-written rules), run in its own environment.\n\n"
        "## ExpDate Products-Real test\n\n" + expdate_table(sets["real-test"]) + "\n\n"
        f"## Store photos (test set v1), by failure tag\n\n" + store_table(sets["store"]) + "\n\n"
        "No store photo is tagged `curved`: no date on a cap or jar visibly bends its\n"
        "line. Tags and counts: [`store_test_set.md`](store_test_set.md).\n\n"
        "## Runs\n\nMade by `python scripts/baseline_table.py` from:\n\n"
        + "\n".join(lines) + "\n")
    OUT.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
