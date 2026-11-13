"""Rebuild the crop experiment's tables (M3 S2) from its reads.

    scripts/pull_cluster_results.sh          # the run record and results/runs/m3-decisions.json
    # the run folder's reads into data/store/runs/, owner-only (README, "Reproduce")
    python experiments/m3-crop-reading/summarise.py [RUN]

RUN is the laptop's copy of the job's run folder (crops.json, reads.jsonl,
read.json), by default the newest data/store/runs/*-m3-crop-experiment.
Every read is scored again with the frozen parser (m3.check_parser,
crop_scores.score_rows); the counts are checked against the job's
summary.json and R1 (m3.crop_recipe) against results/runs/m3-decisions.json,
and the script exits with 1 if either differs. Store set v1's expiry crops
are also set beside baselines A and B's reads of the whole photo (M0's
per-photo records in data/store/runs/, joined through
data/store/train_names.json: laptop only). Prints counts only, never a
photo, a crop or a photo's name.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from bestbefore import m3, runs, store
from bestbefore.reader import crop_scores
from bestbefore.scoring import OUTCOMES

STORE_RUNS = store.RUNS_DIR
TAGS = [t for t in store.TAGS]


def cell(t: dict | None) -> str:
    return f"{t['right']} / {t['wrong']} / {t['none']}" if t else "–"


def configs():
    for reader in m3.CROP_READERS:
        for mode in m3.CROP_MODES:
            for pad in m3.PADS:
                yield reader, mode, pad, crop_scores.config_name(reader, mode, pad)


def table(head: list[str], rows: list[list[str]], left: int = 3) -> str:
    out = ["| " + " | ".join(head) + " |",
           "|" + "---|" * left + "---:|" * (len(head) - left)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def whole_photo(rows: list[dict], pad: float) -> str:
    """v1's expiry crops against baselines A and B on the whole photo, by tag."""
    names = json.loads((store.DATA_DIR / "train_names.json").read_text())["photos"]
    base = {}
    for reader in ("a", "b"):
        run = sorted(STORE_RUNS.glob(f"*-baseline-{reader}-store"))[0]   # M0's, the first run
        with open(run / "results.jsonl") as f:
            base[reader] = {r["id"]: r["score"] for r in map(json.loads, f)}
    picks = [("rapidocr", "full"), ("rapidocr", "turned"), ("ppocrv6", "full"), ("ppocrv6", "turned")]
    by_tag: dict[str, dict[str, Counter]] = {}
    for r in rows:
        if r["set"] != "store" or r["label"] != "expiry" or not r["photo"].startswith("v1-") or r["pad"] != pad:
            continue
        if (r["reader"], r["mode"]) not in picks:
            continue
        photo = names[r["photo"]]["photo"]
        for tag in (r["tag"], "all"):
            group = by_tag.setdefault(tag, {})
            group.setdefault(f"{r['reader']}/{r['mode']}", Counter())[r["outcome"]] += 1
            if r["reader"] == "rapidocr" and r["mode"] == "full":    # once per photo
                for reader in ("a", "b"):
                    group.setdefault(reader.upper(), Counter())[base[reader][photo]] += 1
    cols = ["A", "B"] + [f"{a}/{b}" for a, b in picks]
    rows_out = []
    for tag in [*TAGS, "all"]:
        if tag not in by_tag:
            continue
        g = by_tag[tag]
        n = sum(g["A"].values())
        rows_out.append([tag if tag != "all" else "**all**", str(n),
                         *(cell({o: g[c][o] for o in OUTCOMES}) for c in cols)])
    head = ["v1 expiries, tag", "n", "A: whole photo", "B: whole photo",
            *(f"{c}, pad {pad:g}" for c in cols[2:])]
    return table(head, rows_out, left=1)


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(STORE_RUNS.glob("*-m3-crop-experiment"))[-1]
    info = json.loads((run / "crops.json").read_text())
    with open(run / "reads.jsonl") as f:
        reads = [json.loads(line) for line in f]
    digest = m3.check_parser()
    rows = crop_scores.score_rows(info["crops"], reads)
    counts, ms = crop_scores.tally(rows), crop_scores.mean_ms(rows)
    recipe = m3.crop_recipe(rows)

    record = runs.RUNS_DIR / Path(info["run"]).name
    summary = runs.read_json(record / "summary.json")
    decisions = runs.read_json(m3.DECISIONS_PATH) or {}
    problems = []
    if summary is None:
        problems.append(f"no {record / 'summary.json'}: pull the run records first")
    elif summary["counts"] != counts or summary["crop_recipe"] != recipe or summary["parse_sha256"] != digest:
        problems.append("the job's summary.json differs from the reads scored again here")
    kept = decisions.get("crop_recipe", {})
    if {k: kept.get(k) for k in recipe} != recipe or kept.get("run") != record.name:
        problems.append("results/runs/m3-decisions.json's crop_recipe isn't R1 on this run")

    read_info = json.loads((run / "read.json").read_text())
    n = Counter(c["set"] for c in info["crops"])
    print(f"Run {record.name}: {n['real-val']} real-val crops, {n['store']} store crops, "
          f"{read_info['reads']} reads in {read_info['read_seconds']:.0f} s on {read_info['threads']} threads "
          f"of {read_info['host']}; parser {digest[:12]}.\n")

    print("### 1. Right / wrong / none by configuration\n")
    sets = ["real-val", "store", "store-train", "store-val"]
    rows1 = []
    for reader, mode, pad, cfg in configs():
        c = counts[cfg]
        rows1.append([reader, mode, f"{pad:g}", *(cell(c.get(s)) for s in sets),
                      f"{c['real-val']['exact']}", str(ms[cfg])])
    print(table(["reader", "runs", "pad", *sets, "real-val exact", "ms a crop"], rows1))
    rv = [c for c in info["crops"] if c["set"] == "real-val"]
    truths = [crop_scores.truth_of(c) for c in rv]
    print(f"\nreal-val: {sum(t is None for t, _ in truths)} of {len(rv)} transcriptions give no date "
          f"(left out of right / wrong / none); {sum(w for _, w in truths)} gives one only by the whole-read rule.")
    # RapidOCR's full pipeline reads a crop 30 px high or less whole, without its detector (config.yaml).
    low = {pad: sum(c["box_px"][3] - c["box_px"][1] + 2 * pad * min(c["box_px"][2] - c["box_px"][0],
                                                                    c["box_px"][3] - c["box_px"][1]) <= 30
                    for c in rv) for pad in m3.PADS}
    print("real-val crops about 30 px high or less (RapidOCR's full pipeline skips its detector), by pad: "
          + ", ".join(f"{p:g}: {n}" for p, n in low.items()) + ".")

    print("\n### 2. Store crops by tag\n")
    store_n = {t: counts["rapidocr/full/0"].get(f"tag:{t}", {}).get("n", 0) for t in TAGS}
    tags = [t for t in TAGS if store_n[t]]
    rows2 = [[reader, mode, f"{pad:g}", *(cell(counts[cfg].get(f"tag:{t}")) for t in tags)]
             for reader, mode, pad, cfg in configs()]
    print(table(["reader", "runs", "pad", *(f"{t} ({store_n[t]})" for t in tags)], rows2))

    print("\n### 3. Store crops by shape and label\n")
    groups = ["shape:wide", "shape:tall", "label:expiry", "label:mfg"]
    rows3 = [[reader, mode, f"{pad:g}", *(cell(counts[cfg].get(g)) for g in groups)]
             for reader, mode, pad, cfg in configs()]
    shape_n = {g: counts["rapidocr/full/0"].get(g, {}).get("n", 0) for g in groups}
    print(table(["reader", "runs", "pad", *(f"{g.split(':')[1]} ({shape_n[g]})" for g in groups)], rows3))

    turned = Counter((r["reader"], r["shape"], r["k"]) for r in rows if r["mode"] == "turned" and r["set"] == "store")
    multi = sum(r["dates"] > 1 for r in rows)
    print(f"\nTurned reads kept, store crops, all pads: "
          + "; ".join(f"{reader} {shape} at {k * 90}°: {v}" for (reader, shape, k), v in sorted(turned.items()))
          + f". Reads holding more than one date: {multi} of {len(rows)}.")

    print("\n### 4. R1\n")
    print(f"Right reads, both readers alone with turning, all {len(info['crops'])} crops, by pad: "
          + ", ".join(f"{p}: {v}" for p, v in recipe["right_alone_turned_by_pad"].items())
          + f". **Pad {recipe['pad']:g}.**")
    t = recipe["tall_at_pad"]
    print(f"Tall crops at that pad, both readers: as cut {cell(t['alone'])}, turned {cell(t['turned'])}. "
          f"**Turning {'kept' if recipe['turn'] else 'not kept'}.**")
    used = m3.crop_recipe_used({"crop_recipe": {**recipe, "run": record.name}})
    print(f"Used by M3 (m3.crop_recipe_used): pad {used['pad']:g}, turning {'kept' if used['turn'] else 'off'}"
          + (f"; the owner's departure from R1 on {', '.join(used['departures'])}." if used["departures"] else "."))
    wrong = Counter((r["reader"], r["label"]) for r in rows
                    if r["set"] == "store" and r["mode"] == "turned" and r["pad"] == used["pad"] and r["outcome"] == "wrong")
    print("Wrong reads on store crops, alone with turning at that pad, by label: "
          + (", ".join(f"{reader} {label} {n}" for (reader, label), n in sorted(wrong.items())) or "none") + ".")

    print(f"\n### 5. v1's expiry crops against the whole photo (M0's baselines), pad {recipe['pad']:g}\n")
    print(whole_photo(rows, recipe["pad"]))

    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("\nThe job's summary and R1's record agree with the reads scored again here.")


if __name__ == "__main__":
    main()
