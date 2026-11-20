"""Score trained M1, M2 and M2b detectors on a labelled set, at their validation threshold.

    python scripts/eval_detector.py --set real-test --runs n640 n640-seed1 n640-seed2
    python scripts/eval_detector.py --set store --runs n640 n640-seed1 n640-seed2
    python scripts/eval_detector.py --set real-val --runs n640     # device check
    python scripts/eval_detector.py --set store --runs m2-base-s0 m2-rot-s0
    python scripts/eval_detector.py --set store-v2 --dry-run --runs n640 ...   # the 30 of final_test.RUNS
    python scripts/eval_detector.py --set real-test --runs m2b-n1280-ft-s0 m2b-n1280-ft-s1 m2b-n1280-ft-s2

A name starting with `m2b-` is an M2b run (results/runs/<id>-m2b-<run>/),
one starting with `m2-` an M2 run; any other is M1's (`m1-n640` and `n640`
are the same run). Each run is scored with the confidence threshold it chose
on the validation photos, unchanged: nothing is tuned here. M1's is in its
val.json; M2's is the pooled F2 threshold in its suite.json; M2b's is
threshold_store in its suite-m2b.json (the best F2 on v1-val, M2b decision
13), and on ExpDate test the counts at M2's rule's threshold_m2 sit beside
it, from the same predictions. Every threshold is used exactly as stored,
never rounded (docs/m2-synthetic-generator.md, S7). Writes
results/runs/<run>/<set>.json (counts only) and prints one table, with mean
and range over the runs; on store photos, a second table by failure tag.

The store photos' per-photo predictions, which name photos, go elsewhere:
store set v1's to data/store/runs/, for scripts/detector_misses.py and M2's
store-by-cause tables; store set v2's to data/store/v2/runs/.

`store-v2` is store set v2, the test set once store photos are trained on
(docs/m0-data-and-test-set.md, Part D), scored once at the end of M2b
(docs/m2b-store-and-size.md, S14). Only this script knows the name: it isn't
one of bestbefore.sets' names, so no other script can load v2 by name. It
scores only the final test's records (bestbefore.final_test.RUNS), each
once: --again is refused, and so is a run whose per-photo folder already
exists, or whose weights have no recorded SHA-256. Its record also holds
the counts by pack design and by cause, and our AP (final_test.breakdown).
Runs that share their weights (M2's and M2b's control) share one pass.
v2 is predicted one photo a call (final_test.BATCH: 16 at 1,280 px overflow
the 8 GB laptop), any other set 16 a call unless --batch says otherwise;
--batch reaches Ultralytics' validation too (ExpDate test, one a call on the
laptop: M2b S14's change of 2026-11-20).

The ExpDate test split and the store photos are each meant to be scored
once (docs/m1-date-detector.md, decision 6; M2 decision 17). So a run
already scored on either is refused unless --again is given on purpose
(never for store-v2). --dry-run checks everything a score needs (each run,
its threshold, its weights and their hash, the guard; for store-v2 the
frozen labels, design ids and splits too) and stops: nothing is predicted
or written. Every run is checked before any is scored.

Weights are read from $CHECKPOINT_DIR/m1/<run>/weights/best.pt (M1),
$CHECKPOINT_DIR/m2/<run>/weights/last.pt (M2, decision 10), and M2b's
last.pt where its run.json says (the cluster), else from
$CHECKPOINT_DIR/m2b/<run>/weights/last.pt; M2b's control reads M2's. Each is
checked against its SHA-256 where one is recorded (m1.BEST_PT_SHA256,
m2.LAST_PT_SHA256, m2b.LAST_PT_SHA256) before it predicts. Runs trained on
the cluster need their weights copied first, e.g.
    rsync -a bbhpc:bestbefore-data/checkpoints/m1/<run>/weights/best.pt \\
        data/checkpoints/m1/<run>/weights/
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bestbefore import detector, final_test, m1, m2, m2b, runs, store, tables
from bestbefore.config import checkpoint_dir, expdate_dir
from bestbefore.detection_eval import average_precision, evaluate, summary
from bestbefore.runs import git_state, start_run, write_json
from bestbefore.sets import NAMES, load_set, runs_root
from bestbefore.yolo import CLASSES

SPLITS = {"real-test": "test", "real-val": "val"}   # sets with YOLO labels
SCORED_ONCE = ("real-test", "store", final_test.SET)
PRIVATE = ("store", final_test.SET)                 # sets whose per-photo boxes name photos
SETS = (*NAMES, final_test.SET)


@dataclass(frozen=True)
class Detector:
    """A trained run, as this script scores it."""
    name: str               # n640 (M1), m2-base-s0 (M2) or m2b-n1280-ft-s0 (M2b)
    milestone: str          # m1, m2 or m2b
    run_dir: Path           # its record, results/runs/<id>-m1-n640/
    threshold: float        # chosen on validation, used unchanged
    imgsz: int
    train: list[str]        # the dataset file's train lists (needed for mAP, never read)
    weights: Path           # M1 kept the best epoch on validation; M2 and M2b the final weights
    sha256: str | None = None           # what the weights must hash to, where recorded
    threshold_from: str = ""            # the record and rule the threshold comes from
    threshold_m2: float | None = None   # M2b: M2's rule's threshold, recorded beside it

    @property
    def checkpoints(self) -> Path:
        return checkpoint_dir() / self.milestone / self.run_dir.name


def find(name: str) -> Detector:
    """The newest finished run with this name, and how to score it."""
    if name.startswith("m2b-"):
        return _m2b_run(name)
    if name.startswith("m2-"):
        return _m2_run(name)
    name = name.removeprefix("m1-")
    run_dir = m1.finished_run(name)
    if run_dir is None:
        raise SystemExit(f"no finished run called {name!r} in results/runs")
    cfg = json.loads((run_dir / "run.json").read_text())["config"]
    val = json.loads((run_dir / "val.json").read_text())
    return Detector(name, "m1", run_dir, val["threshold"], cfg["imgsz"], cfg["train"].split(","),
                    checkpoint_dir() / "m1" / run_dir.name / "weights" / "best.pt",
                    m1.BEST_PT_SHA256.get(name), "val.json: M1's best F1 for dates on real-val")


def _m2_run(name: str) -> Detector:
    """An M2 run: its final weights and the threshold in its suite.json."""
    run = name.removeprefix("m2-")
    run_dir = m2.finished_run(run)
    if run_dir is None:
        raise SystemExit(f"no finished run called {name!r} in results/runs")
    suite = m2.read_json(run_dir / "suite.json")
    if suite is None:
        raise SystemExit(f"{name} has no suite.json yet, so no threshold: "
                         f"python scripts/score_suite.py --run {run}")
    cfg = m2.read_json(run_dir / "run.json")["config"]
    # Validating never reads the train entry, and an arm's own lists
    # (turned copies, generated photos) aren't on the laptop.
    return Detector(name, "m2", run_dir, suite["threshold"], cfg["imgsz"], ["train_real"],
                    checkpoint_dir() / "m2" / run_dir.name / "weights" / "last.pt",
                    m2.LAST_PT_SHA256.get(run), "suite.json: M2's best F2 for dates on the real sets, pooled")


def _m2b_run(name: str) -> Detector:
    """An M2b run: its final weights (the control's are M2's run's), its
    threshold_store and M2's rule's threshold, both from its suite-m2b.json."""
    run = name.removeprefix("m2b-")
    run_dir = m2b.finished_run(run)
    if run_dir is None:
        raise SystemExit(f"no finished run called {name!r} in results/runs")
    suite = m2.read_json(run_dir / m2b.SUITE_FILE)
    if suite is None:
        raise SystemExit(f"{name} has no {m2b.SUITE_FILE} yet, so no threshold_store: "
                         f"python scripts/score_suite.py --milestone m2b --suite m2b --run {run}")
    record = m2.read_json(run_dir / "run.json")
    if "weights_from" in record:            # the control: M2's run, as it is (decision 12)
        source = record["weights_from"].split("-m2-", 1)[1]
        weights = m2b.m2_weights(Path(record["weights_from"]), record)
        sha = m2.LAST_PT_SHA256.get(source)
        if sha is not None and record.get("checkpoint_sha256", sha) != sha:
            raise SystemExit(f"{name}'s record holds {record['checkpoint_sha256'][:8]}… for M2's "
                             f"{source}, m2.LAST_PT_SHA256 {sha[:8]}…")
    else:
        recorded = Path(record["checkpoint"])
        weights = recorded if recorded.exists() else checkpoint_dir() / "m2b" / run_dir.name / "weights" / "last.pt"
        sha = m2b.LAST_PT_SHA256.get(run)
    return Detector(name, "m2b", run_dir, suite["threshold"], record["config"]["imgsz"], ["train_real"],
                    weights, sha, f"{m2b.SUITE_FILE}: threshold_store, the best F2 for dates on v1-val",
                    suite["threshold_m2"])


def private_name(run: Detector, set_name: str) -> str:
    return f"detector-{set_name}-{run.name}"


def private_root(set_name: str) -> Path:
    """Where a store set's per-photo boxes go, never results/: v1's
    data/store/runs/, v2's data/store/v2/runs/."""
    return final_test.RUNS_DIR if set_name == final_test.SET else runs_root(set_name)


def problems(run: Detector, set_name: str, again: bool) -> list[str]:
    """Why the run can't be scored on the set now; [] if it can."""
    found = []
    out = run.run_dir / f"{set_name}.json"
    if set_name == final_test.SET:
        if run.name not in final_test.RUNS:
            found.append("not one of the final test's records (final_test.RUNS)")
        if again:
            found.append("--again is refused for store-v2: it is scored once")
        if out.exists():
            found.append(f"already scored on {set_name} ({out})")
        if runs.latest_run(private_name(run, set_name), private_root(set_name)):
            found.append(f"its per-photo folder for {set_name} already exists: look before anything else")
        if run.sha256 is None:
            found.append("no SHA-256 is recorded for its weights")
    elif set_name in SCORED_ONCE and out.exists() and not again:
        found.append(f"already scored on {set_name} ({out}). "
                     "Test sets are scored once; pass --again only on purpose.")
    if not run.weights.exists():
        found.append(f"{run.weights} is missing: copy it from the cluster (see --help)")
    elif run.sha256 is not None and store.sha256(run.weights) != run.sha256:
        found.append(f"{run.weights} isn't the file hashed for it ({run.sha256[:8]}…)")
    return found


def output(run: Detector, set_name: str, again: bool) -> Path:
    """Where the run's scores on the set go, once problems() finds nothing."""
    if found := problems(run, set_name, again):
        raise SystemExit(f"{run.name}: " + "; ".join(found))
    return run.run_dir / f"{set_name}.json"


def torch_threads() -> int | None:
    """torch's threads now: Ultralytics sets them at its first prediction
    on a CPU (CLAUDE.md), so read after one."""
    try:
        import torch
    except ImportError:
        return None
    return torch.get_num_threads()


def score_run(run: Detector, set_name: str, samples, device: str, again: bool,
              cache: dict | None = None, v2: final_test.V2 | None = None,
              batch: int = detector.PREDICT_BATCH) -> dict:
    """Score one run and write its record, `batch` photos a call. Runs whose
    weights hash the same at the same input size share one pass through
    `cache`."""
    if set_name == final_test.SET and batch != final_test.BATCH:
        raise SystemExit(f"store-v2 is predicted {final_test.BATCH} photo a call, not {batch}")
    out = output(run, set_name, again)
    cache = {} if cache is None else cache
    data = None
    if set_name in SPLITS:
        data = detector.dataset_yaml(expdate_dir() / "yolo", run.train, run.run_dir / "data-local.yaml")
    key = (run.sha256 or str(run.weights), run.imgsz)
    shared = key in cache
    if not shared:
        started = time.time()
        cache[key] = (run.name, detector.predict_checkpoint(run.weights, samples, run.imgsz, device, batch=batch),
                      time.time() - started)
    made_by, preds, seconds = cache[key]
    threads = torch_threads()
    if set_name in PRIVATE:
        keep_private(run, preds, set_name)
    scores = detector.score(run.weights, samples, run.imgsz, conf=run.threshold,
                            data=data, split=SPLITS.get(set_name),
                            project=run.checkpoints / "eval",
                            device=device, predictions=preds,
                            **({} if batch == detector.PREDICT_BATCH else {"batch": batch}))
    if v2 is not None:
        scores |= final_test.breakdown(v2, preds, run.threshold)
    if run.threshold_m2 is not None and set_name in SPLITS:
        at_05, at_03 = evaluate(samples, preds, 0.5), evaluate(samples, preds, 0.3)
        scores["at_threshold_m2"] = {"threshold": run.threshold_m2,
                                     "iou0.5": summary(at_05, run.threshold_m2),
                                     "iou0.3": summary(at_03, run.threshold_m2)}
        scores["ap50_ours"] = {"date": average_precision(at_05)}
    scores["scored"] = {"set": set_name, "device": device, **git_state(),
                        "date": datetime.now().isoformat(timespec="seconds"),
                        "threshold": run.threshold, "threshold_from": run.threshold_from,
                        "weights_sha256": run.sha256, "predicted_by": made_by,
                        "predict_seconds": 0.0 if shared else round(seconds, 1),
                        "photos": len(samples), "batch": batch, "torch_threads": threads}
    write_json(out, scores)
    return scores


def keep_private(run: Detector, preds: dict, set_name: str = "store") -> Path:
    """Per-photo boxes on store photos, outside results/ (private_root).
    Store set v1's: its date boxes; M1 rounded them, M2 keeps them exact,
    as its threshold is exact (a confidence rounded to 4 decimals could
    cross it). Store set v2's: exact for every run, and the keyword classes'
    boxes beside the dates, as M3 reads from the M2b detector."""
    v2 = set_name == final_test.SET
    config = {"run": run.run_dir.name, "threshold": run.threshold}
    if v2:
        config["weights_sha256"] = run.sha256
    private = start_run(private_name(run, set_name), config, root=private_root(set_name))
    exact = v2 or run.milestone != "m1"

    def row(p) -> list[float]:
        if exact:
            return [*p.xyxy, p.conf]
        return [*(round(v, 5) for v in p.xyxy), round(p.conf, 4)]

    boxes = {pid: [row(p) for p in ps if p.label == "date"] for pid, ps in preds.items()}
    (private / "predictions.json").write_text(json.dumps(boxes))
    if v2:
        keywords = {pid: {c: [row(p) for p in ps if p.label == c] for c in CLASSES if c != "date"}
                    for pid, ps in preds.items()}
        (private / "keyword_boxes.json").write_text(json.dumps(keywords))
    return private


def plan_table(found: dict[str, Detector], failed: dict[str, list[str]], set_name: str) -> str:
    """The dry run: each run, its threshold and where it comes from, its
    weights' hash, which run's pass it shares, and whether it can be scored."""
    first: dict[tuple, str] = {}
    rows = []
    for name, run in found.items():
        key = (run.sha256 or str(run.weights), run.imgsz)
        first.setdefault(key, name)
        rows.append({"run": name, "milestone": run.milestone, "imgsz": str(run.imgsz),
                     "threshold": f"{run.threshold:.6f}", "from": run.threshold_from.split(":")[0],
                     "weights": f"{run.sha256[:8]}…" if run.sha256 else "not recorded",
                     "pass": "own" if first[key] == name else f"{first[key]}'s",
                     "state": "; ".join(failed.get(name, [])) or "ready"})
    columns = ("milestone", "imgsz", "threshold", "from", "weights", "pass")
    rows += [{"run": name, **dict.fromkeys(columns, "–"), "state": "; ".join(why)}
             for name, why in failed.items() if name not in found]
    return tables.markdown_table(rows, left=("run", "milestone", "from", "pass", "state"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", required=True, choices=SETS)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run names, e.g. n640, m2-base-s0 or m2b-n1280-ft-s0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--again", action="store_true",
                        help="re-score a run already scored on a test set (refused for store-v2)")
    parser.add_argument("--batch", type=int, default=None,
                        help=f"photos a call (default {detector.PREDICT_BATCH}; store-v2 is always {final_test.BATCH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="check everything a score needs, then stop: nothing is predicted or written")
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        raise SystemExit("a run is named twice")
    if args.set == final_test.SET and args.batch not in (None, final_test.BATCH):
        raise SystemExit(f"store-v2 is predicted {final_test.BATCH} photo a call (S14): --batch {args.batch} is refused")
    batch = final_test.BATCH if args.set == final_test.SET else args.batch or detector.PREDICT_BATCH

    found, failed = {}, {}
    for name in args.runs:          # every run is found and checked before any is scored
        try:
            found[name] = find(name)
        except SystemExit as e:
            failed[name] = [str(e)]
    for name, run in found.items():
        if why := problems(run, args.set, args.again):
            failed[name] = why
    v2 = final_test.load() if args.set == final_test.SET else None
    if args.dry_run:
        print(plan_table(found, failed, args.set))
        if v2 is not None:
            print(f"\nStore set v2, frozen and loaded ({len(v2.samples)} photos):\n")
            print(final_test.counts(v2))
        print(f"\n{len(found) - len(set(failed) & set(found))} of {len(args.runs)} ready, {batch} photo(s) a "
              "call; nothing predicted or written (--dry-run)")
        raise SystemExit(1 if failed else 0)
    if failed:
        raise SystemExit("nothing scored:\n" + "\n".join(f"- {n}: {'; '.join(w)}" for n, w in failed.items()))

    samples = v2.samples if v2 is not None else load_set(args.set)
    cache: dict = {}
    results = {name: score_run(run, args.set, samples, args.device, args.again, cache, v2, batch)
               for name, run in found.items()}
    print(m1.scores_table(results))
    if args.set in PRIVATE:
        print("\nBy failure tag (expiry found at IoU 0.5; means over the runs):\n")
        print(m1.tag_table(results))
    if v2 is not None:
        rows = [{"run": name, "AP@0.5 date (ours)": f"{r['ap50']['date']:.4f}",
                 **{g: f"{r['by_design'][g]['iou0.5'].get('expiry_found', 0)}/"
                       f"{r['by_design'][g]['iou0.5'].get('expiry_dates', 0)}"
                    for g in final_test.DESIGN_GROUPS}}
                for name, r in results.items()]
        print("\nExpiry found by pack design (IoU 0.5, at each run's threshold):\n")
        print(tables.markdown_table(rows, left=("run",)))


if __name__ == "__main__":
    main()
