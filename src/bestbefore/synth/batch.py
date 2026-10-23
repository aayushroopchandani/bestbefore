"""M2's generated sets at scale (docs/m2-synthetic-generator.md, S6): the
tasks, making them on many CPUs, and the check that vouches for them.

  gen        50,000 training photos (seed 1) in tasks 0 to 49 of 1,000:
             task N makes photos 1000 N to 1000 N + 999
  synth-val  2,100 photos (seed 2) in task 50: spec.SYNTH_VAL's 7 strata,
             300 photos each, one after another

Photo i of a set comes from default_rng([set seed, i]) alone. A task
writes its photos and label files, then its sidecar lines in photo order
(gen/sidecars/tasks/NN.jsonl), and last a record of how it was made
(NN.json), so a task with a record is whole. However the tasks are split
across processes, runs or jobs, the same files come out.

The check reads everything back: it counts the photos, opens every image,
rebuilds every label file from its sidecar line and compares them byte for
byte, re-checks on what was drawn that training photos used only training
fonts and backgrounds (and synth-val only held-out fonts on validation
photos), and measures every share against the spec. Then it joins the
task sidecars into one per set and writes the lists and a manifest saying
`complete: true`, which is what the GPU chain waits for (m2.list_missing).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import resource
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
from PIL import Image

from .. import derived, m2, runs
from . import camera, compose, fonts, make, render, text, warp
from .compose import Backgrounds
from .spec import GEN, SET_SEEDS, SYNTH_VAL, GenSpec, spec_hash, strata_hash

SHARE_POINTS = 0.03     # a share within 3 points of the spec,
SHARE_SE = 3.0          # or 3 standard errors where the sample is small (synth-val's strata)
CODE = (camera, compose, fonts, make, render, text, warp)   # the modules that draw a photo


@dataclass(frozen=True)
class Task:
    number: int
    set: str            # gen or synth-val
    first: int          # its photos: first to first + count - 1
    count: int

    @property
    def photos(self) -> range:
        return range(self.first, self.first + self.count)


@dataclass(frozen=True)
class Layout:
    """How many photos each set has and how gen is cut into tasks. The
    tests use smaller layouts; the cluster uses LAYOUT."""
    gen: int = 50_000           # training photos
    gen10k: int = 10_000        # the first of them, for the data-size arm (decision 14)
    task_size: int = 1_000      # gen photos per task
    per_stratum: int = 300      # synth-val photos per stratum

    def tasks(self) -> tuple[Task, ...]:
        """gen's tasks, then synth-val's one task."""
        n = math.ceil(self.gen / self.task_size)
        gen = tuple(Task(k, "gen", k * self.task_size, min(self.task_size, self.gen - k * self.task_size))
                    for k in range(n))
        return (*gen, Task(n, "synth-val", 0, self.size("synth-val")))

    def size(self, name: str) -> int:
        return self.gen if name == "gen" else self.per_stratum * len(SYNTH_VAL)

    def spec(self, name: str, i: int) -> tuple[GenSpec, str | None]:
        """The spec photo i of a set is drawn with, and its stratum."""
        if name != "synth-val":
            return GEN, None
        stratum, spec = SYNTH_VAL[i // self.per_stratum]
        return spec, stratum


LAYOUT = Layout()
SETS = ("gen", "synth-val")


def parse_tasks(text_: str, layout: Layout = LAYOUT) -> list[Task]:
    """Tasks from "0-4,50" or "all", in order, each once."""
    tasks = layout.tasks()
    if text_ == "all":
        return list(tasks)
    numbers: set[int] = set()
    for part in text_.split(","):
        lo, _, hi = part.partition("-")
        numbers.update(range(int(lo), int(hi or lo) + 1))
    if bad := sorted(n for n in numbers if not 0 <= n < len(tasks)):
        raise ValueError(f"no task {bad[0]}: tasks are 0 to {len(tasks) - 1}")
    return [t for t in tasks if t.number in numbers]


# --- Where things go ($GENERATED_DIR/m2/gen/) ---------------------------------------------------------

def images_dir(name: str) -> Path:
    return m2.gen_dir() / "images" / m2.GENERATED_SETS[name]


def tasks_dir() -> Path:
    return m2.gen_dir() / "sidecars" / "tasks"


def task_files(task: Task) -> tuple[Path, Path]:
    """A task's sidecar lines and its record."""
    stem = tasks_dir() / f"{task.number:02d}"
    return stem.with_suffix(".jsonl"), stem.with_suffix(".json")


def spec_sha256(name: str) -> str:
    """The hash a set's manifest names: GenSpec's, or synth-val's strata's."""
    return strata_hash() if name == "synth-val" else spec_hash(GEN)


def load_pools(names: Iterable[str]) -> dict[str, Backgrounds]:
    """Each set's backgrounds, through compose.load_backgrounds, which
    refuses a training list holding a validation or test photo or one of the
    14 copies. Every synth-val stratum draws from the same pool."""
    pools = {}
    for name in sorted(set(names)):
        specs = [GEN] if name != "synth-val" else [spec for _, spec in SYNTH_VAL]
        keys = {(s.backgrounds, s.compose.min_side) for s in specs}
        if len(keys) != 1:
            raise ValueError(f"{name}: its strata draw from different backgrounds")
        pools[name] = compose.load_backgrounds(*keys.pop())
    return pools


# --- Making photos ------------------------------------------------------------------------------------

_worker: dict = {}


def _start(pools: dict[str, Backgrounds], layout: Layout) -> None:
    cv2.setNumThreads(1)        # one CPU per worker process
    _worker.update(pools=pools, layout=layout)


def _one(job: tuple[str, int]) -> tuple[str, dict, float]:
    """Make and write photo i of a set: its sidecar line as JSON, the
    seconds per stage, and this process's peak memory so far (MB)."""
    name, i = job
    spec, stratum = _worker["layout"].spec(name, i)
    clock = make.Clock()
    made = make.make_one(i, spec, _worker["pools"][name], SET_SEEDS[name], clock, stratum)
    with clock("write"):
        make.write_one(made, images_dir(name))
    return json.dumps(made.sidecar, separators=(",", ":")), dict(clock.seconds), peak_rss_mb()


def produce(jobs: Sequence[tuple[str, int]], workers: int, pools: dict[str, Backgrounds],
            layout: Layout = LAYOUT) -> Iterator[tuple[str, dict, float]]:
    """Make the photos (set, i) on `workers` processes (in this one if
    1), writing each photo and its label file; yields their sidecar lines,
    stage times and peak memory, in the order asked."""
    for name in {name for name, _ in jobs}:
        images_dir(name).mkdir(parents=True, exist_ok=True)
        derived.label_path(images_dir(name) / "x.jpg").parent.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        _start(pools, layout)
        yield from map(_one, jobs)
        return
    with Pool(workers, initializer=_start, initargs=(pools, layout)) as pool:
        yield from pool.imap(_one, jobs, chunksize=4)


def make_tasks(tasks: Sequence[Task], workers: int, pools: dict[str, Backgrounds] | None = None,
               layout: Layout = LAYOUT, run: str | None = None) -> dict:
    """Make the tasks, all on one pool of `workers` processes. Each task's
    sidecar lines are written once its last photo is, then its record.
    Returns the timing and memory over every photo."""
    pools = pools if pools is not None else load_pools(t.set for t in tasks)
    tasks_dir().mkdir(parents=True, exist_ok=True)
    made_by = {"run": run, **runs.git_state(), "code_sha256": runs.code_sha256(*CODE), "layout": asdict(layout),
               "libraries": libraries()}
    results = produce([(t.set, i) for t in tasks for i in t.photos], workers, pools, layout)
    every: list[dict] = []
    peak = 0.0
    for task in tasks:
        lines_path, record_path = task_files(task)
        record_path.unlink(missing_ok=True)          # a task being remade isn't whole until its record is back
        part = lines_path.with_name(lines_path.name + ".part")
        per_photo = []
        with part.open("w") as out:
            for _ in task.photos:
                line, seconds, rss = next(results)
                out.write(line + "\n")
                per_photo.append(seconds)
                peak = max(peak, rss)
        os.replace(part, lines_path)
        runs.write_json(record_path, {**asdict(task), "spec_sha256": spec_sha256(task.set), **made_by,
                                      "timing": timing(per_photo)})
        every += per_photo
        print(f"task {task.number:2d} ({task.set} {task.first}-{task.photos[-1]}) done", flush=True)
    return {"photos": len(every), "workers": workers, "timing": timing(every),
            "peak_rss_mb_per_worker": round(peak, 1), "main_rss_mb": round(peak_rss_mb(), 1)}


def libraries() -> dict[str, str]:
    """The builds that draw and save the pixels: a machine with others may
    draw slightly different ones (S4, S6)."""
    import numpy
    import PIL
    from PIL import features
    return {"pillow": PIL.__version__, "freetype": features.version("freetype2") or "",
            "libjpeg": features.version("jpg") or "", "libjpeg_turbo": features.version("libjpeg_turbo") or "",
            "opencv": cv2.__version__, "numpy": numpy.__version__,
            "python": sys.version.split()[0], "platform": sys.platform}


def peak_rss_mb() -> float:
    """This process's peak resident memory (MB): ru_maxrss is in bytes on
    macOS, in kB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def timing(per_photo: Sequence[dict]) -> dict:
    """Milliseconds per photo on one CPU, in all and by stage."""
    stages: Counter = Counter()
    for seconds in per_photo:
        stages.update(seconds)
    totals = sorted(sum(s.values()) for s in per_photo)
    n = len(totals)
    if not n:
        return {"photos": 0}
    return {"photos": n, "cpu_seconds": round(sum(totals), 1),
            "mean_ms": round(1000 * sum(totals) / n, 1),
            "median_ms": round(1000 * totals[n // 2], 1),
            "p95_ms": round(1000 * totals[min(n - 1, round(0.95 * n))], 1),
            "max_ms": round(1000 * totals[-1], 1),
            "by_stage_mean_ms": {k: round(1000 * v / n, 1) for k, v in stages.most_common()}}


# --- What was made ----------------------------------------------------------------------------------

def counts(sidecars: Sequence[dict]) -> dict:
    """Photos, blocks placed and skipped (and why), boxes by class."""
    labels = Counter(b.label for s in sidecars for b in make.label_boxes(s))
    return {"photos": len(sidecars),
            "blocks": sum(len(s["blocks"]) for s in sidecars),
            "skipped": dict(Counter(k["why"] for s in sidecars for k in s["skipped"])),
            "redrawn": sum(b["redraws"] > 0 for s in sidecars for b in s["blocks"]),
            "fresh_ink": sum(bool(b["faint_inks"]) for s in sidecars for b in s["blocks"]),
            "shrunk": sum(b["shrunk"] for s in sidecars for b in s["blocks"]),
            "boxes": dict(sorted(labels.items())),
            "generated_dates": sum(t["cls"] == "date" for s in sidecars for b in s["blocks"]
                                   for t in b["tokens"])}


def share_ok(row: dict) -> bool:
    """A measured share against its spec: exact where the spec fixes it at 0
    or 1; else within SHARE_POINTS, or SHARE_SE standard errors where the
    sample is small enough for chance to move it further."""
    p, measured = row["spec"], row["measured"]
    if measured is None:
        return True
    if p in (0, 1):
        return measured == p
    return abs(measured - p) <= max(SHARE_POINTS, SHARE_SE * math.sqrt(p * (1 - p) / row["n"]))


def checked_shares(sidecars: Sequence[dict], spec: GenSpec) -> list[dict]:
    return [{**row, "ok": share_ok(row)} for row in make.shares(sidecars, spec)]


def shares_table(rows: Sequence[dict]) -> str:
    lines = ["| what | spec | measured | n | |", "|---|---:|---:|---:|---|"]
    for r in rows:
        measured = "" if r["measured"] is None else f"{r['measured']:.3f}"
        flag = "" if r.get("ok", True) else "**off**"
        lines.append(f"| {r['what']} | {r['spec']:.3f} | {measured} | {r['n']} | {flag} |")
    return "\n".join(lines)


# --- The check --------------------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify(job: tuple[str, str]) -> tuple[list[str], str | None, str | None, int]:
    """One photo read back: the image opens whole as a JPEG of the
    sidecar's size, and its label file is the one its sidecar line makes,
    byte for byte. Returns the problems, both files' SHA-256 and the
    image's bytes."""
    image, line = Path(job[0]), json.loads(job[1])
    problems, hashes = [], []
    try:
        data = image.read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            im.load()                                       # a cut-off file fails here
            if (im.format, im.mode, list(im.size)) != ("JPEG", "RGB", line["size"]):
                problems.append(f"{image.name}: a {im.format} {im.mode} {im.size}, "
                                f"the sidecar says a JPEG {line['size']}")
        hashes.append(_sha256(data))
    except Exception as e:                                  # missing, unreadable, truncated
        problems.append(f"{image.name}: {type(e).__name__}: {e}")
        data = b""
        hashes.append(None)
    try:
        labels = derived.label_path(image).read_bytes()
        if labels != make.label_text(line).encode():
            problems.append(f"{image.name}: its label file is not the one its sidecar line makes")
        hashes.append(_sha256(labels))
    except Exception as e:
        problems.append(f"{image.name} labels: {type(e).__name__}: {e}")
        hashes.append(None)
    return problems, hashes[0], hashes[1], len(data)


def _lines(task: Task) -> tuple[list[str] | None, list[str]]:
    """A task's sidecar lines (None if it was never finished: no record),
    and what is wrong with them: a record from another spec or layout, or
    too few or many lines."""
    lines_path, record_path = task_files(task)
    record = m2.read_json(record_path)
    if record is None or not lines_path.exists():
        return None, []
    problems = []
    if record["spec_sha256"] != spec_sha256(task.set):
        problems.append(f"task {task.number}: made with spec {record['spec_sha256'][:12]}, "
                        f"not today's {spec_sha256(task.set)[:12]}")
    if (record["set"], record["first"], record["count"]) != (task.set, task.first, task.count):
        problems.append(f"task {task.number}: made as {record['set']} {record['first']}+{record['count']}")
    lines = lines_path.read_text().splitlines()
    if len(lines) != task.count:
        problems.append(f"task {task.number}: {len(lines)} sidecar lines, not {task.count}")
    return lines, problems


def _spans(numbers: Sequence[int]) -> str:
    """[5, 6, 7, 9] -> "5-7, 9"."""
    spans: list[list[int]] = []
    for n in numbers:
        if spans and n == spans[-1][1] + 1:
            spans[-1][1] = n
        else:
            spans.append([n, n])
    return ", ".join(f"{a}-{b}" if b > a else f"{a}" for a, b in spans)


def _photo_problems(name: str, line: dict, i: int, layout: Layout) -> list[str]:
    """A sidecar line against its place: photo i of the set, from its own
    seed, its stratum, and for synth-val exactly one target, boxed."""
    image = make.image_name(i)
    problems = []
    if (line["image"], line["i"], line["seed"]) != (image, i, [SET_SEEDS[name], i]):
        problems.append(f"{name} line {i}: is {line['image']} (i {line['i']}, seed {line['seed']})")
    stratum = layout.spec(name, i)[1]
    if line.get("stratum") != stratum:
        problems.append(f"{name} {image}: stratum {line.get('stratum')}, not {stratum}")
    dates = [t for b in line["blocks"] for t in b["tokens"] if t["cls"] == "date"]
    if any(t["box"] is None for t in dates):
        problems.append(f"{name} {image}: a date without its box")
    targets = [t for t in dates if t["role"] == "target"]
    if name == "synth-val" and (len(targets) != 1 or len(dates) != 1):
        problems.append(f"{name} {image}: {len(targets)} targets among {len(dates)} new dates, not 1 of 1")
    if name != "synth-val" and targets:
        problems.append(f"{name} {image}: a target date outside synth-val")
    return problems


def _guard_problems(name: str, sidecars: Sequence[dict], layout: Layout) -> list[str]:
    """What was drawn, against the guards that keep synth-val apart: every
    background from a list its spec allows, and none that compose.forbidden
    refuses there (no validation or test photo, none of the 14 copies, in
    training); every font family from its spec's split."""
    problems = []
    used: dict[str, set[str]] = {}
    for s in sidecars:
        spec = layout.spec(name, s["i"])[0]
        allowed = {b for b, _ in spec.backgrounds}
        bg = s["background"]
        if bg["list"] not in allowed:
            problems.append(f"{name} {s['image']}: a background from {bg['list']}, not {sorted(allowed)}")
        used.setdefault(bg["list"], set()).add(bg["image"])
        families = {f.name for f in fonts.families(spec.fonts)}
        if wrong := sorted({b["family"] for b in s["blocks"]} - families):
            problems.append(f"{name} {s['image']}: font {', '.join(wrong)}, outside {spec.fonts}")
    for list_name, names in sorted(used.items()):
        paths = {p.name: p for p in derived.read_list(m2.list_file(list_name))}
        if missing := sorted(names - set(paths)):
            problems.append(f"{name}: backgrounds not in {list_name}: {', '.join(missing[:5])}")
        folder = next(iter(paths.values())).parent        # a name the list lacks, as if from its folder
        refused = compose.forbidden(list_name, [paths.get(n, folder / n) for n in sorted(names)])
        problems += [f"{name}: a background no {list_name} photo may be: {p}" for p in refused]
    return problems


def _font_hashes(name: str) -> tuple[dict[str, str], list[str]]:
    """SHA-256 of the font files a set draws with, from disk, against the
    pinned ones."""
    splits = {GEN.fonts} if name != "synth-val" else {spec.fonts for _, spec in SYNTH_VAL}
    found, problems = {}, []
    for split in sorted(splits):
        for family in fonts.families(split):
            for face in fonts.faces(family):
                path = fonts.path(face)
                found[face.file] = fonts.sha256(path) if path.exists() else None
                if found[face.file] != fonts.SHA256[face.file]:
                    problems.append(f"font {face.file}: SHA-256 {found[face.file]}, pinned {fonts.SHA256[face.file]}")
    return found, problems


def check_set(name: str, layout: Layout = LAYOUT, workers: int = 1) -> dict:
    """Check one set (see the module's docstring) and, if nothing is wrong,
    write its joined sidecar, its lists and its manifest saying complete.
    A set with problems gets a manifest saying complete: false and why.
    Returns the manifest and every file's SHA-256."""
    problems: list[str] = []
    raw: list[str] = []                 # the sidecar lines, in photo order
    records, missing = [], []
    for task in (t for t in layout.tasks() if t.set == name):
        lines, found = _lines(task)
        if lines is None:
            missing.append(task.number)
            continue
        problems += found
        records.append(m2.read_json(task_files(task)[1]))
        for i, line in zip(task.photos, lines):
            problems += _photo_problems(name, json.loads(line), i, layout)
        raw += lines
    if missing:
        problems.insert(0, f"{name}: tasks {_spans(missing)} not made (no record)")
    sidecars = [json.loads(line) for line in raw]

    folder = images_dir(name)
    expected = {make.image_name(i)[:-4] for i in range(layout.size(name))}
    for kind, files in (("images", _stems(folder, ".jpg")),
                        ("label files", _stems(derived.label_path(folder / "x.jpg").parent, ".txt"))):
        if extra := sorted(files - expected):
            problems.append(f"{name}: {len(extra)} {kind} beyond the set's {len(expected)}, e.g. {extra[0]}")
    jobs = [(str(folder / s["image"]), line) for s, line in zip(sidecars, raw)]
    hashes: dict[str, dict] = {"images": {}, "labels": {}}
    size = 0
    for s, (found, image_hash, label_hash, n_bytes) in zip(sidecars, _map(_verify, jobs, workers)):
        problems += found
        hashes["images"][s["image"]], hashes["labels"][s["image"][:-4] + ".txt"] = image_hash, label_hash
        size += n_bytes
    problems += _guard_problems(name, sidecars, layout)
    font_hashes, found = _font_hashes(name)
    problems += found

    versions = {json.dumps({k: r.get(k) for k in ("git_commit", "git_dirty", "code_sha256")}, sort_keys=True)
                for r in records}
    if len(versions) > 1:
        problems.append(f"{name}: its tasks were made by {len(versions)} different versions of the code")
    if name == "synth-val":
        shares = {stratum: checked_shares([s for s in sidecars if s.get("stratum") == stratum], spec)
                  for stratum, spec in SYNTH_VAL}
        off = [f"{stratum}: {r['what']}" for stratum, rows in shares.items() for r in rows if not r["ok"]]
    else:
        shares = checked_shares(sidecars, GEN)
        off = [r["what"] for r in shares if not r["ok"]]
    problems += [f"{name}: share off the spec: {what}" for what in off]

    complete = not problems and len(sidecars) == layout.size(name)
    sidecar_text = "".join(f"{line}\n" for line in raw)
    lists = {"gen50k": layout.gen, "gen10k": layout.gen10k} if name == "gen" else {"synth_val": layout.size(name)}
    manifest = {
        "set": name, "complete": complete, "problem_count": len(problems), "problems": problems[:100],
        "photos": len(sidecars), "expected": layout.size(name), "lists": lists, "seed": SET_SEEDS[name],
        "layout": asdict(layout), "spec_sha256": spec_sha256(name),
        "spec": asdict(GEN) if name == "gen" else {stratum: asdict(spec) for stratum, spec in SYNTH_VAL},
        **({"strata": {stratum: {"photos": layout.per_stratum, "spec_sha256": spec_hash(spec)}
                       for stratum, spec in SYNTH_VAL}} if name == "synth-val" else {}),
        "made_by": [json.loads(v) for v in sorted(versions)],
        "made_in_runs": sorted({r["run"] for r in records if r.get("run")}),
        "checked_at": runs.git_state(), "fonts": font_hashes,
        "bytes": {"images": size, "sidecar": len(sidecar_text.encode())},
        "sidecar_sha256": _sha256(sidecar_text.encode()),
        "timing": timing_of(records), "counts": counts(sidecars), "shares": shares,
    }
    if complete:
        _write_set(name, layout, sidecar_text, lists)
    m2.manifest_path(name).parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(m2.manifest_path(name), manifest)
    return {"manifest": manifest, "hashes": hashes}


def _write_set(name: str, layout: Layout, sidecar_text: str, lists: dict[str, int]) -> None:
    """The joined sidecar and the lists (absolute paths), before the
    manifest that vouches for them."""
    m2.sidecar_path(m2.GENERATED_SETS[name]).write_text(sidecar_text)
    images = [images_dir(name) / make.image_name(i) for i in range(layout.size(name))]
    for list_name, n in lists.items():
        derived.write_list(m2.list_path(list_name), images[:n])     # gen10k: gen50k's first 10,000


def _stems(folder: Path, suffix: str) -> set[str]:
    return {p.stem for p in folder.iterdir() if p.suffix == suffix} if folder.exists() else set()


def _map(fn, jobs: Sequence, workers: int) -> Iterator:
    if workers <= 1 or len(jobs) < 2 * workers:
        return map(fn, jobs)
    with Pool(workers) as pool:
        return iter(pool.map(fn, jobs, chunksize=64))


def timing_of(records: Sequence[dict]) -> dict:
    """Milliseconds per photo over the tasks' records."""
    done = [r["timing"] for r in records if r.get("timing", {}).get("photos")]
    n = sum(t["photos"] for t in done)
    if not n:
        return {"photos": 0}
    return {"photos": n, "cpu_seconds": round(sum(t["cpu_seconds"] for t in done), 1),
            "mean_ms": round(sum(t["mean_ms"] * t["photos"] for t in done) / n, 1),
            "slowest_photo_ms": max(t["max_ms"] for t in done)}


def check(layout: Layout = LAYOUT, workers: int = 1) -> dict[str, dict]:
    """Check both sets; see check_set."""
    return {name: check_set(name, layout, workers) for name in SETS}


def passed(results: dict[str, dict]) -> bool:
    """Whether every set is complete, with nothing wrong."""
    return all(r["manifest"]["complete"] for r in results.values())


def release_held(name: str) -> list[str]:
    """After a check that passed: release every job of ours with this name
    that waits held (the GPU chain's, so job B and any held replacement).
    Raises if squeue can't be asked or a release is refused."""
    from .. import slurm

    jobs = slurm.held(name)
    if jobs is None:
        raise RuntimeError("squeue gave no answer: nothing released")
    for job in jobs:
        ok, said = slurm.release(job)
        if not ok:
            raise RuntimeError(f"scontrol release {job} failed: {said}")
    if still := slurm.held(name):
        raise RuntimeError(f"still held after release: {', '.join(still)}")
    return jobs
