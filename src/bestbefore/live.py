"""M2c: the M2b detector as ONNX files for the phone (docs/m2c-live-boxes.md).

    source_weights      the M2b detector's last.pt, its hash checked, copied into
                        $CHECKPOINT_DIR/m2c/ (Ultralytics writes the .onnx beside
                        the .pt, and M2b's run folder must not change)
    export_fp32         Ultralytics' own export: height, width (and batch) dynamic,
                        simplified by onnxslim, never installing anything itself
    decode_nodes        the Detect head's decoding, read from the graph: what int8
                        never quantises and fp16-head32 keeps in float32
    to_float16          ONNX Runtime's float16 converter, inputs and outputs float32
    calibration_photos  training photos only: the store export's upright training
                        photos and ExpDate's real training photos, by a fixed seed
    CalibrationReader   those photos letterboxed tight, as they are scored
    quantize_int8       QDQ, int8 weights per channel, uint8 activations
    manifest            each file's SHA-256, size, opset and node count
    ort_threads         ONNX Runtime's thread count for every session made inside
    r0, r1, r2_accuracy the decision rules S1's records are read by
    rule_g, gate        the stillness gate's threshold T, from a Motion record (S3)

The model files live in $CHECKPOINT_DIR/m2c/ (never in git: `*.onnx` is
ignored). Nothing here is random but the calibration photos' draw (SEED).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from . import derived, final_test, m2b, runs
from .config import checkpoint_dir, expdate_dir, store_train_dir
from .images import load_upright
from .store import sha256

SOURCE = final_test.DETECTOR.removeprefix("m2b-")      # n1280-ft-s0: R3's run, unchanged (decision 4)
IMGSZ = 1280
STRIDE = 32
PREDICT_CONF = 0.001            # every box kept, as M2b's scoring kept them
FILES = {"fp32": "det-fp32.onnx", "fp16": "det-fp16.onnx", "fp16-head32": "det-fp16-head32.onnx",
         "int8": "det-int8.onnx", "int8-pct": "det-int8-pct.onnx"}
FLOAT_FILES = ("fp32", "fp16", "fp16-head32")          # made on the laptop
INT8_FILES = ("int8", "int8-pct")                      # made in the CPU job
MANIFEST = "manifest.json"

# Calibration (S1): training photos only, never a validation photo, never
# store set v2. The store's share is its 80 upright training photos.
SEED = 0
CALIBRATION = {"store": 80, "expdate": 120}
PERCENTILE = 99.999
# int8 (S1): QDQ, int8 weights per channel, uint8 activations. Photos go
# through the calibrator one at a time (ONNX Runtime's CalibStridedMinMax,
# which strides any method, not only MinMax): Percentile otherwise keeps
# every photo's every tensor until the last, and tight photos of different
# shapes can't be stacked into one array.
CALIBRATION_STRIDE = 1

# The parity job's variants (S1): the model file (pt: PyTorch, M2b's own
# weights) and whether it pads tight (rect=True) or to the 1,280 square.
VARIANTS = {"pt-square": ("pt", False),
            "fp32-square": ("fp32", False), "fp32-tight": ("fp32", True),
            "int8-square": ("int8", False), "int8-tight": ("int8", True),
            "int8-pct-square": ("int8-pct", False), "int8-pct-tight": ("int8-pct", True)}

# The rules (docs/m2c-live-boxes.md, "Decision rules"), written before any number.
SETS = ("v1-val", "real-val", "real-val-rot", "real-val-small", "real-val-rot180")
R0_AP = 0.002                   # ONNX float32 square against PyTorch: AP within this
R1_AP, R1_FEWER = 0.01, 1       # tight against square: AP no more than this under, at most 1 date fewer
R2_AP, R2_FEWER = 0.01, 1       # a lower precision against float32, the same way
G_STILL, G_MOVING = 95, 5       # G: T between these percentiles of the still and moving values
# The Motion record rule G reads (S3), under results/runs/m2c/; None until it
# is made. The gate runs on every GATE_EVERY-th frame: 1 if it costs under
# 2 ms a frame on the phone (S3's check), else 2. S3: the OnePlus 13's record
# with the GPU copy, 11.8 ms a frame, so every other frame: T 4.42.
GATE_MOTION: str | None = "20261125-142850-m2c-app/motion-003.json"
GATE_EVERY = 2

_MODULE = re.compile(r"/model\.(\d+)/")
# Ultralytics' Detect keeps its box and class convolutions in two lists named
# cv2 and cv3 (one entry per scale); everything else in the module decodes.
_BRANCHES = ("cv2.", "cv3.")


def models_dir() -> Path:
    return checkpoint_dir() / "m2c"


def threshold() -> float:
    """M2b's threshold_store for the M2b detector, unrounded, as M2b scored
    store set v2 with it (0.1641 rounded): read from its run's own suite
    record, never chosen again."""
    suite = m2b.suite_of(SOURCE)
    if suite is None:
        raise FileNotFoundError(f"m2b-{SOURCE}'s suite-m2b.json isn't in results/runs")
    return float(suite["threshold"])


# --- The weights and the float32 export -----------------------------------------------

def source_weights(folder: Path | None = None) -> tuple[Path, str]:
    """The M2b detector's last.pt copied into `folder` (models_dir()) as
    <SOURCE>.pt: the original's SHA-256 checked against m2b.py first
    (m2b.last_pt), the copy's after. A copy already there is checked, not
    made again. Returns the copy and its hash."""
    import shutil

    original, _ = m2b.last_pt(SOURCE)
    expected = m2b.LAST_PT_SHA256[SOURCE]
    folder = folder or models_dir()
    folder.mkdir(parents=True, exist_ok=True)
    copy_path = folder / f"{SOURCE}.pt"
    if not copy_path.exists():
        shutil.copy2(original, copy_path)
    if (digest := sha256(copy_path)) != expected:
        raise RuntimeError(f"{copy_path} isn't m2b-{SOURCE}'s last.pt ({digest[:8]}…)")
    return copy_path, digest


def check_no_autoinstall() -> None:
    """Ultralytics pip-installs a missing package by itself (onnxslim, for
    one) unless YOLO_AUTOINSTALL is false when it is imported. Here a
    missing package must fail loudly instead."""
    from ultralytics import utils

    if utils.AUTOINSTALL:
        raise RuntimeError("set YOLO_AUTOINSTALL=false before Ultralytics is imported")


def export_fp32(weights: Path, out: Path, imgsz: int = IMGSZ) -> Path:
    """Ultralytics' ONNX export of `weights`, moved to `out`: float32,
    height and width dynamic (Ultralytics makes batch dynamic with them),
    simplified by onnxslim, the opset Ultralytics picks for this torch.
    It writes <weights>.onnx beside `weights` first, so `weights` must be
    a copy (source_weights). An existing `out` is never overwritten."""
    if out.exists():
        raise FileExistsError(f"{out} exists: the export is made once (remove it to make it again)")
    check_no_autoinstall()
    from ultralytics import YOLO

    made = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True,
                                      batch=1, nms=False, device="cpu")
    Path(made).replace(out)
    return out


# --- The Detect head's decoding --------------------------------------------------------

def head_module(model) -> int:
    """The model's last module's index (Ultralytics names every node
    /model.<index>/...): the Detect head, 23 for YOLO11."""
    found = {int(m.group(1)) for n in model.graph.node if (m := _MODULE.match(n.name))}
    if not found:
        raise ValueError("no node named /model.<index>/...: not an Ultralytics export")
    return max(found)


def decode_nodes(model) -> list[str]:
    """The Detect head's decoding, in graph order: every node of the model's
    last module outside its box and class convolutions (cv2, cv3), that
    is the anchors made from the input's shape, the DFL (a softmax and a
    fixed convolution), the boxes scaled by stride, the class sigmoid and
    the join of boxes and scores into the output.

    int8 leaves these in float: the output joins box coordinates (0 to
    1,280 px) and class scores (0 to 1), and one 8-bit scale for both
    (about 5 px a step) rounds every score to 0. fp16-head32 keeps them
    float32. The list is checked against the graph (decode_problems)."""
    head = f"/model.{head_module(model)}/"
    branches = tuple(head + b for b in _BRANCHES)
    nodes = [n.name for n in model.graph.node if n.name.startswith(head) and not n.name.startswith(branches)]
    if problems := decode_problems(model, nodes):
        raise ValueError("the Detect head's decoding isn't what it should be: " + "; ".join(problems))
    return nodes


def decode_problems(model, nodes: Sequence[str]) -> list[str]:
    """What is wrong with `nodes` as the Detect head's decoding: empty; a
    node outside the last module; a convolution that isn't the DFL's; no
    sigmoid or softmax; a graph output it doesn't write; or a node reading
    the head's convolutions (directly or through others) left out of it."""
    graph = model.graph
    if not nodes:
        return ["no node"]
    head = f"/model.{head_module(model)}/"
    branches = tuple(head + b for b in _BRANCHES)
    by_name = {n.name: n for n in graph.node}
    decode = set(nodes)
    problems = [f"{name}: not in the last module" for name in nodes if not name.startswith(head)]
    problems += [f"{name}: a convolution outside the DFL" for name in nodes
                 if by_name[name].op_type == "Conv" and "/dfl/" not in name]
    ops = {by_name[name].op_type for name in nodes}
    problems += [f"no {op}" for op in ("Sigmoid", "Softmax") if op not in ops]
    writers = {t: n.name for n in graph.node for t in n.output}
    problems += [f"output {o.name} is written outside it" for o in graph.output if writers.get(o.name) not in decode]
    readers = defaultdict(list)
    for n in graph.node:
        for t in n.input:
            readers[t].append(n)
    seen, frontier = set(), [t for n in graph.node if n.name.startswith(branches) for t in n.output]
    while frontier:
        for n in readers[frontier.pop()]:
            if n.name.startswith(branches) or n.name in seen:
                continue
            seen.add(n.name)
            if n.name not in decode:
                problems.append(f"{n.name} reads the head's convolutions but is left out")
            frontier += list(n.output)
    return problems


# --- Derived files ------------------------------------------------------------------------

def copy_metadata(source, target) -> None:
    """The float32 file's metadata_props (names, stride, imgsz, task, ...)
    into a derived file, replacing its own: Ultralytics' ONNX backend reads
    the class names from them, and without them it calls the classes
    class0 to class3."""
    del target.metadata_props[:]
    for prop in source.metadata_props:
        target.metadata_props.add(key=prop.key, value=prop.value)


def to_float16(model, keep_float32: Sequence[str] = ()):
    """`model` in float16, inputs and outputs kept float32 (keep_io_types),
    the nodes named in `keep_float32` left in float32 (fp16-head32: the
    head's decoding); metadata copied.

    Through ONNX Runtime's OnnxModel wrapper, which runs
    onnxruntime.transformers.float16's converter and then removes Cast
    chains and Casts to the type they already hold. Called alone (S1), the
    converter casts every kept node's output back to float16 and the next
    kept node's input to float32 again, so the decoding's tensors still
    pass through float16 between its nodes, and a tensor two kept nodes
    read gets two Casts of the same name: an invalid model."""
    from onnxruntime.transformers.onnx_model import OnnxModel

    wrapped = OnnxModel(copy.deepcopy(model))
    wrapped.convert_float_to_float16(keep_io_types=True, node_block_list=list(keep_float32) or None)
    out = wrapped.model
    copy_metadata(model, out)
    return out


def float16_problems(model, keep_float32: Sequence[str], original) -> list[str]:
    """What breaks a float16 file's promise: two nodes of one name; a node
    meant to stay float32 that reads or writes a float16 tensor (by ONNX's
    shape inference); or one gone that isn't a Cast. The Cast cleanup
    (to_float16) merges a kept Cast of integers to float32 (the anchors')
    with the Casts the converter put after it, into one Cast doing the
    same under the converter's name: only Casts may go."""
    import onnx

    names = [n.name for n in model.graph.node]
    problems = [f"{n}: {names.count(n)} nodes of that name" for n in sorted(set(names)) if names.count(n) > 1]
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type.elem_type
             for v in [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]}
    types |= {i.name: i.data_type for i in model.graph.initializer}
    kept = set(keep_float32)
    for n in model.graph.node:
        if n.name in kept:
            half = [t for t in [*n.input, *n.output] if types.get(t) == onnx.TensorProto.FLOAT16]
            if half:
                problems.append(f"{n.name} ({n.op_type}) reads or writes float16: {half}")
    ops = {n.name: n.op_type for n in original.graph.node}
    problems += [f"{name} ({ops.get(name)}): missing from the float16 file"
                 for name in sorted(kept - set(names)) if ops.get(name) != "Cast"]
    return problems


def letterboxed(image, imgsz: int = IMGSZ, stride: int = STRIDE, tight: bool = True) -> np.ndarray:
    """An upright PIL image as the model's input, as Ultralytics' predictor
    makes it: its LetterBox (long side to imgsz, grey 114 around it; tight
    pads to a multiple of stride, else to the imgsz square), RGB, channels
    first, 0 to 1, a batch of one."""
    from ultralytics.data.augment import LetterBox

    padded = LetterBox((imgsz, imgsz), auto=tight, stride=stride)(image=np.asarray(image))
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


# --- Calibration ------------------------------------------------------------------------

def calibration_photos(store: int = CALIBRATION["store"], expdate: int = CALIBRATION["expdate"],
                       seed: int = SEED) -> list[tuple[str, Path]]:
    """The calibration photos as (id, path), training photos only: `store`
    of the store export's training photos, upright (its training list holds
    each photo in 4 turns: only the unturned folder's), then `expdate` of
    ExpDate's real training photos (train_real). Each pool is shuffled by
    `seed` and its first photos taken, so a smaller draw (the laptop's 8)
    is the start of the full one. Ids name a photo by its number in the
    store export or its ExpDate file name, never a store photo's name.

    Refused: a photo in a validation list (v1-val's, real-val's), a store
    photo the export doesn't mark training, one of store set v2, or fewer
    photos than asked."""
    folder = store_train_dir()
    manifest = runs.read_json(folder / "manifest.json")
    settings = manifest["settings"]
    upright = {name for name, quarter in settings["folders"]["train"].items() if quarter == 0}
    store_pool = [p for p in derived.read_list(folder / settings["lists"]["train"]) if p.parent.name in upright]
    store_val = {p.resolve() for p in derived.read_list(folder / settings["lists"]["val"])}
    yolo = expdate_dir() / "yolo"
    expdate_pool = derived.read_list(yolo / "train_real.txt")
    expdate_val = {p.resolve() for p in derived.read_list(yolo / "val.txt")}

    rng = np.random.default_rng(seed)
    store_order, expdate_order = rng.permutation(len(store_pool)), rng.permutation(len(expdate_pool))
    if store > len(store_pool) or expdate > len(expdate_pool):
        raise ValueError(f"asked {store} store and {expdate} ExpDate photos; "
                         f"there are {len(store_pool)} and {len(expdate_pool)}")
    picked = ([(f"store:{store_pool[i].stem}", store_pool[i]) for i in store_order[:store]]
              + [(f"expdate:{expdate_pool[i].name}", expdate_pool[i]) for i in expdate_order[:expdate]])
    problems = []
    for photo_id, path in picked:
        if path.resolve() in store_val | expdate_val:
            problems.append(f"{photo_id} is a validation photo")
        if photo_id.startswith("store:"):
            if manifest["photos"][path.stem]["part"] != "train":
                problems.append(f"{photo_id} isn't a training photo")
            if path.stem.startswith("v2-"):
                problems.append(f"{photo_id} is store set v2's")
    if problems:
        raise ValueError("; ".join(problems))
    return picked


def list_sha256(photos: Sequence[tuple[str, Path]]) -> str:
    """The calibration list's hash, over its ids in order (paths differ
    between the laptop and the cluster; ids don't)."""
    return hashlib.sha256("".join(f"{i}\n" for i, _ in photos).encode()).hexdigest()


class CalibrationReader:
    """ONNX Runtime's calibration reader (get_next, rewind; __len__ and
    set_range for its strided collection): each photo loaded upright and
    letterboxed tight, one at a time, never held after it is read."""

    def __init__(self, paths: Sequence[Path], input_name: str = "images", imgsz: int = IMGSZ):
        self.paths, self.input_name, self.imgsz = list(paths), input_name, imgsz
        self.start, self.end, self.next = 0, len(self.paths), 0
        self.shapes: list[tuple[int, ...]] = []

    def __len__(self) -> int:
        return len(self.paths)

    def set_range(self, start_index: int, end_index: int) -> None:
        self.start, self.end, self.next = start_index, min(end_index, len(self.paths)), start_index

    def rewind(self) -> None:
        self.next = self.start

    def get_next(self) -> dict | None:
        if self.next >= self.end:
            return None
        x = letterboxed(load_upright(self.paths[self.next]), self.imgsz)
        self.next += 1
        self.shapes.append(x.shape)
        return {self.input_name: x}


def quantize_int8(fp32: Path, out: Path, reader: CalibrationReader, method: str,
                  exclude: Sequence[str]) -> dict:
    """quantize_static: QDQ, int8 weights per channel, uint8 activations,
    calibrated by `method` ("minmax", or "percentile" at PERCENTILE) on
    `reader`'s photos one at a time, the nodes in `exclude` (the head's
    decoding) left in float. The float32 file's metadata is copied in.
    Returns the settings, for the manifest."""
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    methods = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}
    extra = {"CalibStridedMinMax": CALIBRATION_STRIDE}
    if method == "percentile":
        extra["CalibPercentile"] = PERCENTILE
    quantize_static(str(fp32), str(out), reader, quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=methods[method], nodes_to_exclude=list(exclude), extra_options=extra)
    model = onnx.load(str(out))
    copy_metadata(onnx.load(str(fp32), load_external_data=False), model)
    onnx.save(model, str(out))
    return {"format": "QDQ", "per_channel": True, "weights": "QInt8", "activations": "QUInt8",
            "method": method, "extra_options": extra, "nodes_excluded": len(exclude)}


# --- The manifest -----------------------------------------------------------------------

def file_entry(path: Path) -> dict:
    """What the manifest keeps of a model file."""
    import onnx

    model = onnx.load(str(path))

    def value(v) -> dict:
        t = v.type.tensor_type
        return {"name": v.name, "type": onnx.TensorProto.DataType.Name(t.elem_type),
                "shape": [d.dim_param or d.dim_value for d in t.shape.dim]}

    return {"sha256": sha256(path), "bytes": path.stat().st_size,
            "opset": {o.domain or "ai.onnx": o.version for o in model.opset_import},
            "ir_version": model.ir_version, "nodes": len(model.graph.node),
            "ops": dict(sorted(_count(n.op_type for n in model.graph.node).items())),
            "inputs": [value(v) for v in model.graph.input], "outputs": [value(v) for v in model.graph.output]}


def structure_problems(path: Path, fp32: Path) -> list[str]:
    """A derived file against the float32 file it was made from: the same
    inputs and outputs (names, float32, the dynamic axes by name) and the
    same metadata."""
    import onnx

    entry, reference, problems = file_entry(path), file_entry(fp32), []
    for side in ("inputs", "outputs"):
        if entry[side] != reference[side]:
            problems.append(f"{path.name}: {side} {entry[side]} differ from float32's {reference[side]}")
    if any(v["type"] != "FLOAT" for side in ("inputs", "outputs") for v in entry[side]):
        problems.append(f"{path.name}: an input or output isn't float32")
    meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    if meta != {p.key: p.value for p in onnx.load(str(fp32)).metadata_props}:
        problems.append(f"{path.name}: metadata differs from float32's")
    return problems


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def versions() -> dict[str, str | None]:
    found = {}
    for name in ("ultralytics", "torch", "onnx", "onnxruntime", "onnxslim", "numpy"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def read_manifest(folder: Path) -> dict:
    return runs.read_json(folder / MANIFEST) or {"files": {}}


def update_manifest(folder: Path, files: Mapping[str, dict], **fields) -> dict:
    """Add file entries (by variant) and top-level fields to `folder`'s
    manifest, made if missing; each update notes when, from which commit."""
    manifest = read_manifest(folder)
    manifest["files"] = {**manifest.get("files", {}), **files}
    manifest.update(fields)
    manifest.setdefault("updates", []).append(
        {"when": datetime.now().isoformat(timespec="seconds"), **runs.git_state(),
         "variants": sorted(files), "versions": versions()})
    runs.write_json(folder / MANIFEST, manifest)
    return manifest


def check_files(folder: Path, variants: Sequence[str]) -> list[str]:
    """Each variant's file in `folder` against its manifest entry: present,
    its SHA-256 and size the ones recorded."""
    manifest = read_manifest(folder)
    problems = []
    for variant in variants:
        entry, path = manifest["files"].get(variant), folder / FILES[variant]
        if entry is None:
            problems.append(f"{variant}: not in {folder / MANIFEST}")
        elif not path.exists():
            problems.append(f"{variant}: {path} is missing")
        elif (digest := sha256(path)) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            problems.append(f"{variant}: {path.name} isn't the file the manifest names ({digest[:8]}…)")
    return problems


# --- ONNX Runtime's threads ---------------------------------------------------------------

@contextmanager
def ort_threads(threads: int) -> Iterator[list[int]]:
    """Every ONNX Runtime session made inside runs `threads` threads (and
    one between operators). Neither Ultralytics' ONNX backend nor ONNX
    Runtime's calibrator sets a count, and the default, a thread per core
    of the whole node, fails to pin each inside a job's cgroup (CLAUDE.md,
    M3 S1). Yields the counts the sessions were made with, to check."""
    import onnxruntime

    made: list[int] = []
    original = onnxruntime.InferenceSession

    def session(path_or_bytes, sess_options=None, *args, **kwargs):
        options = sess_options or onnxruntime.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        made.append(options.intra_op_num_threads)
        return original(path_or_bytes, options, *args, **kwargs)

    onnxruntime.InferenceSession = session
    try:
        yield made
    finally:
        onnxruntime.InferenceSession = original


# --- The rules S1 reads (R0, R1, R2's accuracy half) ----------------------------------

def found(record: Mapping) -> list[bool]:
    """Each labelled date of a parity record's set, in order: found at IoU
    0.5 at the record's threshold."""
    return [c >= record["threshold"] for c in record["dates"]["conf"]]


def r0(pytorch: Mapping, onnx: Mapping) -> dict:
    """R0 on one set: the same dates found, each by both, and AP within R0_AP."""
    same = found(pytorch) == found(onnx)
    ap_diff = abs(onnx["ap50_date"] - pytorch["ap50_date"])
    return {"same_dates_found": same, "ap_diff": round(ap_diff, 6),
            "passed": same and round(ap_diff, 9) <= R0_AP}


def _no_worse(base: Mapping, other: Mapping, ap_drop: float, fewer: int) -> dict:
    n_base, n_other = sum(found(base)), sum(found(other))
    drop = base["ap50_date"] - other["ap50_date"]
    return {"ap_drop": round(drop, 6), "dates_found": [n_base, n_other],
            "passed": round(drop, 9) <= ap_drop and n_base - n_other <= fewer}


def r1(square: Mapping, tight: Mapping) -> dict:
    """R1 on one set: tight's AP no more than R1_AP under square's and at
    most R1_FEWER date fewer; on the store set, every expiry square finds,
    tight finds too."""
    out = _no_worse(square, tight, R1_AP, R1_FEWER)
    expiry = square["dates"]["expiry"]
    if any(expiry):
        lost = sum(e and s and not t for e, s, t in zip(expiry, found(square), found(tight)))
        out["expiries_lost"] = lost
        out["passed"] = out["passed"] and lost == 0
    return out


def r2_accuracy(fp32: Mapping, lower: Mapping) -> dict:
    """R2's accuracy half on one set: the lower precision's AP no more than
    R2_AP under float32's, at most R2_FEWER date fewer (its speed half is
    the phone's, S4)."""
    return _no_worse(fp32, lower, R2_AP, R2_FEWER)


def rule_g(still: Sequence[float], moving: Sequence[float]) -> dict:
    """Rule G: the stillness gate's threshold T, the geometric mean of the
    95th percentile of the still values and the 5th of the moving ones, if
    the first is below the second; otherwise the gate is off (T None)."""
    still_p95 = float(np.percentile(still, G_STILL))
    moving_p5 = float(np.percentile(moving, G_MOVING))
    on = still_p95 < moving_p5
    return {"on": on, "t": float(np.sqrt(still_p95 * moving_p5)) if on else None,
            "still_p95": still_p95, "moving_p5": moving_p5, "still": len(still), "moving": len(moving)}


def gate(record: Path | None = None, every: int | None = None) -> dict | None:
    """Rule G read off a Motion record (GATE_MOTION unless given) at the
    gate's cadence: the motion values against the frame `every` frames
    back, as the app posts them (value1, value2). None until a record is
    pinned. A pinned record is named as pinned: the config goes to the
    phone, and the laptop's own folders stay out of it."""
    name = None if record is None else str(record)
    if record is None:
        if GATE_MOTION is None:
            return None
        record, name = runs.RUNS_DIR / "m2c" / GATE_MOTION, GATE_MOTION
    every = GATE_EVERY if every is None else every
    frames = json.loads(Path(record).read_text())["frames"]
    values = {"still": [], "moving": []}
    for phase, v in zip(frames["phase"], frames[f"value{every}"]):
        if phase in values and v is not None:
            values[phase].append(v)
    return {**rule_g(values["still"], values["moving"]), "every": every, "record": name}


def environment() -> dict:
    """What the scoring ran on, beside the records."""
    return {"YOLO_AUTOINSTALL": os.environ.get("YOLO_AUTOINSTALL"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
            "versions": versions()}
