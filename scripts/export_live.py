"""M2c S1: the M2b detector as ONNX files (docs/m2c-live-boxes.md, "The model files").

    python scripts/export_live.py float                       # the laptop, once
    python scripts/export_live.py int8 [--photos 200] [--out DIR] [--threads N]
    python scripts/export_live.py check [--variant fp32 ...]

`float` (the laptop, with the live extra): last.pt checked against m2b.py
and copied into $CHECKPOINT_DIR/m2c/, then det-fp32.onnx (Ultralytics'
export: dynamic height and width, onnxslim, its own opset), det-fp16.onnx
and det-fp16-head32.onnx (the Detect head's decoding kept float32), each
loaded by ONNX Runtime and its structure checked, and manifest.json.

`int8` (the CPU job; the laptop's smoke with --photos 8 --out .../smoke):
det-int8.onnx (MinMax) and det-int8-pct.onnx (Percentile 99.999) from the
float32 file, calibrated on training photos only (live.calibration_photos),
letterboxed tight, one at a time; the head's decoding never quantised.
Their entries, the calibration list's hash and the time and memory taken
go into --out's manifest.

`check`: the files in $CHECKPOINT_DIR/m2c/ against the manifest, and the
source weights against m2b.py: the job's first step, after the copy.

Ultralytics never installs anything here: YOLO_AUTOINSTALL is false before
it is imported, so a missing package fails loudly.
"""
from __future__ import annotations

import os

os.environ["YOLO_AUTOINSTALL"] = "false"     # before anything imports Ultralytics
os.environ.setdefault("YOLO_OFFLINE", "1")

import argparse
import platform
import resource
import sys
import time
from pathlib import Path


def peak_rss_gb() -> float:
    """This process's peak resident memory so far (macOS counts bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (2**30 if sys.platform == "darwin" else 2**20), 2)


def structure_problems(path: Path, fp32: Path) -> list[str]:
    """A derived file against the float32 file: it loads in ONNX Runtime,
    keeps float32 inputs and outputs with the same names and dynamic axes,
    and carries the same metadata (live.structure_problems)."""
    import onnxruntime

    from bestbefore import live

    problems = live.structure_problems(path, fp32)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    loaded = {v.name: v.type for v in [*session.get_inputs(), *session.get_outputs()]}
    if set(loaded.values()) != {"tensor(float)"}:
        problems.append(f"{path.name}: ONNX Runtime sees {loaded}")
    return problems


def make_float() -> None:
    import onnx

    from bestbefore import live

    folder = live.models_dir()
    weights, digest = live.source_weights(folder)
    print(f"source: m2b-{live.SOURCE}'s last.pt, sha256 {digest}, copied to {weights}")
    t0 = time.perf_counter()
    fp32 = live.export_fp32(weights, folder / live.FILES["fp32"])
    exported_s = time.perf_counter() - t0
    model = onnx.load(str(fp32))
    nodes = live.decode_nodes(model)
    print(f"the Detect head is module {live.head_module(model)}; its decoding is {len(nodes)} nodes")
    problems, merged = [], {}
    for variant, keep in (("fp16", ()), ("fp16-head32", nodes)):
        half = live.to_float16(model, keep)
        problems += [f"{variant}: {p}" for p in live.float16_problems(half, keep, model)]
        merged[variant] = sorted(set(keep) - {n.name for n in half.graph.node})
        onnx.save(half, str(folder / live.FILES[variant]))
    entries = {v: live.file_entry(folder / live.FILES[v]) for v in live.FLOAT_FILES}
    problems += [p for v in ("fp16", "fp16-head32") for p in structure_problems(folder / live.FILES[v], fp32)]
    if problems:
        sys.exit("\n".join(problems))
    by_op = {}
    for name in nodes:
        op = next(n.op_type for n in model.graph.node if n.name == name)
        by_op[op] = by_op.get(op, 0) + 1
    live.update_manifest(
        folder, entries,
        source={"run": f"m2b-{live.SOURCE}", "weights_sha256": digest, "copy": weights.name},
        export={"imgsz": live.IMGSZ, "dynamic": True, "simplify": True, "half": False,
                "opset_chosen_by": "Ultralytics (best_onnx_opset)", "seconds": round(exported_s, 1),
                "peak_rss_gb": peak_rss_gb(), "YOLO_AUTOINSTALL": os.environ["YOLO_AUTOINSTALL"],
                "machine": f"{platform.system()} {platform.machine()}"},
        head={"module": live.head_module(model), "decode_nodes": nodes, "decode_ops": dict(sorted(by_op.items())),
              "casts_merged_in_fp16_head32": merged["fp16-head32"]},
        float16={"how": "onnxruntime.transformers OnnxModel.convert_float_to_float16 (its float16 converter, "
                        "then Cast chains and same-type Casts removed)", "keep_io_types": True})
    for v, e in entries.items():
        print(f"{live.FILES[v]}: {e['bytes'] / 1e6:.1f} MB, opset {e['opset']}, {e['nodes']} nodes, sha256 {e['sha256'][:12]}…")


def make_int8(photos: int, out: Path, threads: int) -> None:
    import onnx

    from bestbefore import live

    folder = live.models_dir()
    if problems := live.check_files(folder, ["fp32"]):
        sys.exit("\n".join(problems))
    fp32 = folder / live.FILES["fp32"]
    nodes = live.decode_nodes(onnx.load(str(fp32)))
    total = sum(live.CALIBRATION.values())
    store = round(photos * live.CALIBRATION["store"] / total)
    picked = live.calibration_photos(store, photos - store)
    print(f"calibration: {store} store and {photos - store} ExpDate training photos, "
          f"list sha256 {live.list_sha256(picked)[:12]}…; {len(nodes)} decoding nodes left in float; {threads} threads")
    out.mkdir(parents=True, exist_ok=True)
    entries, settings, shapes = {}, {}, None
    with live.ort_threads(threads) as made:
        for variant, method in (("int8", "minmax"), ("int8-pct", "percentile")):
            reader = live.CalibrationReader([p for _, p in picked])
            t0 = time.perf_counter()
            settings[variant] = live.quantize_int8(fp32, out / live.FILES[variant], reader, method, nodes)
            settings[variant] |= {"seconds": round(time.perf_counter() - t0, 1), "peak_rss_gb_so_far": peak_rss_gb()}
            if len(reader.shapes) != len(picked):
                sys.exit(f"{variant}: the calibrator read {len(reader.shapes)} photos of {len(picked)}")
            if shapes is not None and reader.shapes != shapes:
                sys.exit(f"{variant}: the photos came in other shapes than for int8")
            shapes = reader.shapes
            entries[variant] = live.file_entry(out / live.FILES[variant])
            print(f"{live.FILES[variant]}: {method}, {settings[variant]['seconds']} s, "
                  f"{entries[variant]['bytes'] / 1e6:.1f} MB, {entries[variant]['nodes']} nodes, "
                  f"peak {settings[variant]['peak_rss_gb_so_far']} GB so far")
    if set(made) != {threads}:
        sys.exit(f"ONNX Runtime sessions made with {made} threads, not {threads} each")
    loose = [s for s in shapes if s[2] % live.STRIDE or s[3] % live.STRIDE or max(s[2:]) != live.IMGSZ]
    if loose:
        sys.exit(f"calibration inputs not letterboxed tight to {live.IMGSZ}: {loose[:3]}")
    if problems := [p for v in live.INT8_FILES for p in structure_problems(out / live.FILES[v], fp32)]:
        sys.exit("\n".join(problems))
    live.update_manifest(out, entries, int8={
        "fp32_sha256": live.read_manifest(folder)["files"]["fp32"]["sha256"],
        "calibration": {"photos": len(picked), "store": store, "expdate": photos - store, "seed": live.SEED,
                        "list_sha256": live.list_sha256(picked), "tight": True, "imgsz": live.IMGSZ,
                        "stride": live.CALIBRATION_STRIDE,
                        "shapes": sorted({f"{s[2]}x{s[3]}" for s in shapes})},
        "threads": threads, "sessions_made": made, "settings": settings,
        "machine": f"{platform.system()} {platform.machine()}", "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID")})


def check(variants: list[str]) -> None:
    from bestbefore import live, m2b

    weights, run = m2b.last_pt(live.SOURCE)     # raises unless its SHA-256 is m2b.py's
    problems = live.check_files(live.models_dir(), variants)
    if problems:
        sys.exit("\n".join(problems))
    print(f"{run}'s weights ({weights}) and {', '.join(live.FILES[v] for v in variants)}: as the manifest says")


def main() -> None:
    from bestbefore import live

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("float")
    q = sub.add_parser("int8")
    q.add_argument("--photos", type=int, default=sum(live.CALIBRATION.values()))
    q.add_argument("--out", type=Path, default=None, help="default: $CHECKPOINT_DIR/m2c")
    q.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    c = sub.add_parser("check")
    c.add_argument("--variant", action="append", choices=list(live.FILES))
    args = p.parse_args()
    if args.cmd == "float":
        make_float()
    elif args.cmd == "int8":
        make_int8(args.photos, args.out or live.models_dir(), args.threads)
    else:
        check(args.variant or list(live.FLOAT_FILES))


if __name__ == "__main__":
    main()
