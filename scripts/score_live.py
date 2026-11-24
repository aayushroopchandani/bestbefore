"""M2c: the app's parity benches against Ultralytics (docs/m2c-live-boxes.md, "Records and scores").

    python scripts/score_live.py RECORD [RECORD ...] [--reference RUN]

Each RECORD is a bench-parity-*.json the app posted (scripts/live_api.py).
Its boxes, every one at conf 0.001 by set and index, are scored with M2b's
own code (detection_eval): AP@0.5 for dates, and dates found at IoU 0.5 at
M2b's threshold_store (expiries too on v1-val). Then box by box against the
CPU job's Ultralytics ONNX float32 record of the same photos (--reference,
default S1's job run; parity-<set>-fp32-square.json, square as the app pads),
rescored from its own boxes on the same photos by the same code:
each box at the threshold on either side matched to a box of its class on
the other at IoU >= 0.9, and how far its confidence moved.

S2's check (the plan's S2): the app finds every date the reference finds at
the threshold but at most one, with AP within 0.01. Records of one browser,
backend and set at float32 and at a float16 variant also get R2's accuracy
half (live.r2_accuracy: AP no more than 0.01 under float32's, at most one
date fewer).

Writes RUN/scores.json (RUN: the records' folder) and prints a table.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "results" / "runs" / "m2c" / "20261123-165009-m2c-parity"
MATCH_IOU = 0.9            # a box "the same" as the reference's
CHECK_AP, CHECK_FEWER = 0.01, 1


def predictions(photos: list[dict], samples) -> dict:
    """The record's boxes as detection_eval's Predictions, keyed by sample id
    (photo i is sample i of the set, as the API serves them)."""
    from bestbefore.detection_eval import Prediction
    from bestbefore.yolo import CLASSES

    if [p["index"] for p in photos] != list(range(len(photos))):
        raise ValueError("a parity record must hold the set's photos 0, 1, 2, ... in order")
    return {samples[p["index"]].id: [Prediction(CLASSES[int(b[0])], tuple(b[1:5]), b[5]) for b in p["boxes"]]
            for p in photos}


def scored(record: dict, samples) -> dict:
    """AP and dates found, in a parity record's shape (live.r2_accuracy reads it)."""
    from bestbefore.detection_eval import average_precision, evaluate, summary

    samples = samples[:len(record["photos"])]
    at_05 = evaluate(samples, predictions(record["photos"], samples), 0.5)
    return {"set": record["set"], "variant": f"app-{record['backend']}-{record['variant']}",
            "threshold": record["threshold"], "photos": len(samples), "ap50_date": average_precision(at_05),
            "at_threshold": summary(at_05, record["threshold"]),
            "dates": {"conf": at_05.truth_conf, "expiry": at_05.truth_expiry}}


def _iou(a, b) -> float:
    from bestbefore.detection_eval import iou

    return iou(a, b)


def box_by_box(app: list[dict], reference: list[dict], threshold: float) -> dict:
    """Each box at the threshold on one side matched, greedily by confidence,
    to an unmatched box of its class on the other side at IoU >= MATCH_IOU,
    whatever that one's confidence. Counts the reference's boxes the app has
    and hasn't, the app's own the reference hasn't, and the confidence moves
    (app minus reference) of every matched pair."""
    kept = {"reference": 0, "app": 0}
    matched, missed, extra, moves, ious = 0, 0, 0, [], []
    for a, r in zip(app, reference, strict=True):
        pairs = {}
        for side, mine, other in (("reference", r["boxes"], a["boxes"]), ("app", a["boxes"], r["boxes"])):
            taken = set()
            for i, b in sorted(enumerate(mine), key=lambda x: -x[1][5]):
                if b[5] < threshold:
                    continue
                kept[side] += 1
                best, best_iou = None, MATCH_IOU
                for j, o in enumerate(other):
                    if j not in taken and o[0] == b[0] and (v := _iou(b[1:5], o[1:5])) >= best_iou:
                        best, best_iou = j, v
                if best is None:
                    if side == "reference":
                        missed += 1
                    else:
                        extra += 1
                    continue
                taken.add(best)
                key = (i, best) if side == "reference" else (best, i)      # (reference box, app box)
                if key not in pairs:
                    pairs[key] = (best_iou, (other[best][5] - b[5]) if side == "reference" else (b[5] - other[best][5]))
        for v, move in pairs.values():
            matched += 1
            ious.append(v)
            moves.append(move)
    absolute = sorted(abs(m) for m in moves)
    return {"iou": MATCH_IOU, "kept": kept, "pairs": matched, "reference_missing_in_app": missed,
            "app_missing_in_reference": extra,
            "conf_move": {"median": statistics.median(moves) if moves else None,
                          "median_abs": statistics.median(absolute) if moves else None,
                          "p95_abs": absolute[int(0.95 * (len(absolute) - 1))] if moves else None,
                          "max_abs": absolute[-1] if moves else None,
                          "over_0.02": sum(m > 0.02 for m in absolute)},
            "iou_min": min(ious) if ious else None, "iou_median": statistics.median(ious) if ious else None}


def s2_check(app: dict, reference: dict) -> dict:
    """The app finds every date the reference finds at the threshold but at
    most CHECK_FEWER, with AP within CHECK_AP."""
    from bestbefore import live

    ref_found, app_found = live.found(reference), live.found(app)
    lost = sum(r and not a for r, a in zip(ref_found, app_found, strict=True))
    ap_diff = app["ap50_date"] - reference["ap50_date"]
    return {"dates_found": [sum(ref_found), sum(app_found)], "lost": lost,
            "gained": sum(a and not r for r, a in zip(ref_found, app_found)), "ap_diff": round(ap_diff, 6),
            "passed": lost <= CHECK_FEWER and round(abs(ap_diff), 9) <= CHECK_AP}


def pixel_summary(photos: list[dict]) -> dict:
    decoded = [p["decoded_equal"] for p in photos if "decoded_equal" in p]
    checks = [p["pixel_check"] for p in photos if "pixel_check" in p]
    return {"decoded_equal": [sum(decoded), len(decoded)],
            "input_checks": len(checks),
            "input_max": max((c["max"] for c in checks), default=None),
            "input_mean": statistics.fmean(c["mean"] for c in checks) if checks else None,
            "input_differing": statistics.fmean(c["differing"] for c in checks) if checks else None}


def device_key(record: dict) -> str:
    return json.dumps([record["device"].get("userAgent"), record["backend"], record["set"]])


def score_records(paths: list[Path], reference_run: Path) -> dict:
    from bestbefore import live, sets

    out, by_device, samples = {}, {}, {}
    for path in paths:
        record = json.loads(path.read_text())
        if record.get("kind") != "bench-parity":
            raise SystemExit(f"{path.name} isn't a parity bench")
        name = record["set"]
        if name not in samples:
            samples[name] = sets.load_set(name)
        app = scored(record, samples[name])
        job = json.loads((reference_run / f"parity-{name}-fp32-square.json").read_text())
        n = len(record["photos"])
        # The reference rescored from its own boxes on the same photos: on a
        # whole set it must give the job's AP (its boxes are kept to 5 decimals).
        reference = scored({"set": name, "backend": "ultralytics", "variant": "fp32-square",
                            "threshold": job["threshold"], "photos": job["per_photo"][:n]}, samples[name])
        whole = n == job["photos"]
        out[path.name] = {"set": name, "backend": record["backend"], "variant": record["variant"], "photos": n,
                          "whole_set": whole, "browser": record["device"].get("userAgent"),
                          "model_sha256": record["model"]["sha256"],
                          "ap50_date": app["ap50_date"], "at_threshold": app["at_threshold"],
                          "reference": {"run": reference_run.name, "ap50_date": reference["ap50_date"],
                                        "at_threshold": reference["at_threshold"],
                                        **({"job_ap50_date": job["ap50_date"]} if whole else {})},
                          "check": s2_check(app, reference),
                          "box_by_box": box_by_box(record["photos"], job["per_photo"][:n], record["threshold"]),
                          "pixels": pixel_summary(record["photos"]), "median_ms": record.get("median_ms")}
        by_device.setdefault(device_key(record), {})[record["variant"]] = (path.name, app)
    # R2's accuracy half, float16 against float32 in the same browser, backend and set.
    for variants in by_device.values():
        if "fp32" not in variants:
            continue
        fp32_name, fp32 = variants["fp32"]
        for v, (name, app) in variants.items():
            if v != "fp32" and len(app["dates"]["conf"]) == len(fp32["dates"]["conf"]):
                out[name]["R2_accuracy"] = {"against": fp32_name, **live.r2_accuracy(fp32, app)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("records", nargs="+", type=Path)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    args = p.parse_args()
    out = score_records(args.records, args.reference)
    folders = {r.parent for r in args.records}
    if len(folders) == 1:
        target = folders.pop() / "scores.json"
        previous = json.loads(target.read_text()) if target.exists() else {}
        target.write_text(json.dumps({**previous, **out}, indent=1) + "\n")
        print(target)
    for name, s in out.items():
        t, rt, b, c = s["at_threshold"], s["reference"]["at_threshold"], s["box_by_box"], s["check"]
        if s["whole_set"] and abs(s["reference"]["ap50_date"] - s["reference"]["job_ap50_date"]) > 1e-6:
            print(f"  note: the reference rescored gives AP {s['reference']['ap50_date']:.6f}, "
                  f"the job recorded {s['reference']['job_ap50_date']:.6f}")
        expiry = f" exp {t['expiry_found']}/{t['expiry_dates']}" if "expiry_dates" in t else ""
        m = b["conf_move"]
        print(f"{s['set']:9} {s['backend']:6} {s['variant']:11} {s['photos']:3} photos  AP {s['ap50_date']:.4f} "
              f"(ref {s['reference']['ap50_date']:.4f})  dates {t['dates_found']}/{t['dates']} (ref {rt['dates_found']})"
              f"{expiry}  check {'passed' if c['passed'] else 'FAILED'} (lost {c['lost']}, gained {c['gained']})"
              f" | boxes {b['pairs']} paired, {b['reference_missing_in_app']} ref-only, "
              f"{b['app_missing_in_reference']} app-only, |conf move| median {m['median_abs']:.4f} max {m['max_abs']:.4f}"
              f" | decoded as PIL {s['pixels']['decoded_equal']}"
              + (f" | R2 {s['R2_accuracy']['passed']} ({s['R2_accuracy']['ap_drop']:+.4f}, {s['R2_accuracy']['dates_found']})"
                 if "R2_accuracy" in s else ""))


if __name__ == "__main__":
    main()
