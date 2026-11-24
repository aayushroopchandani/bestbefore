"""M2c S2: Ultralytics' own answers for the app's letterbox, NMS and box mapping
(docs/m2c-live-boxes.md, Decision 12), as fixtures for Vitest.

    python scripts/live_fixtures.py [--out web/src/detector/__fixtures__]

Writes two JSON files of numbers only (no photo):

letterbox.json  for each frame size, tight and square: LetterBox's resized
                size, its four pads and the input's shape (get_params, the
                input checked by running LetterBox on a blank frame); and
                scale_boxes on boxes in the model's input, some reaching into
                the padding, mapped back to the frame and clipped.
nms.json        non_max_suppression on made-up model outputs [1, 4 + 4, N]
                with planted boxes: overlapping, two classes, IoU exactly at
                the threshold, tied scores, best class under conf, more boxes
                than max_det; and one output of 33,600 anchors (a 1,280 square)
                with every score over conf, so the 30,000-box cut acts. That
                one is made by a formula the TypeScript repeats (`hashed`),
                not stored.

The functions are the ones predict() runs (Ultralytics 8.4.177): LetterBox
with Ultralytics' padding of 114, non_max_suppression with torchvision's
NMS (predict imports torchvision, so Ultralytics uses it), scale_boxes as
construct_result calls it. Boxes stay in the model's input pixels, as
non_max_suppression returns them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "web" / "src" / "detector" / "__fixtures__"

IMGSZ, STRIDE, IOU, MAX_DET, MAX_NMS = 1280, 32, 0.7, 300, 30000
# The plan's five frames (a 16:9 camera both ways, a 4:3 photo, 720p, VGA),
# then shapes whose rounding is a trap: halves Python's round() takes to the
# even side (1001 x 2560 scales to 500.5 px), odd pads split by Ultralytics'
# -0.1/+0.1, and the bench photos' own sizes.
FRAMES = [(1920, 1080), (1080, 1920), (1600, 1200), (1280, 720), (640, 480),
          (1001, 2560), (2560, 1001), (1279, 1280), (1280, 1277), (3, 2),
          (756, 1008), (1008, 756), (899, 1599), (1600, 900), (956, 1275)]


def hashed(i: int) -> float:
    """A fixed value in [0, 1) for index i, exact in float64 in Python and
    JavaScript alike (Knuth's multiplicative hash; i * 2654435761 stays
    under 2^53 for i under 3.3 million)."""
    return ((i * 2654435761) % 4294967296) / 4294967296


def letterbox_cases():
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops

    cases = []
    for w, h in FRAMES:
        for tight in (False, True):
            box = LetterBox((IMGSZ, IMGSZ), auto=tight, stride=STRIDE)
            p = box.get_params({"img": np.zeros((h, w, 3), np.uint8)})
            out = box(image=np.zeros((h, w, 3), np.uint8))
            ih, iw = out.shape[:2]
            if (iw, ih) != (p["new_unpad"][0] + p["left"] + p["right"], p["new_unpad"][1] + p["top"] + p["bottom"]):
                raise SystemExit(f"{w}x{h}: LetterBox's input isn't its params' size")
            # Boxes in the input: one inside the photo, one across the
            # top-left padding, one past the bottom-right, one degenerate.
            model_boxes = [[p["left"] + 10.5, p["top"] + 20.25, p["left"] + 200.0, p["top"] + 64.75],
                           [0.0, 0.0, p["left"] + 30.0, p["top"] + 40.0],
                           [iw - 50.0, ih - 40.0, iw + 10.0, ih + 5.0],
                           [iw / 2, ih / 2, iw / 2, ih / 2]]
            mapped = ops.scale_boxes((ih, iw), torch.tensor(model_boxes, dtype=torch.float32), (h, w))
            cases.append({"frame": [w, h], "tight": tight,
                          "new_unpad": [int(v) for v in p["new_unpad"]],
                          "top": int(p["top"]), "bottom": int(p["bottom"]),
                          "left": int(p["left"]), "right": int(p["right"]),
                          "input": [iw, ih], "ratio": float(p["ratio"][0]),
                          "model_boxes": model_boxes, "frame_boxes": mapped.tolist()})
    return cases


def planted(columns: list[list[float]], n: int, base: int = 0) -> list[list[float]]:
    """A [4 + 4, n] output: the given anchors (cx, cy, w, h, four class
    scores) first, the rest zero (under any conf)."""
    out = [[0.0] * n for _ in range(8)]
    for a, col in enumerate(columns):
        for r in range(8):
            out[r][base + a] = col[r]
    return out


def nms_cases():
    import torch
    import torchvision  # noqa: F401  predict imports it, so non_max_suppression uses its NMS
    from ultralytics.utils import nms

    def anchor(cx, cy, w, h, scores):
        return [cx, cy, w, h, *scores]

    cases = {
        # Two same-class boxes at IoU 0.81 (the weaker goes); the same pair
        # as different classes (both stay); a third far away.
        "overlap": [anchor(100, 100, 40, 20, [0.9, 0, 0, 0]), anchor(102, 100, 40, 20, [0.8, 0, 0, 0]),
                    anchor(300, 300, 40, 20, [0, 0.6, 0, 0])],
        "two_classes": [anchor(100, 100, 40, 20, [0.9, 0, 0, 0]), anchor(102, 100, 40, 20, [0, 0.8, 0, 0])],
        # IoU exactly 0.7 (a 10 x 7 box inside a 10 x 10 one): kept, since
        # NMS removes only IoU over the threshold. Then 0.71: removed.
        "iou_at_threshold": [anchor(105, 105, 10, 10, [0.9, 0, 0, 0]), anchor(105, 103.5, 10, 7, [0.8, 0, 0, 0])],
        "iou_over_threshold": [anchor(105, 105, 10, 10, [0.9, 0, 0, 0]), anchor(105, 103.55, 10, 7.1, [0.8, 0, 0, 0])],
        # Equal scores: apart, both stay; overlapping, which one stays is the
        # sort's tie order.
        "ties_apart": [anchor(100, 100, 20, 20, [0.5, 0, 0, 0]), anchor(400, 100, 20, 20, [0.5, 0, 0, 0])],
        "ties_overlapping": [anchor(100, 100, 20, 20, [0.5, 0, 0, 0]), anchor(101, 100, 20, 20, [0.5, 0, 0, 0])],
        # Under conf: a box whose best score is under; one exactly at conf
        # (kept only if over); one whose best class is the last.
        "under_conf": [anchor(100, 100, 20, 20, [0.2, 0.25, 0, 0]), anchor(300, 100, 20, 20, [0.3, 0, 0, 0]),
                       anchor(500, 100, 20, 20, [0.1, 0.2, 0.3, 0.31])],
        # Best class picks the class; a lower class score of the same box
        # never makes a second box (multi_label is off).
        "best_class": [anchor(100, 100, 20, 20, [0.4, 0.7, 0.6, 0.1])],
    }
    out = []
    for name, cols in cases.items():
        # Never 6 anchors: non_max_suppression takes an output whose last
        # axis is 6 for an end-to-end model's [1, 300, 6] and only filters it.
        n = len(cols) + 10
        conf = 0.3 if name == "under_conf" else 0.25
        out.append(_nms_case(name, planted(cols, n), conf, nms, torch))
    # More boxes than max_det: 320 apart, scores falling, all over conf.
    many = [anchor(20 + 40 * (i % 30), 20 + 40 * (i // 30), 10, 10, [0.9 - i * 0.002, 0, 0, 0]) for i in range(320)]
    out.append(_nms_case("max_det", planted(many, 320), 0.001, nms, torch))
    # The 30,000-box cut: a 1,280 square's 33,600 anchors, every score over
    # conf. The first 30,000 score high and pile up in the middle (NMS keeps
    # about one a class); the last 3,600 score low and sit apart on a grid,
    # so they are kept only if the cut is missing. Made by `cut_output`,
    # which the TypeScript repeats.
    case = _nms_case("max_nms", cut_output(), 0.001, nms, torch)
    case.pop("output")
    case["output_formula"] = "cut_output"
    out.append(case)
    return out


def cut_output(n: int = 33600, high: int = MAX_NMS) -> list[list[float]]:
    out = [[0.0] * n for _ in range(8)]
    for a in range(n):
        u = [hashed(8 * a + k) for k in range(8)]
        if a < high:
            out[0][a], out[1][a] = 640 + 4 * u[0], 640 + 4 * u[1]
            out[2][a], out[3][a] = 200 + 4 * u[2], 100 + 4 * u[3]
            scores = [0.5 + 0.49 * v for v in u[4:]]
        else:
            k = a - high
            out[0][a], out[1][a] = 10 + 21 * (k % 60), 10 + 21 * (k // 60)
            out[2][a] = out[3][a] = 8 + 4 * u[2]
            scores = [0.002 + 0.4 * v for v in u[4:]]
        for c in range(4):
            out[4 + c][a] = scores[c]
    return out


def _nms_case(name, output, conf, nms, torch):
    pred = torch.tensor([output], dtype=torch.float32)
    kept = nms.non_max_suppression(pred, conf, IOU, max_det=MAX_DET, max_nms=MAX_NMS, nc=0)[0]
    return {"name": name, "conf": conf, "iou": IOU, "anchors": len(output[0]), "output": output,
            "kept": [[float(v) for v in row] for row in kept.tolist()]}


def main() -> None:
    import ultralytics

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    made = {"ultralytics": ultralytics.__version__, "imgsz": IMGSZ, "stride": STRIDE,
            "iou": IOU, "max_det": MAX_DET, "max_nms": MAX_NMS}
    (args.out / "letterbox.json").write_text(json.dumps({**made, "cases": letterbox_cases()}, indent=1) + "\n")
    cases = nms_cases()
    (args.out / "nms.json").write_text(json.dumps({**made, "cases": cases}, separators=(",", ":")) + "\n")
    for c in cases:
        print(f"{c['name']:20} {c['anchors']:6} anchors, conf {c['conf']}: {len(c['kept'])} kept")
    print(args.out)


if __name__ == "__main__":
    main()
