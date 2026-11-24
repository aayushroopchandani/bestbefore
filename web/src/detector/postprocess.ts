/**
 * The model's output to boxes: each anchor's best class, the confidence
 * threshold, and NMS per class. Ultralytics 8.4.177's non_max_suppression
 * as its predictor runs it (torchvision's NMS, IoU 0.7, 300 boxes at most),
 * step for step, so the app keeps the boxes M2b scored (tested against
 * fixtures it made, postprocess.test.ts).
 *
 * Torch works in float32 where JavaScript works in float64, so the two
 * comparisons that decide a box are made the way torch makes them (see
 * `conf` and `iou` below). Box coordinates are float32 values too.
 *
 * No React and no app state: this folder is the part Scanify can take.
 */

export const IOU = 0.7; // Ultralytics' prediction default
export const MAX_DET = 300; // boxes kept at most
export const MAX_NMS = 30000; // candidates NMS looks at, the most confident

/** One box in the model's input pixels: xyxy, its class and confidence. */
export interface Box {
  cls: number;
  conf: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface PostprocessOptions {
  conf: number; // a box is kept if its best class scores over this
  iou?: number;
  maxDet?: number;
  maxNms?: number;
}

/** Boxes from `output`, the model's [1, 4 + classes, anchors] (centre x,
 *  centre y, width, height, then one score a class), row by row as ONNX
 *  Runtime returns it. Most confident first; on a tie, the earlier anchor. */
export function postprocess(output: Float32Array, anchors: number, classes: number,
  { conf, iou = IOU, maxDet = MAX_DET, maxNms = MAX_NMS }: PostprocessOptions): Box[] {
  if (output.length !== (4 + classes) * anchors) {
    throw new Error(`the output holds ${output.length} values, not (4 + ${classes}) x ${anchors} = ${(4 + classes) * anchors}`);
  }
  const f = Math.fround;
  // Torch compares a float32 score with Python's conf in float32: a score
  // of exactly 0.3 (float32) isn't over a conf of 0.3.
  const confF = f(conf);

  // Each anchor's best class (the first, on a tie, as torch's max), kept if over conf.
  const candidates: { anchor: number; cls: number; conf: number }[] = [];
  for (let a = 0; a < anchors; a++) {
    let best = output[4 * anchors + a];
    let cls = 0;
    for (let c = 1; c < classes; c++) {
      const s = output[(4 + c) * anchors + a];
      if (s > best) {
        best = s;
        cls = c;
      }
    }
    if (best > confF) candidates.push({ anchor: a, cls, conf: best });
  }
  // Most confident first (a stable sort: ties stay in anchor order), and
  // only the first maxNms of them, as Ultralytics cuts before its NMS.
  candidates.sort((p, q) => q.conf - p.conf);
  if (candidates.length > maxNms) candidates.length = maxNms;

  // Greedy NMS: a box is kept unless a kept box of its class overlaps it by
  // over `iou`. Ultralytics does all classes in one NMS by shifting each
  // class 7,680 px apart; the shift is kept here, in float32, so every IoU
  // is the very number torchvision computes. Kept boxes come out most
  // confident first, so the first maxDet are final as soon as they're kept.
  const kept: Box[] = [];
  const shifted: { x1: number; y1: number; x2: number; y2: number; area: number }[] = [];
  for (const { anchor: a, cls, conf: score } of candidates) {
    const cx = output[a];
    const cy = output[anchors + a];
    const hw = f(output[2 * anchors + a] / 2);
    const hh = f(output[3 * anchors + a] / 2);
    const box = { cls, conf: score, x1: f(cx - hw), y1: f(cy - hh), x2: f(cx + hw), y2: f(cy + hh) };
    const shift = cls * 7680;
    const s = { x1: f(box.x1 + shift), y1: f(box.y1 + shift), x2: f(box.x2 + shift), y2: f(box.y2 + shift), area: 0 };
    s.area = f(f(s.x2 - s.x1) * f(s.y2 - s.y1));
    if (!shifted.some((k) => overlap(k, s) > iou)) {
      kept.push(box);
      shifted.push(s);
      if (kept.length === maxDet) break;
    }
  }
  return kept;
}

/** torchvision's IoU (its CPU kernel), in float32 step by step. Its result
 *  is a float32 compared with the threshold as a double: an IoU of exactly
 *  0.7 is 0.69999999 in float32, so it doesn't remove a box at IoU 0.7. */
function overlap(a: { x1: number; y1: number; x2: number; y2: number; area: number },
  b: { x1: number; y1: number; x2: number; y2: number; area: number }): number {
  const f = Math.fround;
  const w = Math.max(0, f(Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)));
  const h = Math.max(0, f(Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1)));
  const inter = f(w * h);
  return f(inter / f(f(a.area + b.area) - inter));
}
