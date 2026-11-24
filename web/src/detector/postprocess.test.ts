// Best class, the threshold and NMS against Ultralytics 8.4.177's own
// non_max_suppression with torchvision's NMS (fixtures from
// scripts/live_fixtures.py; docs/m2c-live-boxes.md, Decision 12).
import { describe, expect, it } from "vitest";
import fixtures from "./__fixtures__/nms.json";
import { IOU, MAX_DET, MAX_NMS, postprocess } from "./postprocess";

/** The fixtures' `hashed`: a fixed value in [0, 1) for index i, exact in float64. */
function hashed(i: number): number {
  return ((i * 2654435761) % 4294967296) / 4294967296;
}

/** The fixtures' `cut_output`: 30,000 high-scoring boxes piled up, 3,600 low ones apart. */
function cutOutput(n = 33600, high = 30000): Float32Array {
  const out = new Float32Array(8 * n);
  for (let a = 0; a < n; a++) {
    const u = Array.from({ length: 8 }, (_, k) => hashed(8 * a + k));
    let scores: number[];
    if (a < high) {
      [out[a], out[n + a], out[2 * n + a], out[3 * n + a]] = [640 + 4 * u[0], 640 + 4 * u[1], 200 + 4 * u[2], 100 + 4 * u[3]];
      scores = u.slice(4).map((v) => 0.5 + 0.49 * v);
    } else {
      const k = a - high;
      out[a] = 10 + 21 * (k % 60);
      out[n + a] = 10 + 21 * Math.floor(k / 60);
      out[2 * n + a] = out[3 * n + a] = 8 + 4 * u[2];
      scores = u.slice(4).map((v) => 0.002 + 0.4 * v);
    }
    scores.forEach((s, c) => (out[(4 + c) * n + a] = s));
  }
  return out;
}

describe("postprocess", () => {
  it("uses Ultralytics' prediction settings", () => {
    expect([IOU, MAX_DET, MAX_NMS]).toEqual([fixtures.iou, fixtures.max_det, fixtures.max_nms]);
  });

  for (const c of fixtures.cases) {
    it(`${c.name}: the boxes Ultralytics keeps, in its order`, () => {
      const output = "output" in c ? Float32Array.from((c.output as number[][]).flat()) : cutOutput();
      const kept = postprocess(output, c.anchors, 4, { conf: c.conf });
      expect(kept.map((b) => [b.x1, b.y1, b.x2, b.y2, b.conf, b.cls])).toEqual(c.kept);
    });
  }

  it("takes the output as ONNX Runtime returns it: [1, 4 + classes, anchors], row by row", () => {
    // one anchor of three: centre (50, 40), 20 x 10, class 2 at 0.9
    const out = new Float32Array([0, 50, 0, /**/ 0, 40, 0, /**/ 0, 20, 0, /**/ 0, 10, 0,
      0, 0.1, 0, /**/ 0, 0.2, 0, /**/ 0, 0.9, 0, /**/ 0, 0.3, 0]);
    expect(postprocess(out, 3, 4, { conf: 0.25 })).toEqual([{ cls: 2, conf: Math.fround(0.9), x1: 40, y1: 35, x2: 60, y2: 45 }]);
  });

  it("refuses an output whose length isn't (4 + classes) x anchors", () => {
    expect(() => postprocess(new Float32Array(10), 3, 4, { conf: 0.25 })).toThrow(/24/);
  });
});
