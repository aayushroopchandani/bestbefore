// The letterbox against Ultralytics 8.4.177's own LetterBox and scale_boxes
// (fixtures from scripts/live_fixtures.py; docs/m2c-live-boxes.md, Decision 12).
import { describe, expect, it } from "vitest";
import fixtures from "./__fixtures__/letterbox.json";
import { IMGSZ, PAD_VALUE, STRIDE, letterboxParams, pyRound, scaleBoxes, toTensor } from "./letterbox";

describe("pyRound", () => {
  it("rounds halves to the even integer, as Python's round() does", () => {
    expect([0.5, 1.5, 2.5, 3.5, 500.5, -0.5, -1.5, -2.5].map(pyRound)).toEqual([0, 2, 2, 4, 500, 0, -2, -2]);
  });
  it("rounds everything else to the nearest integer", () => {
    expect([0.4, 0.6, 279.9, 280.1, 1.4999, -0.6, 7].map(pyRound)).toEqual([0, 1, 280, 280, 1, -1, 7]);
  });
  it("gives 0, never JavaScript's -0, as Python's integers have no sign of zero", () => {
    expect(Object.is(pyRound(-0.1), 0)).toBe(true);
    expect(Object.is(pyRound(-0.5), 0)).toBe(true);
  });
});

describe("letterboxParams", () => {
  it("uses the fixtures' settings", () => {
    expect([fixtures.imgsz, fixtures.stride]).toEqual([IMGSZ, STRIDE]);
    expect(PAD_VALUE).toBe(114);
  });

  for (const c of fixtures.cases) {
    const [w, h] = c.frame;
    it(`${w} x ${h} ${c.tight ? "tight" : "square"}: Ultralytics' size and pads`, () => {
      const lb = letterboxParams(w, h, c.tight);
      expect([lb.newWidth, lb.newHeight]).toEqual(c.new_unpad);
      expect([lb.top, lb.bottom, lb.left, lb.right]).toEqual([c.top, c.bottom, c.left, c.right]);
      expect([lb.inputWidth, lb.inputHeight]).toEqual(c.input);
      expect([lb.frameWidth, lb.frameHeight]).toEqual([w, h]);
    });
  }

  it("never shrinks the long side below the model's size: 1280 x 720 isn't scaled", () => {
    const lb = letterboxParams(1280, 720, false);
    expect([lb.newWidth, lb.newHeight, lb.inputWidth, lb.inputHeight]).toEqual([1280, 720, 1280, 1280]);
  });
});

describe("scaleBoxes", () => {
  for (const c of fixtures.cases) {
    const [w, h] = c.frame;
    it(`${w} x ${h} ${c.tight ? "tight" : "square"}: boxes back to the frame, clipped`, () => {
      const [iw, ih] = c.input;
      const mapped = scaleBoxes(c.model_boxes, iw, ih, w, h);
      expect(mapped).toHaveLength(c.frame_boxes.length);
      mapped.forEach((box, i) => box.forEach((v, k) => expect(v).toBeCloseTo(c.frame_boxes[i][k], 3)));
    });
  }
});

describe("toTensor", () => {
  it("makes RGBA pixels RGB, channels first, 0 to 1 in float32", () => {
    // 2 x 1 pixels: (255, 0, 114, a) and (1, 128, 254, a); alpha is dropped
    const rgba = new Uint8ClampedArray([255, 0, 114, 255, 1, 128, 254, 7]);
    const x = toTensor(rgba, 2, 1);
    expect(x).toBeInstanceOf(Float32Array);
    const f = (v: number) => Math.fround(v / 255);
    expect(Array.from(x)).toEqual([1, f(1), 0, f(128), f(114), f(254)]);
  });

  it("writes into a given array, so a loop needn't allocate one a frame", () => {
    const out = new Float32Array(3);
    expect(toTensor(new Uint8ClampedArray([51, 102, 153, 255]), 1, 1, out)).toBe(out);
    expect(Array.from(out)).toEqual([Math.fround(0.2), Math.fround(0.4), Math.fround(0.6)]);
  });
});
