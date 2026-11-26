// The stillness gate (docs/m2c-live-boxes.md, "The frame loop"): the
// motion value on made-up 64 px copies, and the gate's timing on made-up
// sequences of values. Written before stillness.ts (Decision 14).
import { describe, expect, it } from "vitest";
import { CLEAR_MS, STILL_MS, StillnessGate, centredGrey, gateSize, motionValue } from "./stillness";

// --- made-up frames ----------------------------------------------------------------

const W = 64;
const H = 36;

/** A fixed pseudo-random sequence in [0, 1) (an LCG), so every run sees the same noise. */
function random(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** A pack-like scene at the gate's size: stripes, a gradient and a dark block,
 *  every grey level between 35 and 194 so a brightness change of 40 never clips. */
function scene(x: number, y: number): number {
  const block = x >= 20 && x < 40 && y >= 10 && y < 22 ? -45 : 0;
  return 115 + 35 * Math.sin(x / 3) + 20 * Math.cos(y / 2.5) + 0.4 * x + block;
}

/** The scene as RGBA bytes (grey: R = G = B), moved right by `dx` px, scaled
 *  by `gain` and `add` levels brighter, with uniform noise of +-`noise` levels
 *  a pixel. (Noise drawn per channel would be averaged down by the luma weights.) */
function frame({ dx = 0, add = 0, gain = 1, noise = 0, seed = 1 } = {}): Uint8ClampedArray {
  const rnd = random(seed);
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const grey = Math.round(gain * scene(x - dx, y) + add + noise * (2 * rnd() - 1));
      const i = 4 * (y * W + x);
      rgba.fill(grey, i, i + 3);
      rgba[i + 3] = 255;
    }
  }
  return rgba;
}

const value = (a: Uint8ClampedArray, b: Uint8ClampedArray) => motionValue(centredGrey(a), centredGrey(b));

// --- the copy's size --------------------------------------------------------------------

describe("gateSize", () => {
  it("is 64 px on the long side, the short side in proportion", () => {
    expect(gateSize(1920, 1080)).toEqual([64, 36]);
    expect(gateSize(1080, 1920)).toEqual([36, 64]);
    expect(gateSize(1280, 720)).toEqual([64, 36]);
    expect(gateSize(1600, 1200)).toEqual([64, 48]);
    expect(gateSize(640, 480)).toEqual([64, 48]);
  });

  it("keeps at least 1 px on the short side, and is empty before the video has a size", () => {
    expect(gateSize(4000, 10)).toEqual([64, 1]);
    expect(gateSize(0, 0)).toEqual([0, 0]);
  });
});

// --- grey, minus its mean ------------------------------------------------------------

describe("centredGrey", () => {
  it("weighs red, green and blue as luma (0.299, 0.587, 0.114), then takes the mean away", () => {
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 255]);
    const grey = [0.299 * 255, 0.587 * 255, 0.114 * 255, 0];
    const mean = grey.reduce((a, b) => a + b) / 4;
    const out = centredGrey(rgba);
    expect(out).toHaveLength(4);
    grey.forEach((g, i) => expect(out[i]).toBeCloseTo(g - mean, 4));
  });

  it("ignores alpha", () => {
    const a = centredGrey(new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]));
    const b = centredGrey(new Uint8ClampedArray([10, 20, 30, 0, 40, 50, 60, 7]));
    expect(a).toEqual(b);
  });

  it("writes into the array it is given, when it is given one", () => {
    const out = new Float32Array(2);
    expect(centredGrey(new Uint8ClampedArray([1, 1, 1, 255, 3, 3, 3, 255]), out)).toBe(out);
  });
});

// --- the motion value -------------------------------------------------------------------

describe("motionValue", () => {
  it("is 0 for the same frame", () => {
    expect(value(frame(), frame())).toBe(0);
  });

  it("is small for a still frame with noise: about 4/3 of a grey level for noise of +-2", () => {
    const v = value(frame({ noise: 2, seed: 1 }), frame({ noise: 2, seed: 2 }));
    expect(v).toBeGreaterThan(0.8);
    expect(v).toBeLessThan(2);
  });

  it("is large for a shift of a few pixels: 2 px is over 5 times the noise", () => {
    const noise = value(frame({ noise: 2, seed: 1 }), frame({ noise: 2, seed: 2 }));
    const shift = value(frame({ noise: 2, seed: 1 }), frame({ dx: 2, noise: 2, seed: 2 }));
    expect(shift).toBeGreaterThan(5 * noise);
    expect(value(frame(), frame({ dx: 1 }))).toBeGreaterThan(5); // even 1 px of the copy (30 of a 1080p frame)
  });

  it("isn't moved by a change of brightness alone (the camera's exposure)", () => {
    expect(value(frame(), frame({ add: 40 }))).toBeLessThan(0.01); // rounding to bytes only
    expect(value(frame(), frame({ add: -25 }))).toBeLessThan(0.01);
    const noise = value(frame({ noise: 2, seed: 1 }), frame({ noise: 2, seed: 2 }));
    const both = value(frame({ noise: 2, seed: 1 }), frame({ add: 40, noise: 2, seed: 2 }));
    expect(Math.abs(both - noise)).toBeLessThan(0.1 * noise);
  });

  it("is moved by a change of contrast (a gain), by the gain's change times the copy's mean deviation", () => {
    // Only the mean is taken away, as the plan says: a gain isn't. 10% more
    // contrast moves the value by about a tenth of the copy's mean deviation.
    const grey = centredGrey(frame());
    const deviation = grey.reduce((s, g) => s + Math.abs(g), 0) / grey.length;
    const v = value(frame(), frame({ gain: 1.1, add: -11.5 }));
    expect(v).toBeGreaterThan(0.07 * deviation); // rounding to bytes adds or takes a little
    expect(v).toBeLessThan(0.13 * deviation);
  });

  it("refuses copies of different sizes (the frame turned)", () => {
    expect(() => motionValue(new Float32Array(4), new Float32Array(6))).toThrow(/4.*6/);
  });
});

// --- the gate's timing -----------------------------------------------------------------

const T = 3;

/** Feeds the gate (value, time) pairs; returns its state after each. */
function feed(gate: StillnessGate, frames: [number | null, number][]) {
  return frames.map(([v, t]) => gate.update(v, t));
}

/** Frames every 33 ms from `from` until before `to`, each with value `v`. */
function run(v: number | null, from: number, to: number): [number | null, number][] {
  const out: [number | null, number][] = [];
  for (let t = from; t < to; t += 33) out.push([v, t]);
  return out;
}

describe("StillnessGate", () => {
  it("uses the plan's times: still after 150 ms, cleared after 400 ms of motion", () => {
    expect([STILL_MS, CLEAR_MS]).toEqual([150, 400]);
  });

  it("is still once the value has been under T for 150 ms, not before", () => {
    const gate = new StillnessGate(T);
    const states = feed(gate, [[1, 0], [1, 33], [1, 66], [1, 99], [1, 132], [1, 149], [1, 150], [1, 183]]);
    expect(states.map((s) => s.still)).toEqual([false, false, false, false, false, false, true, true]);
    expect(states.map((s) => s.motion)).toEqual(["settling", "settling", "settling", "settling", "settling", "settling", "still", "still"]);
  });

  it("compares strictly: a value of exactly T is motion", () => {
    const gate = new StillnessGate(T);
    expect(gate.update(T, 0).below).toBe(false);
    expect(gate.update(T - 1e-9, 33).below).toBe(true);
  });

  it("has no value for the first frame (no copy to compare), and that frame isn't still", () => {
    const gate = new StillnessGate(T);
    const [first, ...rest] = feed(gate, [[null, 0], [1, 33], [1, 183]]);
    expect(first.still).toBe(false);
    expect(rest.map((s) => s.still)).toEqual([false, true]); // 150 ms counted from the first value under T
  });

  it("goes from still to moving at one frame over T, and dims the boxes until still again", () => {
    const gate = new StillnessGate(T);
    feed(gate, run(1, 0, 300));
    const moving = gate.update(9, 300);
    expect([moving.still, moving.motion, moving.dim, moving.clear]).toEqual([false, "moving", true, false]);
    const settling = gate.update(1, 333);
    expect([settling.still, settling.motion, settling.dim]).toEqual([false, "settling", true]);
    expect(gate.update(1, 483).still).toBe(true);
  });

  it("clears the boxes after 400 ms of motion, not at 399", () => {
    const gate = new StillnessGate(T);
    feed(gate, run(1, 0, 300));
    feed(gate, [[9, 300], [9, 500]]);
    expect(gate.update(9, 699).clear).toBe(false);
    expect(gate.update(9, 700).clear).toBe(true);
  });

  it("never clears for a wobble shorter than 400 ms that settles to still", () => {
    const gate = new StillnessGate(T);
    feed(gate, run(1, 0, 300));
    const states = feed(gate, [...run(9, 300, 600), ...run(1, 600, 1200)]);
    expect(states.some((s) => s.clear)).toBe(false);
    expect(states.at(-1)!.still).toBe(true);
  });

  it("counts 400 ms of motion from its first frame, through dips under T shorter than 150 ms", () => {
    const gate = new StillnessGate(T);
    feed(gate, run(1, 0, 300));
    feed(gate, [...run(9, 300, 500), [1, 533], [1, 566]]); // a dip of 2 frames: settling, not still
    expect(gate.update(9, 699).clear).toBe(false);
    expect(gate.update(9, 700).clear).toBe(true);
  });

  it("doesn't clear while the picture settles after motion, though it isn't still yet", () => {
    const gate = new StillnessGate(T);
    feed(gate, run(1, 0, 300));
    feed(gate, run(9, 300, 650));
    expect(gate.update(1, 700).clear).toBe(false); // 400 ms after the motion began, but under T now
  });

  it("counts each clearing once, however long the motion lasts", () => {
    const gate = new StillnessGate(T);
    feed(gate, run(1, 0, 300));
    feed(gate, run(9, 300, 2000));
    expect(gate.update(9, 2000).clears).toBe(1);
    feed(gate, run(1, 2033, 2500));
    feed(gate, run(9, 2500, 3000));
    expect(gate.update(9, 3000).clears).toBe(2);
  });

  it("is always still when it is off (rule G found the values overlapping): no T, no dimming, no clearing", () => {
    const gate = new StillnessGate(null);
    const states = feed(gate, [[null, 0], [50, 33], ...run(99, 66, 1000)]);
    expect(states.every((s) => s.still && !s.dim && !s.clear)).toBe(true);
  });

  it("starts over after a reset (the page was hidden): not still for 150 ms, and in-flight work is stale", () => {
    const gate = new StillnessGate(T);
    feed(gate, run(1, 0, 300));
    const before = gate.update(1, 300).clears;
    gate.reset();
    const after = gate.update(1, 5000);
    expect(after.still).toBe(false);
    expect(after.clears).toBe(before + 1);
    expect(gate.update(1, 5150).still).toBe(true);
  });
});
