// Boxes from the frame on to the screen, where the video is shown whole
// (object-fit: contain) in a view of another shape.
import { describe, expect, it } from "vitest";
import { contain, toView } from "./fit";

describe("contain", () => {
  it("fits an upright 1080 x 1920 frame in a phone's 390 x 844 view: full width, centred in height", () => {
    const fit = contain(1080, 1920, 390, 844);
    expect(fit.scale).toBeCloseTo(390 / 1080, 12);
    expect(fit.x).toBe(0);
    expect(fit.y).toBeCloseTo((844 - 1920 * (390 / 1080)) / 2, 9); // 75.33 px above and below
  });

  // (1000 - 1920 * (1000 / 1920)) is -6e-14 in floating point, not 0: compared to 9 decimals.
  it("fits a 1920 x 1080 frame in a 1000 x 1000 view: full width", () => {
    const fit = contain(1920, 1080, 1000, 1000);
    expect(fit.scale).toBe(1000 / 1920);
    expect(fit.x).toBeCloseTo(0, 9);
    expect(fit.y).toBeCloseTo(218.75, 9);
  });

  it("fits a 1920 x 1080 frame in a wide 2000 x 800 view: full height, centred in width", () => {
    const fit = contain(1920, 1080, 2000, 800);
    expect(fit.scale).toBe(800 / 1080);
    expect(fit.x).toBeCloseTo((2000 - 1920 * (800 / 1080)) / 2, 9);
    expect(fit.y).toBeCloseTo(0, 9);
  });
});

describe("toView", () => {
  it("maps the frame's corners to the shown video's corners", () => {
    const fit = contain(1080, 1920, 390, 844);
    const [x1, y1, x2, y2] = toView({ x1: 0, y1: 0, x2: 1080, y2: 1920 }, fit);
    expect([x1, y1]).toEqual([0, fit.y]);
    expect(x2).toBeCloseTo(390, 9);
    expect(y2).toBeCloseTo(844 - fit.y, 9);
  });

  it("maps a box by the same scale and offset", () => {
    const fit = { scale: 0.5, x: 10, y: 20 };
    expect(toView({ x1: 100, y1: 40, x2: 300, y2: 80 }, fit)).toEqual([60, 40, 160, 60]);
  });
});
