// Percentiles as numpy takes them (linear between the closest ranks), so
// the app's summaries match what Python computes from the same records.
import { describe, expect, it } from "vitest";
import { median, percentile } from "./stats";

describe("percentile", () => {
  it("interpolates between ranks as numpy.percentile does by default", () => {
    // numpy.percentile([1, 2, 3, 4], 95) == 3.85; ([1..10], 5) == 1.45; ([7], 50) == 7
    expect(percentile([1, 2, 3, 4], 95)).toBeCloseTo(3.85, 12);
    expect(percentile([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 5)).toBeCloseTo(1.45, 12);
    expect(percentile([7], 50)).toBe(7);
    expect(percentile([3, 1, 2], 0)).toBe(1);
    expect(percentile([3, 1, 2], 100)).toBe(3);
  });

  it("doesn't reorder the array it is given", () => {
    const xs = [3, 1, 2];
    percentile(xs, 50);
    expect(xs).toEqual([3, 1, 2]);
  });

  it("is NaN for no values", () => {
    expect(percentile([], 50)).toBeNaN();
  });
});

describe("median", () => {
  it("is the middle value, or the mean of the two middle ones", () => {
    expect(median([5, 1, 3])).toBe(3);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([])).toBeNaN();
  });
});
